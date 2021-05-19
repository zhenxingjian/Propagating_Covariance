import argparse
import numpy as np
import time
import os
import pdb

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from Layers_Geometry import RobustLoss, Gaussian_Noise, Certify_Gaussian
from Place_model import PreActResNet18
from rs.certify import certify

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PLACES365_LOC_ENV = "../places365_standard/"

_PLACES365_MEAN = [0.485, 0.456, 0.406]
_PLACES365_STDDEV = [0.229, 0.224, 0.225]

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_dataset(split):
    """Return the dataset as a PyTorch Dataset object"""
    return _places365(split)

def get_num_classes():
    """Return the number of classes in the dataset. """
    return 365


def get_normalize_layer():
    return NormalizeLayer(_PLACES365_MEAN, _PLACES365_STDDEV)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
def _places365(split):

    dir = PLACES365_LOC_ENV
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return datasets.ImageFolder(subdir, transform)


def Places365_train(sigma, lbd, gamma, num_classes, model, trainloader, optimizer, device):
    NoiseDistr = Gaussian_Noise(sigma).to(device)
    rloss = RobustLoss(lbd, gamma)
    cl_total = 0.0
    rl_total = 0.0
    for _, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        inputR, inputSPD = NoiseDistr(inputs)

        if np.random.rand() > 1.0:
            # need to dynamically change to value above between 1.0 and 0.0, from small value to large one
            inputR = inputs # train the model to fine-tune


        inputSPD = inputSPD.to(device)

        outputR, outputSPD = model(inputR, inputSPD)


        loss, loss_C, loss_R = rloss(outputR, outputSPD, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cl_total += loss_C.item()
        rl_total += loss_R.item()

    print('Classification Loss: {}  Robustness Loss: {}'.format(cl_total, rl_total))


def Places365_certify(sigma, num_classes, model, testloader, device):
    print('===certify(sigma={})'.format(sigma))
    NoiseDistr = Gaussian_Noise(sigma).to(device)
    model.eval()
    certify_gauss = Certify_Gaussian(sigma)
    cl_total = 0.0
    Radius = []
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, inputSPD = NoiseDistr(inputs)
            inputR = inputs

            inputSPD = inputSPD.to(device)

            outputR, outputSPD = model(inputR, inputSPD)

            acc, radius = certify_gauss(outputR, outputSPD, targets)

            cl_total += acc.sum()
            Radius.append(radius)
        Radius = torch.cat(Radius,dim=0)
        # pdb.set_trace()
        print('Classification Acc: {}  Robustness Radius: {}'.format(cl_total.float()/len(Radius), (Radius[Radius>0]).sum()/len(Radius)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Places365 demo')
    parser.add_argument('--task', default='train',
                                  type=str, help='Task: train or test')
    parser.add_argument('--root', default='data', type=str, help='Dataset path')
    parser.add_argument('--dataset', default='Places365', type=str, help='Dataset')
    parser.add_argument('--resume_ckpt', default='./Pretrain/Places365.pth', type=str,
                                  help='Checkpoint path to resume')
    parser.add_argument('--ckptdir', default='./data/Places_model/', type=str,
                                  help='Checkpoints save directory')
    parser.add_argument('--matdir', default='./data/Places_mat/', type=str,
                                  help='Matfiles save directory')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                                  help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=120,
                                  type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')

    # params for train
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--sigma', default=0.25, type=float,
                                  help='Standard variance of gaussian noise (also used in test)')
    parser.add_argument('--lbd', default=0.0, type=float,
                                  help='Weight of robustness loss')
    parser.add_argument('--gamma', default=8.0, type=float,
                      help='Hinge factor')

    # params for test
    parser.add_argument('--start_img', default=0,
                                  type=int, help='Image index to start (choose it randomly)')
    parser.add_argument('--num_img', default=365, type=int,
                                  help='Number of test images')
    parser.add_argument('--skip', default=100, type=int,
                                  help='Number of skipped images per test image')

    args = parser.parse_args()

    ckptdir = None if args.ckptdir == 'none' else args.ckptdir
    matdir = None if args.matdir == 'none' else args.matdir
    if matdir is not None and not os.path.isdir(matdir):
      os.makedirs(matdir)
    if ckptdir is not None and not os.path.isdir(ckptdir):
      os.makedirs(ckptdir)
    checkpoint = None if args.resume_ckpt == 'none' else args.resume_ckpt


    model = PreActResNet18()

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')

    pin_memory = True
    train_loader = DataLoaderX(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoaderX(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=pin_memory)


    num_classes = 365

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(
          model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(
          optimizer, milestones=[30, 60, 90], gamma=0.1)



    # Resume from checkpoint if required
    start_epoch = 0
    if checkpoint is not None:
        print('==> Resuming from checkpoint..')
        print(checkpoint)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['net'])
        start_epoch = 0#checkpoint['epoch']
        scheduler.step(start_epoch)


    # Main routine
    if args.task == 'train':
      # Training routine
      for epoch in range(start_epoch + 1, args.epochs + 1):
          print('===train(epoch={})==='.format(epoch))
          t1 = time.time()
          scheduler.step()
          model.train()

          lbd_ = 0.0 if epoch <= 32 else args.lbd

          # lbd_ = args.lbd

          Places365_train(args.sigma, lbd_, args.gamma, num_classes, model, train_loader, optimizer, device)

          t2 = time.time()
          print('Elapsed time: {}'.format(t2 - t1))


          if ckptdir is not None:
              # Save checkpoint
              print('==> Saving {}.pth..'.format(epoch))
              try:
                state = {
                      'net': model.state_dict(),
                      'epoch': epoch,
                }
                torch.save(state, '{}/{}.pth'.format(ckptdir, epoch))
              except OSError:
                print('OSError while saving {}.pth'.format(epoch))
                print('Ignoring...')

          if epoch % 20 == 5:# and epoch >= 20:
            # Certify test
            print('===test(epoch={})==='.format(epoch))
            t1 = time.time()
            model.eval()
            Places365_certify(args.sigma, num_classes, model, test_loader, device)

            certify(model, device, test_dataset, None, num_classes, batch=500,
                mode='hard', start_img=args.start_img, num_img=args.num_img, 
                sigma=args.sigma, matfile=(None if matdir is None else os.path.join(matdir, '{}.mat'.format(epoch))))


            t2 = time.time()
            print('Elapsed time: {}'.format(t2 - t1))

    else:
        # Test routine
        certify(model, device, test_dataset, None, num_classes, batch=1500,
                mode='hard', start_img=args.start_img, num_img=args.num_img, 
                sigma=args.sigma, matfile=(None if matdir is None else os.path.join(matdir, '{}.mat'.format(checkpoint['epoch']))), verbose=True)
