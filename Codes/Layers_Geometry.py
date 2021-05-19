from __future__ import division

import math
import torch
import numpy as np
import time

from scipy import special

import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.distributions.normal import Normal

import warnings
from collections import OrderedDict
from torch._six import container_abcs
from itertools import islice
import operator

import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

rmax = 0.1

std=torch.Tensor([0.229, 0.224, 0.225]).cuda()

class FunctionModifiers(object):
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """
    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from a exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = \
        "if this method is not scripted, copy the python method onto the scripted model"

def _copy_to_script_wrapper(fn):
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
    return fn


class Conv2D_Geometry(nn.Module):
    '''
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values 'zeros' and 'circular' Default: 'zeros'
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If "True", adds a learnable bias to the output. Default: "True"

    Shape:
        - InputR: :math:'(N, C_{in}, H_{in}, W_{in})'
        - InputSPD: :math:'(N, C_{in}, C_{in})'
        - OutputR: :math:'(N, C_{out}, H_{out}, W_{out})' 
        - OutputSPD: :math:'(N, C_{out}, C_{out})' where

          .. math::
              H_{out} = leftlfloorfrac{H_{in}  + 2 times text{padding}[0] - text{dilation}[0]
                        times (text{kernel_size}[0] - 1) - 1}{text{stride}[0]} + 1rightrfloor

          .. math::
              W_{out} = leftlfloorfrac{W_{in}  + 2 times text{padding}[1] - text{dilation}[1]
                        times (text{kernel_size}[1] - 1) - 1}{text{stride}[1]} + 1rightrfloor
    '''
    def __init__(self, in_channels, out_channels, kernel_size, 
                       stride=1, padding=0, dilation=1, groups=1, bias=True, 
                       padding_mode='zeros', independent = False):
        super(Conv2D_Geometry, self).__init__()

        self.k = kernel_size
        self.p = padding

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.independent = independent

        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, inputR, inputSPD):

        kernel_size = self.k
        weight_R = self.weight
        weight_SPD = self.weight.detach() 
        out_channels = self.out_channels
        in_channels = self.in_channels



        Batch_size, in_channels_, H_in, W_in = inputR.shape

        assert in_channels_ == self.in_channels, 'The input channel size does NOT match the Conv2d size.'
        
        outputR = F.conv2d(inputR, weight_R, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)

        if inputSPD is None:
            return outputR, inputSPD

        midSPD = torch.matmul(inputSPD.unsqueeze(1).unsqueeze(1), weight_SPD.permute([2,3,1,0])) # data x W
        outputSPD = torch.matmul(midSPD.permute([0,1,2,4,3]), weight_SPD.permute([2,3,1,0])) # W' x data
        outputSPD = outputSPD.sum([1,2]) # sum with the moving windows
        outputSPD = outputSPD.permute([0,2,1])

        return outputR, outputSPD * (1.+rmax)


class Linear_Geometry(nn.Module):
    '''
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        in_space (int): Size of the input spacial
        out_space (int): Size of the output spacial
        bias (bool, optional): If "True", adds a learnable bias to the output. Default: "True"
    Shape:
        - InputR: :math:'(N, C_{in})'
        - InputSPD: :math:'(N, C_{in}, C_{in})'
        - OutputR: :math:'(N, C_{out})' 
        - OutputR: :math:'(N, C_{out}, C_{out})' where
          .. math::
              C_{in} = in_channels
              C_{out} = out_channels
    '''
    def __init__(self, in_features, out_features, bias=True, independent = False):
        super(Linear_Geometry, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.independent = independent
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, inputR, inputSPD):
        outputR = F.linear(inputR, self.weight, self.bias)

        if inputSPD is None:
            return outputR, inputSPD
        weight_SPD = self.weight.detach() 
        outputSPD = F.linear( F.linear(inputSPD, weight_SPD, bias = None).permute([0,2,1]), weight_SPD, bias = None).permute([0,2,1])

        return outputR, outputSPD


class First_Linear_Geometry(nn.Module):
    '''
    This is the first linear layer (fully connected)
    unlike the reshape and padding proceedure, this is exactly the same with Conv2D without padding
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        bias (bool, optional): If "True", adds a learnable bias to the output. Default: "True"

    Shape:
        - InputR: :math:'(N, C_{in}, H_{in}, W_{in})', where (H_{in}, W_{in}) = kernel_size
        - InputSPD: :math:'(N, C_{in}, C_{in}, H_{in}, W_{in})'
        - OutputR: :math:'(N, C_{out})' 
        - OutputSPD: :math:'(N, C_{out}, C_{out})' where
    '''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, independent = False):
        super(First_Linear_Geometry, self).__init__()
        padding = 0
        self.p = padding

        kernel_size = _pair(kernel_size)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.independent = independent

        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, inputR, inputSPD):
        kernel_size = self.kernel_size
        param = inputR.shape
        assert param[-2] == kernel_size[0] and param[-1] == kernel_size[1], 'The First Linear layer requires the weights has the same size with the data.'
        weight_R = self.weight
        weight_SPD = self.weight.detach() 
        out_channels = self.out_channels
        in_channels = self.in_channels

        Batch_size, in_channels_, H_in, W_in = inputR.shape
        assert in_channels_ == self.in_channels, 'The input channel size does NOT match the Conv2d size.'
        
        outputR = F.conv2d(inputR, weight_R, self.bias, padding = self.padding)
        outputR = outputR.view([param[0], out_channels])

        if inputSPD is None:
            return outputR, inputSPD

        Batch_size, out_channels_ = outputR.shape

        top2 = torch.topk(outputR, 2)

        top2_idx = top2[1]

        weight_SPD = weight_SPD[top2_idx]
        midSPD = torch.matmul(inputSPD.unsqueeze(1).unsqueeze(1), weight_SPD.permute([0,3,4,2,1]))
        outputSPD = torch.matmul(midSPD.permute([0,1,2,4,3]), weight_SPD.permute([0,3,4,2,1]))

        outputSPD = outputSPD.sum([1,2])

        return outputR, outputSPD


class ReLU_Geometry(nn.Module):
    '''
    ReLU activation on R x SPD data
    outputR = E(ReLU(X)) = 0.5 mu - 0.5 erf(-mu/ (sqrt(2) * sigma)) + 1/ (sqrt(2 pi)) * sigma * exp(-mu^2/(2 sigma^2))
    outputSPD = inputSPD + sigma * |inputR|
    Args:
        inplace: can optionally do the operation in-place. Default: "False"

    Shape:
        - InputR: :math:'(N, *, channels)' where '*' means, any number of additional dimensions
        - InputSPD: :math:'(N, *, channels, channels)' where '*' means, any number of additional dimensions
        - OutputR: :math:'(N, *, channels)', same shape as the inputR
        - OutputSPD: :math:'(N, *, channels, channels)', same shape as the inputSPD

    Examples::

        >>> m = nn.ReLU()
        >>> inputR = torch.randn(50, 128, 128, 16)
        >>> inputSPD = torch.randn(50, 128, 128, 16, 16)
        >>> outputR, outputSPD = m(inputR, inputSPD)
    '''
    __constants__ = ['inplace']

    def __init__(self, lbd = 0.1, inplace=False):
        super(ReLU_Geometry, self).__init__()
        self.lbd = lbd
        self.inplace = inplace

    def forward(self, inputR, inputSPD):
        # return F.relu(inputR) , inputSPD
        # From experiment, the above also works
        if inputSPD is None:
            return F.relu(inputR) , inputSPD
        param = inputR.shape
        Batch_size, in_channels_ = param[0], param[1]

        mu = inputR
        sigma = torch.sqrt(inputSPD[:,range(in_channels_), range(in_channels_),...])


        if len(mu.shape) == 4:
            sigma = sigma.unsqueeze(-1).unsqueeze(-1)
        
        outputSPD = inputSPD + self.lbd * torch.eye(in_channels_).to(device)

        outputR = 0.5 * mu - 0.5 * mu * torch.erf(-mu/(np.sqrt(2) * sigma)) + 1./ (np.sqrt(2*np.pi)) * sigma * torch.exp(-mu**2/(2*sigma**2))
        return outputR, outputSPD


class AvePool2D_Geometry(nn.Module):
    '''
    The average pooling for 2d images.

    Args:
        kernel_size: the size of the window

    Shape:
        - InputR: :math:'(N, C, H_{in}, W_{in})'
        - InputSPD: :math:'(N, C, C, H_{in}, W_{in})'
        - OutputR: :math:'(N, C, H_{out}, W_{out})'
        - OutputSPD: :math:'(N, C, C, H_{out}, W_{out})' where

          .. math::
              H_{out} = frac{H_{in}}{text{kernel_size}}}

          .. math::
              W_{out} = frac{W_{in}}{text{kernel_size}}}

    Examples::

        >>> m = AvePool2D_R_SPD(3)
        >>> inputR = torch.randn(20, 16, 50, 32)
        >>> inputSPD = torch.randn(20, 16, 16, 50, 32)
        >>> outputR, outputSPD = m(inputR, inputSPD)
        >>> print(outputR.shape)
        torch.Size([20, 16, 17, 11])
        >>> print(outputSPD.shape)
        torch.Size([20, 16, 16, 17, 11])

    '''
    def __init__(self, kernel_size = 2, independent = True):
        super(AvePool2D_Geometry, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.independent = independent
        
    def forward(self, inputR, inputSPD):
        self.weight = (Parameter(torch.ones([self.kernel_size, self.kernel_size], dtype = torch.float32), 
                                 requires_grad=False)/(self.kernel_size**2)).unsqueeze(0).unsqueeze(0).to(device)
        Batch_size, in_channels_, H_in, W_in = inputR.shape

        weight_R = self.weight
        if self.independent:
            weight_SPD = self.weight**2
        else:
            weight_SPD = torch.abs(self.weight)
        weight_SPD = weight_SPD.detach() 

        outputR = F.conv2d(inputR.reshape([Batch_size * in_channels_, 1, H_in, W_in]), weight_R, bias=None, stride = self.stride)

        _, _, H_out, W_out = outputR.shape
        outputR = outputR.reshape([Batch_size, in_channels_, H_out, W_out])
        if inputSPD is None:
            return outputR, inputSPD
        return outputR, inputSPD * weight_SPD.sum()


class RobustLoss(nn.Module):
    '''
    Compute the rubustness loss
    composed by two parts: classification loss and the robustness loss
    classification loss is: CrossEntropyLoss
    robustness loss is: - certified radius using equation

    Args:
        lbd: the balance between the classification loss and the robustness loss
    '''
    def __init__(self, lbd = 12., Gamma = 8.0):
        super(RobustLoss, self).__init__()
        self.lbd = lbd
        self.Gamma = Gamma
    def forward(self, inputR, inputSPD, label):
        outputs_softmax = F.softmax(inputR, dim=1)
        outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
        loss1 = F.nll_loss(
            outputs_logsoftmax, label, reduction='sum')

        ##################################
        top2 = torch.topk(inputR, 2)
        top2_score = top2[0]
        top2_idx = top2[1]

        indices_correct = (top2_idx[:, 0] == label)

        out0, out1 = top2_score[indices_correct,
                                0], top2_score[indices_correct, 1]

        
        Sigma11 = inputSPD[indices_correct,0,0]
        Sigma22 = inputSPD[indices_correct,1,1]

        Sigma12 = inputSPD[indices_correct,0,1]


        loss2 = F.relu((out1-out0)/torch.sqrt(Sigma11 + Sigma22 + 2 * torch.abs(Sigma12)) + self.Gamma)
        loss2 = loss2.sum()

        beta = 16.0
        m = Normal(torch.tensor([0.0]).to(device),
                   torch.tensor([1.0]).to(device))
        beta_outputs_softmax = F.softmax(inputR * beta, dim = 1)
        top2 = torch.topk(beta_outputs_softmax, 2)
        top2_score = top2[0]
        top2_idx = top2[1]
        indices_correct = (top2_idx[:, 0] == label)  # G_theta

        out0, out1 = top2_score[indices_correct,
                                0], top2_score[indices_correct, 1]
        robustness_loss = m.icdf(out1) - m.icdf(out0)
        indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
            robustness_loss) & (torch.abs(robustness_loss) <= self.Gamma)  # hinge
        out0, out1 = out0[indices], out1[indices]
        robustness_loss = m.icdf(out1) - m.icdf(out0) + self.Gamma
        robustness_loss = robustness_loss.sum() 

        Batch_size, _ = inputR.shape

        loss1 = loss1 / Batch_size
        loss2 = (loss2 + robustness_loss) / Batch_size
        return loss1 + self.lbd * loss2, loss1, loss2

class Certify_Gaussian(nn.Module):
    '''
    Compute the rubustness loss
    composed by two parts: classification loss and the robustness loss
    classification loss is: CrossEntropyLoss
    robustness loss is: - certified radius using equation

    Args:
        lbd: the balance between the classification loss and the robustness loss
    '''
    def __init__(self, sigma):
        super(Certify_Gaussian, self).__init__()
        self.sigma = sigma
    def forward(self, inputR, inputSPD, label):

        top2_score, top2_idx = torch.topk(inputR, 2)
        indices_correct = (top2_idx[:, 0] == label)
        radius = torch.ones_like(indices_correct).float()*(-1)

        out0, out1 = top2_score[indices_correct, 0], top2_score[indices_correct, 1]

        Batch_size, _ = inputR.shape

        Sigma11 = inputSPD[indices_correct,0,0]
        Sigma22 = inputSPD[indices_correct,1,1]

        Sigma12 = inputSPD[indices_correct,0,1]

        loss2 = self.sigma * (out0-out1)/torch.sqrt(Sigma11 + Sigma22 + 2 * torch.abs(Sigma12))
        radius[indices_correct] = loss2

        return indices_correct, radius


class Gaussian_Noise(nn.Module):
    '''
    Generate the noise distribution based on the input size

    Args:
        sigma: the noise energy sigma, n ~ N(mu, sigma ** 2 * I)
    '''
    def __init__(self, sigma = 0.1):
        super(Gaussian_Noise, self).__init__()
        self.sigma = sigma

    def forward(self, inputR):
        Batch, channels, Win, Hin = inputR.shape
        inputR = inputR + torch.randn(Batch, channels, Win, Hin).to(device) * self.sigma / std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        outputSPD_np = np.tile(np.eye(channels).reshape(1,channels,channels),(Batch ,1,1))
        outputSPD_np = outputSPD_np.reshape([Batch, channels, channels]) # [Batch, channels, channels, W_in, H_in]
        outputSPD = torch.from_numpy(outputSPD_np).float().cuda() * (torch.diag(self.sigma/ std)**2)
        return inputR, outputSPD


def Reshape_independent(inputR, inputSPD):
    '''
    inputR: Batch x channel x W x H
    inputSPD: Batch x channel x channel x W x H
    reshape inputR to be Batch x C_S_
    reshape inputSPD to be Batch x C_S_ x C_S_
    '''
    Batch_size, channel, W, H = inputR.shape
    outputR = inputR.reshape([Batch_size, channel, W * H]).permute([0,2,1]).reshape([Batch_size, -1]) # batch, spatial * channel

    if inputSPD is None:
        return outputR, inputSPD

    tempSPD = inputSPD.reshape([Batch_size, channel, channel, W * H, 1])
    tempSPD = torch.cat([tempSPD, torch.zeros([Batch_size, channel, channel, W * H, W * H], dtype = torch.float32).to(device)], dim = 4)
    tempSPD = tempSPD.reshape([Batch_size, channel, channel, W * H +1, W * H])
    tempSPD = tempSPD[:,:,:,:W*H,:]
    outputSPD = tempSPD.permute([0,3,1,4,2]).reshape([Batch_size, W * H * channel, W * H * channel])

    return outputR, outputSPD


class Sequential(nn.Module):
    """A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, inputR, inputSPD):
        for module in self:
            inputR, inputSPD = module(inputR, inputSPD)
        return inputR, inputSPD
    pass


class BatchNorm_Geometry(nn.Module):
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm_Geometry, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
    def forward(self, inputR, inputSPD):
        return inputR, inputSPD
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:

            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  
                    exponential_average_factor = self.momentum

        E_inputR = inputR.mean([0,2,3])
        Std_inputR = inputR.std([0,2,3])
        self.running_mean = exponential_average_factor * self.running_mean + (1-exponential_average_factor)*E_inputR
        self.running_var = exponential_average_factor * self.running_var + (1-exponential_average_factor)*Std_inputR

        self.running_mean = self.running_mean.detach()
        self.running_var = self.running_var.detach()
        outputR = (inputR - self.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))/(self.running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+self.eps)
        outputR = outputR * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1)

        SPD_kernel = torch.matmul((self.weight/(self.running_var + self.eps)).unsqueeze(-1),(self.weight/self.running_var).unsqueeze(-1).transpose(1,0))
        outputSPD = inputSPD/SPD_kernel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


        return outputR, outputSPD


if __name__ == '__main__':
    # inputR = torch.randn([64, 7,15,15])
    # Gn = Gaussian_Noise()
    # outputR, outputSPD = Gn(inputR)
    # # pdb.set_trace()
    # print(outputSPD.shape)


    # inputR = torch.randn([64, 7,15,15])
    # inputSPD = torch.randn([64, 7,7])
    # inputSPD = inputSPD + inputSPD.permute([0,2,1])
    # conv2d = Conv2D_Geometry(7,11,3,padding = 1, bias = False)
    # outputR, outputSPD = conv2d(inputR, inputSPD)
    # print(outputSPD.shape)

    # inputR = torch.randn([64,121])
    # inputSPD = torch.randn([64, 121, 121])
    # inputSPD = inputSPD + inputSPD.permute([0,2,1])
    # linear = Linear_Geometry(121,55)
    # outputR, outputSPD = linear(inputR, inputSPD)
    # print(outputSPD.shape)

    # inputR = torch.randn([64, 7,15,15])
    # inputSPD = torch.randn([64, 7,7])
    # inputSPD = (inputSPD + inputSPD.permute([0,2,1]))**2
    # relu = ReLU_Geometry()
    # outputR, outputSPD = relu(inputR, inputSPD)
    # print(outputSPD.shape)

    # inputR = torch.randn([64, 7])
    # inputSPD = torch.randn([64, 7,7])
    # inputSPD = (inputSPD + inputSPD.permute([0,2,1]))**2
    # relu = ReLU_Geometry()
    # outputR, outputSPD = relu(inputR, inputSPD)
    # print(outputSPD.shape)

    # inputR = torch.randn([64, 7,15,15])
    # inputSPD = torch.randn([64, 7,7])
    # inputSPD = inputSPD + inputSPD.permute([0,2,1])
    # pool2d = AvePool2D_Geometry()
    # outputR, outputSPD = pool2d(inputR, inputSPD)
    # print(outputSPD.shape)

    # inputR = torch.randn([64, 7,15,19])
    # inputSPD = torch.randn([64, 7,7,15,19])
    # inputSPD = inputSPD + inputSPD.permute([0,2,1,3,4])
    # outputR, outputSPD = Reshape_independent(inputR, inputSPD)
    # pdb.set_trace()
    # print(outputSPD.shape)

    inputR = torch.randn([100, 128,7,7])
    Gn = Gaussian_Noise()
    inputR, inputSPD = Gn(inputR)
    # inputSPD = torch.randn([64, 128,128])
    # inputSPD = inputSPD + inputSPD.permute([0,2,1])
    linear = First_Linear_Geometry(in_channels = 128, out_channels = 1000, 
                                                    kernel_size = (7,7), independent = True)
    outputR, outputSPD = linear(inputR, inputSPD)
    print(outputSPD.shape)


    # inputR = torch.randn([64, 7,15,13])
    # inputSPD = torch.randn([64, 7,7,15,13])
    # inputSPD = inputSPD + inputSPD.permute([0,2,1,3,4])
    # bn = BatchNorm_Geometry(7)
    # outputR, outputSPD = bn(inputR, inputSPD)
    # pdb.set_trace()
    # print(outputR.shape)





