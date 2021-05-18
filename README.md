# Simpler Certiﬁed Radius Maximization by Propagating Covariances
This is the official GitHub repo for the paper: [Simpler Certiﬁed Radius Maximization by Propagating Covariances, CVPR 2021 (oral)](https://arxiv.org/abs/2104.05888)

# Video Introduction
During submission, we create an intuitive [introduction video](https://www.youtube.com/watch?v=m1ya2oNf5iE), and we put it on [my YouTube channel](https://www.youtube.com/channel/UCt5acq2GhBpnXb875hiPQYQ). 

# Requirement
## Dependency
pytorch 1.6.0
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

CUDA 10.2

prefetch_generator
```bash
pip install prefetch_generator
```

statsmodels
```bash
pip install statsmodels
```

# Results
The results on MNIST, SVHN, Cifar-10, ImageNet, and Places365 with the certified robustness. The number reported in each column represents the ratio of the test set with the certified radius larger than the header of that column under the perturbation \sigma. ACR is the average certified radius of all the test samples. A larger value is better for all the numbers reported
![Alt text](Results/main_results.png?raw=true "Title")

Ablation experiment on Places365 with \sigma=0.5. We perform the choice of \lambda and r_{max} as the hyper-parameters.
![Alt text](Results/ablation.png?raw=true "Title")


# Citation
```
@InProceedings{zhen2021simpler,
author = {Zhen, Xingjian and Chakraborty, Rudrasis and Singh, Vikas},
title = {Simpler Certified Radius Maximization by Propagating Covariances},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```


The code and more detail will be added shortly.
