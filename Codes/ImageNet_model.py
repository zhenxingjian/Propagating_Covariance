from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import Layers as Gaussian_nn 
import Layers_independent as Independent_nn
import Layers_Geometry as Geometry_nn

import pdb


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = Geometry_nn.ReLU_Geometry(inplace=True)
        self.conv1 = Geometry_nn.Conv2D_Geometry(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Geometry_nn.Conv2D_Geometry(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Geometry_nn.Conv2D_Geometry(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)


    def forward(self, xR, xSPD = None):
        # xR = self.bn1(xR)
        outR, outSPD = self.relu(xR, xSPD)
        shortcutR, shortcutSPD = self.shortcut(outR, outSPD) if hasattr(self, 'shortcut') else (xR, xSPD)
        outR, outSPD = self.conv1(outR, outSPD)
        # outR = self.bn2(outR)
        outR, outSPD = self.relu(outR, outSPD)
        outR, outSPD = self.conv2(outR, outSPD)
        outR += shortcutR
        if outSPD is not None:
            outSPD += shortcutSPD
        return outR, outSPD


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = Geometry_nn.ReLU_Geometry(inplace=True)
        self.conv1 = Geometry_nn.Conv2D_Geometry(in_planes, planes, kernel_size=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Geometry_nn.Conv2D_Geometry(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = Geometry_nn.Conv2D_Geometry(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Geometry_nn.Conv2D_Geometry(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            

    def forward(self, xR, xSPD = None):
        # xR = self.bn1(xR)
        out = self.relu(xR, xSPD)
        shortcutR, shortcutSPD = self.shortcut(outR, outSPD) if hasattr(self, 'shortcut') else (xR, xSPD)
        outR, outSPD = self.conv1(outR, outSPD)
        # outR = self.bn2(outR)
        outR, outSPD = self.relu(outR, outSPD)
        outR, outSPD = self.conv2(outR, outSPD)
        # outR = self.bn3(outR)
        outR, outSPD = self.relu(outR, outSPD)
        outR, outSPD = self.conv3(outR, outSPD)
        outR += shortcutR
        if outSPD is not None:
            outSPD += shortcutSPD
        return outR, outSPD


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(PreActResNet, self).__init__()
        self.in_planes = 16*4

        self.relu = Geometry_nn.ReLU_Geometry(inplace=True)

        self.conv1 = Geometry_nn.Conv2D_Geometry(3, 16*4, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.avepool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16*4, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*4, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128*4, num_blocks[3], stride=2)
        self.avepooling = Geometry_nn.AvePool2D_Geometry(7)
        self.linear = Geometry_nn.First_Linear_Geometry(in_channels = 128*4, out_channels = num_classes, 
                                                    kernel_size = (1,1), independent = True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Geometry_nn.Sequential(*layers)

    def forward(self, xR, xSPD = None):
        outR, outSPD = self.conv1(xR, xSPD)
        outR, outSPD = self.relu(outR, outSPD)
        outR = self.avepool(outR)

        outR, outSPD = self.layer1(outR, outSPD)
        outR, outSPD = self.layer2(outR, outSPD)
        outR, outSPD = self.layer3(outR, outSPD)
        outR, outSPD = self.layer4(outR, outSPD)

        outR, outSPD = self.avepooling(outR, outSPD)

        outR, outSPD = self.linear(outR, outSPD)
        return outR, outSPD


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])








