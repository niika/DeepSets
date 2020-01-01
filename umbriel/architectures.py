#!/usr/bin/env python3
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.stats import multivariate_normal
from scipy import random, linalg
from sklearn.model_selection import train_test_split
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        f1 = 15
        f2 = 1
        f3 = 5
        f = [9,5,1,5]
        padd = [4,2,0,2]
        n1 = 64
        n2 = 32
        n3 = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n1,kernel_size=f[0],padding=padd[0])
        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2,kernel_size=f[1],padding=padd[1])
        self.conv3 = nn.Conv2d(in_channels=n2, out_channels=n3,kernel_size=f[2],padding=padd[2])
        self.conv4 = nn.Conv2d(in_channels=n3, out_channels=3,kernel_size=f[3],padding=padd[3])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x
    
class SiameseNet(nn.Module):
    def __init__(self):

        super(SiameseNet, self).__init__()
        f = [9,5,1,5]
        padd = [4,2,0,2]
        n1 = 64
        n2 = 32
        n3 = 16
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n1,kernel_size=f[0],padding=padd[0])
        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2,kernel_size=f[1],padding=padd[1])
        self.conv3 = nn.Conv2d(in_channels=n2, out_channels=n3,kernel_size=f[2],padding=padd[2])
        self.conv4 = nn.Conv2d(in_channels=n3, out_channels=3,kernel_size=f[3],padding=padd[3])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x
    
class SiameseNet2(nn.Module):
    def __init__(self):

        super(SiameseNet2, self).__init__()
        f = [9,5,3,1,3,5]
        padd = [4,2,1,0,1,2]
        n = [64,32,16,16,9]
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n[0],kernel_size=f[0],padding=padd[0])
        self.conv2 = nn.Conv2d(in_channels=n[0], out_channels=n[1],kernel_size=f[1],padding=padd[1])
        self.conv3 = nn.Conv2d(in_channels=n[1], out_channels=n[2],kernel_size=f[2],padding=padd[2])
        self.conv4 = nn.Conv2d(in_channels=n[2], out_channels=n[3],kernel_size=f[3],padding=padd[3])
        self.conv5 = nn.Conv2d(in_channels=n[3], out_channels=n[4],kernel_size=f[4],padding=padd[4])
        self.conv6 = nn.Conv2d(in_channels=n[4], out_channels=3,kernel_size=f[5],padding=padd[5])



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        return x

    
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        return out
        
class ResNet(nn.Module):
    """ Residual Neural Network """
    def __init__(self, num_blocks=64, planes=20,block=ResidualBlock ):
        super(ResNet, self).__init__()
        self.planes = planes
        self.input = nn.Conv2d(9, self.planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.body = self._make_layer(block,self.planes,num_blocks)
        self.output= nn.Conv2d(self.planes, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    """ Forward pass of ResNet """
    def forward(self, x):
        out = self.input(x)
        out = self.body(out)
        out = self.output(out)
        return out

class GradResNet(nn.Module):
    """ Residual Neural Network fro gradient images """
    def __init__(self, num_blocks=64, planes=20,block=ResidualBlock ):
        super(GradResNet, self).__init__()
        self.planes = planes
        self.input = nn.Conv2d(18, self.planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.body = self._make_layer(block,self.planes,num_blocks)
        self.output= nn.Conv2d(self.planes, 6, kernel_size=3, stride=1, padding=1, bias=True)

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    """ Forward pass of ResNet """
    def forward(self, x):
        out = self.input(x)
        out = self.body(out)
        out = self.output(out)
        return out

class SymmNet(nn.Module):
    """Symmetric Neural Network.
    
       Contains a contracting part and an expanding
    """
    def __init__(self):
        super(SymmNet, self).__init__()
        
        #Contracting
        self.dwnsp1 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=8,stride=2,padding=3)
        self.conv1 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.dwnsp2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=6,stride=2,padding=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.dwnsp3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=6,stride=2,padding=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.dwnsp4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.dwnsp5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.dwnsp6 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1)
        self.conv6 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        
        #Expanding
        self.upsp5 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1)
        self.iconv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.upsp4 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.iconv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.upsp3 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.iconv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upsp2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.iconv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.upsp1 = nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=4,stride=2,padding=1)
        self.iconv1 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1)
        
        self.upsp0 = nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=4,stride=2,padding=1)
        self.iconv0 = nn.Conv2d(in_channels=14,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.pr = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=3,stride=1,padding=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        # Contracting
        x1 = F.relu(self.dwnsp1(x))
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.dwnsp2(x1))
        x2 = F.relu(self.conv2(x2))
        x3 = F.relu(self.dwnsp3(x2))
        x3 = F.relu(self.conv3(x3))
        x4 = F.relu(self.dwnsp4(x3))
        x4 = F.relu(self.conv4(x4))
        x5 = F.relu(self.dwnsp5(x4))
        x5 = F.relu(self.conv5(x5))
#         x6 = F.relu(self.dwnsp6(x5))
#         x6 = F.relu(self.conv6(x6))


        #Expanding
#         y =  F.relu(self.upsp5(x6))
#         y =  F.relu(self.iconv5(y+x5))
        y =  F.relu(self.upsp4(x5))
        y =  F.relu(self.iconv4(y+x4))
        y =  F.relu(self.upsp3(x4))
        y =  F.relu(self.iconv3(y+x3))
        y =  F.relu(self.upsp2(y))
        y =  F.relu(self.iconv2(y+x2))
        y =  F.relu(self.upsp1(y))
        y =  F.relu(self.iconv1(y+x1))
        y =  F.relu(self.upsp0(y))
        y = torch.cat([y,x],dim=1)
        y =  F.relu(self.iconv0(y))
        y = self.pr(y)
    
        ys = torch.split(y,2,dim=1)
        y1 = self.softmax(ys[0])
        y2 = self.softmax(ys[1])
        
        y = torch.cat([y1,y2],dim=1)

        return y



class SymmGradNet(nn.Module):
    """Symmetric Gradient Neural Network .
    
       Contains a contracting part and an expanding
    """
    def __init__(self):
        super(SymmGradNet, self).__init__()
        
        #Contracting
        self.dwnsp1 = nn.Conv2d(in_channels=12,out_channels=16,kernel_size=8,stride=2,padding=3)
        self.conv1 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.dwnsp2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=6,stride=2,padding=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.dwnsp3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=6,stride=2,padding=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.dwnsp4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.dwnsp5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.dwnsp6 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1)
        self.conv6 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        
        #Expanding
        self.upsp5 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1)
        self.iconv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.upsp4 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.iconv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.upsp3 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.iconv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upsp2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.iconv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.upsp1 = nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=4,stride=2,padding=1)
        self.iconv1 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1)
        
        self.upsp0 = nn.ConvTranspose2d(in_channels=16,out_channels=12,kernel_size=4,stride=2,padding=1)
        self.iconv0 = nn.Conv2d(in_channels=12,out_channels=12,kernel_size=1,stride=1,padding=0)
        self.output = nn.Conv2d(in_channels=12,out_channels=6,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        # Contracting
        x1 = F.relu(self.dwnsp1(x))
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.dwnsp2(x1))
        x2 = F.relu(self.conv2(x2))
        x3 = F.relu(self.dwnsp3(x2))
        x3 = F.relu(self.conv3(x3))
        x4 = F.relu(self.dwnsp4(x3))
        x4 = F.relu(self.conv4(x4))
        x5 = F.relu(self.dwnsp5(x4))
        x5 = F.relu(self.conv5(x5))
#         x6 = F.relu(self.dwnsp6(x5))
#         x6 = F.relu(self.conv6(x6))


        #Expanding
#         y =  F.relu(self.upsp5(x6))
#         y =  F.relu(self.iconv5(y+x5))
        y =  F.relu(self.upsp4(x5))
        y =  F.relu(self.iconv4(y+x4))
        y =  F.relu(self.upsp3(x4))
        y =  F.relu(self.iconv3(y+x3))
        y =  F.relu(self.upsp2(y))
        y =  F.relu(self.iconv2(y+x2))
        y =  F.relu(self.upsp1(y))
        y =  F.relu(self.iconv1(y+x1))
        y =  F.relu(self.upsp0(y))
        #y = torch.cat([y,x],dim=1)
        y =  F.relu(self.iconv0(y))
        y = self.output(y+x)
    
        return y

    
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        #self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU()),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1)),
        #self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU()),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=[20],
                 num_init_features=32, bn_size=4, kernel_size=7, drop_rate=0):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(9, num_init_features, kernel_size=7, stride=1, padding=3)),
            #('norm0', nn.BatchNorm2d(num_init_features)),
            #('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        #self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.agg = nn.Conv2d(num_features, 64, kernel_size=7, stride=1, padding=3)
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 18,kernel_size=1, stride=1),
                                        nn.ReLU(),
                                       nn.Conv2d(18, 18,kernel_size=7, stride=1, padding=3))
        self.output= nn.Sequential(nn.Conv2d(18, 18, kernel_size=7, stride=1, padding=3),
                                   nn.ReLU(),
                                  nn.Conv2d(18, 3, kernel_size=7, stride=1, padding=3))


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.agg(features)
        out = self.bottleneck(out)
        out = self.output(out)

        #out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out
    
    
    
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, dilation=2, padding=2, bias=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, dilation=2,padding=2, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        return out
        
class ResNetDilated(nn.Module):
    """ Residual Neural Network """
    def __init__(self, num_blocks=64, planes=20,block=DilatedResidualBlock ):
        super(ResNetDilated, self).__init__()
        self.planes = planes
        self.input = nn.Conv2d(9, self.planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.body = self._make_layer(block,self.planes,num_blocks)
        self.output = nn.Conv2d(self.planes, 3, kernel_size=3, stride=1, padding=1, bias=True)
        

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    """ Forward pass of ResNet """
    def forward(self, x):
        out = self.input(x)
        out = self.body(out)
        out = self.output(out)
        return out