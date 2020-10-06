### dropout has been removed in this code. original code had dropout#####
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def per_image_standardization(x):
    y = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    mean = y.mean(dim=1, keepdim = True).expand_as(y)    
    std = y.std(dim=1, keepdim = True).expand_as(y)      
    adjusted_std = torch.max(std, 1.0/torch.sqrt(torch.cuda.FloatTensor([x.shape[1]*x.shape[2]*x.shape[3]])))    
    y = (y- mean)/ adjusted_std
    standarized_input =  y.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])  
    return standarized_input  

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Encoder(nn.Module):
    
    def __init__(self, depth, widen_factor, per_img_std= False, stride = 1):
        super(Encoder, self).__init__()
        self.per_img_std = per_img_std
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0], stride = stride)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

        self.img_size = [1, 1]
        self.out_channels = 640

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    
    """
    ## Modified WRN architecture###
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor
        #self.mixup_hidden = mixup_hidden
        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.conv1 = conv3x3(3,nStages[0])
        self.bn1 = nn.BatchNorm2d(nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        #self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    """
    def forward(self, x):
        if self.per_img_std:
            x = per_image_standardization(x) 
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        
        return out
        
                  
        
def wrn28_10(dropout = False, per_img_std = False, stride = 1):
    #print ('this')
    model = Encoder(depth=28, widen_factor=10, per_img_std = per_img_std, stride = stride)
    return model

def wrn28_2(dropout = False, per_img_std = False, stride = 1):
    #print ('this')
    model = Encoder(depth =28, widen_factor =2, per_img_std = per_img_std, stride = stride)
    return model