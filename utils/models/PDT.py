"""
    THIS IS FROM HERE: https://gitlab.idiap.ch/bob/bob.paper.tifs2022_hfr_prepended_domain_transformer
"""

from logging import raiseExceptions
import math
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
from torch.nn.modules import module

from typing import Union, Tuple, Optional

import torch 
import torch.nn as nn


from torch.nn import init

def round_channels(channels,
                   divisor=8):
    """
    Round weighted channel number (make divisible operation).

    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.

    Returns:
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels



class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    """
    def __init__(self, channels, reduction=16, round_mid=False, use_conv=True,
                 mid_activation=(lambda: nn.ReLU(inplace=True)),
                 out_activation=(lambda: nn.Sigmoid())):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1,
                                stride=1, groups=1, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        if use_conv:
            self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1,
                                stride=1, groups=1, bias=True)
        else:
            self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual



class PDT(nn.Module): # better than the other InceptionTranslator; ~600 parameters opposed to the huge ones
    """
    Prepended Domain Transformer
    """

    def __init__(self, in_channels=int(3), pool_features=6, use_se=False, use_bias=False, use_cbam=True):
        super(PDT, self).__init__()
        
        self.use_se=use_se
        self.use_bias=use_bias
        self.use_cbam=use_cbam
        self.relu=nn.ReLU()

        self.pool_features=int(pool_features)
        self.branch1x1 = Conv2d(in_channels, self.pool_features, kernel_size=1, bias=self.use_bias)

        self.branch5x5_1 = Conv2d(in_channels, self.pool_features//2, kernel_size=1, bias=self.use_bias)
        self.branch5x5_2 = Conv2d(self.pool_features//2, self.pool_features, kernel_size=5, padding=2, bias=self.use_bias)

        self.branch3x3dbl_1 = Conv2d(in_channels, self.pool_features//2, kernel_size=1, bias=self.use_bias)
        self.branch3x3dbl_2 = Conv2d(self.pool_features//2, self.pool_features, kernel_size=3, padding=1, bias=self.use_bias)
        self.branch3x3dbl_3 = Conv2d(self.pool_features, self.pool_features, kernel_size=3, padding=1, bias=self.use_bias)

        self.branch_pool = Conv2d(in_channels, self.pool_features, kernel_size=1, bias=self.use_bias)
        
        if self.use_cbam:
            self.cbam = CBAMBlock(channel=self.pool_features*4,reduction=4,kernel_size=7)
        if self.use_se:
            self.se=SEBlock(self.pool_features*4,reduction=4)

        self.dec = Conv2d(self.pool_features*4, 3, kernel_size=1, bias=self.use_bias)

    def forward(self, x):
        first = self.branch1x1(x)
        branch1x1 = self.relu(first)
        # branch1x1 = self.relu(self.branch1x1(x))

        branch5x5 = self.relu(self.branch5x5_1(x))
        branch5x5 = self.relu(self.branch5x5_2(branch5x5))

        branch3x3dbl = self.relu(self.branch3x3dbl_1(x))
        branch3x3dbl = self.relu(self.branch3x3dbl_2(branch3x3dbl))
        branch3x3dbl = self.relu(self.branch3x3dbl_3(branch3x3dbl))

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.relu(self.branch_pool(branch_pool))

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        concat = torch.cat(outputs, 1)

        if self.use_se:
            concat=self.se(concat)
        if self.use_cbam:
            concat=self.cbam(concat)
        
        img = self.dec(concat)

        return img
