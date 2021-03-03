#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import math

import numpy as np
import torch
from torch import nn as nn
from torch import Tensor

from disent.model.base import BaseDecoderModule
from disent.model.base import BaseEncoderModule
from disent.model.common import BatchView
from disent.model.common import Flatten3D


# ========================================================================= #
# Params                                                                    #
# ========================================================================= #
from disent.model.common import Print


_START_CAPACITY = 16

# TODO: fix for odd kernel sizes
_LAYER_PARAMS = [
    {'out_channels': _START_CAPACITY * (2**0), 'kernel_size': 4, 'stride': 2, 'padding': 1},  # out: 32*32 = 1024 | channels: 16 *  1 =  16 | out_total: 16384
    {'out_channels': _START_CAPACITY * (2**1), 'kernel_size': 4, 'stride': 2, 'padding': 1},  # out: 16*16 =  256 | channels: 16 *  2 =  32 | out_total: 8192
    {'out_channels': _START_CAPACITY * (2**2), 'kernel_size': 4, 'stride': 2, 'padding': 1},  # out: 8*8   =   64 | channels: 16 *  4 =  64 | out_total: 4096
    {'out_channels': _START_CAPACITY * (2**3), 'kernel_size': 4, 'stride': 2, 'padding': 1},  # out: 4*4   =   16 | channels: 16 *  8 = 128 | out_total: 2048
    {'out_channels': _START_CAPACITY * (2**4), 'kernel_size': 2, 'stride': 2, 'padding': 0},  # out: 2*2   =    4 | channels: 16 * 16 = 256 | out_total: 1024
    {'out_channels': _START_CAPACITY * (2**5), 'kernel_size': 2, 'stride': 2, 'padding': 0},  # out: 1*1   =    1 | channels: 16 * 32 = 512 | out_total: 512
]


# ========================================================================= #
# Conv models                                                               #
# ========================================================================= #


class EncoderLevels64(BaseEncoderModule):

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1, num_levels=4):
        # checks
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        assert x_shape[0] in {1, 3}
        assert 0 <= num_levels <= len(_LAYER_PARAMS)

        # init
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        # make normalisation layer -- we do not activate this
        layers = [nn.Conv2d(in_channels=x_shape[0], out_channels=3, kernel_size=1, stride=1, padding=0)]
        in_shape = (3, 64, 64)

        # make requested layers
        for params in _LAYER_PARAMS[:num_levels]:
            layers.extend([
                nn.Conv2d(in_channels=in_shape[0], **params),
                nn.LeakyReLU(inplace=True)
            ])
            in_shape = conv2d_output_shape(in_shape, **params)

        # make fc layers
        layers.extend([
            Flatten3D(),
            nn.Linear(in_features=int(np.prod(in_shape)), out_features=256),
                nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=self.z_total),
        ])

        # make model
        self.model = nn.Sequential(*layers)

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderLevels64(BaseDecoderModule):

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1, num_levels=4):
        # checks
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        assert x_shape[0] in {1, 3}
        assert 0 <= num_levels <= len(_LAYER_PARAMS)

        # init
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        # compute start shape
        in_shape = (3, 64, 64)
        for params in _LAYER_PARAMS[:num_levels]:
            in_shape = conv2d_output_shape(in_shape, **params)

        # make fc layers
        layers = [
            nn.Linear(in_features=self.z_size, out_features=256),
                nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=int(np.prod(in_shape))),
                nn.LeakyReLU(inplace=True),
            BatchView(in_shape),
        ]

        # make requested layers
        for params in reversed(_LAYER_PARAMS[:num_levels]):
            layers.extend([
                nn.ConvTranspose2d(in_channels=in_shape[0], **params),
                nn.LeakyReLU(inplace=True)
            ])
            in_shape = convtransp2d_output_shape(in_shape, **params)

        # check that we expanded back to the full image
        assert in_shape[1:] == (64, 64)

        # make normalisation layer -- we do not activate this
        layers.append(nn.Conv2d(in_channels=in_shape[0], out_channels=x_shape[0], kernel_size=1, stride=1, padding=0))

        # make model
        self.model = nn.Sequential(*layers)

    def decode(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def num2tuple(num):
    # FROM: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(in_shape, out_channels=None, kernel_size=1, stride=1, padding=0, dilation=1):
    # FROM: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
    in_shape, kernel_size, stride, padding, dilation = num2tuple(in_shape), num2tuple(kernel_size), num2tuple(stride), num2tuple(padding), num2tuple(dilation)
    padding = num2tuple(padding[0]), num2tuple(padding[1])
    h = math.floor((in_shape[-2] + sum(padding[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((in_shape[-1] + sum(padding[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return (out_channels, h, w) if (out_channels is not None) else (*in_shape[:-2], h, w)


def convtransp2d_output_shape(in_shape, out_channels=None, kernel_size=1, stride=1, padding=0, dilation=1, out_pad=0):
    # FROM: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
    in_shape, kernel_size, stride, padding, dilation, out_pad = num2tuple(in_shape), num2tuple(kernel_size), num2tuple(stride), num2tuple(padding), num2tuple(dilation), num2tuple(out_pad)
    padding = num2tuple(padding[0]), num2tuple(padding[1])
    h = (in_shape[-2] - 1) * stride[0] - sum(padding[0]) + dilation[0] * (kernel_size[0] - 1) + out_pad[0] + 1
    w = (in_shape[-1] - 1) * stride[1] - sum(padding[1]) + dilation[1] * (kernel_size[1] - 1) + out_pad[1] + 1
    return (out_channels, h, w) if (out_channels is not None) else (*in_shape[:-2], h, w)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
