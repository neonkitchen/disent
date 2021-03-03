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
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from disent.frameworks.vae.unsupervised import BetaVae


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #


class FixedVarianceVae(BetaVae):

    @dataclass
    class cfg(BetaVae.cfg):
        std: float = 1.0

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        assert self.cfg.std >= 0, 'std must be >= 0'

    def intercept_zs(self, all_params):
        (z_params,) = all_params
        # clamp std
        std = self.cfg.std
        if std < 1e-20:
            warnings.warn('clamping std to min value of 1e-20')
            std = 1e-20
        # calculate logvar
        logvar = 2 * np.log(std)
        # make new logvar
        z_params.logvar = torch.full_like(z_params.logvar, fill_value=logvar)
        # return new value
        return (z_params,), {}


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
