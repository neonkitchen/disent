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

import torch
import disent.transform.functional as F_d


# ========================================================================= #
# Transforms                                                                #
# ========================================================================= #


class Noop(object):
    """
    Transform that does absolutely nothing!
    See: disent.transform.functional.noop
    """

    def __call__(self, obs):
        return obs

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class CheckTensor(object):
    """
    Check that the data is a tensor, the right dtype, and in the required range.
    See: disent.transform.functional.check_tensor
    """

    def __init__(self, low=0., high=1., dtype=torch.float32):
        self._low = low
        self._high = high
        self._dtype = dtype

    def __call__(self, obs):
        return F_d.check_tensor(obs, low=self._low, high=self._high, dtype=self._dtype)

    def __repr__(self):
        return f'{self.__class__.__name__}(low={repr(self._low)}, high={repr(self._high)}, dtype={repr(self._dtype)})'


class ToStandardisedTensor(object):
    """
    Standardise image data after loading, by converting to a tensor
    in range [0, 1], and resizing to a square if specified.
    See: disent.transform.functional.to_standardised_tensor
    """

    def __init__(self, size=None, check=True):
        self._size = size
        self._check = check

    def __call__(self, obs) -> torch.Tensor:
        return F_d.to_standardised_tensor(obs, size=self._size, check=self._check)

    def __repr__(self):
        return f'{self.__class__.__name__}(size={repr(self._size)})'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
