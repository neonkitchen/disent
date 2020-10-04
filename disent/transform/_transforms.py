import kornia
import torch
import torchvision

import disent.transform.functional as F_d


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

    def __init__(self, size=None):
        self._size = size

    def __call__(self, obs) -> torch.Tensor:
        return F_d.to_standardised_tensor(obs, size=self._size)

    def __repr__(self):
        return f'{self.__class__.__name__}(size={repr(self._size)})'


class InputNormalizeTensor(object):
    """
    Basic transform that should be applied after augmentation before
    being passed to a model as the input.

    1. check that tensor is in range [0, 1]
    2. normalise tensor in range [-1, 1]
    """

    def __init__(self, check=False):
        if check:
            self._transform = torchvision.transforms.Compose([
                CheckTensor(low=0, high=1, dtype=torch.float32),
                kornia.augmentation.Normalize(mean=0.5, std=0.5),
            ])
        else:
            self._transform = kornia.augmentation.Normalize(mean=0.5, std=0.5)

    def __call__(self, obs) -> torch.Tensor:
        return self._transform(obs)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
