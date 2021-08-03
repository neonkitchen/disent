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

import os
from numbers import Number
from typing import List
from typing import Optional
from typing import Sequence
from typing import Sized
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import Sampler

from disent.data.groundtruth import Cars3dData
from disent.data.groundtruth import GroundTruthData
from disent.data.groundtruth import Shapes3dData
from disent.data.groundtruth import XYSquaresData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.dataset.groundtruth import GroundTruthDatasetAndFactors
from disent.transform import ToStandardisedTensor
from disent.util import TempNumpySeed
from disent.visualize.visualize_util import make_animated_image_grid
from disent.visualize.visualize_util import make_image_grid
from experiment.exp.util._tasks import IN
from experiment.exp.util._tasks import TASK
from experiment.exp.util._tasks import TaskHandler
from experiment.exp.util._visualise import plt_imshow


# ========================================================================= #
# dataset                                                                   #
# ========================================================================= #


def make_dataset(name: str = 'xysquares', factors: bool = False, data_dir='/Users/neelanpather/dev/disent/data/dataset'):
    Sampler = GroundTruthDatasetAndFactors if factors else GroundTruthDataset
    # make dataset
    if   name == 'xysquares':      dataset = Sampler(XYSquaresData(),              transform=ToStandardisedTensor())
    elif name == 'xysquares_1x1':  dataset = Sampler(XYSquaresData(square_size=1), transform=ToStandardisedTensor())
    elif name == 'xysquares_2x2':  dataset = Sampler(XYSquaresData(square_size=2), transform=ToStandardisedTensor())
    elif name == 'xysquares_4x4':  dataset = Sampler(XYSquaresData(square_size=4), transform=ToStandardisedTensor())
    elif name == 'xysquares_8x8':  dataset = Sampler(XYSquaresData(square_size=8), transform=ToStandardisedTensor())
    elif name == 'cars3d':         dataset = Sampler(Cars3dData(data_dir=os.path.join(data_dir, 'cars3d')),   transform=ToStandardisedTensor(size=64))
    elif name == 'shapes3d':       dataset = Sampler(Shapes3dData(data_dir=os.path.join(data_dir, '3dshapes')), transform=ToStandardisedTensor())
    else: raise KeyError(f'invalid data name: {repr(name)}')
    return dataset


def get_single_batch(dataloader, cuda=True):
    for batch in dataloader:
        (x_targ,) = batch['x_targ']
        break
    if cuda:
        x_targ = x_targ.cuda()
    return x_targ


# ========================================================================= #
# sampling helper                                                           #
# ========================================================================= #


def normalise_factor_idx(dataset, factor: Union[int, str]) -> int:
    if isinstance(factor, str):
        try:
            f_idx = dataset.factor_names.index(factor)
        except:
            raise KeyError(f'{repr(factor)} is not one of: {dataset.factor_names}')
    else:
        f_idx = factor
    assert isinstance(f_idx, (int, np.int32, np.int64, np.uint8))
    assert 0 <= f_idx < dataset.num_factors
    return int(f_idx)


# general type
NonNormalisedFactors = Union[Sequence[Union[int, str]], Union[int, str]]


def normalise_factor_idxs(dataset: GroundTruthDataset, factors: NonNormalisedFactors) -> np.ndarray:
    if isinstance(factors, (int, str)):
        factors = [factors]
    factors = np.array([normalise_factor_idx(dataset, factor) for factor in factors])
    assert len(set(factors)) == len(factors)
    return factors


def get_factor_idxs(dataset: GroundTruthDataset, factors: Optional[NonNormalisedFactors] = None):
    if factors is None:
        return np.arange(dataset.num_factors)
    return normalise_factor_idxs(dataset, factors)


# TODO: clean this up
def sample_factors(dataset, num_obs: int = 1024, factor_mode: str = 'sample_random', factor: Union[int, str] = None):
    # sample multiple random factor traversals
    if factor_mode == 'sample_traversals':
        assert factor is not None, f'factor cannot be None when factor_mode=={repr(factor_mode)}'
        # get traversal
        f_idx = normalise_factor_idx(dataset, factor)
        # generate traversals
        factors = []
        for i in range((num_obs + dataset.factor_sizes[f_idx] - 1) // dataset.factor_sizes[f_idx]):
            factors.append(dataset.sample_random_factor_traversal(f_idx=f_idx))
        factors = np.concatenate(factors, axis=0)
    elif factor_mode == 'sample_random':
        factors = dataset.sample_factors(num_obs)
    else:
        raise KeyError
    return factors


# TODO: move into dataset class
def sample_batch_and_factors(dataset, num_samples: int, factor_mode: str = 'sample_random', factor: Union[int, str] = None, device=None):
    factors = sample_factors(dataset, num_obs=num_samples, factor_mode=factor_mode, factor=factor)
    batch = dataset.dataset_batch_from_factors(factors, mode='target').to(device=device)
    factors = torch.from_numpy(factors).to(dtype=torch.float32, device=device)
    return batch, factors


# ========================================================================= #
# mask helper                                                               #
# ========================================================================= #


def make_changed_mask(batch, masked=True):
    if masked:
        mask = torch.zeros_like(batch[0], dtype=torch.bool)
        for i in range(len(batch)):
            mask |= (batch[0] != batch[i])
    else:
        mask = torch.ones_like(batch[0], dtype=torch.bool)
    return mask


# ========================================================================= #
# dataset indices                                                           #
# ========================================================================= #


def sample_unique_batch_indices(num_obs, num_samples) -> np.ndarray:
    assert num_obs >= num_samples, 'not enough values to sample'
    assert (num_obs - num_samples) / num_obs > 0.5, 'this method might be inefficient'
    # get random sample
    indices = set()
    while len(indices) < num_samples:
        indices.update(np.random.randint(low=0, high=num_obs, size=num_samples - len(indices)))
    # make sure indices are randomly ordered
    indices = np.fromiter(indices, dtype=int)
    # indices = np.array(list(indices), dtype=int)
    np.random.shuffle(indices)
    # return values
    return indices


def generate_epoch_batch_idxs(num_obs: int, num_batches: int, mode: str = 'shuffle') -> List[np.ndarray]:
    """
    Generate `num_batches` batches of indices.
    - Each index is in the range [0, num_obs).
    - If num_obs is not divisible by num_batches, then batches may not all be the same size.

    eg. [0, 1, 2, 3, 4] -> [[0, 1], [2, 3], [4]] -- num_obs=5, num_batches=3, sample_mode='range'
    eg. [0, 1, 2, 3, 4] -> [[1, 4], [2, 0], [3]] -- num_obs=5, num_batches=3, sample_mode='shuffle'
    eg. [0, 1, 0, 3, 2] -> [[0, 1], [0, 3], [2]] -- num_obs=5, num_batches=3, sample_mode='random'
    """
    # generate indices
    if mode == 'range':
        idxs = np.arange(num_obs)
    elif mode == 'shuffle':
        idxs = np.arange(num_obs)
        np.random.shuffle(idxs)
    elif mode == 'random':
        idxs = np.random.randint(0, num_obs, size=(num_obs,))
    else:
        raise KeyError(f'invalid mode={repr(mode)}')
    # return batches
    return np.array_split(idxs, num_batches)


def generate_epochs_batch_idxs(num_obs: int, num_epochs: int, num_epoch_batches: int, mode: str = 'shuffle') -> List[np.ndarray]:
    """
    Like generate_epoch_batch_idxs, but concatenate the batches of calling the function `num_epochs` times.
    - The total number of batches returned is: `num_epochs * num_epoch_batches`
    """
    batches = []
    for i in range(num_epochs):
        batches.extend(generate_epoch_batch_idxs(num_obs=num_obs, num_batches=num_epoch_batches, mode=mode))
    return batches


# ========================================================================= #
# Dataloader Sampler Utilities                                              #
# ========================================================================= #


class StochasticSampler(Sampler):
    """
    Sample random batches, not guaranteed to be unique or cover the entire dataset in one epoch!
    """

    def __init__(self, data_source: Union[Sized, int], batch_size: int = 128):
        super().__init__(data_source)
        if isinstance(data_source, int):
            self._len = data_source
        else:
            self._len = len(data_source)
        self._batch_size = batch_size
        assert isinstance(self._len, int)
        assert self._len > 0
        assert isinstance(self._batch_size, int)
        assert self._batch_size > 0

    def __iter__(self):
        while True:
            yield from np.random.randint(0, self._len, size=self._batch_size)


def yield_dataloader(dataloader, steps: int):
    i = 0
    while True:
        for it in dataloader:
            yield it
            i += 1
            if i >= steps:
                return


def StochasticBatchSampler(data_source: Union[Sized, int], batch_size: int):
    return BatchSampler(
        sampler=StochasticSampler(data_source=data_source, batch_size=batch_size),
        batch_size=batch_size,
        drop_last=True
    )


# ========================================================================= #
# Dataset Visualisation / Traversals -- HELPER                              #
# ========================================================================= #


class _TraversalTasks(object):

    @staticmethod
    def task__factor_idxs(gt_data=IN, factor_names=IN):
        return get_factor_idxs(gt_data, factor_names)

    @staticmethod
    def task__factors(factor_idxs=TASK, gt_data=IN, seed=IN, base_factors=IN, num=IN, traverse_mode=IN):
        with TempNumpySeed(seed):
            return np.stack([
                gt_data.sample_random_factor_traversal(f_idx, base_factors=base_factors, num=num, mode=traverse_mode)
                for f_idx in factor_idxs
            ], axis=0)

    @staticmethod
    def task__raw_grid(factors=TASK, gt_data=IN, data_mode=IN):
        return [gt_data.dataset_batch_from_factors(f, mode=data_mode) for f in factors]

    @staticmethod
    def task__aug_grid(raw_grid=TASK, augment_fn=IN):
        if augment_fn is not None:
            return [augment_fn(batch) for batch in raw_grid]
        return raw_grid

    @staticmethod
    def task__grid(aug_grid=TASK):
        return np.stack(aug_grid, axis=0)

    @staticmethod
    def task__image(grid=TASK, num=IN, pad=IN, border=IN, bg_color=IN):
        return make_image_grid(np.concatenate(grid, axis=0), pad=pad, border=border, bg_color=bg_color, num_cols=num)

    @staticmethod
    def task__animation(grid=TASK, pad=IN, border=IN, bg_color=IN):
        return make_animated_image_grid(np.stack(grid, axis=0), pad=pad, border=border, bg_color=bg_color, num_cols=None)

    @staticmethod
    def task__image_wandb(image=TASK):
        import wandb
        return wandb.Image(image)

    @staticmethod
    def task__animation_wandb(animation=TASK):
        import wandb
        return wandb.Video(np.transpose(animation, [0, 3, 1, 2]), fps=5, format='mp4')

    @staticmethod
    def task__image_plt(image=TASK):
        return plt_imshow(img=image)


# ========================================================================= #
# Dataset Visualisation / Traversals                                        #
# ========================================================================= #


def dataset_traversal_tasks(
    gt_data: Union[GroundTruthData, GroundTruthDataset],
    # task settings
    tasks: Union[str, Tuple[str, ...]] = 'grid',
    # inputs
    factor_names: Optional[NonNormalisedFactors] = None,
    num: int = 9,
    seed: int = 777,
    base_factors=None,
    traverse_mode='cycle',
    # images & animations
    pad: int = 4,
    border: bool = True,
    bg_color: Number = None,
    # augment
    augment_fn: callable = None,
    data_mode: str = 'raw',
):
    """
    Generic function that can return multiple parts of the dataset & factor traversal pipeline.
    - This only evaluates what is needed to compute the next components.

    Tasks include:
        - factor_idxs
        - factors
        - grid
        - image
        - image_wandb
        - image_plt
        - animation
        - animation_wandb
    """
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # normalise dataset
    if not isinstance(gt_data, GroundTruthDataset):
        gt_data = GroundTruthDataset(gt_data)
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    return TaskHandler.compute(
        task_names=tasks,
        task_fns=(
            _TraversalTasks.task__factor_idxs,
            _TraversalTasks.task__factors,
            _TraversalTasks.task__raw_grid,
            _TraversalTasks.task__aug_grid,
            _TraversalTasks.task__grid,
            _TraversalTasks.task__image,
            _TraversalTasks.task__animation,
            _TraversalTasks.task__image_wandb,
            _TraversalTasks.task__animation_wandb,
            _TraversalTasks.task__image_plt,
        ),
        symbols=dict(
            gt_data=gt_data,
            # inputs
            factor_names=factor_names,
            num=num,
            seed=seed,
            base_factors=base_factors,
            traverse_mode=traverse_mode,
            # animation & images
            pad=pad,
            border=border,
            bg_color=bg_color,
            # augment
            augment_fn=augment_fn,
            data_mode=data_mode,
        ),
        strict=True,
        disable_options=True,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
