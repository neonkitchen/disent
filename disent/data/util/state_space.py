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

import numpy as np
from disent.util import LengthIter
from disent.visualize.visualize_util import get_factor_traversal


# ========================================================================= #
# Basic State Space                                                         #
# ========================================================================= #


class StateSpace(LengthIter):
    """
    State space where an index corresponds to coordinates (factors/positions) in the factor space.
    ie. State space with multiple factors of variation, where each factor can be a different size.
    """

    def __init__(self, factor_sizes):
        super().__init__()
        # dimension
        self._factor_sizes = np.array(factor_sizes)
        self._factor_sizes.flags.writeable = False
        # total permutations
        self._size = int(np.prod(factor_sizes))

    def __len__(self):
        """Same as self.size"""
        return self.size

    def __getitem__(self, idx):
        """Data returned based on the idx"""
        return self.idx_to_pos(idx)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Properties                                                            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def size(self) -> int:
        """The number of permutations of factors handled by this state space"""
        return self._size

    @property
    def num_factors(self) -> int:
        """The number of factors handled by this state space"""
        return len(self._factor_sizes)

    @property
    def factor_sizes(self) -> np.ndarray:
        """A list of sizes or dimensionality of factors handled by this state space"""
        return self._factor_sizes

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Coordinate Transform - any dim array, only last axis counts!          #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def pos_to_idx(self, positions) -> np.ndarray:
        """
        Convert a position to an index (or convert a list of positions to a list of indices)
        - positions are lists of integers, with each element < their corresponding factor size
        - indices are integers < size
        """
        positions = np.moveaxis(positions, source=-1, destination=0)
        return np.ravel_multi_index(positions, self._factor_sizes)

    def idx_to_pos(self, indices) -> np.ndarray:
        """
        Convert an index to a position (or convert a list of indices to a list of positions)
        - indices are integers < size
        - positions are lists of integers, with each element < their corresponding factor size
        """
        positions = np.unravel_index(indices, self._factor_sizes)
        return np.moveaxis(positions, source=0, destination=-1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling Functions - any dim array, only last axis counts!            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def sample_factors(self, size=None, factor_indices=None) -> np.ndarray:
        """
        sample randomly from all factors, otherwise the given factor_indices.
        returned values must appear in the same order as factor_indices.

        If factor factor_indices is None, all factors are sampled.
        If size=None then the array returned is the same shape as (len(factor_indices),) or factor_sizes[factor_indices]
        If size is an integer or shape, the samples returned are that shape with the last dimension
            the same size as factor_indices, ie (*size, len(factor_indices))
        """
        # get factor sizes
        sizes = self._factor_sizes if (factor_indices is None) else self._factor_sizes[factor_indices]
        # get resample size
        if size is not None:
            # empty np.array(()) gets dtype float which is incompatible with len
            size = np.append(np.array(size, dtype=int), len(sizes))
        # sample for factors
        return np.random.randint(0, sizes, size=size)

    def sample_missing_factors(self, known_factors, known_factor_indices) -> np.ndarray:
        """
        Samples the remaining factors not given in the known_factor_indices.
        ie. fills in the missing values by sampling from the unused dimensions.
        returned values are ordered by increasing factor index and not factor_indices.
        (known_factors must correspond to known_factor_indices)

        - eg. known_factors=A, known_factor_indices=1
              BECOMES: known_factors=[A], known_factor_indices=[1]
        - eg. known_factors=[A], known_factor_indices=[1]
              = [..., A, ...]
        - eg. known_factors=[[A]], known_factor_indices=[1]
              = [[..., A, ...]]
        - eg. known_factors=[A, B], known_factor_indices=[1, 2]
              = [..., A, B, ...]
        - eg. known_factors=[[A], [B]], known_factor_indices=[1]
              = [[..., A, ...], [..., B, ...]]
        - eg. known_factors=[[A, B], [C, D]], known_factor_indices=[1, 2]
              = [[..., A, B, ...], [..., C, D, ...]]
        """
        # convert & check
        known_factors = np.atleast_1d(known_factors)
        known_factor_indices = np.atleast_1d(known_factor_indices)
        assert known_factor_indices.ndim == 1
        # mask of known factors
        known_mask = np.zeros(self.num_factors, dtype='bool')
        known_mask[known_factor_indices] = True
        # set values
        all_factors = np.zeros((*known_factors.shape[:-1], self.num_factors), dtype='int')
        all_factors[..., known_mask] = known_factors
        all_factors[..., ~known_mask] = self.sample_factors(size=known_factors.shape[:-1], factor_indices=~known_mask)
        return all_factors

    def resample_factors(self, factors, fixed_factor_indices) -> np.ndarray:
        """
        Resample across all the factors, keeping factor_indices constant.
        returned values are ordered by increasing factor index and not factor_indices.
        """
        return self.sample_missing_factors(np.array(factors)[..., fixed_factor_indices], fixed_factor_indices)

    def _get_f_idx_and_factors_and_size(self, f_idx: int = None, base_factors=None, num: int = None):
        """
        :param f_idx: Sampled randomly in the range [0, num_factors) if not given.
        :param base_factors: Sampled randomly from all possible factors if not given. Coerced into the shape (1, num_factors)
        :param num: Set to the factor size `self.factor_sizes[f_idx]` if not given.
        :return: All values above in a tuple.
        """
        # choose a random factor if not given
        if f_idx is None:
            f_idx = np.random.randint(0, self.num_factors)
        # sample factors if not given
        if base_factors is None:
            base_factors = self.sample_factors(size=1)
        else:
            base_factors = np.reshape(base_factors, (1, self.num_factors))
        # get size if not given
        if num is None:
            num = self.factor_sizes[f_idx]
        else:
            assert num > 0
        # generate a traversal
        base_factors = base_factors.repeat(num, axis=0)
        # return everything
        return f_idx, base_factors, num

    def sample_random_factor_traversal(self, f_idx: int = None, base_factors=None, num: int = None, mode='interval') -> np.ndarray:
        f_idx, base_factors, num = self._get_f_idx_and_factors_and_size(f_idx=f_idx, base_factors=base_factors, num=num)
        # generate traversal
        base_factors[:, f_idx] = get_factor_traversal(self.factor_sizes[f_idx], num_frames=num, mode=mode)
        # return factors
        return base_factors


# ========================================================================= #
# Hidden State Space                                                        #
# ========================================================================= #


# class HiddenStateSpace(_BaseStateSpace):
#
#     """
#     State space where an index corresponds to coordinates (factors/positions) in the factor space.
#     HOWEVER: some factors are treated as hidden/unknown and are thus randomly sampled.
#
#     Inputs to functions act as if known_factor_indices is the new factor space (including factor indexes).
#     Outputs from functions act in the full factor space.
#     """
#
#     def __init__(self, factor_sizes, known_factor_indices=None):
#         if known_factor_indices is None:
#             known_factor_indices = np.arange(len(factor_sizes))
#         factor_sizes, known_factor_indices = np.array(factor_sizes), np.array(known_factor_indices)
#         # known factors indices
#         self._known_factor_indices = known_factor_indices
#         self._known_factor_indices.flags.writeable = False
#         self._known_factor_sizes = factor_sizes[known_factor_indices]
#         # known state space does not include hidden variables, and assumes they are randomly sampled
#         self._known_state_space = StateSpace(self._known_factor_sizes)
#         # full state space includes hidden variables
#         self._full_state_space = StateSpace(factor_sizes)
#
#     @property
#     def size(self):
#         return self._known_state_space.size
#
#     @property
#     def num_factors(self):
#         return self._known_state_space.num_factors
#
#     @property
#     def factor_sizes(self):
#         return self._known_state_space.factor_sizes
#
#     def pos_to_idx(self, positions):
#         positions = np.array(positions)
#         assert 1 <= positions.ndim <= 2, f'positions has incorrect number of dimensions: {positions.ndim}'
#         assert positions.shape[-1] == self._full_state_space.num_factors, 'last dimension of positions must equal the full state space size'
#         # remove the unknown dimensions and return the index
#         return self._known_state_space.pos_to_idx(positions[..., self._known_factor_indices])
#
#     def idx_to_pos(self, indices):
#         indices = np.array(indices)
#         assert 0 <= indices.ndim <= 1, f'indices has incorrect number of dimensions: {indices.ndim}'
#         # get factors and return
#         known_factors = self._known_state_space.idx_to_pos(indices.reshape(-1))
#         sampled_factors = self._full_state_space.sample_missing_factors(known_factors, self._known_factor_indices)
#         return sampled_factors.reshape((*indices.shape, self._full_state_space.num_factors))
#
#     def sample_factors(self, size=None, factor_indices=None):
#         return self._full_state_space.sample_factors(
#             size,
#             self._known_factor_indices[factor_indices] if factor_indices else None
#         )
#
#     def sample_missing_factors(self, known_factors, known_factor_indices):
#         return self._full_state_space.sample_missing_factors(
#             known_factors,
#             self._known_factor_indices[known_factor_indices]
#         )
#
#     def resample_factors(self, factors, fixed_factor_indices):
#         return self._full_state_space.resample_factors(
#             factors,
#             self._known_factor_indices[fixed_factor_indices]
#         )


# ========================================================================= #
# Hidden State Space                                                        #
# ========================================================================= #


# class StateSpaceRemapIndex(object):
#     """Mapping from incorrectly ordered factors to state space indices"""
#
#     def __init__(self, factor_sizes, features):
#         self._states = StateSpace(factor_sizes)
#         # get indices of features
#         orig_indices = self._states.pos_to_idx(features)
#         if np.unique(orig_indices).size != self._states.size:
#             raise ValueError("Features do not cover the entire state space.")
#         # get indices of state space
#         state_indices = np.arange(self._states.size)
#         # mapping
#         self._state_to_orig_idx = np.zeros(self._states.size, dtype=np.int64)
#         self._state_to_orig_idx[orig_indices] = state_indices
#
#     def factors_to_orig_idx(self, factors):
#         """
#         get the original index of factors
#         """
#         return self._state_to_orig_idx[self._states.pos_to_idx(factors)]
