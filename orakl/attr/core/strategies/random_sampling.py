#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random

from .strategy import BasePoolStrategy


class RandomSamplingPool(BasePoolStrategy):
    r"""Implements random sampling algorithm.

    Raises:
        AssertionError: if data pool is empty.
    Returns:
        (indexes, samples) (tuple) -- selected indexes and samples.
    """

    def __call__(self,
                 current_model,
                 data_pool=None,
                 excluded_indexes=None,
                 loss_function=None,
                 n_classes=None,
                 n_samples=10,
                 batch_size=32):

        # check if data pool is empty or not.
        if data_pool is None and self.data_pool is None:
            raise AssertionError("data pool is empty!")

        data_pool_ = data_pool if data_pool is not None else self.data_pool

        if excluded_indexes is not None:
            excluded_indexes_ = excluded_indexes
        elif self.excluded_indexes is not None:
            excluded_indexes_ = self.excluded_indexes
        else:
            excluded_indexes_ = []

        n_data_pool = len(list(data_pool))
        candidates = [i for i in range(0, n_data_pool) if i not in excluded_indexes_]

        indexes = random.sample(candidates, n_samples)
        samples = data_pool_[indexes]

        return indexes, samples
