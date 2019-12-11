#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
from operator import itemgetter

from ...base.strategies import BasePoolStrategy


class RandomSamplingPool(BasePoolStrategy):
    def __init__(self):
        self.data_pool = None
        self.excluded_indexes = None

    def __call__(self, current_model, data_pool=None, excluded_indexes=None, n_samples=10):
        if data_pool is None and self.data_pool is None:
            raise AssertionError("data pool is empty!")

        if data_pool is not None:
            data_pool_ = data_pool
        else:
            data_pool_ = self.data_pool

        if excluded_indexes is not None:
            excluded_indexes_ = excluded_indexes
        elif self.excluded_indexes is not None:
            excluded_indexes_ = self.excluded_indexes
        else:
            excluded_indexes_ = []

        candidates = [i for i in range(0, len(list(data_pool))) if i not in excluded_indexes_]
        indexes = random.sample(candidates, n_samples)
        samples = data_pool_[indexes]

        return indexes, samples
