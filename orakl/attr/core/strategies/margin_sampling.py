#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import heapq
import tqdm
import numpy as np
import tensorflow as tf

from .strategy import BasePoolStrategy


class MarginSampling(BasePoolStrategy):
    r"""Implements Margin Sampling algorithm.
    It has been proposed in 'A New Active Labeling Method for Deep Learning'_.
    .._A New Active Labeling Method for Deep Learning, IJCNN, 2014:
    https://ieeexplore.ieee.org/document/6889457

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

        x = tf.convert_to_tensor(data_pool_[candidates])

        preds = current_model(x)
        sorted_preds = tf.sort(preds,
                               axis=1,
                               direction="DESCENDING")

        # compute margin
        margin = sorted_preds[:, 0] - sorted_preds[:, 1]

        indexes = heapq.nsmallest(n_samples,
                                  range(len(candidates)),
                                  margin.__getitem__)

        samples = data_pool_[indexes]

        return indexes, samples
