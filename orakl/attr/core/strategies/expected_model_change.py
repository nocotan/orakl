#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import heapq
import tqdm
import numpy as np
import tensorflow as tf

from ...base.strategies import BasePoolStrategy
from ...utils.functions import grad


class ExpectedModelChange(BasePoolStrategy):
    def __init__(self, loss_function=None, n_classes=None):
        self.data_pool = None
        self.loss_function = loss_function
        self.n_classes = n_classes
        self.excluded_indexes = None

    def __call__(self, current_model, loss_function=None,
                 n_classes=None, data_pool=None,
                 excluded_indexes=None, n_samples=10,
                 batch_size=32):

        if data_pool is None and self.data_pool is None:
            raise AssertionError("data pool is empty!")

        if loss_function is None and self.loss_function is None:
            raise AssertionError("loss function is empty!")

        if n_classes is None and self.n_classes is None:
            raise AssertionError("please specify number of classes!")

        if data_pool is not None:
            data_pool_ = data_pool
        else:
            data_pool_ = self.data_pool

        loss_function_ = loss_function if loss_function is not None else self.loss_function
        n_classes_ = n_classes if n_classes is not None else self.n_classes

        if excluded_indexes is not None:
            excluded_indexes_ = excluded_indexes
        elif self.excluded_indexes is not None:
            excluded_indexes_ = self.excluded_indexes
        else:
            excluded_indexes_ = []

        candidates = [i for i in range(0, len(list(data_pool_))) if i not in excluded_indexes_]
        n_split = (len(candidates) / batch_size) + (len(candidates) % batch_size > 0)
        batches = np.array_split(np.array(candidates), n_split)

        expected_grads = np.zeros((len(candidates),))
        for batch in tqdm.tqdm(batches):
            x = tf.convert_to_tensor(data_pool_[batch])
            for i in range(n_classes_):
                y = tf.constant(i, shape=(len(batch),))
                loss_value, grad_ = grad(current_model, x, y, loss_function_)
                grad_ = tf.reshape(grad_, shape=(len(grad_), -1))
                expected_grads[batch] += tf.abs(tf.reduce_mean(grad_, 1)).numpy()

        # print(expected_grads)
        indexes = heapq.nlargest(n_samples,
                                 range(len(expected_grads)),
                                 expected_grads.__getitem__)

        samples = data_pool_[indexes]

        return indexes, samples
