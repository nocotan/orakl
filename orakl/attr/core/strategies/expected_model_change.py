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
from ...utils.functions import grad


class ExpectedModelChange(BasePoolStrategy):
    r"""Implements Expected Model Change Maximization algorithm.
    It has been proposed in `Maximizing Expected Model Change for Active Learning in Regression`_.
    .. _Maximizing Expected Model Change for Active Learning in Regression:
        https://ieeexplore.ieee.org/abstract/document/6729489/

    Raises:
        AssertionError: if data pool is empty.
        AssertionError: if loss function is empty.
        AssertionError: if n_classes is not specified.
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

        if data_pool is None and self.data_pool is None:
            raise AssertionError("data pool is empty!")

        if loss_function is None and self.loss_function is None:
            raise AssertionError("loss function is empty!")

        if n_classes is None and self.n_classes is None:
            raise AssertionError("please specify number of classes!")

        data_pool_ = data_pool if data_pool is not None else self.data_pool
        loss_function_ = loss_function if loss_function is not None else self.loss_function
        n_classes_ = n_classes if n_classes is not None else self.n_classes

        if excluded_indexes is not None:
            excluded_indexes_ = excluded_indexes
        elif self.excluded_indexes is not None:
            excluded_indexes_ = self.excluded_indexes
        else:
            excluded_indexes_ = []

        n_data_pool = len(list(data_pool))
        candidates = [i for i in range(0, n_data_pool) if i not in excluded_indexes_]

        n_split = (len(candidates) / batch_size) + (len(candidates) % batch_size > 0)
        batches = np.array_split(np.array(candidates), n_split)

        expected_grads = np.zeros((len(candidates),))
        for batch in tqdm.tqdm(batches):

            x = tf.convert_to_tensor(data_pool_[batch])
            ys = []
            for i in range(n_classes_):
                yi = tf.constant(i, shape=(len(batch), 1))
                ys.append*yi

            y = tf.concat(ys, 1)
            loss_value, grad_ = grad(current_model, x, y, loss_function_)
            preds = current_model(x)[:, i]

            # update expected gradients
            # compute expected gradient length
            grad_ = tf.reshape(grad_, shape=(len(grad_), -1))
            grad_ = tf.dtypes.cast(grad_, tf.dtypes.float32)
            grad_ = tf.abs(grad_)
            grad_ = tf.reduce_mean(grad_, 1)

            expected_grads[batch] += (preds * grad_).numpy()

        # print(expected_grads)
        indexes = heapq.nlargest(n_samples,
                                 range(len(expected_grads)),
                                 expected_grads.__getitem__)

        samples = data_pool_[indexes]

        return indexes, samples
