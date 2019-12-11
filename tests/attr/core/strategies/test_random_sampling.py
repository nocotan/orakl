#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from orakl.attr import RandomSamplingPool

from ...helpers.utils import BaseTest


class Test(BaseTest):
    def test_call_with_empty_data_pool(self):
        rsp = RandomSamplingPool()
        model = tf.keras.Model()

        with self.assertRaises(AssertionError):
            rsp(model)

    def test_call_with_random_data_pool(self):
        rsp = RandomSamplingPool()
        model = tf.keras.Model()

        n_samples = 10
        data_pool = np.array(range(100))
        indexes, samples = rsp(model, data_pool=data_pool, n_samples=n_samples)

        assert(len(indexes) == n_samples)
        assert(len(samples) == n_samples)

        # print(indexes)
        # print(samples)
