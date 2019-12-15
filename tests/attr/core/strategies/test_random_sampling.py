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

    def test_call_with_multi_dim_data(self):
        rsp = RandomSamplingPool()
        model = tf.keras.Model()

        n_samples = 10
        data_pool = np.random.rand(100, 10, 10)

        indexes, samples = rsp(model, data_pool=data_pool, n_samples=n_samples)

        assert(len(indexes) == n_samples)
        assert(len(samples) == n_samples)

    def test_repr(self):
        state = {
            "data_pool": np.random.rand(100, 10, 10),
            "excluded_indexes": [0, 1, 2],
            "loss_function": None,
            "n_classes": 10,
            "n_samples": 5,
            "batch_size": 10,
        }

        s = "\n================================"
        s += "\ndata_pool: {}".format(len(state["data_pool"]))
        s += "\nexcluded_indexes: {}".format(state["excluded_indexes"])
        s += "\nloss_function: {}".format(state["loss_function"])
        s += "\nn_classes: {}".format(state["n_classes"])
        s += "\nn_samples: {}".format(state["n_samples"])
        s += "\nbatch_size: {}".format(state["batch_size"])
        s += "\n================================"

        rsp = RandomSamplingPool(
            data_pool=state["data_pool"],
            excluded_indexes=state["excluded_indexes"],
            loss_function=state["loss_function"],
            n_classes=state["n_classes"],
            n_samples=state["n_samples"],
            batch_size=state["batch_size"],
        )

        assert(repr(rsp) == s)
