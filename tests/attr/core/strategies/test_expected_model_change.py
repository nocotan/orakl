#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from orakl.attr import ExpectedModelChange

from ...helpers.utils import BaseTest


class Test(BaseTest):
    def test_call_with_empty_data_pool(self):
        emc = ExpectedModelChange()
        model = tf.keras.Model()

        with self.assertRaises(AssertionError):
            emc(model)

    def test_call_with_random_data_pool(self):
        n_samples = 10
        n_classes = 3

        emc = ExpectedModelChange()

        initializer = tf.initializers.he_normal(seed=0)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                n_classes,
                input_shape=(10,),
                kernel_initializer=initializer),
        ])

        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        data_pool = np.random.rand(100, 10)
        indexes, samples = emc(model,
                               loss_function=loss_function,
                               n_classes=n_classes,
                               data_pool=data_pool,
                               n_samples=n_samples)

        assert(len(indexes) == n_samples)
        assert(len(samples) == n_samples)
