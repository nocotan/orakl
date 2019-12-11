#!/usr/bin/env python3
import tensorflow as tf

from orakl.attr import ExpectedModelChange

from ...helpers.utils import BaseTest


class Test(BaseTest):
    def test_call_with_empty_data_pool(self):
        emc = ExpectedModelChange()
        model = tf.keras.Model()

        with self.assertRaises(AssertionError):
            emc(model)
