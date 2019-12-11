#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from ...base.strategies import BasePoolStrategy


class ExpectedModelChange(BasePoolStrategy):
    def __init__(self):
        self.data_pool = None

    def __call__(self, current_model, data_pool=None, n_samples=10):
        if data_pool is None and self.data_pool is None:
            raise AssertionError("data pool is empty!")

        if data_pool is not None:
            data_pool_ = data_pool
        else:
            data_pool_ = self.data_pool
