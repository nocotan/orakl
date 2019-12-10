#!/usr/bin/env python3
import unittest
import torch.nn as nn

from orakl.attr import ExpectedModelChange

from ...helpers.utils import BaseTest


class Test(BaseTest):
    def test_call_with_empty_data_pool(self):
        emc = ExpectedModelChange()
        model = nn.Module()

        with self.assertRaises(AssertionError):
            emc(model)
