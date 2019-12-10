#!/usr/bin/env python3
import unittest

import random
import numpy as np
import torch


class BaseTest(unittest.TestCase):
    """
    This class provides a basic framework for all Captum tests by providing
    a set up fixture, which sets a fixed random seed. Since many torch
    initializations are random, this ensures that tests run deterministically.
    """

    def setUp(self):
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
