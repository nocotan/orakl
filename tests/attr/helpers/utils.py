#!/usr/bin/env python3
import unittest

import random
import numpy as np


class BaseTest(unittest.TestCase):
    """
    This class provides a basic framework for all Captum tests by providing
    a set up fixture, which sets a fixed random seed.
    """

    def setUp(self):
        random.seed(1234)
        np.random.seed(1234)
