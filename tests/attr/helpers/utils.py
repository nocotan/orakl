#!/usr/bin/env python3
import unittest

import random
import numpy as np
import tensorflow as tf


class BaseTest(unittest.TestCase):
    """
    This class provides a basic framework for all Captum tests by providing
    a set up fixture, which sets a fixed random seed.
    """

    def setUp(self):
        random.seed(1024)
        np.random.seed(1024)
        tf.random.set_seed(1024)
