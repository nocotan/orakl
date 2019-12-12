#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import tensorflow as tf


def grad(model, inputs, targets, loss_function):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        pred = model(inputs)
        loss_value = loss_function(targets, pred)
        loss_value = tf.expand_dims(loss_value, 1)

    return loss_value, tape.batch_jacobian(loss_value, inputs)
