# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: activation functions test (eager + JIT)."""

import numpy as np
import tensorflow_musa  # noqa: F401
import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase


class ActivationOpsTest(PjrtTestCase):
    def testRelu(self):
        for dtype in [tf.float32]:
            x = tf.constant([-1., 0., 1., 2., -0.5], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.relu, x)

    def testRelu6(self):
        for dtype in [tf.float32]:
            x = tf.constant([-1., 0., 5., 10., 3.], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.relu6, x)

    def testSigmoid(self):
        for dtype in [tf.float32]:
            x = tf.constant([-2., -1., 0., 1., 2.], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.sigmoid, x)

    def testTanh(self):
        for dtype in [tf.float32]:
            x = tf.constant([-2., -1., 0., 1., 2.], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.tanh, x)

    def testSoftmax(self):
        for dtype in [tf.float32]:
            x = tf.constant([[1., 2., 3.], [1., 1., 1.]], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.softmax, x)

    def testSoftplus(self):
        for dtype in [tf.float32]:
            x = tf.constant([-2., -1., 0., 1., 2.], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.softplus, x)

    def testElu(self):
        for dtype in [tf.float32]:
            x = tf.constant([-2., -1., 0., 1., 2.], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.elu, x)

    def testSelu(self):
        for dtype in [tf.float32]:
            x = tf.constant([-2., -1., 0., 1., 2.], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.selu, x)

    def testLeakyRelu(self):
        for dtype in [tf.float32]:
            x = tf.constant([-2., -1., 0., 1., 2.], dtype=dtype)
            self._testBothEagerAndJit(tf.nn.leaky_relu, x)

    def testGelu(self):
        for dtype in [tf.float32]:
            x = tf.constant([-2., -1., 0., 1., 2.], dtype=dtype)
            self._testBothEagerAndJit(lambda x: tf.nn.gelu(x, approximate=True), x)
