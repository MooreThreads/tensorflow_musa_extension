# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: matrix operations test (eager + JIT)."""

import numpy as np
import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase


class MatrixOpsTest(PjrtTestCase):
    def testMatMul(self):
        for dtype in [tf.float32]:
            a = tf.constant([[1., 2.], [3., 4.]], dtype=dtype)
            b = tf.constant([[5., 6.], [7., 8.]], dtype=dtype)
            self._testBothEagerAndJit(lambda x, y: tf.matmul(x, y), a, b)

    def testMatMulLarge(self):
        for dtype in [tf.float32]:
            a = tf.constant(np.random.randn(256, 128).astype(np.float32), dtype=dtype)
            b = tf.constant(np.random.randn(128, 256).astype(np.float32), dtype=dtype)
            self._testBothEagerAndJit(lambda x, y: tf.matmul(x, y), a, b)

    def testBatchMatMul(self):
        for dtype in [tf.float32]:
            a = tf.constant(np.random.randn(4, 64, 32).astype(np.float32), dtype=dtype)
            b = tf.constant(np.random.randn(4, 32, 64).astype(np.float32), dtype=dtype)
            self._testBothEagerAndJit(lambda x, y: tf.matmul(x, y), a, b)
