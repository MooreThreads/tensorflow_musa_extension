# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: broadcast and shape ops test (eager + JIT)."""

import numpy as np
import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase


class BroadcastOpsTest(PjrtTestCase):
    def testBiasAdd(self):
        for dtype in [tf.float32]:
            a = tf.constant(np.random.randn(128, 64).astype(np.float32), dtype=dtype)
            b = tf.constant(np.random.randn(64).astype(np.float32), dtype=dtype)
            self._testBothEagerAndJit(lambda x, y: x + y, a, b)

    def testBroadcastVectorMatrix(self):
        for dtype in [tf.float32]:
            a = tf.constant(np.random.randn(128, 256).astype(np.float32), dtype=dtype)
            b = tf.constant(np.random.randn(256).astype(np.float32), dtype=dtype)
            self._testBothEagerAndJit(lambda x, y: x * y, a, b)

    def testBroadcastScalar(self):
        for dtype in [tf.float32]:
            a = tf.constant(np.random.randn(64, 128).astype(np.float32), dtype=dtype)
            b = tf.constant(2.0, dtype=dtype)
            self._testBothEagerAndJit(lambda x, y: x + y, a, b)
            self._testBothEagerAndJit(lambda x, y: x * y, a, b)

    def testConcat(self):
        for dtype in [tf.float32]:
            a = tf.constant([[1., 2.]], dtype=dtype)
            b = tf.constant([[3., 4.]], dtype=dtype)
            self._testBothEagerAndJit(lambda x, y: tf.concat([x, y], 0), a, b)

