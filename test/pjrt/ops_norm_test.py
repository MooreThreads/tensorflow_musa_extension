# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: normalization ops test (eager + JIT)."""

import numpy as np
import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase


class NormOpsTest(PjrtTestCase):
    def testSoftmax(self):
        for dtype in [tf.float32]:
            x = tf.constant(np.random.randn(128, 128).astype(np.float32), dtype=dtype)
            self._testBothEagerAndJit(tf.nn.softmax, x)

    def testLogSoftmax(self):
        for dtype in [tf.float32]:
            x = tf.constant(np.random.randn(64, 64).astype(np.float32), dtype=dtype)
            self._testBothEagerAndJit(tf.nn.log_softmax, x)

