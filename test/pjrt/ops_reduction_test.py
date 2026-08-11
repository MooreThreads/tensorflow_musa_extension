# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: reduction ops test (eager + JIT)."""

import numpy as np
import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase


class ReductionOpsTest(PjrtTestCase):
    def testReduceSum1D(self):
        for dtype in [tf.float32]:
            x = tf.constant([1., 2., 3., 4., 5.], dtype=dtype)
            self._testBothEagerAndJit(tf.reduce_sum, x)

    def testReduceMean1D(self):
        for dtype in [tf.float32]:
            x = tf.constant([1., 2., 3., 4., 5.], dtype=dtype)
            self._testBothEagerAndJit(tf.reduce_mean, x)

