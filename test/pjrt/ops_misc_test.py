# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: miscellaneous ops test (eager + JIT)."""

import numpy as np
import tensorflow_musa  # noqa: F401
import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase


class MiscOpsTest(PjrtTestCase):
    def testIdentity(self):
        for dtype in [tf.float32]:
            x = tf.constant([1., 2., 3.], dtype=dtype)
            self._testBothEagerAndJit(tf.identity, x)
