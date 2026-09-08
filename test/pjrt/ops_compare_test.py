# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: comparison ops test (eager + JIT)."""

import tensorflow_musa  # noqa: F401
import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase


class CompareOpsTest(PjrtTestCase):
    def testEqual(self):
        a = tf.constant([1, 2, 3, 4])
        b = tf.constant([1, 0, 3, 5])
        self._testBothEagerAndJit(lambda x, y: tf.equal(x, y), a, b)

    def testNotEqual(self):
        a = tf.constant([1, 2, 3, 4])
        b = tf.constant([1, 0, 3, 5])
        self._testBothEagerAndJit(lambda x, y: tf.not_equal(x, y), a, b)

    def testGreater(self):
        a = tf.constant([3, 2, 1, 5])
        b = tf.constant([1, 2, 3, 4])
        self._testBothEagerAndJit(lambda x, y: tf.greater(x, y), a, b)

    def testLess(self):
        a = tf.constant([1, 2, 3, 4])
        b = tf.constant([3, 2, 1, 5])
        self._testBothEagerAndJit(lambda x, y: tf.less(x, y), a, b)

    def testGreaterEqual(self):
        a = tf.constant([2, 2, 1, 5])
        b = tf.constant([1, 2, 3, 4])
        self._testBothEagerAndJit(lambda x, y: tf.greater_equal(x, y), a, b)

    def testLessEqual(self):
        a = tf.constant([1, 2, 3, 4])
        b = tf.constant([3, 2, 3, 5])
        self._testBothEagerAndJit(lambda x, y: tf.less_equal(x, y), a, b)
