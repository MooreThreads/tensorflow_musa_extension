# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: logical ops test (eager + JIT)."""

import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase


class LogicalOpsTest(PjrtTestCase):
    def testLogicalAnd(self):
        a = tf.constant([True, True, False, False])
        b = tf.constant([True, False, True, False])
        self._testBothEagerAndJit(lambda x, y: tf.logical_and(x, y), a, b)

    def testLogicalOr(self):
        a = tf.constant([True, True, False, False])
        b = tf.constant([True, False, True, False])
        self._testBothEagerAndJit(lambda x, y: tf.logical_or(x, y), a, b)

    def testLogicalNot(self):
        a = tf.constant([True, False, True])
        self._testBothEagerAndJit(tf.logical_not, a)

