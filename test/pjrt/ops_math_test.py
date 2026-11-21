# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
"""PJRT mode: math operations test (eager + JIT)."""

import numpy as np
import tensorflow_musa  # noqa: F401
import tensorflow as tf
from pjrt.pjrt_test_utils import PjrtTestCase

DTYPES = [tf.float32]
SHAPES = [(128,), (64, 64), (256, 256)]


class MathOpsTest(PjrtTestCase):
    def _test_binary(self, fn, shape_a, shape_b, dtype):
        a = tf.constant(np.random.uniform(-1, 1, shape_a).astype(dtype.as_numpy_dtype), dtype=dtype)
        b = tf.constant(np.random.uniform(0.1, 1, shape_b).astype(dtype.as_numpy_dtype), dtype=dtype)
        self._testBothEagerAndJit(lambda x, y: fn(x, y), a, b, rtol=1e-5)

    def _test_unary(self, fn, shape, dtype):
        x = tf.constant(np.random.uniform(-1, 1, shape).astype(dtype.as_numpy_dtype), dtype=dtype)
        self._testBothEagerAndJit(fn, x, rtol=1e-5)

    def testAdd(self):
        for dtype in DTYPES:
            for s in [(128,), (64, 64)]:
                a = tf.constant(np.ones(s, dtype.as_numpy_dtype), dtype=dtype)
                b = tf.constant(np.ones(s, dtype.as_numpy_dtype) * 2, dtype=dtype)
                self._testBothEagerAndJit(lambda x, y: x + y, a, b)

    def testSub(self):
        for dtype in DTYPES:
            for s in [(128,), (64, 64)]:
                a = tf.constant(np.ones(s, dtype.as_numpy_dtype) * 5, dtype=dtype)
                b = tf.constant(np.ones(s, dtype.as_numpy_dtype) * 3, dtype=dtype)
                self._testBothEagerAndJit(lambda x, y: x - y, a, b)

    def testMul(self):
        for dtype in DTYPES:
            for s in [(128,), (64, 64)]:
                a = tf.constant(np.ones(s, dtype.as_numpy_dtype) * 2, dtype=dtype)
                b = tf.constant(np.ones(s, dtype.as_numpy_dtype) * 3, dtype=dtype)
                self._testBothEagerAndJit(lambda x, y: x * y, a, b)

    def testDiv(self):
        for dtype in DTYPES:
            for s in [(128,), (64, 64)]:
                a = tf.constant(np.ones(s, dtype.as_numpy_dtype) * 6, dtype=dtype)
                b = tf.constant(np.ones(s, dtype.as_numpy_dtype) * 2, dtype=dtype)
                self._testBothEagerAndJit(lambda x, y: x / y, a, b)

    def testNeg(self):
        for dtype in DTYPES:
            for s in [(128,), (64, 64)]:
                x = tf.constant(np.ones(s, dtype.as_numpy_dtype) * 3, dtype=dtype)
                self._testBothEagerAndJit(lambda x: -x, x)

    def testAbs(self):
        for dtype in DTYPES:
            for s in [(128,)]:
                x = tf.constant([-1.0, 2.0, -3.0, 4.0], dtype=dtype)
                self._testBothEagerAndJit(tf.abs, x)

    def testPow(self):
        for dtype in DTYPES:
            a = tf.constant([2.0, 3.0, 4.0], dtype=dtype)
            b = tf.constant([3.0, 2.0, 0.5], dtype=dtype)
            self._testBothEagerAndJit(lambda x, y: tf.pow(x, y), a, b)

    def testSqrt(self):
        for dtype in DTYPES:
            x = tf.constant([1.0, 4.0, 9.0, 16.0], dtype=dtype)
            self._testBothEagerAndJit(tf.sqrt, x)

    def testSquare(self):
        for dtype in DTYPES:
            x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=dtype)
            self._testBothEagerAndJit(tf.square, x)
