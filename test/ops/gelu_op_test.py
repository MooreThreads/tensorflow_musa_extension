# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA GELU gradient."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase, load_musa_ops
from tensorflow_musa import ops as musa_ops


class GeluOpTest(MUSATestCase):
    @classmethod
    def setUpClass(cls):
        super(GeluOpTest, cls).setUpClass()
        cls._musa_ops = load_musa_ops()

    def _test_gelu_grad(self, dtype, approximate):
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
        x_np = np.linspace(-3.0, 3.0, 257).astype(np_dtype)
        dy_np = np.random.uniform(-1.0, 1.0, size=x_np.shape).astype(np_dtype)

        with tf.device("/CPU:0"):
            x_cpu = tf.constant(x_np, dtype=dtype)
            dy_cpu = tf.constant(dy_np, dtype=dtype)
            with tf.GradientTape() as tape:
                tape.watch(x_cpu)
                y_cpu = tf.nn.gelu(x_cpu, approximate=approximate)
            dx_cpu = tape.gradient(y_cpu, x_cpu, output_gradients=dy_cpu)

        with tf.device("/device:MUSA:0"):
            x = tf.constant(x_np, dtype=dtype)
            dy = tf.constant(dy_np, dtype=dtype)
            dx = self._musa_ops.musa_gelu_grad(
                dy=dy, x=x, approximate=approximate
            )

        rtol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-4
        atol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-4
        self.assertAllClose(
            tf.cast(dx_cpu, tf.float32).numpy(),
            tf.cast(dx, tf.float32).numpy(),
            rtol=rtol,
            atol=atol,
        )

    def _test_gelu_gradient_tape(self, dtype, approximate):
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
        x_np = np.linspace(-3.0, 3.0, 257).astype(np_dtype)

        with tf.device("/CPU:0"):
            x_cpu = tf.constant(x_np, dtype=dtype)
            with tf.GradientTape() as tape:
                tape.watch(x_cpu)
                y_cpu = tf.nn.gelu(x_cpu, approximate=approximate)
                loss_cpu = tf.reduce_sum(tf.square(y_cpu))
            dx_cpu = tape.gradient(loss_cpu, x_cpu)

        with tf.device("/device:MUSA:0"):
            x = tf.constant(x_np, dtype=dtype)
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = musa_ops.gelu(x, approximate=approximate)
                loss = tf.reduce_sum(tf.square(y))
            dx = tape.gradient(loss, x)

        rtol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-4
        atol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-4
        self.assertAllClose(
            tf.cast(dx_cpu, tf.float32).numpy(),
            tf.cast(dx, tf.float32).numpy(),
            rtol=rtol,
            atol=atol,
        )

    def testGeluGradMatchesTensorFlow(self):
        for dtype in [tf.float32, tf.float16]:
            for approximate in [False, True]:
                with self.subTest(dtype=dtype, approximate=approximate):
                    self._test_gelu_grad(dtype, approximate)

    def testGeluGradientTape(self):
        for dtype in [tf.float32, tf.float16]:
            for approximate in [False, True]:
                with self.subTest(dtype=dtype, approximate=approximate):
                    self._test_gelu_gradient_tape(dtype, approximate)


if __name__ == "__main__":
    tf.test.main()
