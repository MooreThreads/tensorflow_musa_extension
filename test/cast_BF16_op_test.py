# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA Cast (BFloat16) operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class CastBF16OpTest(MUSATestCase):
  """Tests for MUSA Cast operator, focusing on BFloat16."""

  def testFloat32ToBFloat16(self):
    # Test values: normal, near max fp16, large value (fp16 overflow)
    x_np = np.array([1.0, 10.0, 65504.0, 1e10], dtype=np.float32)
    x = tf.constant(x_np)

    self._compare_cpu_musa_results(
        lambda t: tf.cast(t, dtype=tf.bfloat16),
        [x],
        dtype=tf.bfloat16,
        rtol=1e-2, 
        atol=1e-2
    )

  def testBFloat16ToFloat32(self):
    x_np = np.random.uniform(-100, 100, size=[100]).astype(np.float32)
    x_bf16 = tf.cast(tf.constant(x_np), dtype=tf.bfloat16)

    self._compare_cpu_musa_results(
        lambda t: tf.cast(t, dtype=tf.float32),
        [x_bf16],
        dtype=tf.float32,
        rtol=1e-5,
        atol=1e-5
    )

  def testCastLargeShape(self):
    shape = [1024, 1024]
    x_np = np.random.randn(*shape).astype(np.float32)
    x = tf.constant(x_np)

    self._compare_cpu_musa_results(
        lambda t: tf.cast(t, dtype=tf.bfloat16),
        [x],
        dtype=tf.bfloat16,
        rtol=1e-2,
        atol=1e-2
    )


if __name__ == "__main__":
  tf.test.main()
