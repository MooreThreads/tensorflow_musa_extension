# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA IsNan operator."""

import os

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class IsNanOpTest(MUSATestCase):
  """Tests for MUSA IsNan operator."""

  @classmethod
  def setUpClass(cls):
    # Keep plugin loading/device checks from MUSATestCase.
    super(IsNanOpTest, cls).setUpClass()
    # Model-level behavior: allow fallback and print device placement logs.
    tf.config.set_soft_device_placement(True)
    # Can be disabled by setting MUSA_LOG_DEVICE_PLACEMENT=0.
    tf.debugging.set_log_device_placement(
        os.environ.get("MUSA_LOG_DEVICE_PLACEMENT", "1") == "1")

  def _compare_cpu_musa_isnan(self, x):
    # Compare TF CPU reference vs. MUSA plugin result exactly (bool output).
    with tf.device("/CPU:0"):
      y_cpu = tf.raw_ops.IsNan(x=x)
    with tf.device("/device:MUSA:0"):
      y_musa = tf.raw_ops.IsNan(x=x)

    self.assertEqual(y_cpu.dtype, tf.bool)
    self.assertEqual(y_musa.dtype, tf.bool)
    self.assertAllEqual(y_cpu.shape, y_musa.shape)
    self.assertAllEqual(y_cpu.numpy(), y_musa.numpy())

  def _make_random_input(self, shape, dtype):
    np_dtype = dtype.as_numpy_dtype
    x_np = np.random.randn(*shape).astype(np_dtype) if shape else np.array(
        np.random.randn(), dtype=np_dtype)

    if x_np.size > 0:
      flat = x_np.reshape(-1)
      # Inject representative edge values: NaN/Inf/zeros/normal numbers.
      flat[0] = np.nan
      if x_np.size > 1:
        flat[1] = np.PINF
      if x_np.size > 2:
        flat[2] = np.NINF
      if x_np.size > 3:
        flat[3] = np_dtype(0.0)
      if x_np.size > 4:
        flat[4] = np_dtype(-0.0)

    return tf.constant(x_np, dtype=dtype)

  def testIsNanNormalShapes(self):
    # Normal elementwise cases across multiple ranks/shapes.
    for dtype in [tf.float16, tf.float32, tf.float64]:
      for shape in [[17], [3, 5], [2, 3, 4], [2, 1, 3, 4]]:
        with self.subTest(dtype=dtype.name, shape=shape):
          x = self._make_random_input(shape, dtype)
          self._compare_cpu_musa_isnan(x)

  def testIsNanEdgeValues(self):
    # Explicitly validate semantics on NaN/Inf/+0/-0 and regular values.
    for dtype in [tf.float16, tf.float32, tf.float64]:
      np_dtype = dtype.as_numpy_dtype
      values = np.array(
          [np.nan, np.PINF, np.NINF, 0.0, -0.0, 1.0, -1.0, 3.25, -7.5],
          dtype=np_dtype)
      x = tf.constant(values, dtype=dtype)
      with self.subTest(dtype=dtype.name):
        self._compare_cpu_musa_isnan(x)

  def testIsNanEmptyTensor(self):
    # Empty input should produce empty bool output with the same shape.
    for dtype in [tf.float16, tf.float32, tf.float64]:
      x = tf.constant(np.array([], dtype=dtype.as_numpy_dtype).reshape(0, 3),
                      dtype=dtype)
      with self.subTest(dtype=dtype.name):
        self._compare_cpu_musa_isnan(x)

  def testIsNanUnsupportedDType(self):
    # Only floating-point dtypes are registered for MUSA IsNan.
    old_soft_placement = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(False)
    try:
      with self.assertRaises(
          (tf.errors.InvalidArgumentError, tf.errors.NotFoundError)):
        with tf.device("/device:MUSA:0"):
          tf.raw_ops.IsNan(x=tf.constant([1, 2, 3], dtype=tf.int32))
    finally:
      tf.config.set_soft_device_placement(old_soft_placement)


if __name__ == "__main__":
  tf.test.main()
