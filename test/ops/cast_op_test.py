# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA Cast operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class CastOpTest(MUSATestCase):
  """Tests for MUSA Cast operator."""

  def _test_cast(self, src_dtype, dst_dtype, shape=(10, 10)):
    """
    Helper function to test casting between types.
    """
    # 1. Generate random data based on source dtype
    if src_dtype == tf.bool:
        x_np = np.random.choice([True, False], size=shape)
    elif src_dtype.is_integer:
        # Integers: range [-100, 100]
        x_np = np.random.randint(-100, 100, size=shape).astype(src_dtype.as_numpy_dtype)
    else:
        # Floats: range [-10.0, 10.0]
        x_np = (np.random.rand(*shape) * 20 - 10).astype(src_dtype.as_numpy_dtype)

    x = tf.constant(x_np, dtype=src_dtype)

    # 2. Define wrapper
    def op_wrapper(input_tensor):
        return tf.cast(input_tensor, dtype=dst_dtype)

    # 3. Compare CPU vs MUSA
    self._compare_cpu_musa_results(
        op_wrapper,
        [x],
        dtype=dst_dtype
    )

  def testFloat32ToInt32(self):
    """Test Float32 -> Int32 (Truncation check)."""
    self._test_cast(tf.float32, tf.int32)

  def testInt64ToFloat32(self):
    """Test Int64 -> Float32 (Embedding ID preprocessing)."""
    self._test_cast(tf.int64, tf.float32)

  def testFloat32ToFloat16(self):
    """Test Float32 -> Float16 (Mixed Precision)."""
    # Note: The tool class automatically handles casting fp16 back to fp32 for comparison
    self._test_cast(tf.float32, tf.float16)

  def testBoolToFloat32(self):
    """Test Bool -> Float32 (Masking operations)."""
    self._test_cast(tf.bool, tf.float32)

  def testInt32ToInt64(self):
    """Test Int32 -> Int64 (Safe upcasting)."""
    self._test_cast(tf.int32, tf.int64)


if __name__ == "__main__":
  tf.test.main()
