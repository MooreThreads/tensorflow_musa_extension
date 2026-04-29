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

  def testEmptyTensorPreservesDstDtype(self):
    """Empty tensor cast must output DstT dtype, not SrcT dtype.

    Regression test for: Cast(bool->float) on empty tensor returned bool
    output because the early-exit path did `set_output(0, inp)` which kept
    the source dtype.  The fix allocates a new output with the correct DstT.
    """
    # Cover the exact failure case from the Dropout/Cast bug:
    # bool -> float on an empty tensor
    cross_type_pairs = [
        (tf.bool,    tf.float32),
        (tf.bool,    tf.float16),
        (tf.bool,    tf.int32),
        (tf.float32, tf.bool),
        (tf.float32, tf.int32),
        (tf.int32,   tf.float32),
        (tf.int64,   tf.float32),
    ]
    empty_shapes = [
        (0,),
        (0, 4),
        (2, 0, 8),
    ]
    for src_dtype, dst_dtype in cross_type_pairs:
      for shape in empty_shapes:
        with self.subTest(src=src_dtype.name, dst=dst_dtype.name, shape=shape):
          x = tf.zeros(shape, dtype=src_dtype)
          with tf.device('/device:MUSA:0'):
            y = tf.cast(x, dtype=dst_dtype)
          # Shape must be preserved
          self.assertEqual(y.shape.as_list(), list(shape))
          # Output dtype must be DstT, not SrcT
          self.assertEqual(
              y.dtype, dst_dtype,
              msg=f"Cast({src_dtype.name}->{dst_dtype.name}) empty tensor: "
                  f"expected output dtype {dst_dtype.name}, got {y.dtype.name}")

  def testEmptyTensorInGraphMode(self):
    """Verify empty-tensor Cast dtype is correct inside @tf.function (graph mode).

    The original bug manifested during gradient computation where TF constructs
    empty tensors for shape inference inside @tf.function.
    """
    @tf.function
    def cast_in_graph(x):
      with tf.device('/device:MUSA:0'):
        return tf.cast(x, tf.float32)

    for shape in [(0,), (0, 8), (3, 0, 4)]:
      with self.subTest(shape=shape):
        x = tf.zeros(shape, dtype=tf.bool)
        y = cast_in_graph(x)
        self.assertEqual(y.dtype, tf.float32)
        self.assertEqual(y.shape.as_list(), list(shape))


if __name__ == "__main__":
  tf.test.main()
