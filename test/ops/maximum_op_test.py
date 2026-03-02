"""Tests for MUSA Maximum operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class MaximumOpTest(MUSATestCase):
  """Tests for MUSA Maximum operator."""

  def _test_maximum(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    """Test maximum operation with given shapes and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    x_np = np.random.uniform(-2, 2, size=shape_x).astype(np_dtype)
    y_np = np.random.uniform(-2, 2, size=shape_y).astype(np_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.maximum, [x, y], dtype, rtol=rtol, atol=atol)

  def testMaximumBasic(self):
    """Test basic maximum operation with same shapes."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_maximum([1024, 1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testMaximumBroadcastVectorMatrix(self):
    """Test maximum with vector-matrix broadcasting."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_maximum([1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testMaximumBroadcastColumnRow(self):
    """Test maximum with column-row broadcasting."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_maximum([1024, 1], [1, 1024], dtype, rtol=rtol, atol=atol)

  def testMaximumScalar(self):
    """Test maximum with scalar values."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_maximum([], [], dtype, rtol=rtol, atol=atol)

  def testMaximumDifferentShapes(self):
    """Test maximum with various different shapes."""
    test_cases = [
        ([1], [1]),
        ([5], [5]),
        ([3, 4], [3, 4]),
        ([2, 3, 4], [2, 3, 4]),
        ([1, 1, 10], [5, 3, 10]),
        ([2, 1, 3], [2, 4, 3]),
    ]
    for dtype in [tf.float32]:
      for shape_x, shape_y in test_cases:
        self._test_maximum(shape_x, shape_y, dtype)

  def testMaximumSpecialValues(self):
    """Test maximum with special values including negative numbers."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      # 创建包含负数的测试数据
      x_np = np.array([1.0, -2.0, 3.0, -4.0, 0.0], dtype=np.float32)
      y_np = np.array([-1.0, 2.0, -3.0, 4.0, 0.0], dtype=np.float32)
      
      if dtype != tf.float32:
        x_np = x_np.astype(np.float16) if dtype == tf.float16 else x_np
        y_np = y_np.astype(np.float16) if dtype == tf.float16 else y_np
      
      x = tf.constant(x_np, dtype=dtype)
      y = tf.constant(y_np, dtype=dtype)
      
      self._compare_cpu_musa_results(tf.maximum, [x, y], dtype, rtol=rtol, atol=atol)

  def testMaximumLargeTensors(self):
    """Test maximum with large tensors for performance validation."""
    for dtype in [tf.float32]:
      self._test_maximum([1024, 1024, 4], [1024, 1024, 4], dtype)


if __name__ == "__main__":
  tf.test.main()
