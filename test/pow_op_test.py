"""Tests for MUSA Pow operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class PowOpTest(MUSATestCase):

  def _test_pow(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
   
    x_np = np.random.uniform(-1, 1, size=shape_x).astype(np_dtype)
    y_np = np.random.uniform(-1, 1, size=shape_y).astype(np_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.pow, [x, y], dtype, rtol=rtol, atol=atol)

  def testPowDifferentShapes(self):
    """Test add with various different shapes."""
    test_cases = [
        ([1], [1]),
        ([5], [5]),
        ([3, 4], [3, 4]),
        ([2, 3, 4], [2, 3, 4]),
        ([1, 1, 10], [5, 3, 10]),
    ]
    for dtype in [tf.float32]:
      for shape_x, shape_y in test_cases:
        self._test_pow(shape_x, shape_y, dtype)


if __name__ == "__main__":
  tf.test.main()