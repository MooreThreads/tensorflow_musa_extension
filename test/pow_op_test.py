import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class PowOpTest(MUSATestCase):

  def _test_pow(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if shape_x:
      x_np = np.abs(np.random.randn(*shape_x)).astype(np_dtype) + 0.1
    else:
      x_np = np.array(np.abs(np.random.randn()) + 0.1, dtype=np_dtype)
    
    if shape_y:
      y_np = np.random.randn(*shape_y).astype(np_dtype)
    else:
      y_np = np.array(np.random.randn(), dtype=np_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.math.pow, [x, y], dtype, rtol=rtol, atol=atol)

  def testPowBasic(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_pow([1024, 1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testPowBroadcastVectorMatrix(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_pow([1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testPowBroadcastColumnRow(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_pow([1024, 1], [1, 1024], dtype, rtol=rtol, atol=atol)

  def testPowScalar(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_pow([], [], dtype, rtol=rtol, atol=atol)

  def testPowFloat64(self):
    self._test_pow([1024, 1024], [1024, 1024], tf.float64)

  def testPowDifferentShapes(self):
    test_cases = [
        ([5, 5], [5, 5]),
        ([5, 5], [1, 1]),
        ([5, 5], [1, 5]),
        ([1, 5], [5, 1]),
        ([2, 3, 4], [2, 3, 4]),
        ([1, 1, 10], [5, 3, 10]),
        ([2, 1, 3], [2, 4, 3]),
    ]
    for dtype in [tf.float32]:
      for shape_x, shape_y in test_cases:
        self._test_pow(shape_x, shape_y, dtype)

  def testPowLargeTensors(self):
    for dtype in [tf.float32]:
      self._test_pow([1024, 1024, 2], [1024, 1024, 2], dtype)


if __name__ == "__main__":
  tf.test.main()
