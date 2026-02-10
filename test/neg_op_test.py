import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class NegOpTest(MUSATestCase):

  def _test_neg(self, shape, dtype, rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    data_np = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
    
    data_tensor = tf.constant(data_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.math.negative, [data_tensor], dtype, rtol=rtol, atol=atol)

  def testNegFloatBasic(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_neg([5], dtype, rtol=rtol, atol=atol)

  def testNegIntBasic(self):
    self._test_neg([5], tf.int32, rtol=0, atol=0)
    self._test_neg([5], tf.int64, rtol=0, atol=0)

  def testNegLargeTensor(self):
    for dtype in [tf.float32, tf.int32]:
      rtol = 0 if dtype in [tf.int32] else 1e-5
      atol = 0 if dtype in [tf.int32] else 1e-8
      self._test_neg([1024, 1024], dtype, rtol=rtol, atol=atol)

  def testNegDifferentShapes(self):
    test_shapes = [[], [1], [5], [2, 3], [2, 3, 4], [2, 3, 4, 5]]
    for shape in test_shapes:
      self._test_neg(shape, tf.float32)
      self._test_neg(shape, tf.int32, rtol=0, atol=0)

  def testNegSpecialValues(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      data_np = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
      if dtype != tf.float32:
        data_np = data_np.astype(np.float16) if dtype == tf.float16 else data_np
      
      data_tensor = tf.constant(data_np, dtype=dtype)
      
      self._compare_cpu_musa_results(tf.math.negative, [data_tensor], dtype, rtol=rtol, atol=atol)

  def testNegIntSpecialValues(self):
    data_np = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    data_tensor = tf.constant(data_np, dtype=tf.int32)
    self._compare_cpu_musa_results(tf.math.negative, [data_tensor], tf.int32, rtol=0, atol=0)


if __name__ == "__main__":
  tf.test.main()
