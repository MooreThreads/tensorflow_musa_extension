import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ReciprocalOpTest(MUSATestCase):

  def _test_reciprocal(self, shape, dtype, rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16]:
      x_np = np.random.uniform(0.1, 5, size=shape).astype(np_dtype)
      x_np = np.where(x_np == 0, 0.1, x_np)
    else:
      x_np = np.random.randint(1, 10, size=shape).astype(np_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.math.reciprocal, [x], dtype, rtol=rtol, atol=atol)

  def testReciprocalBasic(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_reciprocal([4], dtype, rtol=rtol, atol=atol)
      self._test_reciprocal([2, 3], dtype, rtol=rtol, atol=atol)

  def testReciprocalDifferentShapes(self):
    test_shapes = [[], [1], [5], [2, 3], [2, 3, 4], [1, 5], [5, 1]]
    for shape in test_shapes:
      self._test_reciprocal(shape, tf.float32)

  def testReciprocalLargeTensor(self):
    self._test_reciprocal([1024, 1024], tf.float32)
    self._test_reciprocal([2048], tf.float32)

  def testReciprocalEdgeCases(self):
    test_np = np.array([0.5, 1.0, 2.0, 4.0, 10.0], dtype=np.float32)
    test_tensor = tf.constant(test_np, dtype=tf.float32)
    
    self._compare_cpu_musa_results(tf.math.reciprocal, [test_tensor], tf.float32)

  def testReciprocalFloat16(self):
    self._test_reciprocal([4, 4], tf.float16)

  def testReciprocalBfloat16(self):
    self._test_reciprocal([3, 3], tf.bfloat16)

  def testReciprocalExactValues(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      test_np = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0], dtype=np.float32)
      if dtype != tf.float32:
        test_np = test_np.astype(np.float16) if dtype == tf.float16 else test_np
      
      test_tensor = tf.constant(test_np, dtype=dtype)
      self._compare_cpu_musa_results(tf.math.reciprocal, [test_tensor], dtype, rtol=rtol, atol=atol)


if __name__ == "__main__":
  tf.test.main()
