import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class RealDivOpTest(MUSATestCase):

  def _test_realdiv(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    x_np = np.random.uniform(1, 10, size=shape_x).astype(np_dtype)
    y_np = np.random.uniform(1, 5, size=shape_y).astype(np_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.divide, [x, y], dtype, rtol=rtol, atol=atol)

  def testRealDivBasic(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_realdiv([1024, 1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testRealDivScalarBroadcast(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_realdiv([], [1024, 1024], dtype, rtol=rtol, atol=atol)
      self._test_realdiv([1024, 1024], [], dtype, rtol=rtol, atol=atol)

  def testRealDivMatrixBroadcast(self):
    for dtype in [tf.float32]:
      self._test_realdiv([1024, 1], [1, 1024], dtype)
      self._test_realdiv([1024], [1024, 1024], dtype)

  def testRealDivDifferentShapes(self):
    test_cases = [
        ([4], [4]),
        ([2, 3], [2, 3]),
        ([1, 5], [5, 1]),
        ([2, 3, 4], [2, 3, 4]),
        ([1, 1, 10], [5, 3, 10]),
        ([2, 1, 3], [2, 4, 3]),
    ]
    for dtype in [tf.float32]:
      for shape_x, shape_y in test_cases:
        self._test_realdiv(shape_x, shape_y, dtype)

  def testRealDivExactValues(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      x_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
      y_np = np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float32)
      
      if dtype != tf.float32:
        x_np = x_np.astype(np.float16) if dtype == tf.float16 else x_np
        y_np = y_np.astype(np.float16) if dtype == tf.float16 else y_np
      
      x = tf.constant(x_np, dtype=dtype)
      y = tf.constant(y_np, dtype=dtype)
      
      self._compare_cpu_musa_results(tf.divide, [x, y], dtype, rtol=rtol, atol=atol)

  def testRealDivScalarExact(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      scalar_np = np.array(100.0, dtype=np.float32)
      vector_np = np.array([2.0, 4.0, 5.0, 10.0], dtype=np.float32)
      
      if dtype != tf.float32:
        scalar_np = scalar_np.astype(np.float16) if dtype == tf.float16 else scalar_np
        vector_np = vector_np.astype(np.float16) if dtype == tf.float16 else vector_np
      
      scalar = tf.constant(scalar_np, dtype=dtype)
      vector = tf.constant(vector_np, dtype=dtype)
      
      def scalar_div_vector(x, y):
        return tf.divide(x, y)
      
      def vector_div_scalar(x, y):
        return tf.divide(x, y)
      
      self._compare_cpu_musa_results(scalar_div_vector, [scalar, vector], dtype, rtol=rtol, atol=atol)
      self._compare_cpu_musa_results(vector_div_scalar, [vector, scalar], dtype, rtol=rtol, atol=atol)

  def testRealDivMatrixExact(self):
    matrix_np = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]], dtype=np.float32)
    vector_np = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    vector2_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    
    matrix = tf.constant(matrix_np, dtype=tf.float32)
    vector = tf.constant(vector_np, dtype=tf.float32)
    vector2 = tf.constant(vector2_np, dtype=tf.float32)
    
    def matrix_div_vector(x, y):
      return tf.divide(x, y)
    
    def vector_div_matrix(x, y):
      return tf.divide(x, y)
    
    self._compare_cpu_musa_results(matrix_div_vector, [matrix, vector], tf.float32)
    self._compare_cpu_musa_results(vector_div_matrix, [vector2, matrix], tf.float32)

  def testRealDivLargeTensors(self):
    for dtype in [tf.float32]:
      self._test_realdiv([1024, 1024, 2], [1024, 1024, 2], dtype)


if __name__ == "__main__":
  tf.test.main()
