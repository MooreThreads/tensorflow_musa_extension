import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ProdOpTest(MUSATestCase):

  def _test_prod(self, shape, axis, dtype, keepdims=False, rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if dtype in [tf.int32, tf.int64]:
      data_np = np.random.randint(1, 10, size=shape).astype(np_dtype)
    else:
      data_np = np.random.uniform(0.5, 2.0, size=shape).astype(np_dtype)
    
    data_tensor = tf.constant(data_np, dtype=dtype)
    
    def prod_func(x):
      return tf.reduce_prod(x, axis=axis, keepdims=keepdims)
    
    self._compare_cpu_musa_results(prod_func, [data_tensor], dtype, rtol=rtol, atol=atol)

  def testProdGlobal(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_prod([2, 3, 4], None, dtype, rtol=rtol, atol=atol)

  def testProdAxis0(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_prod([2, 3, 4], 0, dtype, rtol=rtol, atol=atol)

  def testProdAxis1(self):
    for dtype in [tf.float32]:
      self._test_prod([2, 3], 1, dtype)

  def testProdNegativeAxis(self):
    for dtype in [tf.float32]:
      self._test_prod([2, 3, 4], -1, dtype)

  def testProdMultipleAxes(self):
    for dtype in [tf.float32]:
      self._test_prod([2, 3, 4], [0, 2], dtype)

  def testProdKeepDims(self):
    for dtype in [tf.float32]:
      self._test_prod([2, 3, 4], 1, dtype, keepdims=True)

  def testProdInt32(self):
    self._test_prod([2, 3, 4], 0, tf.int32, rtol=0, atol=0)

  def testProdFloat16(self):
    atol = 1e-3
    self._test_prod([2, 3], 0, tf.float16, atol=atol)

  def testProdWithZeros(self):
    data_np = np.array([[1.0, 2.0, 0.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    data_tensor = tf.constant(data_np, dtype=tf.float32)
    
    def prod_func(x):
      return tf.reduce_prod(x, axis=1)
    
    self._compare_cpu_musa_results(prod_func, [data_tensor], tf.float32)

  def testProdWithNegatives(self):
    data_np = np.array([[-1.0, 2.0], [-3.0, -4.0]], dtype=np.float32)
    data_tensor = tf.constant(data_np, dtype=tf.float32)
    
    def prod_func(x):
      return tf.reduce_prod(x, axis=1)
    
    self._compare_cpu_musa_results(prod_func, [data_tensor], tf.float32)

  def testProdEmptyAxis(self):
    data_np = np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    data_tensor = tf.constant(data_np, dtype=tf.float32)
    
    def prod_func(x):
      return tf.reduce_prod(x, axis=[])
    
    self._compare_cpu_musa_results(prod_func, [data_tensor], tf.float32)

  def testProdLargeTensor(self):
    data_tensor = tf.ones([100, 100], dtype=tf.float32)
    
    def prod_func(x):
      return tf.reduce_prod(x)
    
    self._compare_cpu_musa_results(prod_func, [data_tensor], tf.float32)

  def testProdSingleElement(self):
    data_np = np.array([5.0], dtype=np.float32)
    data_tensor = tf.constant(data_np, dtype=tf.float32)
    
    def prod_func(x):
      return tf.reduce_prod(x, axis=0)
    
    self._compare_cpu_musa_results(prod_func, [data_tensor], tf.float32)

  def testProdDifferentShapes(self):
    test_cases = [
        ([4], None),
        ([2, 3], 0),
        ([2, 3], 1),
        ([2, 3, 4], [0, 1]),
        ([2, 3, 4, 5], [1, 2, 3]),
    ]
    for shape, axis in test_cases:
      self._test_prod(shape, axis, tf.float32)


if __name__ == "__main__":
  tf.test.main()
