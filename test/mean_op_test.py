import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class MeanOpTest(MUSATestCase):

  def _test_mean(self, shape, axis, dtype, keepdims=False, rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    data_np = np.random.randn(*shape).astype(np_dtype)
    
    data_tensor = tf.constant(data_np, dtype=dtype)
    
    def mean_func(x):
      return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
    
    self._compare_cpu_musa_results(mean_func, [data_tensor], dtype, rtol=rtol, atol=atol)

  def testMeanAxisLast(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_mean([2, 512, 1024], -1, dtype, rtol=rtol, atol=atol)

  def testMeanAxisFirst(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_mean([4, 256, 512], 0, dtype, rtol=rtol, atol=atol)

  def testMeanAxisMiddle(self):
    for dtype in [tf.float32]:
      self._test_mean([2, 8, 16, 32], 2, dtype)

  def testMeanMultipleAxes(self):
    for dtype in [tf.float32]:
      self._test_mean([4, 8, 16, 32], [1, 2], dtype)

  def testMeanAllAxes(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_mean([2, 4, 8], None, dtype, rtol=rtol, atol=atol)

  def testMeanKeepDims(self):
    for dtype in [tf.float32]:
      atol = 2e-7
      self._test_mean([2, 512, 1024], -1, dtype, keepdims=True, atol=atol)
      self._test_mean([4, 8, 16, 32], [1, 2], dtype, keepdims=True, atol=atol)

  def testMeanDifferentShapes(self):
    test_cases = [
        ([4], -1),
        ([8, 8], 0),
        ([8, 8], 1),
        ([2, 4, 8], [0, 1]),
        ([2, 4, 8, 16], [1, 2, 3]),
    ]
    for dtype in [tf.float32]:
      for shape, axis in test_cases:
        self._test_mean(shape, axis, dtype)

  def testMeanLargeTensor(self):
    for dtype in [tf.float32]:
      atol = 2e-7
      self._test_mean([2, 512, 1024, 2], -2, dtype, atol=atol)


if __name__ == "__main__":
  tf.test.main()
