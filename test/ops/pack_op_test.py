import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class PackOpTest(MUSATestCase):

  def _test_pack(self, shape, axis, num_inputs, dtype, rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    inputs_np = []
    for i in range(num_inputs):
      if dtype in [tf.int32, tf.int64]:
        data = np.random.randint(i*10, (i+1)*10, size=shape).astype(np_dtype)
      else:
        data = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
      inputs_np.append(data)
    
    def pack_func(*args):
      return tf.stack(args, axis=axis)
    
    input_tensors = [tf.constant(x, dtype=dtype) for x in inputs_np]
    self._compare_cpu_musa_results(pack_func, input_tensors, dtype, rtol=rtol, atol=atol)

  def testPackBasic(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      rtol = 0 if dtype in [tf.int32] else rtol
      atol = 0 if dtype in [tf.int32] else atol
      self._test_pack([1024, 512], 0, 2, dtype, rtol=rtol, atol=atol)

  def testPackDifferentAxes(self):
    test_cases = [
        ([256, 4096], 1, 4),
        ([1024, 1024], -1, 8),
        ([64, 128, 256], 2, 2),
    ]
    for shape, axis, num_inputs in test_cases:
      self._test_pack(shape, axis, num_inputs, tf.float32)

  def testPackDifferentShapes(self):
    test_cases = [
        ([], 0, 2),
        ([1], 0, 3),
        ([2, 3], 0, 4),
        ([2, 3, 4], 1, 2),
        ([2, 3, 4, 5], 2, 3),
    ]
    for shape, axis, num_inputs in test_cases:
      self._test_pack(shape, axis, num_inputs, tf.float32)
      self._test_pack(shape, axis, num_inputs, tf.int32, rtol=0, atol=0)

  def testPackInt64(self):
    self._test_pack([1024, 512], 0, 2, tf.int64, rtol=0, atol=0)

  def testPackDifferentNumInputs(self):
    for num_inputs in [1, 2, 4, 8, 16]:
      self._test_pack([8, 8], 0, num_inputs, tf.float32)
      self._test_pack([8, 8], 0, num_inputs, tf.int32, rtol=0, atol=0)

  def testPackEdgeCaseSingleInput(self):
    self._test_pack([4, 4], 0, 1, tf.float32)
    self._test_pack([4, 4], 0, 1, tf.int32, rtol=0, atol=0)


if __name__ == "__main__":
  tf.test.main()
