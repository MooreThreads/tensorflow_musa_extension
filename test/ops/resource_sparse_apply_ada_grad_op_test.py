import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class ResourceSparseApplyAdaGradV2Test(MUSATestCase):
  def _test_logic(self, var_np, accum_np, lr_np, epsilon_np, grad_np, indices_np, dtype):
    # Expected manual calculation (always in float32 for reference)
    expected_accum = accum_np.astype(np.float32).copy()
    for i, idx in enumerate(indices_np):
      # Handle negative indices if needed (TF's behavior for invalid indices can vary,
      # but standard sparse ops often ignore or clamp. Our kernel ignores negative.)
      if idx < 0: continue
      expected_accum[idx] += grad_np[i].astype(np.float32)**2

    expected_var = var_np.astype(np.float32).copy()
    for i, idx in enumerate(indices_np):
      if idx < 0: continue
      expected_var[idx] -= lr_np.astype(np.float32) * grad_np[i].astype(np.float32) / (np.sqrt(expected_accum[idx]) + epsilon_np.astype(np.float32))

    with tf.device("/device:MUSA:0"):
      var = tf.Variable(var_np, dtype=dtype)
      accum = tf.Variable(accum_np, dtype=dtype)
      lr = tf.constant(lr_np, dtype=dtype)
      epsilon = tf.constant(epsilon_np, dtype=dtype)
      grad = tf.constant(grad_np, dtype=dtype)
      indices = tf.constant(indices_np)

      tf.raw_ops.ResourceSparseApplyAdagradV2(
          var=var.handle,
          accum=accum.handle,
          lr=lr,
          epsilon=epsilon,
          grad=grad,
          indices=indices,
          use_locking=False)

      out_var = var.read_value().numpy()
      out_accum = accum.read_value().numpy()

    # Using higher tolerance for half precision
      if dtype in [tf.float16, tf.bfloat16]:
        self.assertAllClose(expected_var, out_var, atol=1e-2, rtol=1e-2)
        self.assertAllClose(expected_accum, out_accum, atol=1e-2, rtol=1e-2)
      else:
        self.assertAllClose(expected_var, out_var)
        self.assertAllClose(expected_accum, out_accum)

  def testAllDtypes(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      for index_dtype in [np.int32, np.int64]:
        np_type = dtype.as_numpy_dtype
        var_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_type)
        accum_np = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np_type)
        lr_np = np.array(0.01, dtype=np_type)
        epsilon_np = np.array(1e-7, dtype=np_type)
        grad_np = np.array([[0.1, 0.1]], dtype=np_type)
        indices_np = np.array([1], dtype=index_dtype)
        self._test_logic(var_np, accum_np, lr_np, epsilon_np, grad_np, indices_np, dtype)

  def testLargeHalf(self):
    # Test larger scale with half precision
    for dtype in [tf.float16, tf.bfloat16]:
      np_type = dtype.as_numpy_dtype
      rows = 50
      cols = 32
      var_np = np.random.random([rows, cols]).astype(np_type)
      accum_np = np.random.random([rows, cols]).astype(np_type)
      lr_np = np.array(0.01, dtype=np_type)
      epsilon_np = np.array(1e-4, dtype=np_type)

      num_updates = 10
      indices_np = np.random.choice(rows, num_updates, replace=False).astype(np.int32)
      grad_np = np.random.random([num_updates, cols]).astype(np_type)

      self._test_logic(var_np, accum_np, lr_np, epsilon_np, grad_np, indices_np, dtype)

  def testEmptyIndices(self):
    # Test with empty indices (no-op)
    dtype = tf.float32
    var_np = np.array([[1.0, 2.0]], dtype=np.float32)
    accum_np = np.array([[0.1, 0.1]], dtype=np.float32)
    lr_np = np.array(0.01, dtype=np.float32)
    epsilon_np = np.array(1e-7, dtype=np.float32)
    indices_np = np.array([], dtype=np.int32)
    grad_np = np.zeros([0, 2], dtype=np.float32)

    self._test_logic(var_np, accum_np, lr_np, epsilon_np, grad_np, indices_np, dtype)

  def testLargeStrideRows(self):
    # Test large rows to verify 64-bit indexing (int64 indices)
    dtype = tf.float32
    rows = 200
    cols = 2
    var_np = np.random.random([rows, cols]).astype(np.float32)
    accum_np = np.random.random([rows, cols]).astype(np.float32)
    lr_np = np.array(0.01, dtype=np.float32)
    epsilon_np = np.array(1e-8, dtype=np.float32)
    indices_np = np.array([0, 199], dtype=np.int64)
    grad_np = np.random.random([2, cols]).astype(np.float32)

    self._test_logic(var_np, accum_np, lr_np, epsilon_np, grad_np, indices_np, dtype)

  def testZeroLR(self):
    # Test with learning rate = 0 (var should not change, but accum should)
    dtype = tf.float32
    var_np = np.array([[1.0, 2.0]], dtype=np.float32)
    accum_np = np.array([[0.1, 0.1]], dtype=np.float32)
    lr_np = np.array(0.0, dtype=np.float32)
    epsilon_np = np.array(1e-7, dtype=np.float32)
    indices_np = np.array([0], dtype=np.int32)
    grad_np = np.array([[0.5, 0.5]], dtype=np.float32)

    self._test_logic(var_np, accum_np, lr_np, epsilon_np, grad_np, indices_np, dtype)

if __name__ == "__main__":
  tf.test.main()
