import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class ResourceApplyFtrlTest(MUSATestCase):
  def _numpy_ftrl(self, var, accum, linear, grad, lr, l1, l2, lr_power, l2_shrinkage=0.0):
    accum_new = accum + grad * grad
    grad_with_shrinkage = grad + 2.0 * l2_shrinkage * var
    
    linear_new = linear + grad_with_shrinkage - (np.power(accum_new, -lr_power) - np.power(accum, -lr_power)) / lr * var
    quadratic = 1.0 / (lr * np.power(accum_new, lr_power)) + 2.0 * l2
    
    var_new = np.where(np.abs(linear_new) > l1,
                       (np.sign(linear_new) * l1 - linear_new) / quadratic,
                       0.0)
    return var_new.astype(var.dtype), accum_new.astype(accum.dtype), linear_new.astype(linear.dtype)

  def _test_dense_logic(self, var_np, accum_np, linear_np, grad_np, lr_np, l1_np, l2_np, lr_power_np, dtype, l2_shrinkage_np=None):
    is_v2 = l2_shrinkage_np is not None
    
    expected_var, expected_accum, expected_linear = self._numpy_ftrl(
        var_np.astype(np.float64), accum_np.astype(np.float64), linear_np.astype(np.float64),
        grad_np.astype(np.float64), lr_np.astype(np.float64), l1_np.astype(np.float64),
        l2_np.astype(np.float64), lr_power_np.astype(np.float64),
        l2_shrinkage_np.astype(np.float64) if is_v2 else 0.0)

    with tf.device("/device:MUSA:0"):
      var = tf.Variable(var_np, dtype=dtype)
      accum = tf.Variable(accum_np, dtype=dtype)
      linear = tf.Variable(linear_np, dtype=dtype)
      
      if is_v2:
        tf.raw_ops.ResourceApplyFtrlV2(
            var=var.handle, accum=accum.handle, linear=linear.handle,
            grad=grad_np, lr=lr_np, l1=l1_np, l2=l2_np, 
            l2_shrinkage=l2_shrinkage_np, lr_power=lr_power_np)
      else:
        tf.raw_ops.ResourceApplyFtrl(
            var=var.handle, accum=accum.handle, linear=linear.handle,
            grad=grad_np, lr=lr_np, l1=l1_np, l2=l2_np, lr_power=lr_power_np)
      
      out_var = var.read_value().numpy()
      out_accum = accum.read_value().numpy()
      out_linear = linear.read_value().numpy()

    tol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
    self.assertAllClose(expected_var, out_var, atol=tol, rtol=tol)
    self.assertAllClose(expected_accum, out_accum, atol=tol, rtol=tol)
    self.assertAllClose(expected_linear, out_linear, atol=tol, rtol=tol)

  def testDenseV1(self):
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      var = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      accum = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      linear = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
      grad = np.array([0.1, -0.1], dtype=dtype.as_numpy_dtype)
      lr = np.array(0.01, dtype=dtype.as_numpy_dtype)
      l1 = np.array(0.0, dtype=dtype.as_numpy_dtype)
      l2 = np.array(0.0, dtype=dtype.as_numpy_dtype)
      lr_power = np.array(-0.5, dtype=dtype.as_numpy_dtype)
      self._test_dense_logic(var, accum, linear, grad, lr, l1, l2, lr_power, dtype)

  def testDenseV2(self):
    for dtype in [tf.float32]:
      var = np.array([0.5, -0.5], dtype=dtype.as_numpy_dtype)
      accum = np.array([0.1, 0.2], dtype=dtype.as_numpy_dtype)
      linear = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      grad = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      lr = np.array(0.01, dtype=dtype.as_numpy_dtype)
      l1 = np.array(0.01, dtype=dtype.as_numpy_dtype)
      l2 = np.array(0.01, dtype=dtype.as_numpy_dtype)
      l2_shrinkage = np.array(0.01, dtype=dtype.as_numpy_dtype)
      lr_power = np.array(-0.5, dtype=dtype.as_numpy_dtype)
      self._test_dense_logic(var, accum, linear, grad, lr, l1, l2, lr_power, dtype, l2_shrinkage)

  def _test_sparse_logic(self, var_np, accum_np, linear_np, grad_np, indices_np, lr_np, l1_np, l2_np, lr_power_np, dtype, l2_shrinkage_np=None):
    is_v2 = l2_shrinkage_np is not None
    expected_var = var_np.copy().astype(np.float64)
    expected_accum = accum_np.copy().astype(np.float64)
    expected_linear = linear_np.copy().astype(np.float64)
    
    for i, idx in enumerate(indices_np):
      if idx < 0: continue
      v_row, a_row, l_row = self._numpy_ftrl(
          expected_var[idx], expected_accum[idx], expected_linear[idx],
          grad_np[i].astype(np.float64), lr_np.astype(np.float64), l1_np.astype(np.float64),
          l2_np.astype(np.float64), lr_power_np.astype(np.float64),
          l2_shrinkage_np.astype(np.float64) if is_v2 else 0.0)
      expected_var[idx] = v_row
      expected_accum[idx] = a_row
      expected_linear[idx] = l_row

    with tf.device("/device:MUSA:0"):
      var = tf.Variable(var_np, dtype=dtype)
      accum = tf.Variable(accum_np, dtype=dtype)
      linear = tf.Variable(linear_np, dtype=dtype)
      indices = tf.constant(indices_np)
      
      if is_v2:
        tf.raw_ops.ResourceSparseApplyFtrlV2(
            var=var.handle, accum=accum.handle, linear=linear.handle,
            grad=grad_np, indices=indices, lr=lr_np, l1=l1_np, l2=l2_np, 
            l2_shrinkage=l2_shrinkage_np, lr_power=lr_power_np)
      else:
        tf.raw_ops.ResourceSparseApplyFtrl(
            var=var.handle, accum=accum.handle, linear=linear.handle,
            grad=grad_np, indices=indices, lr=lr_np, l1=l1_np, l2=l2_np, lr_power=lr_power_np)
      
      out_var = var.read_value().numpy()
      out_accum = accum.read_value().numpy()
      out_linear = linear.read_value().numpy()

    tol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
    self.assertAllClose(expected_var, out_var, atol=tol, rtol=tol)
    self.assertAllClose(expected_accum, out_accum, atol=tol, rtol=tol)
    self.assertAllClose(expected_linear, out_linear, atol=tol, rtol=tol)

  def testSparseV1(self):
    for dtype in [tf.float32]:
      for index_dtype in [np.int32, np.int64]:
        var = np.random.random([10, 4]).astype(dtype.as_numpy_dtype)
        accum = np.random.random([10, 4]).astype(dtype.as_numpy_dtype)
        linear = np.random.random([10, 4]).astype(dtype.as_numpy_dtype)
        indices = np.array([2, 5], dtype=index_dtype)
        grad = np.random.random([2, 4]).astype(dtype.as_numpy_dtype)
        lr = np.array(0.01, dtype=dtype.as_numpy_dtype)
        l1 = np.array(0.0, dtype=dtype.as_numpy_dtype)
        l2 = np.array(0.0, dtype=dtype.as_numpy_dtype)
        lr_power = np.array(-0.5, dtype=dtype.as_numpy_dtype)
        self._test_sparse_logic(var, accum, linear, grad, indices, lr, l1, l2, lr_power, dtype)

  def testSparseV2(self):
    for dtype in [tf.float32]:
      var = np.random.random([10, 4]).astype(dtype.as_numpy_dtype)
      accum = np.random.random([10, 4]).astype(dtype.as_numpy_dtype)
      linear = np.random.random([10, 4]).astype(dtype.as_numpy_dtype)
      indices = np.array([1, 4], dtype=np.int32)
      grad = np.random.random([2, 4]).astype(dtype.as_numpy_dtype)
      lr = np.array(0.01, dtype=dtype.as_numpy_dtype)
      l1 = np.array(0.01, dtype=dtype.as_numpy_dtype)
      l2 = np.array(0.01, dtype=dtype.as_numpy_dtype)
      l2_shrinkage = np.array(0.01, dtype=dtype.as_numpy_dtype)
      lr_power = np.array(-0.5, dtype=dtype.as_numpy_dtype)
      self._test_sparse_logic(var, accum, linear, grad, indices, lr, l1, l2, lr_power, dtype, l2_shrinkage)

if __name__ == "__main__":
  tf.test.main()
