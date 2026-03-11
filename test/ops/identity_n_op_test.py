# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA IdentityN operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class IdentityNOpTest(MUSATestCase):
  """Tests for MUSA IdentityN operator."""

  def testIdentityNBasic(self):
    """Test basic IdentityN operation with multiple inputs."""
    for dtype in [tf.float32, tf.float16, tf.int32, tf.int64]:
      shape1 = [2, 3]
      shape2 = [4]
      
      x1_np = np.random.uniform(-10, 10, size=shape1).astype(dtype.as_numpy_dtype if dtype != tf.bfloat16 else np.float32)
      x2_np = np.random.uniform(-10, 10, size=shape2).astype(dtype.as_numpy_dtype if dtype != tf.bfloat16 else np.float32)
      
      x1 = tf.constant(x1_np, dtype=dtype)
      x2 = tf.constant(x2_np, dtype=dtype)
      
      # IdentityN returns a list of tensors
      def identity_n_wrapper(*args):
        return tf.identity_n(list(args))

      cpu_results = self._test_op_device_placement(identity_n_wrapper, [x1, x2], '/CPU:0')
      musa_results = self._test_op_device_placement(identity_n_wrapper, [x1, x2], '/device:MUSA:0')
      
      for cpu_res, musa_res in zip(cpu_results, musa_results):
        self.assertAllClose(cpu_res.numpy(), musa_res.numpy())

  def testIdentityNSingleInput(self):
    """Test IdentityN with a single input."""
    dtype = tf.float32
    shape = [10, 10]
    x_np = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    x = tf.constant(x_np, dtype=dtype)
    
    def identity_n_wrapper(arg):
      return tf.identity_n([arg])

    cpu_results = self._test_op_device_placement(identity_n_wrapper, [x], '/CPU:0')
    musa_results = self._test_op_device_placement(identity_n_wrapper, [x], '/device:MUSA:0')
    
    for cpu_res, musa_res in zip(cpu_results, musa_results):
      self.assertAllClose(cpu_res.numpy(), musa_res.numpy())

  def testIdentityNManyInputs(self):
    """Test IdentityN with many inputs of different shapes."""
    dtype = tf.float32
    inputs = []
    for i in range(1, 6):
        shape = [i, i]
        x_np = np.random.uniform(-10, 10, size=shape).astype(np.float32)
        inputs.append(tf.constant(x_np, dtype=dtype))
    
    def identity_n_wrapper(*args):
      return tf.identity_n(list(args))

    cpu_results = self._test_op_device_placement(identity_n_wrapper, inputs, '/CPU:0')
    musa_results = self._test_op_device_placement(identity_n_wrapper, inputs, '/device:MUSA:0')
    
    for cpu_res, musa_res in zip(cpu_results, musa_results):
      self.assertAllClose(cpu_res.numpy(), musa_res.numpy())

  def testIdentityNEmptyInput(self):
    """Test IdentityN with empty input list (not supported by TF, should raise)."""
    with self.assertRaises(Exception):
      tf.identity_n([])

if __name__ == "__main__":
  tf.test.main()
