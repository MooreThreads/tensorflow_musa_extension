# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for MUSA ApplyAdagradV2 operators."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase

# Enable device placement logging to verify ops run on MUSA
# Uncomment the line below to see device placement for each op
# tf.debugging.set_log_device_placement(True)


class ApplyAdagradV2OpTest(MUSATestCase):
  """Tests for MUSA ApplyAdagradV2 operators."""

  def setUp(self):
    super(ApplyAdagradV2OpTest, self).setUp()
    # Verify MUSA device is available
    musa_devices = tf.config.list_physical_devices('MUSA')
    self.assertTrue(len(musa_devices) > 0, "No MUSA devices found")
    self.musa_device = musa_devices[0]

  def _numpy_dtype(self, dtype):
    return np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

  def _assert_by_dtype(self, expected, actual, dtype):
    if dtype in [tf.float16, tf.bfloat16]:
      self.assertAllClose(
          np.asarray(expected, dtype=np.float32),
          np.asarray(actual, dtype=np.float32),
          rtol=1e-2,
          atol=1e-2)
    elif dtype == tf.float64:
      self.assertAllClose(expected, actual, rtol=1e-10, atol=1e-12)
    else:
      self.assertAllClose(expected, actual, rtol=1e-5, atol=1e-8)

  def _expected_apply_adagrad_v2(self, var_np, accum_np, lr_np, epsilon_np, grad_np):
    """Compute expected Adagrad V2 update using numpy.

    Adagrad V2 update rule:
      accum = accum + grad * grad
      var = var - lr * grad / (sqrt(accum) + epsilon)
    """
    accum_updated = accum_np + grad_np * grad_np
    var_updated = var_np - lr_np * grad_np / (np.sqrt(accum_updated) + epsilon_np)
    return var_updated, accum_updated

  def _run_resource_apply_adagrad_v2(self,
                                     device,
                                     init_var_np,
                                     init_accum_np,
                                     lr_np,
                                     epsilon_np,
                                     grad_np,
                                     dtype,
                                     use_locking=False):
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(device):
        var = tf.Variable(init_var_np, dtype=dtype, name="var")
        accum = tf.Variable(init_accum_np, dtype=dtype, name="accum")
        grad = tf.constant(grad_np, dtype=dtype, name="grad")

      with tf.device("/CPU:0"):
        lr = tf.constant(lr_np, dtype=dtype, name="lr")
        epsilon = tf.constant(epsilon_np, dtype=dtype, name="epsilon")

      update = tf.raw_ops.ResourceApplyAdagradV2(
          var=var.handle,
          accum=accum.handle,
          lr=lr,
          epsilon=epsilon,
          grad=grad,
          use_locking=use_locking)

      with tf.control_dependencies([update]):
        read_var = tf.identity(var.read_value(), name="updated_var")
        read_accum = tf.identity(accum.read_value(), name="updated_accum")

      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      return sess.run(read_var), sess.run(read_accum)

  def testResourceApplyAdagradV2Basic(self):
    """Test basic ResourceApplyAdagradV2 operation."""
    cases = [
        # 1D case
        (
            np.array([1.0, -2.0, 3.5, -4.5], dtype=np.float32),
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            0.01,
            1e-7,
            np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32),
        ),
        # 2D case
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
            0.01,
            1e-7,
            np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        ),
    ]

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = self._numpy_dtype(dtype)
      for init_var, init_accum, lr, epsilon, grad in cases:
        with self.subTest(dtype=dtype.name, shape=init_var.shape):
          init_var_np = init_var.astype(np_dtype)
          init_accum_np = init_accum.astype(np_dtype)
          grad_np = grad.astype(np_dtype)
          lr_np = np_dtype(lr)
          epsilon_np = np_dtype(epsilon)

          expected_var, expected_accum = self._expected_apply_adagrad_v2(
              init_var_np, init_accum_np, lr_np, epsilon_np, grad_np)

          cpu_var, cpu_accum = self._run_resource_apply_adagrad_v2(
              "/CPU:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)
          musa_var, musa_accum = self._run_resource_apply_adagrad_v2(
              "/device:MUSA:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)

          self._assert_by_dtype(cpu_var, musa_var, dtype)
          self._assert_by_dtype(cpu_accum, musa_accum, dtype)

  def testResourceApplyAdagradV2LargeScale(self):
    """Test ResourceApplyAdagradV2 with larger tensors."""
    np.random.seed(42)

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = self._numpy_dtype(dtype)

      # Large 2D tensor
      shape = (128, 64)
      init_var_np = np.random.random(shape).astype(np_dtype)
      init_accum_np = np.random.random(shape).astype(np_dtype) + 0.01  # Avoid zero accum
      lr_np = np_dtype(0.01)
      epsilon_np = np_dtype(1e-7)
      grad_np = np.random.random(shape).astype(np_dtype) * 0.1

      with self.subTest(dtype=dtype.name, shape=shape):
        cpu_var, cpu_accum = self._run_resource_apply_adagrad_v2(
            "/CPU:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)
        musa_var, musa_accum = self._run_resource_apply_adagrad_v2(
            "/device:MUSA:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_accum, musa_accum, dtype)

  def testResourceApplyAdagradV2WithUseLocking(self):
    """Test ResourceApplyAdagradV2 with use_locking=True."""
    init_var_np = np.array([1.0, -2.0, 3.5, -4.5], dtype=np.float32)
    init_accum_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    lr_np = np.float32(0.01)
    epsilon_np = np.float32(1e-7)
    grad_np = np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32)

    cpu_var, cpu_accum = self._run_resource_apply_adagrad_v2(
        "/CPU:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np,
        tf.float32, use_locking=True)
    musa_var, musa_accum = self._run_resource_apply_adagrad_v2(
        "/device:MUSA:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np,
        tf.float32, use_locking=True)

    self._assert_by_dtype(cpu_var, musa_var, tf.float32)
    self._assert_by_dtype(cpu_accum, musa_accum, tf.float32)

  def testResourceApplyAdagradV2ZeroLearningRate(self):
    """Test ResourceApplyAdagradV2 with lr=0 (var unchanged, accum updated)."""
    dtype = tf.float32
    init_var_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    init_accum_np = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    lr_np = np.float32(0.0)
    epsilon_np = np.float32(1e-7)
    grad_np = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    cpu_var, cpu_accum = self._run_resource_apply_adagrad_v2(
        "/CPU:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)
    musa_var, musa_accum = self._run_resource_apply_adagrad_v2(
        "/device:MUSA:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)

    # Var should remain unchanged (lr=0)
    self._assert_by_dtype(init_var_np, musa_var, dtype)
    # Accum should be updated
    self._assert_by_dtype(cpu_accum, musa_accum, dtype)

  def testResourceApplyAdagradV2SmallEpsilon(self):
    """Test ResourceApplyAdagradV2 with very small epsilon."""
    dtype = tf.float32
    init_var_np = np.array([1.0, 2.0], dtype=np.float32)
    init_accum_np = np.array([1e-8, 1e-8], dtype=np.float32)  # Very small accum
    lr_np = np.float32(0.1)
    epsilon_np = np.float32(1e-10)  # Very small epsilon
    grad_np = np.array([0.1, 0.2], dtype=np.float32)

    cpu_var, cpu_accum = self._run_resource_apply_adagrad_v2(
        "/CPU:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)
    musa_var, musa_accum = self._run_resource_apply_adagrad_v2(
        "/device:MUSA:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_accum, musa_accum, dtype)

  def testResourceApplyAdagradV2ZeroGradient(self):
    """Test ResourceApplyAdagradV2 with zero gradient."""
    dtype = tf.float32
    init_var_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    init_accum_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr_np = np.float32(0.01)
    epsilon_np = np.float32(1e-7)
    grad_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    cpu_var, cpu_accum = self._run_resource_apply_adagrad_v2(
        "/CPU:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)
    musa_var, musa_accum = self._run_resource_apply_adagrad_v2(
        "/device:MUSA:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)

    # Both var and accum should remain unchanged (grad=0)
    self._assert_by_dtype(init_var_np, musa_var, dtype)
    self._assert_by_dtype(init_accum_np, musa_accum, dtype)

  def testResourceApplyAdagradV2NegativeValues(self):
    """Test ResourceApplyAdagradV2 with negative values."""
    dtype = tf.float32
    init_var_np = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    init_accum_np = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    lr_np = np.float32(0.01)
    epsilon_np = np.float32(1e-7)
    grad_np = np.array([-0.5, 0.5, -1.0], dtype=np.float32)

    cpu_var, cpu_accum = self._run_resource_apply_adagrad_v2(
        "/CPU:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)
    musa_var, musa_accum = self._run_resource_apply_adagrad_v2(
        "/device:MUSA:0", init_var_np, init_accum_np, lr_np, epsilon_np, grad_np, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_accum, musa_accum, dtype)

  def testDevicePlacement(self):
    """Verify that ResourceApplyAdagradV2 runs on MUSA device."""
    dtype = tf.float32
    init_var_np = np.array([1.0, 2.0], dtype=np.float32)
    init_accum_np = np.array([0.1, 0.1], dtype=np.float32)
    lr_np = np.float32(0.01)
    epsilon_np = np.float32(1e-7)
    grad_np = np.array([0.5, 0.5], dtype=np.float32)

    # Create variables on MUSA device and check their device
    with tf.device("/device:MUSA:0"):
      var = tf.Variable(init_var_np, dtype=dtype, name="var_musa")
      accum = tf.Variable(init_accum_np, dtype=dtype, name="accum_musa")

      # Verify variables are placed on MUSA
      self.assertIn("MUSA", var.device)
      self.assertIn("MUSA", accum.device)

      # Run the operation
      lr = tf.constant(lr_np, dtype=dtype)
      epsilon = tf.constant(epsilon_np, dtype=dtype)
      grad = tf.constant(grad_np, dtype=dtype)

      # Use tf.function to trace execution
      @tf.function
      def run_adagrad():
        tf.raw_ops.ResourceApplyAdagradV2(
            var=var.handle,
            accum=accum.handle,
            lr=lr,
            epsilon=epsilon,
            grad=grad,
            use_locking=False)
        return var.read_value(), accum.read_value()

      # Execute and verify results
      result_var, result_accum = run_adagrad()

      # Check that result tensors are on MUSA device
      self.assertIn("MUSA", result_var.device)
      self.assertIn("MUSA", result_accum.device)

      # Verify correctness
      expected_var, expected_accum = self._expected_apply_adagrad_v2(
          init_var_np, init_accum_np, lr_np, epsilon_np, grad_np)
      self._assert_by_dtype(expected_var, result_var.numpy(), dtype)
      self._assert_by_dtype(expected_accum, result_accum.numpy(), dtype)

  def testOpKernelRegisteredOnMUSA(self):
    """Verify that ResourceApplyAdagradV2 kernel is registered for MUSA device."""
    # Check if the kernel is available for MUSA by inspecting registered kernels
    # This is done indirectly by successfully running the op on MUSA

    with tf.device("/device:MUSA:0"):
      var = tf.Variable([1.0], dtype=tf.float32)
      accum = tf.Variable([0.1], dtype=tf.float32)

      # This will fail if kernel is not registered for MUSA
      try:
        tf.raw_ops.ResourceApplyAdagradV2(
            var=var.handle,
            accum=accum.handle,
            lr=tf.constant(0.01),
            epsilon=tf.constant(1e-7),
            grad=tf.constant([0.1]),
            use_locking=False)
        # If we reach here, the kernel is registered
        kernel_registered = True
      except Exception as e:
        # Check if error is about missing kernel
        if "Could not satisfy explicit device specification" in str(e) or \
           "no supported kernel" in str(e):
          kernel_registered = False
        else:
          raise

    self.assertTrue(kernel_registered,
                    "ResourceApplyAdagradV2 kernel is NOT registered for MUSA device")


if __name__ == "__main__":
  tf.test.main()