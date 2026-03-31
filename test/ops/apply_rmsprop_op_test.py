# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================
#
# Tests for MUSA ResourceApplyRMSProp and ResourceApplyCenteredRMSProp operators.

"""Tests for MUSA ResourceApplyRMSProp and ResourceApplyCenteredRMSProp operators."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ResourceApplyRMSPropTest(MUSATestCase):
  """Tests for MUSA ResourceApplyRMSProp operator."""

  def _run_resource_apply_rmsprop(self, device, init_var, init_ms, init_mom, grad_val,
                                   lr=0.01, rho=0.9, momentum=0.9, epsilon=1e-8):
    """Run one ResourceApplyRMSProp update in graph mode on the requested device."""
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(device):
        var = tf.Variable(init_var, dtype=tf.float32, name="var")
        ms = tf.Variable(init_ms, dtype=tf.float32, name="ms")
        mom = tf.Variable(init_mom, dtype=tf.float32, name="mom")
        grad = tf.constant(grad_val, dtype=tf.float32, name="grad")

      # Resource handles and scalar hyper-parameters stay on host memory.
      with tf.device("/CPU:0"):
        lr_t = tf.constant(lr, dtype=tf.float32, name="lr")
        rho_t = tf.constant(rho, dtype=tf.float32, name="rho")
        momentum_t = tf.constant(momentum, dtype=tf.float32, name="momentum")
        epsilon_t = tf.constant(epsilon, dtype=tf.float32, name="epsilon")

      update = tf.raw_ops.ResourceApplyRMSProp(
          var=var.handle,
          ms=ms.handle,
          mom=mom.handle,
          lr=lr_t,
          rho=rho_t,
          momentum=momentum_t,
          epsilon=epsilon_t,
          grad=grad,
          use_locking=False)

      with tf.control_dependencies([update]):
        read_var = tf.identity(var.read_value(), name="updated_var")
        read_ms = tf.identity(ms.read_value(), name="updated_ms")
        read_mom = tf.identity(mom.read_value(), name="updated_mom")

      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      return sess.run([read_var, read_ms, read_mom])

  def _compute_expected_rmsprop(self, var, ms, mom, grad, lr, rho, momentum, epsilon):
    """Compute expected RMSProp update using NumPy."""
    # ms <- rho * ms + (1-rho) * grad^2
    new_ms = rho * ms + (1 - rho) * grad * grad
    # mom <- momentum * mom + lr * grad / sqrt(ms + epsilon)
    new_mom = momentum * mom + lr * grad / np.sqrt(new_ms + epsilon)
    # var <- var - mom
    new_var = var - new_mom
    return new_var, new_ms, new_mom

  def test_rmsprop_basic_update(self):
    """Test basic RMSProp update on MUSA device."""
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_ms = np.zeros(4, dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)
    grad_val = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    lr = 0.01
    rho = 0.9
    momentum = 0.9
    epsilon = 1e-8

    # Run on MUSA
    result_var, result_ms, result_mom = self._run_resource_apply_rmsprop(
        "/device:MUSA:0", init_var, init_ms, init_mom, grad_val,
        lr=lr, rho=rho, momentum=momentum, epsilon=epsilon)

    # Compute expected
    expected_var, expected_ms, expected_mom = self._compute_expected_rmsprop(
        init_var, init_ms, init_mom, grad_val, lr, rho, momentum, epsilon)

    self.assertAllClose(result_var, expected_var, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_ms, expected_ms, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_mom, expected_mom, rtol=1e-5, atol=1e-6)

  def test_rmsprop_multiple_steps(self):
    """Test RMSProp with multiple update steps."""
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_ms = np.zeros(4, dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)

    lr = 0.01
    rho = 0.9
    momentum = 0.9
    epsilon = 1e-8

    var = init_var.copy()
    ms = init_ms.copy()
    mom = init_mom.copy()

    for step in range(3):
      grad_val = np.array([0.1 * (step + 1), 0.2 * (step + 1),
                           0.3 * (step + 1), 0.4 * (step + 1)], dtype=np.float32)

      result_var, result_ms, result_mom = self._run_resource_apply_rmsprop(
          "/device:MUSA:0", var, ms, mom, grad_val,
          lr=lr, rho=rho, momentum=momentum, epsilon=epsilon)

      # Compute expected
      expected_var, expected_ms, expected_mom = self._compute_expected_rmsprop(
          var, ms, mom, grad_val, lr, rho, momentum, epsilon)

      self.assertAllClose(result_var, expected_var, rtol=1e-5, atol=1e-6)
      self.assertAllClose(result_ms, expected_ms, rtol=1e-5, atol=1e-6)
      self.assertAllClose(result_mom, expected_mom, rtol=1e-5, atol=1e-6)

      # Update for next iteration
      var = result_var
      ms = result_ms
      mom = result_mom

  def test_rmsprop_2d_tensor(self):
    """Test RMSProp with 2D tensor."""
    init_var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    init_ms = np.zeros((2, 2), dtype=np.float32)
    init_mom = np.zeros((2, 2), dtype=np.float32)
    grad_val = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    result_var, result_ms, result_mom = self._run_resource_apply_rmsprop(
        "/device:MUSA:0", init_var, init_ms, init_mom, grad_val)

    expected_var, expected_ms, expected_mom = self._compute_expected_rmsprop(
        init_var, init_ms, init_mom, grad_val, 0.01, 0.9, 0.9, 1e-8)

    self.assertAllClose(result_var, expected_var, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_ms, expected_ms, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_mom, expected_mom, rtol=1e-5, atol=1e-6)

  def test_rmsprop_large_tensor(self):
    """Test RMSProp with larger tensor."""
    np.random.seed(42)
    init_var = np.random.randn(128, 64).astype(np.float32)
    init_ms = np.zeros((128, 64), dtype=np.float32)
    init_mom = np.zeros((128, 64), dtype=np.float32)
    grad_val = np.random.randn(128, 64).astype(np.float32) * 0.1

    result_var, result_ms, result_mom = self._run_resource_apply_rmsprop(
        "/device:MUSA:0", init_var, init_ms, init_mom, grad_val)

    expected_var, expected_ms, expected_mom = self._compute_expected_rmsprop(
        init_var, init_ms, init_mom, grad_val, 0.01, 0.9, 0.9, 1e-8)

    self.assertAllClose(result_var, expected_var, rtol=1e-4, atol=1e-5)
    self.assertAllClose(result_ms, expected_ms, rtol=1e-4, atol=1e-5)
    self.assertAllClose(result_mom, expected_mom, rtol=1e-4, atol=1e-5)


class ResourceApplyCenteredRMSPropTest(MUSATestCase):
  """Tests for MUSA ResourceApplyCenteredRMSProp operator."""

  def _run_resource_apply_centered_rmsprop(self, device, init_var, init_mg, init_ms,
                                            init_mom, grad_val, lr=0.01, rho=0.9,
                                            momentum=0.9, epsilon=1e-8):
    """Run one ResourceApplyCenteredRMSProp update in graph mode on the requested device."""
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(device):
        var = tf.Variable(init_var, dtype=tf.float32, name="var")
        mg = tf.Variable(init_mg, dtype=tf.float32, name="mg")
        ms = tf.Variable(init_ms, dtype=tf.float32, name="ms")
        mom = tf.Variable(init_mom, dtype=tf.float32, name="mom")
        grad = tf.constant(grad_val, dtype=tf.float32, name="grad")

      # Resource handles and scalar hyper-parameters stay on host memory.
      with tf.device("/CPU:0"):
        lr_t = tf.constant(lr, dtype=tf.float32, name="lr")
        rho_t = tf.constant(rho, dtype=tf.float32, name="rho")
        momentum_t = tf.constant(momentum, dtype=tf.float32, name="momentum")
        epsilon_t = tf.constant(epsilon, dtype=tf.float32, name="epsilon")

      update = tf.raw_ops.ResourceApplyCenteredRMSProp(
          var=var.handle,
          mg=mg.handle,
          ms=ms.handle,
          mom=mom.handle,
          lr=lr_t,
          rho=rho_t,
          momentum=momentum_t,
          epsilon=epsilon_t,
          grad=grad,
          use_locking=False)

      with tf.control_dependencies([update]):
        read_var = tf.identity(var.read_value(), name="updated_var")
        read_mg = tf.identity(mg.read_value(), name="updated_mg")
        read_ms = tf.identity(ms.read_value(), name="updated_ms")
        read_mom = tf.identity(mom.read_value(), name="updated_mom")

      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      return sess.run([read_var, read_mg, read_ms, read_mom])

  def _compute_expected_centered_rmsprop(self, var, mg, ms, mom, grad,
                                          lr, rho, momentum, epsilon):
    """Compute expected Centered RMSProp update using NumPy."""
    # mg <- rho * mg + (1-rho) * grad
    new_mg = rho * mg + (1 - rho) * grad
    # ms <- rho * ms + (1-rho) * grad^2
    new_ms = rho * ms + (1 - rho) * grad * grad
    # mom <- momentum * mom + lr * grad / sqrt(ms - mg^2 + epsilon)
    new_mom = momentum * mom + lr * grad / np.sqrt(new_ms - new_mg * new_mg + epsilon)
    # var <- var - mom
    new_var = var - new_mom
    return new_var, new_mg, new_ms, new_mom

  def test_centered_rmsprop_basic_update(self):
    """Test basic CenteredRMSProp update on MUSA device."""
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_mg = np.zeros(4, dtype=np.float32)
    init_ms = np.zeros(4, dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)
    grad_val = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    lr = 0.01
    rho = 0.9
    momentum = 0.9
    epsilon = 1e-8

    # Run on MUSA
    result_var, result_mg, result_ms, result_mom = self._run_resource_apply_centered_rmsprop(
        "/device:MUSA:0", init_var, init_mg, init_ms, init_mom, grad_val,
        lr=lr, rho=rho, momentum=momentum, epsilon=epsilon)

    # Compute expected
    expected_var, expected_mg, expected_ms, expected_mom = self._compute_expected_centered_rmsprop(
        init_var, init_mg, init_ms, init_mom, grad_val, lr, rho, momentum, epsilon)

    self.assertAllClose(result_var, expected_var, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_mg, expected_mg, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_ms, expected_ms, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_mom, expected_mom, rtol=1e-5, atol=1e-6)

  def test_centered_rmsprop_multiple_steps(self):
    """Test CenteredRMSProp with multiple update steps."""
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_mg = np.zeros(4, dtype=np.float32)
    init_ms = np.zeros(4, dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)

    lr = 0.01
    rho = 0.9
    momentum = 0.9
    epsilon = 1e-8

    var = init_var.copy()
    mg = init_mg.copy()
    ms = init_ms.copy()
    mom = init_mom.copy()

    for step in range(3):
      grad_val = np.array([0.1 * (step + 1), 0.2 * (step + 1),
                           0.3 * (step + 1), 0.4 * (step + 1)], dtype=np.float32)

      result_var, result_mg, result_ms, result_mom = self._run_resource_apply_centered_rmsprop(
          "/device:MUSA:0", var, mg, ms, mom, grad_val,
          lr=lr, rho=rho, momentum=momentum, epsilon=epsilon)

      # Compute expected
      expected_var, expected_mg, expected_ms, expected_mom = self._compute_expected_centered_rmsprop(
          var, mg, ms, mom, grad_val, lr, rho, momentum, epsilon)

      self.assertAllClose(result_var, expected_var, rtol=1e-5, atol=1e-6)
      self.assertAllClose(result_mg, expected_mg, rtol=1e-5, atol=1e-6)
      self.assertAllClose(result_ms, expected_ms, rtol=1e-5, atol=1e-6)
      self.assertAllClose(result_mom, expected_mom, rtol=1e-5, atol=1e-6)

      # Update for next iteration
      var = result_var
      mg = result_mg
      ms = result_ms
      mom = result_mom

  def test_centered_rmsprop_2d_tensor(self):
    """Test CenteredRMSProp with 2D tensor."""
    init_var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    init_mg = np.zeros((2, 2), dtype=np.float32)
    init_ms = np.zeros((2, 2), dtype=np.float32)
    init_mom = np.zeros((2, 2), dtype=np.float32)
    grad_val = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    result_var, result_mg, result_ms, result_mom = self._run_resource_apply_centered_rmsprop(
        "/device:MUSA:0", init_var, init_mg, init_ms, init_mom, grad_val)

    expected_var, expected_mg, expected_ms, expected_mom = self._compute_expected_centered_rmsprop(
        init_var, init_mg, init_ms, init_mom, grad_val, 0.01, 0.9, 0.9, 1e-8)

    self.assertAllClose(result_var, expected_var, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_mg, expected_mg, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_ms, expected_ms, rtol=1e-5, atol=1e-6)
    self.assertAllClose(result_mom, expected_mom, rtol=1e-5, atol=1e-6)

  def test_centered_rmsprop_large_tensor(self):
    """Test CenteredRMSProp with larger tensor."""
    np.random.seed(42)
    init_var = np.random.randn(128, 64).astype(np.float32)
    init_mg = np.zeros((128, 64), dtype=np.float32)
    init_ms = np.zeros((128, 64), dtype=np.float32)
    init_mom = np.zeros((128, 64), dtype=np.float32)
    grad_val = np.random.randn(128, 64).astype(np.float32) * 0.1

    result_var, result_mg, result_ms, result_mom = self._run_resource_apply_centered_rmsprop(
        "/device:MUSA:0", init_var, init_mg, init_ms, init_mom, grad_val)

    expected_var, expected_mg, expected_ms, expected_mom = self._compute_expected_centered_rmsprop(
        init_var, init_mg, init_ms, init_mom, grad_val, 0.01, 0.9, 0.9, 1e-8)

    self.assertAllClose(result_var, expected_var, rtol=1e-4, atol=1e-5)
    self.assertAllClose(result_mg, expected_mg, rtol=1e-4, atol=1e-5)
    self.assertAllClose(result_ms, expected_ms, rtol=1e-4, atol=1e-5)
    self.assertAllClose(result_mom, expected_mom, rtol=1e-4, atol=1e-5)

  def test_centered_rmsprop_compare_with_cpu(self):
    """Compare CenteredRMSProp results between CPU and MUSA."""
    init_var = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    init_mg = np.zeros(6, dtype=np.float32)
    init_ms = np.zeros(6, dtype=np.float32)
    init_mom = np.zeros(6, dtype=np.float32)
    grad_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

    # Run on CPU
    cpu_var, cpu_mg, cpu_ms, cpu_mom = self._run_resource_apply_centered_rmsprop(
        "/CPU:0", init_var, init_mg, init_ms, init_mom, grad_val)

    # Run on MUSA
    musa_var, musa_mg, musa_ms, musa_mom = self._run_resource_apply_centered_rmsprop(
        "/device:MUSA:0", init_var, init_mg, init_ms, init_mom, grad_val)

    self.assertAllClose(cpu_var, musa_var, rtol=1e-5, atol=1e-6)
    self.assertAllClose(cpu_mg, musa_mg, rtol=1e-5, atol=1e-6)
    self.assertAllClose(cpu_ms, musa_ms, rtol=1e-5, atol=1e-6)
    self.assertAllClose(cpu_mom, musa_mom, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()