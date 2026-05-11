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

"""Tests for MUSA ApplyAdaMax operators."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ApplyAdaMaxOpTest(MUSATestCase):
  """Tests for MUSA ResourceApplyAdaMax operators."""

  def setUp(self):
    super(ApplyAdaMaxOpTest, self).setUp()
    musa_devices = tf.config.list_physical_devices('MUSA')
    self.assertTrue(len(musa_devices) > 0, "No MUSA devices found")

  def _numpy_dtype(self, dtype):
    return dtype.as_numpy_dtype

  def _calc_dtype(self, dtype):
    return dtype.as_numpy_dtype

  def _assert_by_dtype(self, expected, actual, dtype):
    if dtype == tf.bfloat16:
      self.assertAllClose(
          np.asarray(expected, dtype=np.float32),
          np.asarray(actual, dtype=np.float32),
          rtol=2e-2,
          atol=5e-2)
    elif dtype == tf.float16:
      self.assertAllClose(
          np.asarray(expected, dtype=np.float32),
          np.asarray(actual, dtype=np.float32),
          rtol=1e-2,
          atol=1e-2)
    else:
      self.assertAllClose(expected, actual, rtol=1e-5, atol=1e-8)

  def _expected_apply_adamax(self, var, m, v, grad, beta1_power, lr, beta1,
                              beta2, epsilon, dtype):
    """Compute expected AdaMax update using numpy.

    AdaMax update rule:
      m = beta1 * m + (1 - beta1) * grad
      v = max(beta2 * v, |grad|)
      var = var - (lr / (1 - beta1_power)) * m / (v + epsilon)
    """
    calc_dtype = self._calc_dtype(dtype)
    var = np.asarray(var, dtype=calc_dtype)
    m = np.asarray(m, dtype=calc_dtype)
    v = np.asarray(v, dtype=calc_dtype)
    grad = np.asarray(grad, dtype=calc_dtype)
    beta1_power = calc_dtype(beta1_power)
    lr = calc_dtype(lr)
    beta1 = calc_dtype(beta1)
    beta2 = calc_dtype(beta2)
    epsilon = calc_dtype(epsilon)

    new_m = beta1 * m + (calc_dtype(1.0) - beta1) * grad
    new_v = np.maximum(beta2 * v, np.abs(grad))
    lr_t = lr / (calc_dtype(1.0) - beta1_power)
    new_var = var - lr_t * new_m / (new_v + epsilon)
    return new_var, new_m, new_v

  def _run_resource_apply_adamax(self, device, init_var, init_m, init_v, grad,
                                  beta1_power, lr, beta1, beta2, epsilon, dtype,
                                  use_locking=False):
    if device.upper().endswith("CPU:0"):
      return self._expected_apply_adamax(
          init_var, init_m, init_v, grad, beta1_power, lr, beta1, beta2,
          epsilon, dtype)

    np_dtype = self._numpy_dtype(dtype)
    with tf.device(device):
      var = tf.Variable(np.asarray(init_var, dtype=np_dtype), dtype=dtype)
      m = tf.Variable(np.asarray(init_m, dtype=np_dtype), dtype=dtype)
      v = tf.Variable(np.asarray(init_v, dtype=np_dtype), dtype=dtype)
      grad_t = tf.constant(np.asarray(grad, dtype=np_dtype), dtype=dtype)

    with tf.device("/CPU:0"):
      beta1_power_t = tf.constant(np_dtype(beta1_power), dtype=dtype)
      lr_t = tf.constant(np_dtype(lr), dtype=dtype)
      beta1_t = tf.constant(np_dtype(beta1), dtype=dtype)
      beta2_t = tf.constant(np_dtype(beta2), dtype=dtype)
      epsilon_t = tf.constant(np_dtype(epsilon), dtype=dtype)

    tf.raw_ops.ResourceApplyAdaMax(
        var=var.handle,
        m=m.handle,
        v=v.handle,
        beta1_power=beta1_power_t,
        lr=lr_t,
        beta1=beta1_t,
        beta2=beta2_t,
        epsilon=epsilon_t,
        grad=grad_t,
        use_locking=use_locking)

    return var.numpy(), m.numpy(), v.numpy()

  def testResourceApplyAdaMaxBasic(self):
    """Test basic ResourceApplyAdaMax operation against numpy reference."""
    cases = [
        # 1D case
        (
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32),
        ),
        # 2D case
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.array([[0.2, -0.4], [0.6, -0.8]], dtype=np.float32),
        ),
    ]
    beta1_power, lr, beta1, beta2, epsilon = 0.9, 0.01, 0.9, 0.999, 1e-8
    dtype = tf.float32

    for init_var, init_m, init_v, grad in cases:
      with self.subTest(shape=init_var.shape):
        cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
            "/CPU:0", init_var, init_m, init_v, grad,
            beta1_power, lr, beta1, beta2, epsilon, dtype)
        musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
            "/device:MUSA:0", init_var, init_m, init_v, grad,
            beta1_power, lr, beta1, beta2, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_m, musa_m, dtype)
        self._assert_by_dtype(cpu_v, musa_v, dtype)

  def testResourceApplyAdaMaxMultipleDtypes(self):
    """Test ResourceApplyAdaMax across float32, float16, bfloat16."""
    init_var = np.array([1.0, -2.0, 3.5, -4.5], dtype=np.float32)
    init_m = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    init_v = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    grad = np.array([0.25, -0.75, 1.5, -2.0], dtype=np.float32)
    beta1_power, lr, beta1, beta2, epsilon = 0.81, 0.001, 0.9, 0.999, 1e-7

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = self._numpy_dtype(dtype)
      with self.subTest(dtype=dtype.name):
        # CPU kernel does not support bfloat16; use numpy reference instead.
        if dtype == tf.bfloat16:
          ref_var, ref_m, ref_v = self._expected_apply_adamax(
              init_var, init_m, init_v, grad,
              beta1_power, lr, beta1, beta2, epsilon, dtype)
        else:
          ref_var, ref_m, ref_v = self._run_resource_apply_adamax(
              "/CPU:0",
              init_var.astype(np_dtype), init_m.astype(np_dtype),
              init_v.astype(np_dtype), grad.astype(np_dtype),
              beta1_power, lr, beta1, beta2, epsilon, dtype)
        musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
            "/device:MUSA:0",
            init_var.astype(np_dtype), init_m.astype(np_dtype),
            init_v.astype(np_dtype), grad.astype(np_dtype),
            beta1_power, lr, beta1, beta2, epsilon, dtype)

        self._assert_by_dtype(ref_var, musa_var, dtype)
        self._assert_by_dtype(ref_m, musa_m, dtype)
        self._assert_by_dtype(ref_v, musa_v, dtype)

  def testResourceApplyAdaMaxMultipleShapes(self):
    """Test ResourceApplyAdaMax with 2D and 3D tensor shapes."""
    beta1_power, lr, beta1, beta2, epsilon = 0.9, 0.01, 0.9, 0.999, 1e-8
    dtype = tf.float32

    for shape in [(2, 3), (2, 3, 4)]:
      with self.subTest(shape=shape):
        rng = np.random.RandomState(sum(shape))
        init_var = rng.randn(*shape).astype(np.float32)
        init_m = rng.randn(*shape).astype(np.float32) * 0.1
        init_v = np.abs(rng.randn(*shape).astype(np.float32))
        grad = rng.randn(*shape).astype(np.float32)

        cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
            "/CPU:0", init_var, init_m, init_v, grad,
            beta1_power, lr, beta1, beta2, epsilon, dtype)
        musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
            "/device:MUSA:0", init_var, init_m, init_v, grad,
            beta1_power, lr, beta1, beta2, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_m, musa_m, dtype)
        self._assert_by_dtype(cpu_v, musa_v, dtype)

  def testResourceApplyAdaMaxZeroGradient(self):
    """Test ResourceApplyAdaMax with zero gradient."""
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    init_v = np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    grad = np.zeros(4, dtype=np.float32)

    cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
        "/CPU:0", init_var, init_m, init_v, grad,
        0.81, 0.01, 0.9, 0.999, 1e-8, dtype)
    musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
        "/device:MUSA:0", init_var, init_m, init_v, grad,
        0.81, 0.01, 0.9, 0.999, 1e-8, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_m, musa_m, dtype)
    self._assert_by_dtype(cpu_v, musa_v, dtype)

  def testResourceApplyAdaMaxNegativeGradient(self):
    """Test ResourceApplyAdaMax with negative gradient values."""
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.zeros(4, dtype=np.float32)
    init_v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    grad = np.array([-10.0, -5.0, -2.0, -1.0], dtype=np.float32)

    cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
        "/CPU:0", init_var, init_m, init_v, grad,
        0.9, 0.01, 0.9, 0.999, 1e-8, dtype)
    musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
        "/device:MUSA:0", init_var, init_m, init_v, grad,
        0.9, 0.01, 0.9, 0.999, 1e-8, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_m, musa_m, dtype)
    self._assert_by_dtype(cpu_v, musa_v, dtype)

  def testResourceApplyAdaMaxMaxBranches(self):
    """Test the max() branch in the v update: max(beta2*v, |grad|)."""
    dtype = tf.float32
    # init_v has mixed values relative to |grad| to exercise both branches
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.zeros(4, dtype=np.float32)
    init_v = np.array([100.0, 0.001, 50.0, 0.01], dtype=np.float32)
    grad = np.array([10.0, 5.0, 100.0, 1.0], dtype=np.float32)

    cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
        "/CPU:0", init_var, init_m, init_v, grad,
        0.9, 0.01, 0.9, 0.5, 1e-8, dtype)
    musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
        "/device:MUSA:0", init_var, init_m, init_v, grad,
        0.9, 0.01, 0.9, 0.5, 1e-8, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_m, musa_m, dtype)
    self._assert_by_dtype(cpu_v, musa_v, dtype)

  def testResourceApplyAdaMaxMultipleSteps(self):
    """Test ResourceApplyAdaMax across multiple optimizer steps."""
    dtype = tf.float32
    beta1, beta2, epsilon, lr = 0.9, 0.999, 1e-8, 0.01
    grads = [
        np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32),
        np.array([0.5, 0.5, -1.0, 1.0], dtype=np.float32),
        np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32),
    ]

    var = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    m = np.zeros(4, dtype=np.float32)
    v = np.zeros(4, dtype=np.float32)

    for step, grad in enumerate(grads, start=1):
      with self.subTest(step=step):
        beta1_power = beta1 ** step
        cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
            "/CPU:0", var, m, v, grad,
            beta1_power, lr, beta1, beta2, epsilon, dtype)
        musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
            "/device:MUSA:0", var, m, v, grad,
            beta1_power, lr, beta1, beta2, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_m, musa_m, dtype)
        self._assert_by_dtype(cpu_v, musa_v, dtype)

        # Advance state using numpy reference for next step
        var, m, v = self._expected_apply_adamax(
            var, m, v, grad, beta1_power, lr, beta1, beta2, epsilon, dtype)

  def testResourceApplyAdaMaxUseLocking(self):
    """Test ResourceApplyAdaMax with use_locking=True."""
    dtype = tf.float32
    init_var = np.array([1.25, -2.5, 5.0, -10.0], dtype=np.float32)
    init_m = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    init_v = np.array([0.5, 0.25, 1.0, 2.0], dtype=np.float32)
    grad = np.array([0.5, 0.25, -1.0, 2.0], dtype=np.float32)

    cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
        "/CPU:0", init_var, init_m, init_v, grad,
        0.9, 0.01, 0.9, 0.999, 1e-8, dtype, use_locking=True)
    musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
        "/device:MUSA:0", init_var, init_m, init_v, grad,
        0.9, 0.01, 0.9, 0.999, 1e-8, dtype, use_locking=True)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_m, musa_m, dtype)
    self._assert_by_dtype(cpu_v, musa_v, dtype)

  def testResourceApplyAdaMaxLargeScale(self):
    """Test ResourceApplyAdaMax with larger tensors."""
    np.random.seed(42)

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = self._numpy_dtype(dtype)
      shape = (128, 64)
      with self.subTest(dtype=dtype.name, shape=shape):
        rng = np.random.RandomState(0)
        init_var = rng.randn(*shape).astype(np_dtype)
        init_m = (rng.randn(*shape) * 0.1).astype(np_dtype)
        init_v = np.abs(rng.randn(*shape)).astype(np_dtype)
        grad = (rng.randn(*shape) * 0.1).astype(np_dtype)

        # CPU kernel does not support bfloat16; use numpy reference instead.
        if dtype == tf.bfloat16:
          ref_var, ref_m, ref_v = self._expected_apply_adamax(
              init_var, init_m, init_v, grad,
              0.9, 0.01, 0.9, 0.999, 1e-8, dtype)
        else:
          ref_var, ref_m, ref_v = self._run_resource_apply_adamax(
              "/CPU:0", init_var, init_m, init_v, grad,
              0.9, 0.01, 0.9, 0.999, 1e-8, dtype)
        musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
            "/device:MUSA:0", init_var, init_m, init_v, grad,
            0.9, 0.01, 0.9, 0.999, 1e-8, dtype)

        self._assert_by_dtype(ref_var, musa_var, dtype)
        self._assert_by_dtype(ref_m, musa_m, dtype)
        self._assert_by_dtype(ref_v, musa_v, dtype)

  def testResourceApplyAdaMaxNumericsAgainstReference(self):
    """Verify MUSA result matches numpy reference formula."""
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.zeros(4, dtype=np.float32)
    init_v = np.zeros(4, dtype=np.float32)
    grad = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    beta1_power, lr, beta1, beta2, epsilon = 0.9, 0.01, 0.9, 0.999, 1e-8

    musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
        "/device:MUSA:0", init_var, init_m, init_v, grad,
        beta1_power, lr, beta1, beta2, epsilon, dtype)

    exp_var, exp_m, exp_v = self._expected_apply_adamax(
        init_var, init_m, init_v, grad,
        beta1_power, lr, beta1, beta2, epsilon, dtype)

    self._assert_by_dtype(exp_var, musa_var, dtype)
    self._assert_by_dtype(exp_m, musa_m, dtype)
    self._assert_by_dtype(exp_v, musa_v, dtype)

  def testDevicePlacement(self):
    """Verify that ResourceApplyAdaMax runs on MUSA device."""
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    init_m = np.zeros(3, dtype=np.float32)
    init_v = np.zeros(3, dtype=np.float32)
    grad = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    with tf.device("/device:MUSA:0"):
      var = tf.Variable(init_var, dtype=dtype)
      m = tf.Variable(init_m, dtype=dtype)
      v = tf.Variable(init_v, dtype=dtype)
      grad_t = tf.constant(grad, dtype=dtype)

      self.assertIn("MUSA", var.device)

    with tf.device("/CPU:0"):
      beta1_power_t = tf.constant(np.float32(0.9), dtype=dtype)
      lr_t = tf.constant(np.float32(0.01), dtype=dtype)
      beta1_t = tf.constant(np.float32(0.9), dtype=dtype)
      beta2_t = tf.constant(np.float32(0.999), dtype=dtype)
      epsilon_t = tf.constant(np.float32(1e-8), dtype=dtype)

    tf.raw_ops.ResourceApplyAdaMax(
        var=var.handle, m=m.handle, v=v.handle,
        beta1_power=beta1_power_t, lr=lr_t, beta1=beta1_t,
        beta2=beta2_t, epsilon=epsilon_t, grad=grad_t, use_locking=False)

    exp_var, exp_m, exp_v = self._expected_apply_adamax(
        init_var, init_m, init_v, grad, 0.9, 0.01, 0.9, 0.999, 1e-8, dtype)
    self._assert_by_dtype(exp_var, var.numpy(), dtype)
    self._assert_by_dtype(exp_m, m.numpy(), dtype)
    self._assert_by_dtype(exp_v, v.numpy(), dtype)


if __name__ == "__main__":
  tf.test.main()
