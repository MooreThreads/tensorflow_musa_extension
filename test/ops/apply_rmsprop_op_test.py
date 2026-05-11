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

"""Tests for MUSA ResourceApplyRMSProp and ResourceApplyCenteredRMSProp operators."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ResourceApplyRMSPropTest(MUSATestCase):
  """Tests for MUSA ResourceApplyRMSProp operator."""

  def setUp(self):
    super(ResourceApplyRMSPropTest, self).setUp()
    musa_devices = tf.config.list_physical_devices('MUSA')
    self.assertTrue(len(musa_devices) > 0, "No MUSA devices found")

  def _numpy_dtype(self, dtype):
    return dtype.as_numpy_dtype

  def _calc_dtype(self, dtype):
    return np.float32 if dtype in [tf.float16, tf.bfloat16] else dtype.as_numpy_dtype

  def _assert_by_dtype(self, expected, actual, dtype):
    if dtype in [tf.float16, tf.bfloat16]:
      self.assertAllClose(
          np.asarray(expected, dtype=np.float32),
          np.asarray(actual, dtype=np.float32),
          rtol=1e-2,
          atol=1e-2)
    else:
      self.assertAllClose(expected, actual, rtol=1e-5, atol=1e-6)

  def _run_resource_apply_rmsprop(self, device, init_var, init_ms, init_mom,
                                   grad, lr, rho, momentum, epsilon, dtype,
                                   use_locking=False):
    np_dtype = self._numpy_dtype(dtype)
    with tf.device(device):
      var = tf.Variable(np.asarray(init_var, dtype=np_dtype), dtype=dtype)
      ms = tf.Variable(np.asarray(init_ms, dtype=np_dtype), dtype=dtype)
      mom = tf.Variable(np.asarray(init_mom, dtype=np_dtype), dtype=dtype)
      grad_t = tf.constant(np.asarray(grad, dtype=np_dtype), dtype=dtype)

    with tf.device("/CPU:0"):
      lr_t = tf.constant(np_dtype(lr), dtype=dtype)
      rho_t = tf.constant(np_dtype(rho), dtype=dtype)
      momentum_t = tf.constant(np_dtype(momentum), dtype=dtype)
      epsilon_t = tf.constant(np_dtype(epsilon), dtype=dtype)

    tf.raw_ops.ResourceApplyRMSProp(
        var=var.handle,
        ms=ms.handle,
        mom=mom.handle,
        lr=lr_t,
        rho=rho_t,
        momentum=momentum_t,
        epsilon=epsilon_t,
        grad=grad_t,
        use_locking=use_locking)

    return var.numpy(), ms.numpy(), mom.numpy()

  def testResourceApplyRMSPropBasic(self):
    """Test basic ResourceApplyRMSProp operation with 1D and 2D tensors."""
    cases = [
        # 1D case
        (
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        ),
        # 2D case
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        ),
    ]
    lr, rho, momentum, epsilon = 0.01, 0.9, 0.9, 1e-8
    dtype = tf.float32

    for init_var, init_ms, init_mom, grad in cases:
      with self.subTest(shape=init_var.shape):
        cpu_var, cpu_ms, cpu_mom = self._run_resource_apply_rmsprop(
            "/CPU:0", init_var, init_ms, init_mom, grad,
            lr, rho, momentum, epsilon, dtype)
        musa_var, musa_ms, musa_mom = self._run_resource_apply_rmsprop(
            "/device:MUSA:0", init_var, init_ms, init_mom, grad,
            lr, rho, momentum, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_ms, musa_ms, dtype)
        self._assert_by_dtype(cpu_mom, musa_mom, dtype)

  def testResourceApplyRMSPropMultipleDtypes(self):
    """Test ResourceApplyRMSProp across float32, float16, bfloat16."""
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_ms = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    lr, rho, momentum, epsilon = 0.01, 0.9, 0.9, 1e-7

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = self._numpy_dtype(dtype)
      with self.subTest(dtype=dtype.name):
        cpu_var, cpu_ms, cpu_mom = self._run_resource_apply_rmsprop(
            "/CPU:0",
            init_var.astype(np_dtype), init_ms.astype(np_dtype),
            init_mom.astype(np_dtype), grad.astype(np_dtype),
            lr, rho, momentum, epsilon, dtype)
        musa_var, musa_ms, musa_mom = self._run_resource_apply_rmsprop(
            "/device:MUSA:0",
            init_var.astype(np_dtype), init_ms.astype(np_dtype),
            init_mom.astype(np_dtype), grad.astype(np_dtype),
            lr, rho, momentum, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_ms, musa_ms, dtype)
        self._assert_by_dtype(cpu_mom, musa_mom, dtype)

  def testResourceApplyRMSPropMultipleShapes(self):
    """Test ResourceApplyRMSProp with 2D and 3D tensor shapes."""
    lr, rho, momentum, epsilon = 0.01, 0.9, 0.9, 1e-8
    dtype = tf.float32

    for shape in [(2, 3), (2, 3, 4)]:
      with self.subTest(shape=shape):
        rng = np.random.RandomState(sum(shape))
        init_var = rng.randn(*shape).astype(np.float32)
        init_ms = np.abs(rng.randn(*shape).astype(np.float32)) * 0.1
        init_mom = rng.randn(*shape).astype(np.float32) * 0.01
        grad = rng.randn(*shape).astype(np.float32)

        cpu_var, cpu_ms, cpu_mom = self._run_resource_apply_rmsprop(
            "/CPU:0", init_var, init_ms, init_mom, grad,
            lr, rho, momentum, epsilon, dtype)
        musa_var, musa_ms, musa_mom = self._run_resource_apply_rmsprop(
            "/device:MUSA:0", init_var, init_ms, init_mom, grad,
            lr, rho, momentum, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_ms, musa_ms, dtype)
        self._assert_by_dtype(cpu_mom, musa_mom, dtype)

  def testResourceApplyRMSPropZeroGradient(self):
    """Test ResourceApplyRMSProp with zero gradient."""
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_ms = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)
    grad = np.zeros(4, dtype=np.float32)

    cpu_var, cpu_ms, cpu_mom = self._run_resource_apply_rmsprop(
        "/CPU:0", init_var, init_ms, init_mom, grad,
        0.01, 0.9, 0.9, 1e-8, dtype)
    musa_var, musa_ms, musa_mom = self._run_resource_apply_rmsprop(
        "/device:MUSA:0", init_var, init_ms, init_mom, grad,
        0.01, 0.9, 0.9, 1e-8, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_ms, musa_ms, dtype)
    self._assert_by_dtype(cpu_mom, musa_mom, dtype)

  def testResourceApplyRMSPropNegativeGradient(self):
    """Test ResourceApplyRMSProp with negative gradient values."""
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_ms = np.zeros(4, dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)
    grad = np.array([-10.0, -5.0, -2.0, -1.0], dtype=np.float32)

    cpu_var, cpu_ms, cpu_mom = self._run_resource_apply_rmsprop(
        "/CPU:0", init_var, init_ms, init_mom, grad,
        0.01, 0.9, 0.9, 1e-8, dtype)
    musa_var, musa_ms, musa_mom = self._run_resource_apply_rmsprop(
        "/device:MUSA:0", init_var, init_ms, init_mom, grad,
        0.01, 0.9, 0.9, 1e-8, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_ms, musa_ms, dtype)
    self._assert_by_dtype(cpu_mom, musa_mom, dtype)


class ResourceApplyCenteredRMSPropTest(MUSATestCase):
  """Tests for MUSA ResourceApplyCenteredRMSProp operator."""

  def setUp(self):
    super(ResourceApplyCenteredRMSPropTest, self).setUp()
    musa_devices = tf.config.list_physical_devices('MUSA')
    self.assertTrue(len(musa_devices) > 0, "No MUSA devices found")

  def _numpy_dtype(self, dtype):
    return dtype.as_numpy_dtype

  def _calc_dtype(self, dtype):
    return np.float32 if dtype in [tf.float16, tf.bfloat16] else dtype.as_numpy_dtype

  def _assert_by_dtype(self, expected, actual, dtype):
    if dtype in [tf.float16, tf.bfloat16]:
      self.assertAllClose(
          np.asarray(expected, dtype=np.float32),
          np.asarray(actual, dtype=np.float32),
          rtol=1e-2,
          atol=1e-2)
    else:
      self.assertAllClose(expected, actual, rtol=1e-5, atol=1e-6)

  def _run_resource_apply_centered_rmsprop(self, device, init_var, init_mg,
                                            init_ms, init_mom, grad, lr, rho,
                                            momentum, epsilon, dtype,
                                            use_locking=False):
    np_dtype = self._numpy_dtype(dtype)
    with tf.device(device):
      var = tf.Variable(np.asarray(init_var, dtype=np_dtype), dtype=dtype)
      mg = tf.Variable(np.asarray(init_mg, dtype=np_dtype), dtype=dtype)
      ms = tf.Variable(np.asarray(init_ms, dtype=np_dtype), dtype=dtype)
      mom = tf.Variable(np.asarray(init_mom, dtype=np_dtype), dtype=dtype)
      grad_t = tf.constant(np.asarray(grad, dtype=np_dtype), dtype=dtype)

    with tf.device("/CPU:0"):
      lr_t = tf.constant(np_dtype(lr), dtype=dtype)
      rho_t = tf.constant(np_dtype(rho), dtype=dtype)
      momentum_t = tf.constant(np_dtype(momentum), dtype=dtype)
      epsilon_t = tf.constant(np_dtype(epsilon), dtype=dtype)

    tf.raw_ops.ResourceApplyCenteredRMSProp(
        var=var.handle,
        mg=mg.handle,
        ms=ms.handle,
        mom=mom.handle,
        lr=lr_t,
        rho=rho_t,
        momentum=momentum_t,
        epsilon=epsilon_t,
        grad=grad_t,
        use_locking=use_locking)

    return var.numpy(), mg.numpy(), ms.numpy(), mom.numpy()

  def testResourceApplyCenteredRMSPropBasic(self):
    """Test basic ResourceApplyCenteredRMSProp operation with 1D and 2D tensors."""
    cases = [
        # 1D case
        (
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        ),
        # 2D case
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        ),
    ]
    lr, rho, momentum, epsilon = 0.01, 0.9, 0.9, 1e-8
    dtype = tf.float32

    for init_var, init_mg, init_ms, init_mom, grad in cases:
      with self.subTest(shape=init_var.shape):
        cpu_var, cpu_mg, cpu_ms, cpu_mom = self._run_resource_apply_centered_rmsprop(
            "/CPU:0", init_var, init_mg, init_ms, init_mom, grad,
            lr, rho, momentum, epsilon, dtype)
        musa_var, musa_mg, musa_ms, musa_mom = self._run_resource_apply_centered_rmsprop(
            "/device:MUSA:0", init_var, init_mg, init_ms, init_mom, grad,
            lr, rho, momentum, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_mg, musa_mg, dtype)
        self._assert_by_dtype(cpu_ms, musa_ms, dtype)
        self._assert_by_dtype(cpu_mom, musa_mom, dtype)

  def testResourceApplyCenteredRMSPropMultipleDtypes(self):
    """Test ResourceApplyCenteredRMSProp across float32, float16, bfloat16."""
    init_var = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    init_mg = np.zeros(6, dtype=np.float32)
    init_ms = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    init_mom = np.zeros(6, dtype=np.float32)
    grad = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    lr, rho, momentum, epsilon = 0.01, 0.9, 0.9, 1e-7

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = self._numpy_dtype(dtype)
      with self.subTest(dtype=dtype.name):
        cpu_var, cpu_mg, cpu_ms, cpu_mom = self._run_resource_apply_centered_rmsprop(
            "/CPU:0",
            init_var.astype(np_dtype), init_mg.astype(np_dtype),
            init_ms.astype(np_dtype), init_mom.astype(np_dtype),
            grad.astype(np_dtype), lr, rho, momentum, epsilon, dtype)
        musa_var, musa_mg, musa_ms, musa_mom = self._run_resource_apply_centered_rmsprop(
            "/device:MUSA:0",
            init_var.astype(np_dtype), init_mg.astype(np_dtype),
            init_ms.astype(np_dtype), init_mom.astype(np_dtype),
            grad.astype(np_dtype), lr, rho, momentum, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_mg, musa_mg, dtype)
        self._assert_by_dtype(cpu_ms, musa_ms, dtype)
        self._assert_by_dtype(cpu_mom, musa_mom, dtype)

  def testResourceApplyCenteredRMSPropMultipleShapes(self):
    """Test ResourceApplyCenteredRMSProp with 2D and 3D tensor shapes."""
    lr, rho, momentum, epsilon = 0.01, 0.9, 0.9, 1e-8
    dtype = tf.float32

    for shape in [(2, 3), (2, 3, 4)]:
      with self.subTest(shape=shape):
        rng = np.random.RandomState(sum(shape))
        init_var = rng.randn(*shape).astype(np.float32)
        init_mg = rng.randn(*shape).astype(np.float32) * 0.1
        init_ms_noise = np.abs(rng.randn(*shape).astype(np.float32)) * 0.1
        init_ms = np.square(init_mg) + init_ms_noise
        init_mom = rng.randn(*shape).astype(np.float32) * 0.01
        grad = rng.randn(*shape).astype(np.float32)

        cpu_var, cpu_mg, cpu_ms, cpu_mom = self._run_resource_apply_centered_rmsprop(
            "/CPU:0", init_var, init_mg, init_ms, init_mom, grad,
            lr, rho, momentum, epsilon, dtype)
        musa_var, musa_mg, musa_ms, musa_mom = self._run_resource_apply_centered_rmsprop(
            "/device:MUSA:0", init_var, init_mg, init_ms, init_mom, grad,
            lr, rho, momentum, epsilon, dtype)

        self._assert_by_dtype(cpu_var, musa_var, dtype)
        self._assert_by_dtype(cpu_mg, musa_mg, dtype)
        self._assert_by_dtype(cpu_ms, musa_ms, dtype)
        self._assert_by_dtype(cpu_mom, musa_mom, dtype)

  def testResourceApplyCenteredRMSPropZeroGradient(self):
    """Test ResourceApplyCenteredRMSProp with zero gradient."""
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_mg = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    init_ms = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)
    grad = np.zeros(4, dtype=np.float32)

    cpu_var, cpu_mg, cpu_ms, cpu_mom = self._run_resource_apply_centered_rmsprop(
        "/CPU:0", init_var, init_mg, init_ms, init_mom, grad,
        0.01, 0.9, 0.9, 1e-8, dtype)
    musa_var, musa_mg, musa_ms, musa_mom = self._run_resource_apply_centered_rmsprop(
        "/device:MUSA:0", init_var, init_mg, init_ms, init_mom, grad,
        0.01, 0.9, 0.9, 1e-8, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_mg, musa_mg, dtype)
    self._assert_by_dtype(cpu_ms, musa_ms, dtype)
    self._assert_by_dtype(cpu_mom, musa_mom, dtype)

  def testResourceApplyCenteredRMSPropNegativeGradient(self):
    """Test ResourceApplyCenteredRMSProp with negative gradient values."""
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_mg = np.zeros(4, dtype=np.float32)
    init_ms = np.zeros(4, dtype=np.float32)
    init_mom = np.zeros(4, dtype=np.float32)
    grad = np.array([-10.0, -5.0, -2.0, -1.0], dtype=np.float32)

    cpu_var, cpu_mg, cpu_ms, cpu_mom = self._run_resource_apply_centered_rmsprop(
        "/CPU:0", init_var, init_mg, init_ms, init_mom, grad,
        0.01, 0.9, 0.9, 1e-8, dtype)
    musa_var, musa_mg, musa_ms, musa_mom = self._run_resource_apply_centered_rmsprop(
        "/device:MUSA:0", init_var, init_mg, init_ms, init_mom, grad,
        0.01, 0.9, 0.9, 1e-8, dtype)

    self._assert_by_dtype(cpu_var, musa_var, dtype)
    self._assert_by_dtype(cpu_mg, musa_mg, dtype)
    self._assert_by_dtype(cpu_ms, musa_ms, dtype)
    self._assert_by_dtype(cpu_mom, musa_mom, dtype)


if __name__ == "__main__":
  tf.test.main()
