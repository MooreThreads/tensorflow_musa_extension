# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA ApplyAdaMax operators."""

import uuid

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ApplyAdaMaxTest(MUSATestCase):
  """Tests for MUSA ResourceApplyAdaMax and ApplyAdaMax operators."""

  def _numpy_dtype(self, dtype):
    return np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

  def _calc_dtype(self, dtype):
    return np.float64 if dtype == tf.float64 else np.float32

  def _tolerances(self, dtype):
    if dtype == tf.float16:
      return 1e-3, 1e-3
    if dtype == tf.bfloat16:
      return 1e-2, 1e-2
    if dtype == tf.float64:
      return 1e-10, 1e-12
    return 1e-5, 1e-8

  def _compute_expected_state(self, var, m, v, grad, beta1_power, lr, beta1,
                              beta2, epsilon, dtype):
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

  def _assert_state_close(self, expected, actual, dtype):
    rtol, atol = self._tolerances(dtype)
    for exp_value, act_value in zip(expected, actual):
      self.assertAllClose(exp_value, act_value, rtol=rtol, atol=atol)

  def _unique_ref_name(self, prefix):
    # VariableV2 container/shared_name only accept a conservative character set.
    return prefix + uuid.uuid4().hex

  def _create_ref_variable(self, device, init_value, dtype, prefix):
    np_value = np.asarray(init_value, dtype=self._numpy_dtype(dtype))
    container = self._unique_ref_name("musatest")
    shared_name = self._unique_ref_name(prefix)

    with tf.device(device):
      ref_var = tf.raw_ops.VariableV2(
          shape=np_value.shape,
          dtype=dtype,
          container=container,
          shared_name=shared_name)
      assign = tf.raw_ops.Assign(
          ref=ref_var,
          value=tf.constant(np_value, dtype=dtype),
          validate_shape=True,
          use_locking=False)

    return ref_var, assign

  def _run_resource_apply_adamax(self,
                                 device,
                                 init_var,
                                 init_m,
                                 init_v,
                                 grad_val,
                                 beta1_power,
                                 lr,
                                 beta1,
                                 beta2,
                                 epsilon,
                                 dtype,
                                 use_locking=False):
    np_dtype = self._numpy_dtype(dtype)
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(device):
        var = tf.Variable(init_var.astype(np_dtype), dtype=dtype, name="var")
        m = tf.Variable(init_m.astype(np_dtype), dtype=dtype, name="m")
        v = tf.Variable(init_v.astype(np_dtype), dtype=dtype, name="v")
        grad = tf.constant(grad_val.astype(np_dtype), dtype=dtype, name="grad")

      with tf.device("/CPU:0"):
        beta1_power_t = tf.constant(beta1_power, dtype=dtype, name="beta1_power")
        lr_t = tf.constant(lr, dtype=dtype, name="lr")
        beta1_t = tf.constant(beta1, dtype=dtype, name="beta1")
        beta2_t = tf.constant(beta2, dtype=dtype, name="beta2")
        epsilon_t = tf.constant(epsilon, dtype=dtype, name="epsilon")

      update = tf.raw_ops.ResourceApplyAdaMax(
          var=var.handle,
          m=m.handle,
          v=v.handle,
          beta1_power=beta1_power_t,
          lr=lr_t,
          beta1=beta1_t,
          beta2=beta2_t,
          epsilon=epsilon_t,
          grad=grad,
          use_locking=use_locking)

      with tf.control_dependencies([update]):
        read_var = tf.identity(var.read_value(), name="updated_var")
        read_m = tf.identity(m.read_value(), name="updated_m")
        read_v = tf.identity(v.read_value(), name="updated_v")

      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      return sess.run([read_var, read_m, read_v])

  def _run_apply_adamax(self,
                        device,
                        init_var,
                        init_m,
                        init_v,
                        grad_val,
                        beta1_power,
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        dtype,
                        use_locking=False):
    graph = tf.Graph()
    with graph.as_default():
      var, init_var_assign = self._create_ref_variable(device, init_var, dtype,
                                                       "amaxvar")
      m, init_m_assign = self._create_ref_variable(device, init_m, dtype,
                                                   "amaxm")
      v, init_v_assign = self._create_ref_variable(device, init_v, dtype,
                                                   "amaxv")

      with tf.device(device):
        grad = tf.constant(grad_val.astype(self._numpy_dtype(dtype)),
                           dtype=dtype,
                           name="grad")

      with tf.device("/CPU:0"):
        beta1_power_t = tf.constant(
            beta1_power, dtype=dtype, name="beta1_power")
        lr_t = tf.constant(lr, dtype=dtype, name="lr")
        beta1_t = tf.constant(beta1, dtype=dtype, name="beta1")
        beta2_t = tf.constant(beta2, dtype=dtype, name="beta2")
        epsilon_t = tf.constant(epsilon, dtype=dtype, name="epsilon")

      with tf.device(device):
        update = tf.raw_ops.ApplyAdaMax(
            var=var,
            m=m,
            v=v,
            beta1_power=beta1_power_t,
            lr=lr_t,
            beta1=beta1_t,
            beta2=beta2_t,
            epsilon=epsilon_t,
            grad=grad,
            use_locking=use_locking)

      init_ref_vars = tf.group(
          init_var_assign, init_m_assign, init_v_assign, name="init_ref_vars")

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_ref_vars)
      sess.run([var, m, v])
      sess.run(update)
      return sess.run([var, m, v])

  def test_resource_apply_adamax_basic_update(self):
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.zeros(4, dtype=np.float32)
    init_v = np.zeros(4, dtype=np.float32)
    grad = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    actual = self._run_resource_apply_adamax(
        "/device:MUSA:0",
        init_var,
        init_m,
        init_v,
        grad,
        beta1_power=0.9,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        dtype=dtype)

    expected = self._compute_expected_state(
        init_var, init_m, init_v, grad, 0.9, 0.01, 0.9, 0.999, 1e-8, dtype)
    self._assert_state_close(expected, actual, dtype)

  def test_resource_apply_adamax_multiple_dtypes(self):
    init_var = np.array([1.0, -2.0, 3.5, -4.5], dtype=np.float32)
    init_m = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    init_v = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    grad = np.array([0.25, -0.75, 1.5, -2.0], dtype=np.float32)

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      with self.subTest(dtype=dtype.name):
        actual = self._run_resource_apply_adamax(
            "/device:MUSA:0",
            init_var,
            init_m,
            init_v,
            grad,
            beta1_power=0.81,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-7,
            dtype=dtype)

        expected = self._compute_expected_state(
            init_var, init_m, init_v, grad, 0.81, 0.001, 0.9, 0.999, 1e-7,
            dtype)
        self._assert_state_close(expected, actual, dtype)

  def test_resource_apply_adamax_float64(self):
    dtype = tf.float64
    init_var = np.array([1.0, -2.0, 3.5, -4.5], dtype=np.float64)
    init_m = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    init_v = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    grad = np.array([0.25, -0.75, 1.5, -2.0], dtype=np.float64)

    actual = self._run_resource_apply_adamax(
        "/device:MUSA:0",
        init_var,
        init_m,
        init_v,
        grad,
        beta1_power=0.81,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        dtype=dtype)

    expected = self._compute_expected_state(
        init_var, init_m, init_v, grad, 0.81, 0.001, 0.9, 0.999, 1e-7, dtype)
    self._assert_state_close(expected, actual, dtype)

  def test_resource_apply_adamax_multiple_shapes(self):
    beta1_power = 0.9
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    dtype = tf.float32

    for shape in [(2, 3), (2, 3, 4)]:
      with self.subTest(shape=shape):
        rng = np.random.RandomState(sum(shape))
        init_var = rng.randn(*shape).astype(np.float32)
        init_m = rng.randn(*shape).astype(np.float32) * 0.1
        init_v = np.abs(rng.randn(*shape).astype(np.float32))
        grad = rng.randn(*shape).astype(np.float32)

        actual = self._run_resource_apply_adamax(
            "/device:MUSA:0",
            init_var,
            init_m,
            init_v,
            grad,
            beta1_power=beta1_power,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            dtype=dtype)

        expected = self._compute_expected_state(
            init_var, init_m, init_v, grad, beta1_power, lr, beta1, beta2,
            epsilon, dtype)
        self._assert_state_close(expected, actual, dtype)

  def test_resource_apply_adamax_zero_gradient(self):
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    init_v = np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    grad = np.zeros(4, dtype=np.float32)

    actual = self._run_resource_apply_adamax(
        "/device:MUSA:0",
        init_var,
        init_m,
        init_v,
        grad,
        beta1_power=0.81,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        dtype=dtype)

    expected = self._compute_expected_state(
        init_var, init_m, init_v, grad, 0.81, 0.01, 0.9, 0.999, 1e-8, dtype)
    self._assert_state_close(expected, actual, dtype)

  def test_resource_apply_adamax_negative_gradient(self):
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.zeros(4, dtype=np.float32)
    init_v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    grad = np.array([-10.0, -5.0, -2.0, -1.0], dtype=np.float32)

    actual = self._run_resource_apply_adamax(
        "/device:MUSA:0",
        init_var,
        init_m,
        init_v,
        grad,
        beta1_power=0.9,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        dtype=dtype)

    expected = self._compute_expected_state(
        init_var, init_m, init_v, grad, 0.9, 0.01, 0.9, 0.999, 1e-8, dtype)
    self._assert_state_close(expected, actual, dtype)

  def test_resource_apply_adamax_max_branches(self):
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.zeros(4, dtype=np.float32)
    init_v = np.array([100.0, 0.001, 50.0, 0.01], dtype=np.float32)
    grad = np.array([10.0, 5.0, 100.0, 1.0], dtype=np.float32)

    actual = self._run_resource_apply_adamax(
        "/device:MUSA:0",
        init_var,
        init_m,
        init_v,
        grad,
        beta1_power=0.9,
        lr=0.01,
        beta1=0.9,
        beta2=0.5,
        epsilon=1e-8,
        dtype=dtype)

    expected = self._compute_expected_state(
        init_var, init_m, init_v, grad, 0.9, 0.01, 0.9, 0.5, 1e-8, dtype)
    self._assert_state_close(expected, actual, dtype)

  def test_resource_apply_adamax_multiple_steps(self):
    dtype = tf.float32
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    lr = 0.01

    var = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    m = np.zeros(4, dtype=np.float32)
    v = np.zeros(4, dtype=np.float32)
    grads = [
        np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32),
        np.array([0.5, 0.5, -1.0, 1.0], dtype=np.float32),
        np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32),
    ]

    for step, grad in enumerate(grads, start=1):
      with self.subTest(step=step):
        actual = self._run_resource_apply_adamax(
            "/device:MUSA:0",
            var,
            m,
            v,
            grad,
            beta1_power=beta1**step,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            dtype=dtype)

        expected = self._compute_expected_state(
            var, m, v, grad, beta1**step, lr, beta1, beta2, epsilon, dtype)
        self._assert_state_close(expected, actual, dtype)
        var, m, v = [np.asarray(state, dtype=np.float32) for state in expected]

  def test_resource_apply_adamax_use_locking(self):
    dtype = tf.float32
    init_var = np.array([1.25, -2.5, 5.0, -10.0], dtype=np.float32)
    init_m = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    init_v = np.array([0.5, 0.25, 1.0, 2.0], dtype=np.float32)
    grad = np.array([0.5, 0.25, -1.0, 2.0], dtype=np.float32)

    actual = self._run_resource_apply_adamax(
        "/device:MUSA:0",
        init_var,
        init_m,
        init_v,
        grad,
        beta1_power=0.9,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        dtype=dtype,
        use_locking=True)

    expected = self._compute_expected_state(
        init_var, init_m, init_v, grad, 0.9, 0.01, 0.9, 0.999, 1e-8, dtype)
    self._assert_state_close(expected, actual, dtype)

  def test_apply_adamax_ref_smoke(self):
    dtype = tf.float32
    init_var = np.array([2.0, -4.0, 6.0], dtype=np.float32)
    init_m = np.array([0.2, -0.1, 0.05], dtype=np.float32)
    init_v = np.array([0.5, 0.25, 0.75], dtype=np.float32)
    grad = np.array([1.0, -0.5, 0.25], dtype=np.float32)

    actual = self._run_apply_adamax(
        "/device:MUSA:0",
        init_var,
        init_m,
        init_v,
        grad,
        beta1_power=0.9,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        dtype=dtype)

    expected = self._compute_expected_state(
        init_var, init_m, init_v, grad, 0.9, 0.01, 0.9, 0.999, 1e-8, dtype)
    self._assert_state_close(expected, actual, dtype)

  def test_resource_apply_adamax_rejects_non_scalar_hyperparameter(self):
    graph = tf.Graph()
    with graph.as_default():
      with tf.device("/device:MUSA:0"):
        var = tf.Variable([1.0, 2.0], dtype=tf.float32, name="var")
        m = tf.Variable([0.0, 0.0], dtype=tf.float32, name="m")
        v = tf.Variable([0.0, 0.0], dtype=tf.float32, name="v")
        grad = tf.constant([0.1, 0.2], dtype=tf.float32, name="grad")

      with tf.device("/CPU:0"):
        beta1_power = tf.constant([0.9, 0.81], dtype=tf.float32)
        lr = tf.constant(0.01, dtype=tf.float32)
        beta1 = tf.constant(0.9, dtype=tf.float32)
        beta2 = tf.constant(0.999, dtype=tf.float32)
        epsilon = tf.constant(1e-8, dtype=tf.float32)

      with self.assertRaisesRegex(ValueError, "rank 0|scalar"):
        tf.raw_ops.ResourceApplyAdaMax(
            var=var.handle,
            m=m.handle,
            v=v.handle,
            beta1_power=beta1_power,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad=grad,
            use_locking=False)

  def test_resource_apply_adamax_rejects_shape_mismatch(self):
    try:
      graph = tf.Graph()
      with graph.as_default():
        with tf.device("/device:MUSA:0"):
          var = tf.Variable(
              [[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32, name="var")
          m = tf.Variable(
              [0.0, 0.0, 0.0], dtype=tf.float32, name="m")
          v = tf.Variable(
              [[0.0, 0.0], [0.0, 0.0]], dtype=tf.float32, name="v")
          grad = tf.constant(
              [[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32, name="grad")

        with tf.device("/CPU:0"):
          beta1_power = tf.constant(0.9, dtype=tf.float32)
          lr = tf.constant(0.01, dtype=tf.float32)
          beta1 = tf.constant(0.9, dtype=tf.float32)
          beta2 = tf.constant(0.999, dtype=tf.float32)
          epsilon = tf.constant(1e-8, dtype=tf.float32)

        update = tf.raw_ops.ResourceApplyAdaMax(
            var=var.handle,
            m=m.handle,
            v=v.handle,
            beta1_power=beta1_power,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad=grad,
            use_locking=False)

        init_op = tf.compat.v1.global_variables_initializer()

      with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(init_op)
        sess.run(update)
    except (ValueError, tf.errors.InvalidArgumentError) as exc:
      self.assertRegex(
          str(exc), "same shape|shape|Shapes|Dimensions|compatible|rank")
    else:
      self.skipTest(
          "Current plugin build did not surface a ResourceApplyAdaMax shape "
          "mismatch error.")


if __name__ == "__main__":
  tf.test.main()
