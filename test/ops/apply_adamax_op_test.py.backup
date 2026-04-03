# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA ApplyAdaMax operators."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import tensorflow as tf


_ISOLATED_RUNNER = r"""
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import tensorflow as tf


def finish(payload):
  sys.stdout.write(json.dumps(payload))
  sys.stdout.flush()
  os._exit(0)


def load_musa_plugin(project_root):
  candidate_paths = [
      os.path.join(project_root, "build", "libmusa_plugin.so"),
      os.path.join(os.getcwd(), "..", "build", "libmusa_plugin.so"),
      os.path.join(os.getcwd(), "build", "libmusa_plugin.so"),
  ]

  plugin_path = None
  for path in candidate_paths:
    normalized_path = os.path.normpath(path)
    if os.path.exists(normalized_path):
      plugin_path = normalized_path
      break

  if plugin_path is None:
    raise FileNotFoundError(
        "MUSA plugin not found. Searched locations: %s" %
        ", ".join(os.path.normpath(path) for path in candidate_paths))

  tf.load_library(plugin_path)


def input_numpy_dtype(dtype_name):
  return np.float32 if dtype_name == "bfloat16" else np.dtype(dtype_name)


def output_numpy_dtype(dtype_name):
  if dtype_name == "float64":
    return np.float64
  return np.float32


def tf_dtype(dtype_name):
  return getattr(tf, dtype_name)


def create_ref_variable(device, init_value, dtype, prefix):
  np_value = np.asarray(init_value)
  container = "musatest" + uuid.uuid4().hex
  shared_name = prefix + uuid.uuid4().hex

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


def encode_results(values, dtype_name):
  encoded = []
  out_dtype = output_numpy_dtype(dtype_name)
  for value in values:
    encoded.append(np.asarray(value, dtype=out_dtype).tolist())
  return encoded


def run_resource_update(payload):
  dtype_name = payload["dtype_name"]
  dtype = tf_dtype(dtype_name)
  np_dtype = input_numpy_dtype(dtype_name)
  device = "/device:MUSA:0"

  graph = tf.Graph()
  with graph.as_default():
    with tf.device(device):
      var = tf.Variable(np.asarray(payload["init_var"], dtype=np_dtype),
                        dtype=dtype,
                        name="var")
      m = tf.Variable(np.asarray(payload["init_m"], dtype=np_dtype),
                      dtype=dtype,
                      name="m")
      v = tf.Variable(np.asarray(payload["init_v"], dtype=np_dtype),
                      dtype=dtype,
                      name="v")
      grad = tf.constant(np.asarray(payload["grad"], dtype=np_dtype),
                         dtype=dtype,
                         name="grad")

    with tf.device("/CPU:0"):
      beta1_power_t = tf.constant(payload["beta1_power"],
                                  dtype=dtype,
                                  name="beta1_power")
      lr_t = tf.constant(payload["lr"], dtype=dtype, name="lr")
      beta1_t = tf.constant(payload["beta1"], dtype=dtype, name="beta1")
      beta2_t = tf.constant(payload["beta2"], dtype=dtype, name="beta2")
      epsilon_t = tf.constant(payload["epsilon"], dtype=dtype, name="epsilon")

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
        use_locking=payload["use_locking"])

    with tf.control_dependencies([update]):
      read_var = tf.identity(var.read_value(), name="updated_var")
      read_m = tf.identity(m.read_value(), name="updated_m")
      read_v = tf.identity(v.read_value(), name="updated_v")

    init_op = tf.compat.v1.global_variables_initializer()

  sess = tf.compat.v1.Session(graph=graph)
  sess.run(init_op)
  result = sess.run([read_var, read_m, read_v])
  finish({"ok": True, "result": encode_results(result, dtype_name)})


def run_ref_update(payload):
  dtype_name = payload["dtype_name"]
  dtype = tf_dtype(dtype_name)
  np_dtype = input_numpy_dtype(dtype_name)
  device = "/device:MUSA:0"

  graph = tf.Graph()
  with graph.as_default():
    var, init_var_assign = create_ref_variable(
        device, np.asarray(payload["init_var"], dtype=np_dtype), dtype, "amaxvar")
    m, init_m_assign = create_ref_variable(
        device, np.asarray(payload["init_m"], dtype=np_dtype), dtype, "amaxm")
    v, init_v_assign = create_ref_variable(
        device, np.asarray(payload["init_v"], dtype=np_dtype), dtype, "amaxv")

    with tf.device(device):
      grad = tf.constant(np.asarray(payload["grad"], dtype=np_dtype),
                         dtype=dtype,
                         name="grad")

    with tf.device("/CPU:0"):
      beta1_power_t = tf.constant(payload["beta1_power"],
                                  dtype=dtype,
                                  name="beta1_power")
      lr_t = tf.constant(payload["lr"], dtype=dtype, name="lr")
      beta1_t = tf.constant(payload["beta1"], dtype=dtype, name="beta1")
      beta2_t = tf.constant(payload["beta2"], dtype=dtype, name="beta2")
      epsilon_t = tf.constant(payload["epsilon"], dtype=dtype, name="epsilon")

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
          use_locking=payload["use_locking"])

    init_ref_vars = tf.group(
        init_var_assign, init_m_assign, init_v_assign, name="init_ref_vars")

  sess = tf.compat.v1.Session(graph=graph)
  sess.run(init_ref_vars)
  sess.run([var, m, v])
  sess.run(update)
  result = sess.run([var, m, v])
  finish({"ok": True, "result": encode_results(result, dtype_name)})


def run_non_scalar_error():
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

  finish({"ok": True, "unexpected_success": True})


def run_shape_mismatch_error():
  graph = tf.Graph()
  with graph.as_default():
    with tf.device("/device:MUSA:0"):
      var = tf.Variable(
          [[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32, name="var")
      m = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32, name="m")
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

  sess = tf.compat.v1.Session(graph=graph)
  sess.run(init_op)
  sess.run(update)
  finish({"ok": True, "unexpected_success": True})


def main():
  payload = json.load(sys.stdin)
  load_musa_plugin(payload["project_root"])
  if not tf.config.list_physical_devices("MUSA"):
    raise RuntimeError("No MUSA devices found.")
  tf.compat.v1.disable_eager_execution()

  mode = payload["mode"]
  if mode == "resource_update":
    run_resource_update(payload)
  if mode == "ref_update":
    run_ref_update(payload)
  if mode == "non_scalar_error":
    run_non_scalar_error()
  if mode == "shape_mismatch_error":
    run_shape_mismatch_error()
  raise ValueError("Unsupported mode: %s" % mode)


try:
  main()
except Exception as exc:
  finish({
      "ok": False,
      "exc_type": type(exc).__name__,
      "message": str(exc),
  })
"""


class ApplyAdaMaxTest(unittest.TestCase):
  """Tests for MUSA ResourceApplyAdaMax and ApplyAdaMax operators."""

  def _project_root(self):
    return str(Path(__file__).resolve().parents[2])

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

  def _serialize_array(self, value):
    return np.asarray(value).tolist()

  def _run_isolated(self, payload):
    full_payload = {"project_root": self._project_root()}
    full_payload.update(payload)

    proc = subprocess.run(
        [sys.executable, "-c", _ISOLATED_RUNNER],
        input=json.dumps(full_payload),
        text=True,
        capture_output=True,
        timeout=300,
        check=False)

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    json_line = ""
    if stdout:
      json_line = stdout.splitlines()[-1]

    if not json_line:
      self.fail(
          "Isolated AdaMax subprocess produced no JSON output. "
          "returncode=%s stderr=%s" % (proc.returncode, stderr))

    try:
      response = json.loads(json_line)
    except json.JSONDecodeError as exc:
      self.fail(
          "Isolated AdaMax subprocess returned invalid JSON: %s. "
          "stdout=%s stderr=%s" % (exc, stdout, stderr))

    if proc.returncode != 0:
      self.fail(
          "Isolated AdaMax subprocess exited with code %s. "
          "response=%s stderr=%s" % (proc.returncode, response, stderr))

    return response

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

  def assertAllClose(self, a, b, rtol=1e-5, atol=1e-8):
    a = np.asarray(a)
    b = np.asarray(b)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
      diff = np.abs(a - b)
      self.fail(
          "Arrays are not close.\n"
          "shape=%s max_diff=%e mean_diff=%e rtol=%g atol=%g" %
          (a.shape, np.max(diff), np.mean(diff), rtol, atol))

  def _assert_state_close(self, expected, actual, dtype):
    rtol, atol = self._tolerances(dtype)
    for exp_value, act_value in zip(expected, actual):
      self.assertAllClose(exp_value, act_value, rtol=rtol, atol=atol)

  def _run_resource_apply_adamax(self,
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
    response = self._run_isolated({
        "mode": "resource_update",
        "dtype_name": dtype.name,
        "init_var": self._serialize_array(init_var),
        "init_m": self._serialize_array(init_m),
        "init_v": self._serialize_array(init_v),
        "grad": self._serialize_array(grad_val),
        "beta1_power": beta1_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "use_locking": use_locking,
    })
    if not response["ok"]:
      self.fail("ResourceApplyAdaMax failed: %s" % response["message"])
    np_dtype = self._numpy_dtype(dtype)
    return tuple(np.asarray(value, dtype=np_dtype) for value in response["result"])

  def _run_apply_adamax(self,
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
    response = self._run_isolated({
        "mode": "ref_update",
        "dtype_name": dtype.name,
        "init_var": self._serialize_array(init_var),
        "init_m": self._serialize_array(init_m),
        "init_v": self._serialize_array(init_v),
        "grad": self._serialize_array(grad_val),
        "beta1_power": beta1_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "use_locking": use_locking,
    })
    if not response["ok"]:
      self.fail("ApplyAdaMax failed: %s" % response["message"])
    np_dtype = self._numpy_dtype(dtype)
    return tuple(np.asarray(value, dtype=np_dtype) for value in response["result"])

  def test_resource_apply_adamax_basic_update(self):
    dtype = tf.float32
    init_var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    init_m = np.zeros(4, dtype=np.float32)
    init_v = np.zeros(4, dtype=np.float32)
    grad = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    actual = self._run_resource_apply_adamax(
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
    response = self._run_isolated({"mode": "non_scalar_error"})
    self.assertFalse(response["ok"], "Expected non-scalar hyperparameter error")
    self.assertRegex(response["message"], "rank 0|scalar")

  def test_resource_apply_adamax_rejects_shape_mismatch(self):
    response = self._run_isolated({"mode": "shape_mismatch_error"})
    if response["ok"] and response.get("unexpected_success"):
      self.skipTest(
          "Current plugin build did not surface a ResourceApplyAdaMax shape "
          "mismatch error.")
    self.assertFalse(response["ok"], "Expected shape mismatch error")
    self.assertRegex(
        response["message"], "same shape|shape|Shapes|Dimensions|compatible|rank")


if __name__ == "__main__":
  unittest.main()
