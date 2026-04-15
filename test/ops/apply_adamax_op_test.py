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

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import math

import numpy as np
import tensorflow as tf


_ISOLATED_RUNNER = r"""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import re
import sys
import uuid

import numpy as np
import tensorflow as tf


def finish(payload):
  sys.stdout.write(json.dumps(payload))
  sys.stdout.flush()
  os._exit(0)


def create_session(graph):
  config = tf.compat.v1.ConfigProto()
  config.allow_soft_placement = True
  return tf.compat.v1.Session(graph=graph, config=config)


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


def resolve_musa_device():
  explicit = (os.environ.get("MUSA_TEST_DEVICE") or
              os.environ.get("TF_MUSA_TEST_DEVICE"))
  logical_devices = tf.config.list_logical_devices("MUSA")
  if not logical_devices:
    raise RuntimeError("No MUSA logical devices found.")

  if explicit:
    explicit = explicit.strip()
    if explicit.startswith("/device:"):
      return explicit
    if explicit.isdigit():
      index = int(explicit)
      if len(logical_devices) == 1:
        return logical_devices[0].name
      if 0 <= index < len(logical_devices):
        return logical_devices[index].name
      raise RuntimeError(
          "Requested MUSA_TEST_DEVICE=%s, but only %d logical devices are visible." %
          (explicit, len(logical_devices)))

  local_rank = os.environ.get("LOCAL_RANK")
  if local_rank and local_rank.isdigit():
    index = int(local_rank)
    if len(logical_devices) == 1:
      return logical_devices[0].name
    if 0 <= index < len(logical_devices):
      return logical_devices[index].name

  return logical_devices[0].name


def parse_musa_device_index(device_name):
  match = re.search(r"MUSA:(\d+)", device_name)
  if match:
    return int(match.group(1))
  return 0


def query_musa_memory_info(device_name):
  candidates = [
      os.environ.get("ADAMAX_MUSART_PATH"),
      ctypes.util.find_library("musart"),
      ctypes.util.find_library("musa"),
      "/usr/local/musa-3.1.2/lib/libmusart.so.1.0.0",
      "/usr/local/musa-3.1.2/lib/libmusa.so.1.0.0",
  ]

  lib = None
  errors = []
  for candidate in candidates:
    if not candidate:
      continue
    if os.path.sep in candidate and not os.path.exists(candidate):
      continue
    try:
      lib = ctypes.CDLL(candidate)
      break
    except OSError as exc:
      errors.append("%s: %s" % (candidate, exc))

  if lib is None:
    finish({
        "ok": False,
        "exc_type": "RuntimeError",
        "message": "Unable to load MUSA runtime library for memory query. Tried: %s" %
                   "; ".join(errors if errors else ["<none>"]),
    })

  device_index = parse_musa_device_index(device_name)
  if hasattr(lib, "musaSetDevice"):
    lib.musaSetDevice.argtypes = [ctypes.c_int]
    lib.musaSetDevice.restype = ctypes.c_int
    set_status = lib.musaSetDevice(device_index)
    if set_status != 0:
      finish({
          "ok": False,
          "exc_type": "RuntimeError",
          "message": "musaSetDevice(%d) failed with status %d" %
                     (device_index, set_status),
      })

  if not hasattr(lib, "musaMemGetInfo"):
    finish({
        "ok": False,
        "exc_type": "RuntimeError",
        "message": "Loaded runtime library does not expose musaMemGetInfo",
    })

  free_bytes = ctypes.c_size_t()
  total_bytes = ctypes.c_size_t()
  lib.musaMemGetInfo.argtypes = [
      ctypes.POINTER(ctypes.c_size_t),
      ctypes.POINTER(ctypes.c_size_t),
  ]
  lib.musaMemGetInfo.restype = ctypes.c_int
  status = lib.musaMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes))
  if status != 0:
    finish({
        "ok": False,
        "exc_type": "RuntimeError",
        "message": "musaMemGetInfo failed with status %d" % status,
    })

  finish({
      "ok": True,
      "memory_info": {
          "device_name": device_name,
          "device_index": device_index,
          "free_bytes": int(free_bytes.value),
          "total_bytes": int(total_bytes.value),
      },
  })


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


def encode_scalar(value):
  arr = np.asarray(value)
  if arr.shape == ():
    return float(arr)
  return float(arr.reshape(()))


def run_resource_update(payload):
  dtype_name = payload["dtype_name"]
  dtype = tf_dtype(dtype_name)
  np_dtype = input_numpy_dtype(dtype_name)
  device = payload["device_name"]

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

    with tf.device(device):
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

  sess = create_session(graph)
  sess.run(init_op)
  result = sess.run([read_var, read_m, read_v])
  finish({"ok": True, "result": encode_results(result, dtype_name)})


def run_large_resource_update(payload):
  dtype_name = payload["dtype_name"]
  dtype = tf_dtype(dtype_name)
  device = payload["device_name"]
  shape = payload["shape"]
  include_mean = bool(payload.get("include_mean", False))

  graph = tf.Graph()
  with graph.as_default():
    with tf.device(device):
      init_var = tf.fill(shape, tf.cast(payload["init_var_scalar"], dtype))
      init_m = tf.fill(shape, tf.cast(payload["init_m_scalar"], dtype))
      init_v = tf.fill(shape, tf.cast(payload["init_v_scalar"], dtype))
      grad = tf.fill(shape, tf.cast(payload["grad_scalar"], dtype))

      var = tf.Variable(init_var, dtype=dtype, name="var")
      m = tf.Variable(init_m, dtype=dtype, name="m")
      v = tf.Variable(init_v, dtype=dtype, name="v")

    with tf.device("/CPU:0"):
      beta1_power_t = tf.constant(payload["beta1_power"], dtype=dtype)
      lr_t = tf.constant(payload["lr"], dtype=dtype)
      beta1_t = tf.constant(payload["beta1"], dtype=dtype)
      beta2_t = tf.constant(payload["beta2"], dtype=dtype)
      epsilon_t = tf.constant(payload["epsilon"], dtype=dtype)

    with tf.device(device):
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
        var_value = var.read_value()
        m_value = m.read_value()
        v_value = v.read_value()
        flat_var = tf.reshape(var_value, [-1])
        flat_m = tf.reshape(m_value, [-1])
        flat_v = tf.reshape(v_value, [-1])
        summary_tensors = {
            "shape": tf.shape(var_value),
            "num_elements": tf.size(var_value),
            "var_first": flat_var[0],
            "m_first": flat_m[0],
            "v_first": flat_v[0],
        }
        if include_mean:
          summary_tensors["var_mean"] = tf.reduce_mean(tf.cast(var_value, tf.float32))
          summary_tensors["m_mean"] = tf.reduce_mean(tf.cast(m_value, tf.float32))
          summary_tensors["v_mean"] = tf.reduce_mean(tf.cast(v_value, tf.float32))

      init_op = tf.compat.v1.global_variables_initializer()

  sess = create_session(graph)
  sess.run(init_op)
  summary = sess.run(summary_tensors)
  encoded_summary = {
      "shape": np.asarray(summary["shape"], dtype=np.int64).tolist(),
      "num_elements": int(summary["num_elements"]),
      "var_first": encode_scalar(summary["var_first"]),
      "m_first": encode_scalar(summary["m_first"]),
      "v_first": encode_scalar(summary["v_first"]),
  }
  if include_mean:
    encoded_summary["var_mean"] = encode_scalar(summary["var_mean"])
    encoded_summary["m_mean"] = encode_scalar(summary["m_mean"])
    encoded_summary["v_mean"] = encode_scalar(summary["v_mean"])
  finish({"ok": True, "summary": encoded_summary})


def run_ref_update(payload):
  dtype_name = payload["dtype_name"]
  dtype = tf_dtype(dtype_name)
  np_dtype = input_numpy_dtype(dtype_name)
  device = payload["device_name"]

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

  sess = create_session(graph)
  sess.run(init_ref_vars)
  sess.run([var, m, v])
  sess.run(update)
  result = sess.run([var, m, v])
  finish({"ok": True, "result": encode_results(result, dtype_name)})


def run_non_scalar_error(device):
  graph = tf.Graph()
  with graph.as_default():
    with tf.device(device):
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

    with tf.device(device):
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

  sess = create_session(graph)
  sess.run(init_op)
  sess.run(update)
  finish({"ok": True, "unexpected_success": True})


def run_shape_mismatch_error(device):
  graph = tf.Graph()
  with graph.as_default():
    with tf.device(device):
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

    with tf.device(device):
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

  sess = create_session(graph)
  sess.run(init_op)
  sess.run(update)
  finish({"ok": True, "unexpected_success": True})


def run_invalid_beta1_power_error(device):
  graph = tf.Graph()
  with graph.as_default():
    with tf.device(device):
      var = tf.Variable([1.0, 2.0], dtype=tf.float32, name="var")
      m = tf.Variable([0.0, 0.0], dtype=tf.float32, name="m")
      v = tf.Variable([0.0, 0.0], dtype=tf.float32, name="v")
      grad = tf.constant([0.1, 0.2], dtype=tf.float32, name="grad")

    with tf.device("/CPU:0"):
      beta1_power = tf.constant(1.0, dtype=tf.float32)
      lr = tf.constant(0.01, dtype=tf.float32)
      beta1 = tf.constant(0.9, dtype=tf.float32)
      beta2 = tf.constant(0.999, dtype=tf.float32)
      epsilon = tf.constant(1e-8, dtype=tf.float32)

    with tf.device(device):
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

  sess = create_session(graph)
  sess.run(init_op)
  sess.run(update)
  finish({"ok": True, "unexpected_success": True})


def main():
  payload = json.load(sys.stdin)
  load_musa_plugin(payload["project_root"])
  if not tf.config.list_physical_devices("MUSA"):
    raise RuntimeError("No MUSA devices found.")
  tf.compat.v1.disable_eager_execution()
  device_name = payload.get("device_name") or resolve_musa_device()

  mode = payload["mode"]
  if mode == "memory_info":
    query_musa_memory_info(device_name)
  if mode == "resource_update":
    payload["device_name"] = device_name
    run_resource_update(payload)
  if mode == "large_resource_update":
    payload["device_name"] = device_name
    run_large_resource_update(payload)
  if mode == "ref_update":
    payload["device_name"] = device_name
    run_ref_update(payload)
  if mode == "non_scalar_error":
    run_non_scalar_error(device_name)
  if mode == "shape_mismatch_error":
    run_shape_mismatch_error(device_name)
  if mode == "invalid_beta1_power_error":
    run_invalid_beta1_power_error(device_name)
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
    return np.float32 if dtype in [tf.float16, tf.bfloat16] else dtype.as_numpy_dtype

  def _assert_by_dtype(self, expected, actual, dtype):
    if dtype in [tf.float16, tf.bfloat16]:
      self.assertAllClose(
          np.asarray(expected, dtype=np.float32),
          np.asarray(actual, dtype=np.float32),
          rtol=1e-2,
          atol=1e-2)
    else:
      self.assertAllClose(expected, actual, rtol=1e-5, atol=1e-8)

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
    for expected_tensor, actual_tensor in zip(expected, actual):
      self._assert_by_dtype(expected_tensor, actual_tensor, dtype)

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

  def _parse_large_test_shape(self, num_elements):
    shape_env = os.environ.get("ADAMAX_LARGE_TEST_SHAPE")
    if shape_env:
      dims = [int(part.strip()) for part in shape_env.split(",") if part.strip()]
      if not dims:
        raise ValueError("ADAMAX_LARGE_TEST_SHAPE is empty.")
      product = 1
      for dim in dims:
        product *= dim
      if product != num_elements:
        raise ValueError(
            "ADAMAX_LARGE_TEST_SHAPE product %d does not match computed "
            "num_elements %d" % (product, num_elements))
      return dims
    return [num_elements]

  def _bytes_per_element(self, dtype):
    return tf.as_dtype(dtype).size

  def _estimate_large_test_required_bytes(self, dtype, num_elements):
    # Current MuDNN AdaMax path keeps 18 full-shape tensors alive at peak:
    # 4 state/input tensors + 6 broadcasted scalar tensors + 8 intermediates.
    full_shape_tensor_count = int(
        os.environ.get("ADAMAX_LARGE_TEST_FULL_TENSOR_COUNT", "18"))
    safety_factor = float(
        os.environ.get("ADAMAX_LARGE_TEST_MEMORY_FACTOR", "1.35"))
    bytes_per_elem = self._bytes_per_element(dtype)
    required = num_elements * bytes_per_elem * full_shape_tensor_count
    return int(math.ceil(required * safety_factor))

  def _format_gib(self, value_bytes):
    return "%.2f GiB" % (float(value_bytes) / (1024 ** 3))

  def _query_large_test_memory_info(self):
    response = self._run_isolated({"mode": "memory_info"})
    if not response["ok"]:
      return None, response["message"]
    return response["memory_info"], None

  def _estimated_safe_raw_input_gib(self, available_bytes):
    full_shape_tensor_count = float(
        os.environ.get("ADAMAX_LARGE_TEST_FULL_TENSOR_COUNT", "18"))
    safety_factor = float(
        os.environ.get("ADAMAX_LARGE_TEST_MEMORY_FACTOR", "1.35"))
    if full_shape_tensor_count <= 0 or safety_factor <= 0:
      return 0.0
    # raw_input_bytes counts only var/m/v/grad, i.e. 4 full-shape tensors.
    raw_input_bytes = available_bytes * 4.0 / (full_shape_tensor_count *
                                               safety_factor)
    return raw_input_bytes / (1024 ** 3)

  def _run_large_resource_apply_adamax(self,
                                       shape,
                                       dtype,
                                       init_var_scalar=1.0,
                                       init_m_scalar=0.0,
                                       init_v_scalar=0.0,
                                       grad_scalar=0.25,
                                       beta1_power=0.9,
                                       lr=0.01,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-8,
                                       use_locking=False,
                                       include_mean=False,
                                       error_context=None):
    response = self._run_isolated({
        "mode": "large_resource_update",
        "dtype_name": dtype.name,
        "shape": shape,
        "init_var_scalar": init_var_scalar,
        "init_m_scalar": init_m_scalar,
        "init_v_scalar": init_v_scalar,
        "grad_scalar": grad_scalar,
        "beta1_power": beta1_power,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "use_locking": use_locking,
        "include_mean": include_mean,
    })
    if not response["ok"]:
      message = "Large ResourceApplyAdaMax failed: %s" % response["message"]
      if error_context:
        message = "%s\nPrecheck: %s" % (message, error_context)
      self.fail(message)
    return response["summary"]

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

  def test_resource_apply_adamax_multiple_shapes(self):
    beta1_power = 0.9
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

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
            dtype=tf.float32)

        expected = self._compute_expected_state(
            init_var, init_m, init_v, grad, beta1_power, lr, beta1, beta2,
            epsilon, tf.float32)
        self._assert_state_close(expected, actual, tf.float32)

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

    response = self._run_isolated({
        "mode": "ref_update",
        "dtype_name": dtype.name,
        "init_var": self._serialize_array(init_var),
        "init_m": self._serialize_array(init_m),
        "init_v": self._serialize_array(init_v),
        "grad": self._serialize_array(grad),
        "beta1_power": 0.9,
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "use_locking": False,
    })
    if not response["ok"]:
      self.skipTest(
          "RefVariable ApplyAdaMax remains unstable in TF2.6.1 graph mode: %s" %
          response["message"])

    actual = tuple(np.asarray(value, dtype=np.float32) for value in response["result"])
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

  def test_resource_apply_adamax_rejects_beta1_power_one(self):
    response = self._run_isolated({"mode": "invalid_beta1_power_error"})
    self.assertFalse(response["ok"], "Expected beta1_power validation error")
    self.assertRegex(response["message"], "beta1_power must not be 1")

  def test_resource_apply_adamax_large_stress(self):
    if os.environ.get("ADAMAX_ENABLE_LARGE_TEST") != "1":
      self.skipTest("Set ADAMAX_ENABLE_LARGE_TEST=1 to run the large stress test.")

    dtype_name = os.environ.get("ADAMAX_LARGE_TEST_DTYPE", "float32")
    dtype = getattr(tf, dtype_name)
    target_gib = float(os.environ.get("ADAMAX_LARGE_TEST_GIB", "16"))
    include_mean = os.environ.get("ADAMAX_LARGE_TEST_INCLUDE_MEAN") == "1"
    if (target_gib >= 16.0 and
        os.environ.get("ADAMAX_LARGE_TEST_SKIP_16G", "1") == "1"):
      self.skipTest(
          "16 GiB AdaMax large-stress runs are skipped by default because the "
          "current MuDNN chain implementation is known to OOM at this scale. "
          "Set ADAMAX_LARGE_TEST_SKIP_16G=0 to force execution.")

    # Count only var/m/v/grad to size the logical test payload.
    num_elements = int((target_gib * (1024 ** 3)) /
                       (4 * self._bytes_per_element(dtype)))
    if num_elements <= 0:
      self.fail("Computed num_elements must be positive for the large stress test.")

    shape = self._parse_large_test_shape(num_elements)
    required_bytes = self._estimate_large_test_required_bytes(dtype, num_elements)
    memory_info, memory_error = self._query_large_test_memory_info()
    skip_if_insufficient = (
        os.environ.get("ADAMAX_LARGE_TEST_SKIP_IF_INSUFFICIENT_MEMORY") == "1")
    precheck_message = (
        "estimated_required=%s, raw_input=%s GiB, dtype=%s, shape=%s" %
        (self._format_gib(required_bytes), target_gib, dtype.name, shape))
    if memory_info is not None:
      free_bytes = int(memory_info["free_bytes"])
      total_bytes = int(memory_info["total_bytes"])
      safe_raw_gib = self._estimated_safe_raw_input_gib(free_bytes)
      precheck_message = (
          "%s, current_free=%s, total=%s, device=%s, estimated_safe_raw_upper_bound=%.2f GiB" %
          (precheck_message, self._format_gib(free_bytes),
           self._format_gib(total_bytes), memory_info["device_name"], safe_raw_gib))
      if required_bytes > free_bytes and skip_if_insufficient:
        self.skipTest(
            "Estimated AdaMax peak memory %s exceeds current free MUSA memory %s "
            "(total %s) on %s. Current implementation likely needs at least %s "
            "free to run this shape. Estimated safe raw-input upper bound is about "
            "%.2f GiB. Unset ADAMAX_LARGE_TEST_SKIP_IF_INSUFFICIENT_MEMORY to "
            "run anyway and capture the OOM path." %
            (self._format_gib(required_bytes), self._format_gib(free_bytes),
             self._format_gib(total_bytes), memory_info["device_name"],
             self._format_gib(required_bytes), safe_raw_gib))
    elif memory_error:
      precheck_message = "%s, memory_query_error=%s" % (precheck_message,
                                                         memory_error)
      if os.environ.get("ADAMAX_LARGE_TEST_REQUIRE_MEMORY_QUERY") == "1":
        self.skipTest(
            "Estimated AdaMax peak memory is %s for dtype=%s shape=%s, but current "
            "MUSA free memory could not be queried (%s). The large stress test is "
            "skipped because ADAMAX_LARGE_TEST_REQUIRE_MEMORY_QUERY=1 is set." %
            (self._format_gib(required_bytes), dtype.name, shape, memory_error))

    grad_scalar = float(os.environ.get("ADAMAX_LARGE_TEST_GRAD", "0.25"))
    init_var_scalar = float(os.environ.get("ADAMAX_LARGE_TEST_INIT_VAR", "1.0"))
    init_m_scalar = float(os.environ.get("ADAMAX_LARGE_TEST_INIT_M", "0.0"))
    init_v_scalar = float(os.environ.get("ADAMAX_LARGE_TEST_INIT_V", "0.0"))

    summary = self._run_large_resource_apply_adamax(
        shape=shape,
        dtype=dtype,
        init_var_scalar=init_var_scalar,
        init_m_scalar=init_m_scalar,
        init_v_scalar=init_v_scalar,
        grad_scalar=grad_scalar,
        beta1_power=0.9,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        use_locking=False,
        include_mean=include_mean,
        error_context=precheck_message)

    expected_var, expected_m, expected_v = self._compute_expected_state(
        np.array([init_var_scalar], dtype=np.float32),
        np.array([init_m_scalar], dtype=np.float32),
        np.array([init_v_scalar], dtype=np.float32),
        np.array([grad_scalar], dtype=np.float32),
        0.9,
        0.01,
        0.9,
        0.999,
        1e-8,
        dtype)

    expected_shape = [int(dim) for dim in shape]
    self.assertEqual(summary["shape"], expected_shape)
    self.assertEqual(summary["num_elements"], num_elements)
    self._assert_by_dtype(expected_var[0], summary["var_first"], dtype)
    self._assert_by_dtype(expected_m[0], summary["m_first"], dtype)
    self._assert_by_dtype(expected_v[0], summary["v_first"], dtype)
    if include_mean:
      self._assert_by_dtype(expected_var[0], summary["var_mean"], dtype)
      self._assert_by_dtype(expected_m[0], summary["m_mean"], dtype)
      self._assert_by_dtype(expected_v[0], summary["v_mean"], dtype)


if __name__ == "__main__":
  unittest.main()
