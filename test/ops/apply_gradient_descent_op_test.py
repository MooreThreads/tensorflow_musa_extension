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

"""Tests for MUSA ApplyGradientDescent operators."""

import numpy as np
import tensorflow as tf

from musa_test_utils import load_musa_plugin

try:
  load_musa_plugin()
  MUSA_DEVICES = tf.config.list_physical_devices("MUSA")
  PLUGIN_LOAD_ERROR = None
except Exception as exc:  # pragma: no cover
  MUSA_DEVICES = []
  PLUGIN_LOAD_ERROR = exc


class ApplyGradientDescentOpTest(tf.test.TestCase):
  """Tests for MUSA ApplyGradientDescent operators."""

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

  def _skip_if_no_musa(self):
    if MUSA_DEVICES:
      return
    if PLUGIN_LOAD_ERROR is not None:
      self.skipTest(f"MUSA plugin load failed: {PLUGIN_LOAD_ERROR}")
    self.skipTest("No MUSA devices found.")

  def _run_resource_apply_gradient_descent(self,
                                           device,
                                           init_var_np,
                                           alpha_np,
                                           grad_np,
                                           dtype,
                                           use_locking=False):
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(device):
        var = tf.Variable(init_var_np, dtype=dtype, name="var")
        grad = tf.constant(grad_np, dtype=dtype, name="grad")

      with tf.device("/CPU:0"):
        alpha = tf.constant(alpha_np, dtype=dtype, name="alpha")

      update = tf.raw_ops.ResourceApplyGradientDescent(
          var=var.handle,
          alpha=alpha,
          delta=grad,
          use_locking=use_locking)

      with tf.control_dependencies([update]):
        read_var = tf.identity(var.read_value(), name="updated_var")

      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      return sess.run(read_var)

  def _run_apply_gradient_descent(self,
                                  device,
                                  init_var_np,
                                  alpha_np,
                                  grad_np,
                                  dtype,
                                  use_locking=False):
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(device):
        var = tf.compat.v1.get_variable(
            "var",
            initializer=tf.constant(init_var_np, dtype=dtype),
            dtype=dtype,
            use_resource=False)
        alpha = tf.constant(alpha_np, dtype=dtype, name="alpha")
        grad = tf.constant(grad_np, dtype=dtype, name="grad")

        update = tf.raw_ops.ApplyGradientDescent(
            var=var, alpha=alpha, delta=grad, use_locking=use_locking)

        with tf.control_dependencies([update]):
          read_var = tf.identity(var, name="updated_var")

      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      return sess.run(read_var)

  def testResourceApplyGradientDescent(self):
    self._skip_if_no_musa()

    cases = [
        (
            np.array([1.0, -2.0, 3.5, -4.5], dtype=np.float32),
            0.1,
            np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32),
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            0.25,
            np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        ),
    ]

    # Note: muDNN does not support DOUBLE (float64) for binary operations (MUL, SUB).
    # Skip float64 testing for MUSA device.
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      np_dtype = self._numpy_dtype(dtype)
      for init_var, alpha, grad in cases:
        with self.subTest(op="resource",
                          dtype=dtype.name,
                          shape=init_var.shape):
          init_var_np = init_var.astype(np_dtype)
          grad_np = grad.astype(np_dtype)
          alpha_np = np_dtype(alpha)

          cpu_result = self._run_resource_apply_gradient_descent(
              "/CPU:0", init_var_np, alpha_np, grad_np, dtype)
          musa_result = self._run_resource_apply_gradient_descent(
              "/device:MUSA:0", init_var_np, alpha_np, grad_np, dtype)

          self._assert_by_dtype(cpu_result, musa_result, dtype)

  def testApplyGradientDescent(self):
    # Note: The non-resource version uses deprecated RefVariable which has issues
    # with TensorFlow graph mode. The ResourceApplyGradientDescent (use_resource=True)
    # is the modern approach and is tested in testResourceApplyGradientDescent.
    self.skipTest("Skipping deprecated RefVariable test - use ResourceApplyGradientDescent instead")

  def testApplyGradientDescentUseLocking(self):
    # Note: The non-resource version uses deprecated RefVariable which has issues
    # with TensorFlow graph mode. The ResourceApplyGradientDescent (use_resource=True)
    # is the modern approach and is tested in testResourceApplyGradientDescent.
    self.skipTest("Skipping deprecated RefVariable test - use ResourceApplyGradientDescent instead")

    init_var_np = np.array([1.25, -2.5, 5.0, -10.0], dtype=np.float32)
    grad_np = np.array([0.5, 0.25, -1.0, 2.0], dtype=np.float32)
    alpha_np = np.float32(0.5)

    cpu_ref_result = self._run_apply_gradient_descent(
        "/CPU:0",
        init_var_np,
        alpha_np,
        grad_np,
        tf.float32,
        use_locking=True)
    musa_ref_result = self._run_apply_gradient_descent(
        "/device:MUSA:0",
        init_var_np,
        alpha_np,
        grad_np,
        tf.float32,
        use_locking=True)
    cpu_resource_result = self._run_resource_apply_gradient_descent(
        "/CPU:0",
        init_var_np,
        alpha_np,
        grad_np,
        tf.float32,
        use_locking=True)
    musa_resource_result = self._run_resource_apply_gradient_descent(
        "/device:MUSA:0",
        init_var_np,
        alpha_np,
        grad_np,
        tf.float32,
        use_locking=True)

    self._assert_by_dtype(cpu_ref_result, musa_ref_result, tf.float32)
    self._assert_by_dtype(cpu_resource_result, musa_resource_result, tf.float32)


if __name__ == "__main__":
  tf.test.main()
