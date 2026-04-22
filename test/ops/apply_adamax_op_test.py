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

"""Tests for MUSA ApplyAdaMax operators.

Mirrors the structure of ``apply_gradient_descent_op_test.py``: each test case
runs the op in a ``tf.Graph`` / ``tf.compat.v1.Session`` on CPU and on MUSA
and compares the post-update values of ``var``, ``m`` and ``v``.

AdaMax update rule (TF's ``ResourceApplyAdaMax`` / ``ApplyAdaMax``):

    m_t   = beta1 * m + (1 - beta1) * grad
    v_t   = max(beta2 * v, abs(grad))
    var_t = var - (lr / (1 - beta1_power)) * m_t / (v_t + epsilon)

Note: the MUSA kernel is registered only for float32 / float16 / bfloat16
(see ``musa_applyadamax_op.cc``); float64 is intentionally not exercised.
"""

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


class ApplyAdaMaxOpTest(tf.test.TestCase):
  """Tests for MUSA ApplyAdaMax operators."""

  def _numpy_dtype(self, dtype):
    return np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

  def _tolerances(self, dtype):
    # AdaMax involves a division by (v_t + epsilon), so fp16 / bf16 lose a
    # few bits of precision vs the fp32 reference path. Match the tolerances
    # used by apply_gradient_descent_op_test.py.
    if dtype == tf.float16:
      return dict(rtol=1e-2, atol=1e-2)
    if dtype == tf.bfloat16:
      return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=1e-5, atol=1e-6)

  def _assert_by_dtype(self, expected, actual, dtype):
    tols = self._tolerances(dtype)
    if dtype in (tf.float16, tf.bfloat16):
      self.assertAllClose(
          np.asarray(expected, dtype=np.float32),
          np.asarray(actual, dtype=np.float32),
          **tols)
    else:
      self.assertAllClose(expected, actual, **tols)

  def _skip_if_no_musa(self):
    if MUSA_DEVICES:
      return
    if PLUGIN_LOAD_ERROR is not None:
      self.skipTest(f"MUSA plugin load failed: {PLUGIN_LOAD_ERROR}")
    self.skipTest("No MUSA devices found.")

  def _adamax_reference(self, var, m, v, beta1_power, lr, beta1, beta2,
                        epsilon, grad):
    """NumPy AdaMax reference in float64 for stable comparisons."""
    var64 = var.astype(np.float64)
    m64 = m.astype(np.float64)
    v64 = v.astype(np.float64)
    grad64 = grad.astype(np.float64)

    beta1_power64 = float(beta1_power)
    lr64 = float(lr)
    beta1_64 = float(beta1)
    beta2_64 = float(beta2)
    epsilon64 = float(epsilon)

    m_new = beta1_64 * m64 + (1.0 - beta1_64) * grad64
    v_new = np.maximum(beta2_64 * v64, np.abs(grad64))
    var_new = var64 - (lr64 / (1.0 - beta1_power64)) * m_new / (v_new +
                                                                epsilon64)
    return var_new, m_new, v_new

  def _run_resource_apply_adamax(self,
                                 device,
                                 init_var_np,
                                 init_m_np,
                                 init_v_np,
                                 beta1_power_np,
                                 lr_np,
                                 beta1_np,
                                 beta2_np,
                                 epsilon_np,
                                 grad_np,
                                 dtype,
                                 use_locking=False):
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(device):
        var = tf.Variable(init_var_np, dtype=dtype, name="var")
        m = tf.Variable(init_m_np, dtype=dtype, name="m")
        v = tf.Variable(init_v_np, dtype=dtype, name="v")
        grad = tf.constant(grad_np, dtype=dtype, name="grad")

      # beta1_power / lr / beta1 / beta2 / epsilon are registered as
      # .HostMemory() on MUSA, so pinning them to CPU matches the kernel
      # expectation and avoids a spurious H2D for every step.
      with tf.device("/CPU:0"):
        beta1_power = tf.constant(beta1_power_np, dtype=dtype,
                                  name="beta1_power")
        lr = tf.constant(lr_np, dtype=dtype, name="lr")
        beta1 = tf.constant(beta1_np, dtype=dtype, name="beta1")
        beta2 = tf.constant(beta2_np, dtype=dtype, name="beta2")
        epsilon = tf.constant(epsilon_np, dtype=dtype, name="epsilon")

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
          use_locking=use_locking)

      with tf.control_dependencies([update]):
        read_var = tf.identity(var.read_value(), name="updated_var")
        read_m = tf.identity(m.read_value(), name="updated_m")
        read_v = tf.identity(v.read_value(), name="updated_v")

      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      return sess.run([read_var, read_m, read_v])

  def _standard_hyperparams(self):
    """Shared, numerically well-conditioned AdaMax hyperparameters."""
    return dict(
        beta1=0.9,
        beta2=0.999,
        beta1_power=0.9,  # i.e. step 1; 1 - beta1_power != 0
        lr=0.01,
        epsilon=1e-7,
    )

  def _random_state(self, shape, seed):
    """Generate (var, m, v, grad) with controlled magnitudes per shape."""
    rng = np.random.default_rng(seed)
    var = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    m = rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)
    # AdaMax's v state is the running max of |grad|, so it should stay >= 0.
    v = rng.uniform(0.0, 0.5, size=shape).astype(np.float32)
    grad = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return var, m, v, grad

  def testResourceApplyAdaMax(self):
    self._skip_if_no_musa()

    shapes = [
        (4,),
        (2, 3),
        (3, 4, 5),
    ]
    hp = self._standard_hyperparams()

    for dtype in (tf.float32, tf.float16, tf.bfloat16):
      np_dtype = self._numpy_dtype(dtype)
      for shape in shapes:
        with self.subTest(op="resource", dtype=dtype.name, shape=shape):
          var_f32, m_f32, v_f32, grad_f32 = self._random_state(shape,
                                                               seed=hash(
                                                                   (dtype.name,
                                                                    shape)) &
                                                               0xFFFFFFFF)

          init_var_np = var_f32.astype(np_dtype)
          init_m_np = m_f32.astype(np_dtype)
          init_v_np = v_f32.astype(np_dtype)
          grad_np = grad_f32.astype(np_dtype)
          beta1_power_np = np_dtype(hp["beta1_power"])
          lr_np = np_dtype(hp["lr"])
          beta1_np = np_dtype(hp["beta1"])
          beta2_np = np_dtype(hp["beta2"])
          epsilon_np = np_dtype(hp["epsilon"])

          cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
              "/CPU:0", init_var_np, init_m_np, init_v_np, beta1_power_np,
              lr_np, beta1_np, beta2_np, epsilon_np, grad_np, dtype)
          musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
              "/device:MUSA:0", init_var_np, init_m_np, init_v_np,
              beta1_power_np, lr_np, beta1_np, beta2_np, epsilon_np, grad_np,
              dtype)

          self._assert_by_dtype(cpu_var, musa_var, dtype)
          self._assert_by_dtype(cpu_m, musa_m, dtype)
          self._assert_by_dtype(cpu_v, musa_v, dtype)

  def testResourceApplyAdaMaxMatchesNumpyReference(self):
    """MUSA output must also match an independent fp64 NumPy reference.

    Comparing only against CPU TF would hide a bug where both CPU and MUSA
    regress in the same direction. Running a NumPy reference in fp64 and
    casting down anchors the expected value independently.
    """
    self._skip_if_no_musa()

    hp = self._standard_hyperparams()
    shape = (2, 4)
    var_f32, m_f32, v_f32, grad_f32 = self._random_state(shape, seed=42)

    for dtype in (tf.float32, tf.float16, tf.bfloat16):
      np_dtype = self._numpy_dtype(dtype)
      with self.subTest(dtype=dtype.name):
        init_var_np = var_f32.astype(np_dtype)
        init_m_np = m_f32.astype(np_dtype)
        init_v_np = v_f32.astype(np_dtype)
        grad_np = grad_f32.astype(np_dtype)

        expected_var, expected_m, expected_v = self._adamax_reference(
            init_var_np, init_m_np, init_v_np, hp["beta1_power"], hp["lr"],
            hp["beta1"], hp["beta2"], hp["epsilon"], grad_np)

        musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
            "/device:MUSA:0", init_var_np, init_m_np, init_v_np,
            np_dtype(hp["beta1_power"]), np_dtype(hp["lr"]),
            np_dtype(hp["beta1"]), np_dtype(hp["beta2"]),
            np_dtype(hp["epsilon"]), grad_np, dtype)

        self._assert_by_dtype(expected_var, musa_var, dtype)
        self._assert_by_dtype(expected_m, musa_m, dtype)
        self._assert_by_dtype(expected_v, musa_v, dtype)

  def testResourceApplyAdaMaxZeroGrad(self):
    """Zero gradient: exercises the ``max(beta2 * v, |grad|)`` branch where
    ``|grad|`` is the smaller operand. Note that ``var`` still changes
    because the m momentum from a previous step is non-zero."""
    self._skip_if_no_musa()

    hp = self._standard_hyperparams()
    shape = (5,)
    np_dtype = np.float32

    init_var_np = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np_dtype)
    init_m_np = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np_dtype)
    init_v_np = np.array([0.05, 0.06, 0.07, 0.08, 0.09], dtype=np_dtype)
    grad_np = np.zeros(shape, dtype=np_dtype)

    cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
        "/CPU:0", init_var_np, init_m_np, init_v_np,
        np_dtype(hp["beta1_power"]), np_dtype(hp["lr"]),
        np_dtype(hp["beta1"]), np_dtype(hp["beta2"]),
        np_dtype(hp["epsilon"]), grad_np, tf.float32)
    musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
        "/device:MUSA:0", init_var_np, init_m_np, init_v_np,
        np_dtype(hp["beta1_power"]), np_dtype(hp["lr"]),
        np_dtype(hp["beta1"]), np_dtype(hp["beta2"]),
        np_dtype(hp["epsilon"]), grad_np, tf.float32)

    self._assert_by_dtype(cpu_var, musa_var, tf.float32)
    self._assert_by_dtype(cpu_m, musa_m, tf.float32)
    self._assert_by_dtype(cpu_v, musa_v, tf.float32)

  def testResourceApplyAdaMaxUseLocking(self):
    """use_locking=True must not change the numerical result."""
    self._skip_if_no_musa()

    hp = self._standard_hyperparams()
    shape = (3, 3)
    np_dtype = np.float32
    var_f32, m_f32, v_f32, grad_f32 = self._random_state(shape, seed=7)

    for use_locking in (False, True):
      with self.subTest(use_locking=use_locking):
        cpu_var, cpu_m, cpu_v = self._run_resource_apply_adamax(
            "/CPU:0", var_f32.astype(np_dtype), m_f32.astype(np_dtype),
            v_f32.astype(np_dtype), np_dtype(hp["beta1_power"]),
            np_dtype(hp["lr"]), np_dtype(hp["beta1"]),
            np_dtype(hp["beta2"]), np_dtype(hp["epsilon"]),
            grad_f32.astype(np_dtype), tf.float32,
            use_locking=use_locking)
        musa_var, musa_m, musa_v = self._run_resource_apply_adamax(
            "/device:MUSA:0", var_f32.astype(np_dtype),
            m_f32.astype(np_dtype), v_f32.astype(np_dtype),
            np_dtype(hp["beta1_power"]), np_dtype(hp["lr"]),
            np_dtype(hp["beta1"]), np_dtype(hp["beta2"]),
            np_dtype(hp["epsilon"]), grad_f32.astype(np_dtype),
            tf.float32, use_locking=use_locking)

        self._assert_by_dtype(cpu_var, musa_var, tf.float32)
        self._assert_by_dtype(cpu_m, musa_m, tf.float32)
        self._assert_by_dtype(cpu_v, musa_v, tf.float32)

  def testApplyAdaMax(self):
    # The non-resource ApplyAdaMax op operates on deprecated RefVariable
    # handles which don't round-trip cleanly through TF 2.6's graph mode.
    # ResourceApplyAdaMax (use_resource=True) is the modern codepath and is
    # covered by testResourceApplyAdaMax above; mirrors the treatment of
    # ApplyGradientDescent in apply_gradient_descent_op_test.py.
    self.skipTest(
        "Skipping deprecated RefVariable test - use ResourceApplyAdaMax "
        "instead")


if __name__ == "__main__":
  tf.test.main()
