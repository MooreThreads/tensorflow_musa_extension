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

"""Utilities for MUSA kernel tests."""

import logging
import os
import sys
import unittest
import tensorflow as tf


def _get_test_full_name(test_method):
  """Get the full name of a test method in format 'module.class.method'."""
  test_class = test_method.__self__.__class__
  module_name = test_class.__module__
  # Extract just the filename without path and extension
  module_file = module_name.split('.')[-1] if module_name else 'unknown'
  class_name = test_class.__name__
  method_name = test_method.__name__
  return f"{module_file}.{class_name}.{method_name}"


# Idempotency guard. PluggableDevice loading is NOT safe to call twice: every
# invocation of ``tf.load_pluggable_device_library`` re-runs ``SE_InitPlugin``,
# which in turn tries to ``RegisterPlatform("MUSA")``. The second call fails
# with a fatal CHECK ("platform is already registered with name: MUSA") and
# aborts the process. Callers can freely ``load_musa_plugin()`` as many times
# as they want; we dedupe here.
_MUSA_PLUGIN_LOADED_PATH = None
# Module returned by ``tf.load_op_library``. Exposes Python wrappers for every
# op whose C++ registration lives in ``libmusa_plugin.so`` (MusaInteract,
# MusaLayerNorm, ResourceApplyNadam, ...). This module only gets populated when
# ``load_op_library`` runs *before* ``load_pluggable_device_library`` -- if the
# pluggable-device path dlopens the .so first, TF tags the library as "already
# loaded" and returns an empty module from any subsequent ``load_op_library``
# call. We therefore load in the order: ops-first, device-second.
_MUSA_OPS_MODULE = None


def load_musa_plugin():
  """Load the MUSA PluggableDevice plugin (idempotent).

  The plugin is registered with TensorFlow via the public
  ``SE_InitPlugin`` C API, which is only invoked through
  ``load_library.load_pluggable_device_library``. We additionally call
  ``tf.load_op_library`` first so tests can reach custom ops
  (``MusaInteract``, ``MusaLayerNorm``, ``ResourceApplyNadam``, ...) via
  a returned module. ``tf.raw_ops`` is populated at TF build time and does
  not expose dynamically-loaded custom ops, so a fresh op_library module is
  the only portable entry point for them in eager mode.
  """
  global _MUSA_PLUGIN_LOADED_PATH, _MUSA_OPS_MODULE
  if _MUSA_PLUGIN_LOADED_PATH is not None:
    return _MUSA_PLUGIN_LOADED_PATH

  plugin_path = None
  current_dir = os.path.dirname(os.path.abspath(__file__))

  candidate_paths = [
    # Relative to test directory (most common case)
    os.path.join(current_dir, "..", "build", "libmusa_plugin.so"),
    # Relative to project root (when running from project root)
    os.path.join(os.path.dirname(current_dir), "build", "libmusa_plugin.so"),
    # Current working directory build
    os.path.join(os.getcwd(), "build", "libmusa_plugin.so"),
  ]

  for path in candidate_paths:
    normalized_path = os.path.normpath(path)
    if os.path.exists(normalized_path):
      plugin_path = normalized_path
      break

  if not (plugin_path and os.path.exists(plugin_path)):
    searched_locations = [os.path.normpath(path) for path in candidate_paths]
    raise FileNotFoundError(
        "MUSA plugin not found. Searched locations:\n" +
        "\n".join(f"  - {loc}" for loc in searched_locations)
    )

  try:
    # Step 1: load ops. This dlopens the .so (firing C++ static initializers
    # -- REGISTER_OP, REGISTER_KERNEL_BUILDER -- into the op registry) and
    # returns a Python module exposing wrappers for every newly-registered op.
    # Must run before the pluggable-device path, which marks the library as
    # already-loaded and makes the follow-up load_op_library return an empty
    # module.
    try:
      _MUSA_OPS_MODULE = tf.load_op_library(plugin_path)
    except Exception:
      _MUSA_OPS_MODULE = None

    # Step 2: register the device platform via the pluggable-device C API.
    load_pluggable = None
    try:
      from tensorflow.python.framework import load_library as _ll
      load_pluggable = getattr(_ll, "load_pluggable_device_library", None)
    except ImportError:
      load_pluggable = None
    if load_pluggable is None:
      load_pluggable = getattr(tf, "load_pluggable_device_library", None)

    if load_pluggable is not None:
      load_pluggable(plugin_path)
    elif _MUSA_OPS_MODULE is None:
      # No pluggable-device entry point and load_op_library failed too.
      tf.load_library(plugin_path)
  except Exception as e:
    print(f"FAILED: Error loading MUSA plugin from {plugin_path}: {e}")
    raise

  _MUSA_PLUGIN_LOADED_PATH = plugin_path
  return plugin_path


def get_musa_ops():
  """Return the module exposing this plugin's custom op wrappers.

  The module's attributes map to the ``REGISTER_OP(Name)`` declarations in
  ``libmusa_plugin.so`` -- e.g. ``get_musa_ops().musa_interact(...)``.
  Returns ``None`` if the plugin hasn't been loaded or if load_op_library
  couldn't resolve it (older TF builds).
  """
  if _MUSA_PLUGIN_LOADED_PATH is None:
    load_musa_plugin()
  return _MUSA_OPS_MODULE


# Import tensorflow first (load_musa_plugin needs it)
import tensorflow as tf

# Load plugin immediately after importing tensorflow
load_musa_plugin()


class MUSATestCase(tf.test.TestCase):
  """Base test class for MUSA kernel tests."""

  @classmethod
  def setUpClass(cls):
    """Set up the test class."""
    super(MUSATestCase, cls).setUpClass()

    # Verify MUSA device is available (plugin already loaded at module import)
    if not tf.config.list_physical_devices('MUSA'):
      raise unittest.SkipTest("No MUSA devices found.")

  def _test_op_device_placement(self, op_func, input_tensors, device):
    """Test operation on specified device."""
    with tf.device(device):
      result = op_func(*input_tensors)
    return result

  def _compare_cpu_musa_results(self,
                               op_func,
                               input_tensors,
                               dtype,
                               rtol=1e-5,
                               atol=1e-8):
    """Compare results between CPU and MUSA devices."""
    # Test on CPU
    cpu_result = self._test_op_device_placement(op_func, input_tensors, '/CPU:0')

    # Test on MUSA
    musa_result = self._test_op_device_placement(op_func, input_tensors, '/device:MUSA:0')

    # Convert to float32 for comparison if needed
    if dtype in [tf.float16, tf.bfloat16]:
      cpu_result_f32 = tf.cast(cpu_result, tf.float32)
      musa_result_f32 = tf.cast(musa_result, tf.float32)
      self.assertAllClose(cpu_result_f32.numpy(),
                         musa_result_f32.numpy(),
                         rtol=rtol,
                         atol=atol)
    else:
      self.assertAllClose(cpu_result.numpy(),
                         musa_result.numpy(),
                         rtol=rtol,
                         atol=atol)

  def assertAllClose(self, a, b, rtol=1e-5, atol=1e-8, max_diffs_to_show=5):
    """
    Custom assertAllClose that limits output to avoid excessive printing.
    This overrides the parent class method to provide more concise error messages.

    Args:
      a, b: Arrays to compare (can be numpy arrays or TensorFlow tensors)
      rtol, atol: Relative and absolute tolerance
      max_diffs_to_show: Maximum number of differing elements to show
    """
    import numpy as np
    import tensorflow as tf

    # Convert TensorFlow tensors to numpy arrays
    if hasattr(a, 'numpy'):
      a = a.numpy()
    if hasattr(b, 'numpy'):
      b = b.numpy()

    # Handle bfloat16 by converting to float32 for comparison
    # bfloat16 is not a standard numpy type
    if hasattr(a, 'dtype') and a.dtype == tf.bfloat16.as_numpy_dtype:
      a = a.astype(np.float32) if hasattr(a, 'astype') else np.array(a, dtype=np.float32)
    if hasattr(b, 'dtype') and b.dtype == tf.bfloat16.as_numpy_dtype:
      b = b.astype(np.float32) if hasattr(b, 'astype') else np.array(b, dtype=np.float32)

    # Ensure both are numpy arrays
    a = np.array(a)
    b = np.array(b)

    # Use numpy's allclose for the actual comparison
    if np.allclose(a, b, rtol=rtol, atol=atol):
      return  # Success, no assertion error

    # If they're not close, provide limited diagnostic info
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    total_elements = a.size
    mismatched_mask = diff > (atol + rtol * np.abs(b))
    mismatched_count = np.sum(mismatched_mask)

    # Build concise error message
    msg_parts = [
        f"Arrays are not close (shape: {a.shape})",
        f"Total elements: {total_elements}, Mismatched: {mismatched_count}",
        f"Max difference: {max_diff:.6e}, Mean difference: {mean_diff:.6e}",
        f"Tolerance: rtol={rtol}, atol={atol}"
    ]

    # Show first few mismatched values
    if mismatched_count > 0:
        mismatched_indices = np.where(mismatched_mask)
        msg_parts.append(f"First {min(max_diffs_to_show, mismatched_count)} mismatched values:")
        for i in range(min(max_diffs_to_show, mismatched_count)):
            idx = tuple(mismatched_indices[j][i] for j in range(len(mismatched_indices)))
            msg_parts.append(f"  Index {idx}: {a[idx]:.6e} vs {b[idx]:.6e} (diff: {diff[idx]:.6e})")

    if mismatched_count > max_diffs_to_show:
        msg_parts.append(f"  ... and {mismatched_count - max_diffs_to_show} more")

    self.fail("\n".join(msg_parts))
