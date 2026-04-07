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

import argparse
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


def load_musa_plugin():
  """Load the MUSA plugin library with simplified path resolution.
  With __init__.py handling path setup, we can use a simpler approach
  focusing on the most common locations.
  """
  plugin_path = None
  current_dir = os.path.dirname(os.path.abspath(__file__))

  # Common plugin locations to try
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

  if plugin_path and os.path.exists(plugin_path):
    try:
      tf.load_library(plugin_path)
      # print(f"SUCCESS: MUSA plugin loaded from {plugin_path}!")
    except Exception as e:
      print(f"FAILED: Error loading plugin from {plugin_path}: {e}")
      raise
  else:
    # Provide helpful error message
    searched_locations = [os.path.normpath(path) for path in candidate_paths]
    raise FileNotFoundError(
        f"MUSA plugin not found. Searched locations:\n" +
        "\n".join(f"  - {loc}" for loc in searched_locations)
    )


# Import tensorflow first (load_musa_plugin needs it)
import tensorflow as tf

# Load plugin immediately after importing tensorflow
load_musa_plugin()


_ORIGINAL_TF_TEST_MAIN = tf.test.main


def _parse_progress_runner_args(argv):
  """Parse lightweight flags for direct test execution."""
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--detail', '-d', action='store_true')
  parser.add_argument('--quiet', '-q', action='store_true')
  parser.add_argument('--log-file', default='test_results.log')
  return parser.parse_known_args(argv[1:])



def _use_progress_runner_for_direct_execution():
  return os.environ.get('MUSA_TEST_DISABLE_PROGRESS_RUNNER') != '1'



def _musa_tf_test_main(argv=None):
  """Run direct test files with the custom progress runner when possible."""
  args = list(sys.argv if argv is None else argv)

  if not _use_progress_runner_for_direct_execution():
    return _ORIGINAL_TF_TEST_MAIN(args)

  parsed_args, remaining = _parse_progress_runner_args(args)
  if remaining:
    return _ORIGINAL_TF_TEST_MAIN(args)

  main_module = sys.modules.get('__main__')
  if main_module is None:
    return _ORIGINAL_TF_TEST_MAIN(args)

  from test_runner import run_module_tests

  quiet_mode = not parsed_args.detail
  if parsed_args.quiet:
    quiet_mode = True

  return run_module_tests(main_module,
                          quiet=quiet_mode,
                          detail_mode=parsed_args.detail,
                          log_file=parsed_args.log_file if parsed_args.detail else None)


# Make direct `python test/ops/foo_test.py` executions use the same
# progress bar runner as `test_runner.py`, while preserving an opt-out.
tf.test.main = _musa_tf_test_main


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
