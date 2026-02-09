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

import os
import tensorflow as tf


def load_musa_plugin():
  """Load the MUSA plugin library."""
  plugin_path = "/workspace/tensorflow_musa_extension/build/libmusa_plugin.so"
  if os.path.exists(plugin_path):
    try:
      tf.load_library(plugin_path)
      print("SUCCESS: MUSA plugin loaded!")
    except Exception as e:
      print(f"FAILED: Error loading plugin: {e}")
      raise
  else:
    raise FileNotFoundError(f"Plugin not found at {plugin_path}")


class MUSATestCase(tf.test.TestCase):
  """Base test class for MUSA kernel tests."""

  @classmethod
  def setUpClass(cls):
    """Set up the test class by loading the MUSA plugin."""
    super(MUSATestCase, cls).setUpClass()
    load_musa_plugin()
    
    # Verify MUSA device is available
    if not tf.config.list_physical_devices('MUSA'):
      raise RuntimeError("No MUSA devices found.")

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
      a, b: Arrays to compare
      rtol, atol: Relative and absolute tolerance
      max_diffs_to_show: Maximum number of differing elements to show
    """
    import numpy as np
    
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