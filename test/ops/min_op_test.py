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

"""Tests for MUSA ReduceMin operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import load_musa_plugin

# Load plugin before test discovery/runtime checks.
load_musa_plugin()
MUSA_DEVICES = tf.config.list_physical_devices('MUSA')


class MinOpTest(tf.test.TestCase):
  """Tests for MUSA ReduceMin operator."""

  def _compare_cpu_musa_results(self, op_func, input_tensors, dtype,
                                rtol=1e-5, atol=1e-8):
    if not MUSA_DEVICES:
      self.skipTest("No MUSA devices found.")

    with tf.device('/CPU:0'):
      cpu_result = op_func(*input_tensors)

    with tf.device('/device:MUSA:0'):
      musa_result = op_func(*input_tensors)

    if dtype in [tf.float16, tf.bfloat16]:
      self.assertAllClose(tf.cast(cpu_result, tf.float32).numpy(),
                          tf.cast(musa_result, tf.float32).numpy(),
                          rtol=rtol, atol=atol)
    else:
      self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), rtol=rtol, atol=atol)

  def _test_min(self, shape, dtype, axis=None, keepdims=False, rtol=1e-5, atol=1e-8):
    """Test reduce_min operation with given parameters."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    if np.issubdtype(np_dtype, np.integer):
      x_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)
    else:
      x_np = np.random.uniform(-10, 10, size=shape).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)

    def op_func(input_tensor):
      return tf.reduce_min(input_tensor, axis=axis, keepdims=keepdims)

    self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def testMinBasic(self):
    """Test basic min operation (Global Min)."""
    shape = [10, 10]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_min(shape, dtype, axis=None, rtol=rtol, atol=atol)

  def testMinIntegerTypes(self):
    """Test min operation with integer types."""
    shape = [5, 5]
    for dtype in [tf.int32, tf.int64]:
      self._test_min(shape, dtype, axis=0)
      self._test_min(shape, dtype, axis=1)
      self._test_min(shape, dtype, axis=None)

  def testMinDouble(self):
    """Test min operation with float64."""
    shape = [5, 5]
    self._test_min(shape, tf.float64)

  def testMinAxes(self):
    """Test min along specific axes (Rows, Cols, Negative, List)."""
    shape = [2, 3, 4]
    dtype = tf.float32

    self._test_min(shape, dtype, axis=0)
    self._test_min(shape, dtype, axis=1)
    self._test_min(shape, dtype, axis=2)
    self._test_min(shape, dtype, axis=-1)
    self._test_min(shape, dtype, axis=[0, 1])
    self._test_min(shape, dtype, axis=[0, 1, 2])

  def testMinKeepDims(self):
    """Test min with keepdims=True."""
    shape = [4, 4]
    dtype = tf.float32

    self._test_min(shape, dtype, axis=0, keepdims=True)
    self._test_min(shape, dtype, axis=1, keepdims=True)
    self._test_min(shape, dtype, axis=[0, 1], keepdims=True)

  def testMin1D(self):
    """Test min on 1D tensor."""
    shape = [100]
    dtype = tf.float32
    self._test_min(shape, dtype, axis=0)
    self._test_min(shape, dtype, axis=None)

  def testMinSpecificValues(self):
    """Test min with specific known values."""
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = tf.constant(x_np, dtype=tf.float32)

    def op_func(input_tensor):
      return tf.reduce_min(input_tensor, axis=0)

    self._compare_cpu_musa_results(op_func, [x], tf.float32)

  def testMinNegativeValues(self):
    """Test min with negative values."""
    x_np = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    x = tf.constant(x_np, dtype=tf.float32)

    def op_func(input_tensor):
      return tf.reduce_min(input_tensor, axis=1)

    self._compare_cpu_musa_results(op_func, [x], tf.float32)

  def testMinMixedValues(self):
    """Test min with mixed positive and negative values."""
    x_np = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    x = tf.constant(x_np, dtype=tf.float32)

    def op_func(input_tensor):
      return tf.reduce_min(input_tensor, axis=None)

    self._compare_cpu_musa_results(op_func, [x], tf.float32)


if __name__ == "__main__":
  tf.test.main()
