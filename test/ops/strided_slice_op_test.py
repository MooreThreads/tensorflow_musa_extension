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

"""Tests for MUSA StridedSlice and Cast operators."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class CastOpTest(MUSATestCase):
  """Tests for MUSA Cast operator."""

  def _test_cast(self, src_dtype, dst_dtype, data, rtol=1e-5, atol=1e-8):
    """Test cast operation."""
    # Prepare Data
    if data is None:
        shape = [2, 5]
        if src_dtype == tf.bool:
            x_np = np.random.choice([True, False], size=shape)
        elif src_dtype.is_integer:
            x_np = np.random.randint(-100, 100, size=shape).astype(src_dtype.as_numpy_dtype)
        else:
            np_dtype = np.float32 if src_dtype == tf.bfloat16 else src_dtype.as_numpy_dtype
            x_np = np.random.randn(*shape).astype(np_dtype) * 10.0
    else:
        x_np = np.array(data)
        
    x = tf.constant(x_np, dtype=src_dtype)

    # Define Operator Wrapper
    def op_func(input_tensor):
        return tf.cast(input_tensor, dtype=dst_dtype)

    # Compare Results
    self._compare_cpu_musa_results(op_func, [x], dst_dtype, rtol=rtol, atol=atol)

  def testCastBasic(self):
    """Test basic numeric casts."""
    test_cases = [
        (tf.float32, tf.int32, [1.5, 2.9, -3.1, 0.0]),
        (tf.int32, tf.float32, [10, 20, -5, 0]),
        (tf.float32, tf.int64, [1e9, 2e9, -1e9]),
    ]
    for src, dst, data in test_cases:
        self._test_cast(src, dst, data)

  def testCastBool(self):
    """Test boolean casting (important interceptor logic)."""
    data = [True, False, True, True]
    self._test_cast(tf.bool, tf.float32, data)
    self._test_cast(tf.bool, tf.int32, data)

  def testCastBfloat16(self):
    """Test mixed precision casting."""
    data = [1.0, 65504.0, 1e10]
    self._test_cast(tf.float32, tf.bfloat16, data, rtol=1e-2, atol=1e-2)


class StridedSliceOpTest(MUSATestCase):
  """Tests for MUSA StridedSlice operator."""

  def _test_slice(self, slice_spec, input_data=None, shape=None, dtype=tf.float32):
    """Test strided_slice operation."""
    # Prepare Data
    if input_data is not None:
        x_np = input_data
    else:
        if shape is None: shape = [3, 5]
        x_np = np.arange(np.prod(shape)).reshape(shape).astype(dtype.as_numpy_dtype)
        
    x = tf.constant(x_np, dtype=dtype)

    # Define Operator Wrapper
    def op_func(input_tensor):
        return input_tensor[slice_spec]

    # Compare Results
    self._compare_cpu_musa_results(op_func, [x], dtype)

  def testSliceFull(self):
    """Test basic full slice [:]"""
    self._test_slice(slice(None))

  def testSliceColumns(self):
    """Test extracting specific columns (e.g., [:, 0:2])."""
    # Equivalent to x[:, 0:2]
    self._test_slice((slice(None), slice(0, 2)))

  def testSliceStrided(self):
    """Test strided slicing (e.g., [::2, ::2])."""
    # Equivalent to x[::2, ::2]
    self._test_slice((slice(None, None, 2), slice(None, None, 2)))

  def testSliceShrinkAxis(self):
    """Test shrink axis (reducing dimension, e.g., x[0, :])."""
    # Equivalent to x[0, :]
    self._test_slice((0, slice(None)))

  def testSliceEmpty(self):
    """Test empty tensor slicing."""
    # Equivalent to x[0:0, 0:0]
    self._test_slice((slice(0, 0), slice(0, 0)))

  def testSliceHighDim(self):
    """Test slicing on higher dimensional tensors."""
    shape = [2, 3, 4, 5]
    # x[0, :, 1:3, ::2]
    spec = (0, slice(None), slice(1, 3), slice(None, None, 2))
    self._test_slice(spec, shape=shape)


if __name__ == "__main__":
  tf.test.main()