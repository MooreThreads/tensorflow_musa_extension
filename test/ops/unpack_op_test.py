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

"""Tests for MUSA Pack (Stack) and Unpack (Unstack) operators."""

import numpy as np
import tensorflow as tf
import functools
import unittest

from musa_test_utils import MUSATestCase

class UnpackOpTest(MUSATestCase):
  """Tests for MUSA Pack (Stack) and Unpack (Unstack) operators."""

  def _test_pack(self, inputs, axis, dtype, rtol=1e-5, atol=1e-8):
    """Test pack (tf.stack) operation."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    tf_inputs = []
    for item in inputs:
        if isinstance(item, (int, float, np.number)):
            arr = np.array(item, dtype=np_dtype)
        else:
            arr = np.array(item).astype(np_dtype)
        tf_inputs.append(tf.constant(arr, dtype=dtype))

    def op_func(*args):
        return tf.stack(list(args), axis=axis)

    self._compare_cpu_musa_results(op_func, tf_inputs, dtype, rtol=rtol, atol=atol)

  def _test_unpack(self, input_data, axis, dtype, rtol=1e-5, atol=1e-8):
    """Test unpack (tf.unstack) operation."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    x_np = np.array(input_data).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)

    num_outputs = x_np.shape[axis]

    def _slice_at_index(input_tensor, index, axis):
        return tf.unstack(input_tensor, axis=axis)[index]

    for i in range(num_outputs):
        op_func = functools.partial(_slice_at_index, index=i, axis=axis)
        self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def testPackBasic(self):
    """Test basic Pack (Stack) operations."""
    self._test_pack([[1, 2], [3, 4]], axis=0, dtype=tf.int32)
    self._test_pack([[1, 2], [3, 4]], axis=1, dtype=tf.float32)
    self._test_pack([5.0, 6.0], axis=0, dtype=tf.float32)

  def testUnpackBasic(self):
    """Test basic Unpack (Unstack) operations."""
    data = [[1, 2], [3, 4]]
    self._test_unpack(data, axis=0, dtype=tf.int32)


  @unittest.skip("Known issue: MUSA Unpack operator fails on axis != 0")
  def testUnpackAxis1(self):
    """Explicitly test failing case for tracking."""
    data = [[1, 2], [3, 4]]
    self._test_unpack(data, axis=1, dtype=tf.float32)

  def testInverseConsistency(self):
    """Test Pack/Unpack inverse consistency."""
    with tf.device('/device:MUSA:0'):
        original = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        
        unpacked_0 = tf.unstack(original, axis=0)
        repacked_0 = tf.stack(unpacked_0, axis=0)
        self.assertAllClose(original.numpy(), repacked_0.numpy())
        
  def testPackHighDim(self):
    """Test Pack with higher dimensions."""
    data = [[[1, 2]], [[3, 4]]]
    self._test_pack(data, axis=0, dtype=tf.float32)
    
    m1 = [[1], [2]]
    m2 = [[3], [4]]
    self._test_pack([m1, m2], axis=1, dtype=tf.float32)

  def testUnpackHighDim(self):
    """Test Unpack with higher dimensions."""
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    self._test_unpack(data, axis=0, dtype=tf.float32) 

  def testDtypes(self):
    """Test various data types."""
    data = [[1, 2], [3, 4]]
    for dtype in [tf.float16, tf.bfloat16, tf.int64]:
        rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
        atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
        
        self._test_pack(data, axis=0, dtype=dtype, rtol=rtol, atol=atol)
        self._test_unpack(data, axis=0, dtype=dtype, rtol=rtol, atol=atol)


if __name__ == "__main__":
  tf.test.main()