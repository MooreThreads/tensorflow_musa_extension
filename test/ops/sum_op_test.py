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

"""Tests for MUSA ReduceSum operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SumOpTest(MUSATestCase):
  """Tests for MUSA ReduceSum operator."""

  def _test_sum(self, shape, dtype, axis=None, keepdims=False, rtol=1e-5, atol=1e-8):
    """Test reduce_sum operation with given parameters."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if np.issubdtype(np_dtype, np.integer):
        x_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)
    else:
        x_np = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
        
    x = tf.constant(x_np, dtype=dtype)

    def op_func(input_tensor):
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)

    self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def testSumBasic(self):
    """Test basic sum operation (Global Sum)."""
    shape = [10, 10]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      self._test_sum(shape, dtype, axis=None, rtol=rtol, atol=atol)

  def testSumIntegerTypes(self):
    """Test sum operation with integer types."""
    shape = [5, 5]
    for dtype in [tf.int32, tf.int64]:
      self._test_sum(shape, dtype, axis=0)
      self._test_sum(shape, dtype, axis=1)

  def testSumDouble(self):
    """Test sum operation with float64."""
    shape = [5, 5]
    self._test_sum(shape, tf.float64)

  def testSumAxes(self):
    """Test sum along specific axes (Rows, Cols, Negative, List)."""
    # 3D Tensor
    shape = [2, 3, 4]
    dtype = tf.float32
    
    self._test_sum(shape, dtype, axis=0)
    self._test_sum(shape, dtype, axis=1)
    self._test_sum(shape, dtype, axis=2)
    self._test_sum(shape, dtype, axis=-1)
    self._test_sum(shape, dtype, axis=[0, 1])
    self._test_sum(shape, dtype, axis=[0, 1, 2])

  def testSumKeepDims(self):
    """Test sum with keepdims=True."""
    shape = [4, 4]
    dtype = tf.float32
    
    self._test_sum(shape, dtype, axis=0, keepdims=True)
    self._test_sum(shape, dtype, axis=1, keepdims=True)
    self._test_sum(shape, dtype, axis=[0, 1], keepdims=True)

  def testSum1D(self):
    """Test sum on 1D tensor."""
    shape = [100]
    dtype = tf.float32
    self._test_sum(shape, dtype, axis=0)
    self._test_sum(shape, dtype, axis=None)


if __name__ == "__main__":
  tf.test.main()