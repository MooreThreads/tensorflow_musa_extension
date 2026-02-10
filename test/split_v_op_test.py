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

"""Tests for MUSA SplitV operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SplitVOpTest(MUSATestCase):
  """Tests for MUSA SplitV operator (split with variable sizes)."""

  def _test_splitv(self, shape, dtype, size_splits, axis, rtol=1e-5, atol=1e-8):
    """Test splitv operation with given parameters."""

    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    # Generate data based on dtype
    if dtype in [tf.int32, tf.int64]:
        x_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)
    else:
        x_np = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
        
    x = tf.constant(x_np, dtype=dtype)

    # Compare Results individually for each split output
    num_outputs = len(size_splits)
    
    for i in range(num_outputs):
        def op_func(input_tensor, index=i):
            return tf.split(input_tensor, num_or_size_splits=size_splits, axis=axis)[index]

        current_rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else rtol
        current_atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else atol

        self._compare_cpu_musa_results(op_func, [x], dtype, rtol=current_rtol, atol=current_atol)

  def testSplitV1DInt32(self):
    """Test 1D Int32 split (from original test_splitv.py)."""
    # Data: [1, 2, 3, 4, 5, 6], Splits: [3, 2, 1]
    shape = [6]
    size_splits = [3, 2, 1]
    axis = 0
    self._test_splitv(shape, tf.int32, size_splits, axis)

  def testSplitV2DFloat32Axis1(self):
    """Test 2D Float32 split along axis 1 (from original test_splitv.py)."""
    # Shape: (2, 5), Splits: [2, 1, 2], Axis: 1
    shape = [2, 5]
    size_splits = [2, 1, 2]
    axis = 1
    self._test_splitv(shape, tf.float32, size_splits, axis)

  def testSplitV2DFloat32Axis0(self):
    """Test 2D Float32 split along axis 0 (from original test_splitv.py)."""
    # Shape: (4, 2), Splits: [1, 3], Axis: 0
    shape = [4, 2]
    size_splits = [1, 3]
    axis = 0
    self._test_splitv(shape, tf.float32, size_splits, axis)

  def testSplitVEmpty(self):
    """Test SplitV with empty tensors (size 0) (from original test_splitv.py)."""
    # Shape: (3,), Splits: [0, 2, 1, 0] -> Expects outputs with shape (0,), (2,), (1,), (0,)
    shape = [3]
    size_splits = [0, 2, 1, 0]
    axis = 0
    self._test_splitv(shape, tf.float32, size_splits, axis)

  def testSplitVNegativeAxis(self):
    """Test SplitV with negative axis (from original test_splitv.py)."""
    # Shape: (2, 3, 4), Splits: [1, 1, 2], Axis: -1 (last dim, which is 4)
    shape = [2, 3, 4]
    size_splits = [1, 1, 2]
    axis = -1
    self._test_splitv(shape, tf.float32, size_splits, axis)

  def testSplitVFloat16(self):
    """Test SplitV with Float16 (Half precision)."""
    # Shape: (4, 4), Splits: [2, 2], Axis: 0
    shape = [4, 4]
    size_splits = [2, 2]
    axis = 0
    self._test_splitv(shape, tf.float16, size_splits, axis)

  def testSplitVBfloat16(self):
    """Test SplitV with Bfloat16 (Extra coverage)."""
    shape = [4, 4]
    size_splits = [1, 3]
    axis = 1
    self._test_splitv(shape, tf.bfloat16, size_splits, axis)


if __name__ == "__main__":
  tf.test.main()