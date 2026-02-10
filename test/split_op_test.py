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

"""Tests for MUSA Split operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SplitOpTest(MUSATestCase):
  """Tests for MUSA Split operator."""

  def _test_split(self, shape, dtype, num_or_size_splits, axis, rtol=1e-5, atol=1e-8):
    """Test split operation with given parameters."""
    
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    x_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)

    # Determine number of outputs to iterate over
    if isinstance(num_or_size_splits, int):
        num_outputs = num_or_size_splits
    else:
        num_outputs = len(num_or_size_splits)

    # Compare Results individually for each split output
    for i in range(num_outputs):
        
        def op_func_for_index(input_tensor, index=i): 
            return tf.split(input_tensor, num_or_size_splits=num_or_size_splits, axis=axis)[index]

        # print(f"  Verifying Split output index {i}/{num_outputs}...")
        self._compare_cpu_musa_results(op_func_for_index, [x], dtype, rtol=rtol, atol=atol)

  def testSplitBasic(self):
    """Test basic equal split operation with common data types."""
    # Shape: (4, 6), Axis: 1, Num Splits: 3 -> Expecting 3 tensors of shape (4, 2)
    shape = [4, 6]
    num_splits = 3
    axis = 1

    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_split(shape, dtype, num_splits, axis, rtol=rtol, atol=atol)

  def testSplitVariableSizes(self):
    """Test split with explicit sizes (uneven split)."""
    # Shape: (4, 10), Axis: 1, Sizes: [2, 3, 5]
    shape = [4, 10]
    sizes = [2, 3, 5]
    axis = 1
    self._test_split(shape, tf.float32, sizes, axis)

  def testSplitNegativeAxis(self):
    """Test split with negative axis index."""
    # Shape: (10, 4), Axis: -1 (last dim), Num Splits: 2
    shape = [10, 4]
    num_splits = 2
    axis = -1
    self._test_split(shape, tf.float32, num_splits, axis)

  def testSplitFirstDimension(self):
    """Test split along the first dimension (batch dimension)."""
    # Shape: (8, 4), Axis: 0, Num Splits: 4
    shape = [8, 4]
    num_splits = 4
    axis = 0
    self._test_split(shape, tf.float32, num_splits, axis)

  def testSplitLargeFP16(self):
    """Test larger shape with FP16 to verify AMP/Performance stability."""
    # Shape: (128, 64), Axis: 1, Num Splits: 8
    shape = [128, 64]
    num_splits = 8
    axis = 1
    self._test_split(shape, tf.float16, num_splits, axis, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  tf.test.main()