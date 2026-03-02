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

"""Tests for MUSA Transpose operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class TransposeOpTest(MUSATestCase):
  """Tests for MUSA Transpose operator."""

  def _test_transpose(self, shape, perm, dtype, rtol=1e-5, atol=1e-8):
    """Test transpose operation with given shape and permutation."""
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if np.issubdtype(np_dtype, np.integer):
        x_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)
    else:
        x_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
        
    x = tf.constant(x_np, dtype=dtype)

    # Define Operator Wrapper
    def op_func(input_tensor):
        return tf.transpose(input_tensor, perm=perm)

    # Compare Results
    self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def testTransposeNHWCtoNCHW(self):
    """Test NHWC to NCHW transpose (common in CV)."""
    # Matches original test case: (1, 4, 4, 3) -> [0, 3, 1, 2]
    shape = [1, 4, 4, 3]
    perm = [0, 3, 1, 2]
    
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      self._test_transpose(shape, perm, dtype, rtol=rtol, atol=atol)

  def testTransposeBasic(self):
    """Test basic 2D matrix transpose."""
    shape = [5, 10]
    perm = [1, 0]
    
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      self._test_transpose(shape, perm, dtype, rtol=rtol, atol=atol)

  def testTransposeInt(self):
    """Test transpose with integer types."""
    shape = [4, 4]
    perm = [1, 0]
    for dtype in [tf.int32, tf.int64]:
      self._test_transpose(shape, perm, dtype)

  def testTransposeComplex(self):
    """Test higher dimensional transpose permutations."""
    # 3D: (Batch, Seq, Feature) -> (Seq, Batch, Feature)
    shape = [2, 3, 4]
    perm = [1, 0, 2]
    self._test_transpose(shape, perm, tf.float32)
    
    # 3D: Reverse dimensions
    perm_rev = [2, 1, 0]
    self._test_transpose(shape, perm_rev, tf.float32)


if __name__ == "__main__":
  tf.test.main()