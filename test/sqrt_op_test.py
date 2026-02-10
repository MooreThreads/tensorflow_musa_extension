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

"""Tests for MUSA Sqrt operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SqrtOpTest(MUSATestCase):
  """Tests for MUSA Sqrt operator."""

  def _test_sqrt(self, shape, dtype, rtol=1e-5, atol=1e-8):
    """Test sqrt operation with given shapes and dtype."""
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    x_np = np.random.uniform(1.0, 100.0, size=shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)

    # Define Operator Wrapper
    def op_func(input_tensor):
        return tf.math.sqrt(input_tensor)

    # Compare Results
    self._compare_cpu_musa_results(op_func, [x], dtype, rtol=rtol, atol=atol)

  def testSqrtBasic(self):
    """Test basic sqrt operation with standard shapes."""
    shape = [256, 256]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      self._test_sqrt(shape, dtype, rtol=rtol, atol=atol)

  def testSqrtLarge(self):
    """Test sqrt with larger shape (from original test_sqrt.py)."""
    shape = [256, 4096]
    
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      self._test_sqrt(shape, dtype, rtol=rtol, atol=atol)

  def testSqrtVector(self):
    """Test sqrt with 1D vector."""
    shape = [1024]
    self._test_sqrt(shape, tf.float32)

  def testSqrtSmallValues(self):
    """Test sqrt with very small positive values."""
    # Testing numerical stability near zero
    shape = [100, 100]
    dtype = tf.float32
    np_dtype = np.float32
    
    x_np = np.random.uniform(1e-5, 1.0, size=shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.math.sqrt, [x], dtype, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
  tf.test.main()