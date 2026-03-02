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

"""Tests for MUSA Fill operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class FillOpTest(MUSATestCase):
  """Tests for MUSA Fill operator."""

  def _test_fill(self, shape, value, dtype, rtol=1e-5, atol=1e-8):
    """Test fill operation with given shape and value."""
    dims = tf.constant(shape, dtype=tf.int32)
    
    if dtype == tf.bfloat16:
        val_tensor = tf.cast(tf.constant(value, dtype=tf.float32), dtype=tf.bfloat16)
    else:
        val_tensor = tf.constant(value, dtype=dtype)

    self._compare_cpu_musa_results(tf.fill, [dims, val_tensor], dtype, rtol=rtol, atol=atol)

  def testFillBasic(self):
    """Test basic fill with floating point types."""
    shape = [2, 3]
    value = 3.14
    
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      self._test_fill(shape, value, dtype, rtol=rtol, atol=atol)

  def testFillShapes(self):
    """Test fill with various shapes (1D, 2D, 3D, Scalar)."""
    # 1D
    self._test_fill([5], 1.0, tf.float32)
    
    # 2D
    self._test_fill([3, 4], 2.5, tf.float32)
    
    # 3D
    self._test_fill([2, 3, 4], -1.0, tf.float32)

    self._test_fill([], 42.0, tf.float32)

  def testFillInt(self):
    """Test fill with integer types."""
    shape = [2, 2]
    # Int32
    self._test_fill(shape, 42, tf.int32)
    # Int64
    self._test_fill(shape, 100, tf.int64)

  def testFillDouble(self):
    """Test fill with float64."""
    shape = [2, 2]
    self._test_fill(shape, 2.71828, tf.float64)


if __name__ == "__main__":
  tf.test.main()