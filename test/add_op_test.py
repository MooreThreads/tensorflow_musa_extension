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

"""Tests for MUSA Add operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class AddOpTest(MUSATestCase):
  """Tests for MUSA Add operator."""

  def _test_add(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    """Test add operation with given shapes and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    x_np = np.random.uniform(-1, 1, size=shape_x).astype(np_dtype)
    y_np = np.random.uniform(-1, 1, size=shape_y).astype(np_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.add, [x, y], dtype, rtol=rtol, atol=atol)

  def testAddBasic(self):
    """Test basic add operation with same shapes."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_add([1024, 1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testAddBroadcastVectorMatrix(self):
    """Test add with vector-matrix broadcasting."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_add([1024], [1024, 1024], dtype, rtol=rtol, atol=atol)

  def testAddBroadcastColumnRow(self):
    """Test add with column-row broadcasting."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_add([1024, 1], [1, 1024], dtype, rtol=rtol, atol=atol)

  def testAddScalar(self):
    """Test add with scalar values."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_add([], [], dtype, rtol=rtol, atol=atol)

  def testAddDifferentShapes(self):
    """Test add with various different shapes."""
    test_cases = [
        ([1], [1]),
        ([5], [5]),
        ([3, 4], [3, 4]),
        ([2, 3, 4], [2, 3, 4]),
        ([1, 1, 10], [5, 3, 10]),
    ]
    for dtype in [tf.float32]:
      for shape_x, shape_y in test_cases:
        self._test_add(shape_x, shape_y, dtype)


if __name__ == "__main__":
  tf.test.main()