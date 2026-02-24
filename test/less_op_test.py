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

"""Tests for MUSA Less operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class LessOpTest(MUSATestCase):
  """Tests for MUSA Less operator."""

  def _test_less(self, shape_x, shape_y, dtype):
    """Test less operation with given shapes and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    x_np = np.random.uniform(-1, 1, size=shape_x).astype(np_dtype)
    y_np = np.random.uniform(-1, 1, size=shape_y).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)

    self._compare_cpu_musa_results(tf.less, [x, y], dtype, rtol=0, atol=0)

  def testLessBasic(self):
    """Test basic less operation with same shapes."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_less([1024, 1024], [1024, 1024], dtype)

  def testLessBroadcastVectorMatrix(self):
    """Test less with vector-matrix broadcasting."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_less([1024], [1024, 1024], dtype)

  def testLessBroadcastColumnRow(self):
    """Test less with column-row broadcasting."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_less([1024, 1], [1, 1024], dtype)

  def testLessScalar(self):
    """Test less with scalar values."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_less([], [], dtype)

  def testLessDifferentShapes(self):
    """Test less with various different shapes."""
    test_cases = [
        ([1], [1]),
        ([5], [5]),
        ([3, 4], [3, 4]),
        ([2, 3, 4], [2, 3, 4]),
        ([1, 1, 10], [5, 3, 10]),
    ]
    for dtype in [tf.float32]:
      for shape_x, shape_y in test_cases:
        self._test_less(shape_x, shape_y, dtype)

  def testLessIntTypes(self):
    """Test less with integer types."""
    for dtype in [tf.int32, tf.int64, tf.uint8]:
      np_dtype = dtype.as_numpy_dtype
      x_np = np.random.randint(0, 100, size=[256, 256]).astype(np_dtype)
      y_np = np.random.randint(0, 100, size=[256, 256]).astype(np_dtype)

      x = tf.constant(x_np, dtype=dtype)
      y = tf.constant(y_np, dtype=dtype)

      self._compare_cpu_musa_results(tf.less, [x, y], dtype, rtol=0, atol=0)

  def testLessDouble(self):
    """Test less with double precision."""
    self._test_less([512, 512], [512, 512], tf.float64)

  def testLessEqualValues(self):
    """Test less with equal values to ensure strict less-than behavior."""
    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    x = tf.constant(x_np, dtype=tf.float32)
    y = tf.constant(y_np, dtype=tf.float32)

    self._compare_cpu_musa_results(
        tf.less, [x, y], tf.float32, rtol=0, atol=0)


if __name__ == "__main__":
  tf.test.main()