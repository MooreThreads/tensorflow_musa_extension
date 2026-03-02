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

"""Tests for MUSA FloorMod operator."""

import numpy as np
import tensorflow as tf
import unittest

from musa_test_utils import MUSATestCase


class FloorModOpTest(MUSATestCase):
  """Tests for MUSA FloorMod operator."""

  def _test_floor_mod(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    """Test floormod operation with given shapes and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if np.issubdtype(np_dtype, np.integer):
        x_np = np.random.randint(-100, 100, size=shape_x).astype(np_dtype)
        y_np = np.random.randint(1, 100, size=shape_y).astype(np_dtype)
        mask = np.random.choice([-1, 1], size=shape_y).astype(np_dtype)
        y_np = y_np * mask
    else:
        x_np = np.random.uniform(-50, 50, size=shape_x).astype(np_dtype)
        y_np = np.random.uniform(0.1, 10, size=shape_y).astype(np_dtype)
        mask = np.random.choice([-1, 1], size=shape_y).astype(np_dtype)
        y_np = y_np * mask
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.math.floormod, [x, y], dtype, rtol=rtol, atol=atol)

  def testFloat32(self):
    """Test floor_mod with Float32 (Standard Case)."""
    shape = [100, 100]
    self._test_floor_mod(shape, shape, tf.float32, rtol=1e-4, atol=1e-4)

  def testBroadcast(self):
    """Test floormod with broadcasting (Float32)."""
    self._test_floor_mod([100], [], tf.float32, rtol=1e-4, atol=1e-4)
    self._test_floor_mod([10, 10], [10], tf.float32, rtol=1e-4, atol=1e-4)

  def testSigns(self):
    """Test floormod with specific sign combinations (Edge Cases)."""
    x = tf.constant([10.0, -10.0, 10.0, -10.0], dtype=tf.float32)
    y = tf.constant([3.0, 3.0, -3.0, -3.0], dtype=tf.float32)
    
    self._compare_cpu_musa_results(tf.math.floormod, [x, y], tf.float32, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  tf.test.main()