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

"""Tests for MUSA LessEqual operator."""

import numpy as np
import tensorflow as tf
import unittest

from musa_test_utils import MUSATestCase


class LessEqualOpTest(MUSATestCase):
  """Tests for MUSA LessEqual operator."""

  def _test_less_equal(self, shape_x, shape_y, dtype):
    """Test less_equal operation with given shapes and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if np.issubdtype(np_dtype, np.integer):
        x_np = np.random.randint(-100, 100, size=shape_x).astype(np_dtype)
        y_np = np.random.randint(-100, 100, size=shape_y).astype(np_dtype)
    else:
        x_np = np.array(np.random.randn(*shape_x)).astype(np_dtype)
        y_np = np.array(np.random.randn(*shape_y)).astype(np_dtype)
        
    if x_np.size > 0 and y_np.size > 0 and x_np.shape == y_np.shape:
        mask = np.random.rand(*shape_x) < 0.2
        y_np[mask] = x_np[mask]
        
        if x_np.size > 1:
            flat_x = x_np.ravel()
            flat_y = y_np.ravel()
            flat_x[0] = -100
            flat_y[0] = 100
            flat_x[1] = 100
            flat_y[1] = -100
            x_np = flat_x.reshape(shape_x)
            y_np = flat_y.reshape(shape_y)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    def op_func(input_x, input_y):
        return tf.math.less_equal(input_x, input_y)

    self._compare_cpu_musa_results(op_func, [x, y], dtype)

  def testBasic1D(self):
    """Test basic comparison with 1D shapes (Should Pass if Kernel flattens)."""
    shape = [100]
    for dtype in [tf.float32, tf.float16, tf.int32]:
        self._test_less_equal(shape, shape, dtype)

  def testScalar(self):
    """Test scalar inputs."""
    self._test_less_equal([], [], tf.float32)
    self._test_less_equal([], [], tf.int32)


if __name__ == "__main__":
  tf.test.main()