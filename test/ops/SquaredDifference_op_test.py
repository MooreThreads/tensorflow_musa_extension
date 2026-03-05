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

"""Tests for MUSA SquaredDifference operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SquaredDifferenceOpTest(MUSATestCase):
  """Tests for MUSA SquaredDifference operator."""

  def _test_squared_difference(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8):
    """Test squared_difference operation with given shapes and dtype."""
    # Prepare Data
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    x_np = np.random.randn(*shape_x).astype(np_dtype)
    y_np = np.random.randn(*shape_y).astype(np_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    # Compare Results
    self._compare_cpu_musa_results(tf.math.squared_difference, [x, y], dtype, rtol=rtol, atol=atol)

  def testSquaredDifferenceBasic(self):
    """Test basic squared difference with same shapes."""
    shape = [2, 3]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      self._test_squared_difference(shape, shape, dtype, rtol=rtol, atol=atol)

  def testSquaredDifferenceBroadcastScalar(self):
    """Test squared difference with scalar/vector broadcasting."""
    shape_x = [5]
    shape_y = [1]
    
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      self._test_squared_difference(shape_x, shape_y, dtype, rtol=rtol, atol=atol)

  def testSquaredDifferenceBroadcastCNN(self):
    """Test squared difference with typical CNN broadcasting shapes."""
    shape_x = [1, 3, 224, 224]
    shape_y = [1, 3, 1, 1]
    
    for dtype in [tf.float32, tf.float16]:
      rtol = 1e-2 if dtype == tf.float16 else 1e-5
      atol = 1e-2 if dtype == tf.float16 else 1e-8
      
      self._test_squared_difference(shape_x, shape_y, dtype, rtol=rtol, atol=atol)

  def testSquaredDifferenceDouble(self):
    """Test squared difference with float64 (verification fallback)."""
    shape = [2, 3]
    self._test_squared_difference(shape, shape, tf.float64)


if __name__ == "__main__":
  tf.test.main()