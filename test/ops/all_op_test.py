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

"""Tests for MUSA All operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class AllOpTest(MUSATestCase):
  """Tests for MUSA All operator."""

  def _test_all(self, shape, axis=None, keepdims=False):
    """Test reduce_all operation with given parameters."""
    # All operator typically works on boolean tensors
    dtype = tf.bool
    x_np = np.random.choice([True, False], size=shape)
    x = tf.constant(x_np, dtype=dtype)

    def op_func(input_tensor):
        return tf.reduce_all(input_tensor, axis=axis, keepdims=keepdims)

    self._compare_cpu_musa_results(op_func, [x], dtype)

  def testAllBasic(self):
    """Test basic all operation (Global All)."""
    shapes = [[10, 10], [2, 3, 4], [100]]
    for shape in shapes:
      self._test_all(shape, axis=None)

  def testAllAxes(self):
    """Test all along specific axes."""
    shape = [2, 3, 4]
    self._test_all(shape, axis=0)
    self._test_all(shape, axis=1)
    self._test_all(shape, axis=2)
    self._test_all(shape, axis=-1)
    self._test_all(shape, axis=[0, 1])
    self._test_all(shape, axis=[0, 1, 2])

  def testAllKeepDims(self):
    """Test all with keepdims=True."""
    shape = [4, 4]
    self._test_all(shape, axis=0, keepdims=True)
    self._test_all(shape, axis=1, keepdims=True)
    self._test_all(shape, axis=[0, 1], keepdims=True)

  def testAllTrue(self):
    """Test all with all True values."""
    shape = [5, 5]
    x_np = np.ones(shape, dtype=bool)
    x = tf.constant(x_np, dtype=tf.bool)
    
    def op_func(input_tensor):
        return tf.reduce_all(input_tensor)
        
    self._compare_cpu_musa_results(op_func, [x], tf.bool)

  def testAllFalse(self):
    """Test all with all False values."""
    shape = [5, 5]
    x_np = np.zeros(shape, dtype=bool)
    x = tf.constant(x_np, dtype=tf.bool)
    
    def op_func(input_tensor):
        return tf.reduce_all(input_tensor)
        
    self._compare_cpu_musa_results(op_func, [x], tf.bool)

  def testAllLarge(self):
    """Test all with a large tensor."""
    shape = [1024, 1024]
    self._test_all(shape, axis=0)

  def testEmptyInput(self):
    """Test all with empty input."""
    shape = [0, 5]
    # TensorFlow's behavior for reduce_all on empty dimension depends on the axis
    # For axis=0, it returns a tensor of True with shape [5]
    self._test_all(shape, axis=0)


if __name__ == "__main__":
  tf.test.main()
