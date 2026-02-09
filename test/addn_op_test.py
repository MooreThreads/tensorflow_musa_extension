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

"""Tests for MUSA AddN operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class AddNOpTest(MUSATestCase):
  """Tests for MUSA AddN operator."""

  def _test_addn(self, shapes, dtype, rtol=1e-5, atol=1e-8):
    """Test addn operation with given shapes and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    # Generate input tensors
    inputs_np = []
    for shape in shapes:
      if np_dtype in [np.int32, np.int64]:
        # For integer types, use small range to avoid overflow
        input_np = np.random.randint(-10, 10, size=shape).astype(np_dtype)
      else:
        # For floating point types, use uniform distribution
        input_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
      inputs_np.append(input_np)
    
    # Create TensorFlow constants
    inputs_tf = [tf.constant(inp, dtype=dtype) for inp in inputs_np]
    
    # Special handling for tf.add_n - pass as list, not unpacked arguments
    def addn_wrapper(*args):
      return tf.add_n(args)
    
    self._compare_cpu_musa_results(addn_wrapper, inputs_tf, dtype, rtol=rtol, atol=atol)

  def testAddNSingleInput(self):
    """Test AddN with single input."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_addn([[10]], dtype, rtol=rtol, atol=atol)
      self._test_addn([[2048, 2048]], dtype, rtol=rtol, atol=atol)

  def testAddNTwoInputs(self):
    """Test AddN with two inputs."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_addn([[10], [10]], dtype, rtol=rtol, atol=atol)
      self._test_addn([[256, 256], [256, 256]], dtype, rtol=rtol, atol=atol)

  def testAddNThreeInputs(self):
    """Test AddN with three inputs."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_addn([[10], [10], [10]], dtype, rtol=rtol, atol=atol)
      self._test_addn([[256, 256], [256, 256], [256, 256]], dtype, rtol=rtol, atol=atol)

  def testAddNFourInputs(self):
    """Test AddN with four inputs."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_addn([[10], [10], [10], [10]], dtype, rtol=rtol, atol=atol)
      self._test_addn([[256, 256], [256, 256], [256, 256], [256, 256]], dtype, rtol=rtol, atol=atol)

  def testAddNEmptyTensor(self):
    """Test AddN with empty tensors."""
    for dtype in [tf.float32, tf.int32]:
      # Empty tensor with shape [0]
      self._test_addn([[0]], dtype)
      # Empty tensor with shape [0, 5]
      self._test_addn([[0, 5]], dtype)
      # Multiple empty tensors
      self._test_addn([[0], [0]], dtype)

  def testAddNLargeNumberOfInputs(self):
    """Test AddN with large number of inputs (stress test)."""
    for dtype in [tf.float32, tf.int32]:
      # Test with 8 inputs
      shapes = [[100]] * 8
      self._test_addn(shapes, dtype)
      # Test with 16 inputs (smaller size to avoid memory issues)
      shapes = [[10]] * 16
      self._test_addn(shapes, dtype)

  def testAddNDifferentShapesBroadcasting(self):
    """Test AddN with different shapes that require broadcasting."""
    # Note: AddN requires all inputs to have the same shape, so this tests
    # that we properly handle shape validation
    for dtype in [tf.float32, tf.int32]:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self._test_addn([[10], [5]], dtype)
      
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self._test_addn([[2, 3], [3, 2]], dtype)

  def testAddNZeroValues(self):
    """Test AddN with zero values to ensure numerical stability."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      # All zeros
      shapes = [[100]]
      inputs_np = [np.zeros(shape, dtype=np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype) 
                  for shape in shapes]
      inputs_tf = [tf.constant(inp, dtype=dtype) for inp in inputs_np]
      
      def addn_wrapper(*args):
        return tf.add_n(args)
      
      self._compare_cpu_musa_results(addn_wrapper, inputs_tf, dtype, rtol=rtol, atol=atol)

  def testAddNMixedPositiveNegative(self):
    """Test AddN with mixed positive and negative values."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      # Create inputs with mixed positive and negative values
      shapes = [[50], [50]]
      inputs_np = []
      for shape in shapes:
        if dtype in [tf.int32, tf.int64]:
          inp = np.random.randint(-100, 100, size=shape).astype(dtype.as_numpy_dtype)
        else:
          inp = np.random.uniform(-10, 10, size=shape).astype(np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype)
        inputs_np.append(inp)
      
      inputs_tf = [tf.constant(inp, dtype=dtype) for inp in inputs_np]
      
      def addn_wrapper(*args):
        return tf.add_n(args)
      
      self._compare_cpu_musa_results(addn_wrapper, inputs_tf, dtype, rtol=rtol, atol=atol)

  def testAddNEdgeCases(self):
    """Test AddN edge cases like very small/large values."""
    # Only test floating point types for extreme values
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      
      # Test with very small values (close to machine epsilon)
      shapes = [[20]]
      if dtype == tf.float16:
        small_val = np.finfo(np.float16).eps
        inputs_np = [np.full(shape, small_val, dtype=np.float16) for shape in shapes * 2]
      elif dtype == tf.bfloat16:
        # bfloat16 has larger epsilon
        small_val = 1e-3
        inputs_np = [np.full(shape, small_val, dtype=np.float32) for shape in shapes * 2]
      else:  # float32
        small_val = np.finfo(np.float32).eps
        inputs_np = [np.full(shape, small_val, dtype=np.float32) for shape in shapes * 2]
      
      inputs_tf = [tf.constant(inp, dtype=dtype) for inp in inputs_np]
      
      def addn_wrapper(*args):
        return tf.add_n(args)
      
      self._compare_cpu_musa_results(addn_wrapper, inputs_tf, dtype, rtol=rtol, atol=atol)

  def testAddNMaxInputs(self):
    """Test AddN with maximum reasonable number of inputs."""
    # Test with 32 inputs (reasonable upper limit)
    for dtype in [tf.float32, tf.int32]:
      shapes = [[5]] * 32  # Small size to avoid memory issues
      self._test_addn(shapes, dtype)

  def testAddNShapeValidation(self):
    """Test that AddN properly validates input shapes."""
    # This test ensures our implementation correctly validates shapes
    # before attempting computation
    with self.assertRaises(tf.errors.InvalidArgumentError):
      # Different ranks
      inputs = [tf.constant([1, 2, 3], dtype=tf.float32), 
                tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
      def addn_wrapper(*args):
        return tf.add_n(args)
      addn_wrapper(*inputs)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      # Same rank but incompatible dimensions
      inputs = [tf.constant([1, 2, 3], dtype=tf.float32), 
                tf.constant([1, 2], dtype=tf.float32)]
      def addn_wrapper(*args):
        return tf.add_n(args)
      addn_wrapper(*inputs)


if __name__ == "__main__":
  tf.test.main()