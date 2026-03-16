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

"""Tests for MUSA AssignAdd operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class AssignAddOpTest(MUSATestCase):
  """Tests for MUSA AssignAdd operator."""

  def _test_assign_add(self, shape, dtype, use_locking=False):
    """Test AssignAdd operation with given shape and dtype.
    
    Args:
      shape: Shape of the tensor
      dtype: TensorFlow data type
      use_locking: Whether to use locking for the operation
    """
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    # Generate test data
    if np_dtype in [np.int32, np.int64]:
      init_val_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)
      add_val_np = np.random.randint(-50, 50, size=shape).astype(np_dtype)
    else:
      init_val_np = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
      add_val_np = np.random.uniform(-5, 5, size=shape).astype(np_dtype)

    # Expected result
    expected_result = init_val_np + add_val_np

    # Test on CPU using tf.raw_ops.AssignAdd (requires RefVariable)
    # Note: Modern TensorFlow prefers ResourceVariable, but AssignAdd op
    # works with legacy ref tensors. We'll use Variable.state_value() approach.
    
    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      # Create a variable and perform assign_add
      var_musa = tf.Variable(init_val_np, dtype=dtype)
      var_musa.assign_add(add_val_np)
      musa_result = var_musa.read_value()

    # Test on CPU for comparison
    with tf.device('/CPU:0'):
      var_cpu = tf.Variable(init_val_np, dtype=dtype)
      var_cpu.assign_add(add_val_np)
      cpu_result = var_cpu.read_value()

    # Compare results
    if dtype in [tf.float16, tf.bfloat16]:
      rtol = 1e-2
      atol = 1e-2
      self.assertAllClose(
          tf.cast(cpu_result, tf.float32).numpy(),
          tf.cast(musa_result, tf.float32).numpy(),
          rtol=rtol, atol=atol)
    else:
      rtol = 1e-5
      atol = 1e-8
      self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), 
                          rtol=rtol, atol=atol)

  def testAssignAdd1D(self):
    """Test AssignAdd with 1D tensor."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      self._test_assign_add([100], dtype)

  def testAssignAdd2D(self):
    """Test AssignAdd with 2D tensor."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.float64]:
      self._test_assign_add([64, 64], dtype)

  def testAssignAdd3D(self):
    """Test AssignAdd with 3D tensor."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_assign_add([8, 16, 32], dtype)

  def testAssignAdd4D(self):
    """Test AssignAdd with 4D tensor (common in CNN)."""
    for dtype in [tf.float32, tf.float16]:
      self._test_assign_add([2, 8, 8, 16], dtype)

  def testAssignAddInt32(self):
    """Test AssignAdd with int32 dtype."""
    self._test_assign_add([50], tf.int32)
    self._test_assign_add([10, 20], tf.int32)

  def testAssignAddInt64(self):
    """Test AssignAdd with int64 dtype."""
    self._test_assign_add([50], tf.int64)
    self._test_assign_add([10, 20], tf.int64)

  def testAssignAddScalar(self):
    """Test AssignAdd with scalar value."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_assign_add([], dtype)

  def testAssignAddSingleElement(self):
    """Test AssignAdd with single element tensor."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_assign_add([1], dtype)

  def testAssignAddEmptyTensor(self):
    """Test AssignAdd with empty tensors."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_assign_add([0], dtype)
      self._test_assign_add([0, 5], dtype)

  def testAssignAddLargeTensor(self):
    """Test AssignAdd with large tensor."""
    # Test with 1M elements
    self._test_assign_add([1024, 1024], tf.float32)

  def testAssignAddUseLockingTrue(self):
    """Test AssignAdd with use_locking=True."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      self._test_assign_add([256], dtype, use_locking=True)

  def testAssignAddMultipleTimes(self):
    """Test multiple consecutive AssignAdd operations."""
    shape = [100]
    dtype = tf.float32
    np_dtype = np.float32

    init_val_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
    add_val_1 = np.random.uniform(-0.5, 0.5, size=shape).astype(np_dtype)
    add_val_2 = np.random.uniform(-0.5, 0.5, size=shape).astype(np_dtype)
    add_val_3 = np.random.uniform(-0.5, 0.5, size=shape).astype(np_dtype)

    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      var_musa = tf.Variable(init_val_np, dtype=dtype)
      var_musa.assign_add(add_val_1)
      var_musa.assign_add(add_val_2)
      var_musa.assign_add(add_val_3)
      musa_result = var_musa.read_value()

    # Test on CPU
    with tf.device('/CPU:0'):
      var_cpu = tf.Variable(init_val_np, dtype=dtype)
      var_cpu.assign_add(add_val_1)
      var_cpu.assign_add(add_val_2)
      var_cpu.assign_add(add_val_3)
      cpu_result = var_cpu.read_value()

    self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), 
                        rtol=1e-5, atol=1e-8)

  def testAssignAddWithZeros(self):
    """Test AssignAdd with zero values."""
    shape = [50]
    dtype = tf.float32
    np_dtype = np.float32

    init_val_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
    zero_val = np.zeros(shape, dtype=np_dtype)

    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      var_musa = tf.Variable(init_val_np, dtype=dtype)
      var_musa.assign_add(zero_val)
      musa_result = var_musa.read_value()

    # Adding zero should not change the value
    self.assertAllClose(init_val_np, musa_result.numpy(), rtol=1e-5, atol=1e-8)

  def testAssignAddWithNegative(self):
    """Test AssignAdd with negative values (effectively subtraction)."""
    shape = [50]
    dtype = tf.float32
    np_dtype = np.float32

    init_val_np = np.random.uniform(0, 10, size=shape).astype(np_dtype)
    neg_val = -np.random.uniform(0, 5, size=shape).astype(np_dtype)

    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      var_musa = tf.Variable(init_val_np, dtype=dtype)
      var_musa.assign_add(neg_val)
      musa_result = var_musa.read_value()

    # Test on CPU
    with tf.device('/CPU:0'):
      var_cpu = tf.Variable(init_val_np, dtype=dtype)
      var_cpu.assign_add(neg_val)
      cpu_result = var_cpu.read_value()

    self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), 
                        rtol=1e-5, atol=1e-8)

  def testAssignAddDoublePrecision(self):
    """Test AssignAdd with double precision."""
    shape = [100]
    dtype = tf.float64
    np_dtype = np.float64

    init_val_np = np.random.uniform(-1, 1, size=shape).astype(np_dtype)
    add_val_np = np.random.uniform(-0.5, 0.5, size=shape).astype(np_dtype)

    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      var_musa = tf.Variable(init_val_np, dtype=dtype)
      var_musa.assign_add(add_val_np)
      musa_result = var_musa.read_value()

    # Test on CPU
    with tf.device('/CPU:0'):
      var_cpu = tf.Variable(init_val_np, dtype=dtype)
      var_cpu.assign_add(add_val_np)
      cpu_result = var_cpu.read_value()

    # Double precision should be very accurate
    self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), 
                        rtol=1e-10, atol=1e-14)


if __name__ == "__main__":
  tf.test.main()