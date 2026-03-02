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

"""Tests for MUSA BiasAdd operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class BiasAddOpTest(MUSATestCase):
  """Tests for MUSA BiasAdd operator."""

  def _test_bias_add(self, value_shape, bias_shape, data_format="NHWC", 
                     dtype=tf.float32, rtol=None, atol=None):
    """Test bias add operation with given shapes and data format."""
    # Set default tolerances based on data type
    if rtol is None:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
    if atol is None:
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
    
    if dtype == tf.bfloat16:
      value_np = np.random.uniform(-1, 1, size=value_shape).astype(np.float32)
      bias_np = np.random.uniform(-1, 1, size=bias_shape).astype(np.float32)
    else:
      value_np = np.random.uniform(-1, 1, size=value_shape).astype(dtype.as_numpy_dtype)
      bias_np = np.random.uniform(-1, 1, size=bias_shape).astype(dtype.as_numpy_dtype)
    
    value = tf.constant(value_np, dtype=dtype)
    bias = tf.constant(bias_np, dtype=dtype)
    
    # Test on CPU
    with tf.device('/CPU:0'):
      cpu_result = tf.nn.bias_add(value, bias, data_format=data_format)
    
    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      musa_result = tf.nn.bias_add(value, bias, data_format=data_format)
    
    # Compare results using the overridden assertAllClose method
    if dtype in [tf.float16, tf.bfloat16]:
      cpu_result_f32 = tf.cast(cpu_result, tf.float32)
      musa_result_f32 = tf.cast(musa_result, tf.float32)
      self.assertAllClose(cpu_result_f32.numpy(), 
                         musa_result_f32.numpy(),
                         rtol=rtol, 
                         atol=atol)
    elif dtype in [tf.int32, tf.int64]:
      # For integer types, use exact comparison (no tolerance needed)
      self.assertAllClose(cpu_result.numpy(), 
                         musa_result.numpy(),
                         rtol=0, 
                         atol=0)
    else:
      self.assertAllClose(cpu_result.numpy(), 
                         musa_result.numpy(),
                         rtol=rtol, 
                         atol=atol)

  def testBiasAdd2D(self):
    """2D tensor bias add test."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_bias_add([10, 5], [5], dtype=dtype, rtol=rtol, atol=atol)

  def testBiasAdd4DNHWC(self):
    """4D tensor bias add with NHWC format."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_bias_add([2, 3, 4, 5], [5], "NHWC", dtype=dtype, rtol=rtol, atol=atol)

  def testBiasAdd4DNCHW(self):
    """4D tensor bias add with NCHW format."""
    for dtype in [tf.float32]:
      self._test_bias_add([2, 5, 3, 4], [5], "NCHW", dtype=dtype)

  def testBiasAdd3D(self):
    """3D tensor bias add test."""
    for dtype in [tf.float32]:
      self._test_bias_add([2, 3, 4], [4], dtype=dtype)

  def testBiasAddDifferentDtypes(self):
    """Test bias add with different data types."""
    value_shape = [5, 3]
    bias_shape = [3]
    dtypes = [tf.float32, tf.int32, tf.int64]
    
    for dtype in dtypes:
      with self.subTest(dtype=dtype):
        if dtype == tf.float32:
          # Use default float32 tolerances
          self._test_bias_add(value_shape, bias_shape, dtype=dtype)
        else:
          # For integer types, use exact comparison
          self._test_bias_add(value_shape, bias_shape, dtype=dtype, rtol=0, atol=0)

  def testBiasAddLargeTensor(self):
    """Bias add with larger tensors."""
    for dtype in [tf.float32]:
      self._test_bias_add([100, 50], [50], dtype=dtype)


if __name__ == "__main__":
  tf.test.main()