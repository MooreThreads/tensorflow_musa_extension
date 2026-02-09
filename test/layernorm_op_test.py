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

"""Tests for MUSA LayerNorm operator."""

import tensorflow as tf
import numpy as np

from musa_test_utils import MUSATestCase


class LayerNormOpTest(MUSATestCase):
  """Tests for MUSA LayerNorm operator."""

  def _test_layer_norm(self, x_shape, gamma_shape, beta_shape, epsilon=1e-5,
                      dtype=tf.float32, rtol=1e-5, atol=1e-8):
    """Test layer normalization operation."""
    if dtype == tf.bfloat16:
      x_np = np.random.uniform(-1, 1, size=x_shape).astype(np.float32)
      gamma_np = np.random.uniform(0.5, 1.5, size=gamma_shape).astype(np.float32)
      beta_np = np.random.uniform(-0.5, 0.5, size=beta_shape).astype(np.float32)
    else:
      x_np = np.random.uniform(-1, 1, size=x_shape).astype(dtype.as_numpy_dtype)
      gamma_np = np.random.uniform(0.5, 1.5, size=gamma_shape).astype(dtype.as_numpy_dtype)
      beta_np = np.random.uniform(-0.5, 0.5, size=beta_shape).astype(dtype.as_numpy_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    gamma = tf.constant(gamma_np, dtype=dtype)
    beta = tf.constant(beta_np, dtype=dtype)
    
    # Get the custom LayerNorm op
    try:
      layernorm_op = getattr(tf.raw_ops, 'MusaLayerNorm', None)
      if layernorm_op is None:
        self.skipTest("MusaLayerNorm op not available")
    except AttributeError:
      self.skipTest("MusaLayerNorm op not available")
    
    # Test on CPU reference (using standard tf.nn.l2_normalize approach)
    # For LayerNorm, we compute manually for comparison
    def compute_layernorm_ref(x_val, gamma_val, beta_val, eps):
      mean = np.mean(x_val, axis=-1, keepdims=True)
      variance = np.var(x_val, axis=-1, keepdims=True)
      normalized = (x_val - mean) / np.sqrt(variance + eps)
      return normalized * gamma_val + beta_val
    
    # Test on MUSA
    with tf.device('/device:MUSA:0'):
      musa_result = layernorm_op(x=x, gamma=gamma, beta=beta, epsilon=epsilon)
    
    # Compute reference result
    ref_result = compute_layernorm_ref(x_np, gamma_np, beta_np, epsilon)
    
    # Compare results
    if dtype in [tf.float16, tf.bfloat16]:
      musa_result_f32 = tf.cast(musa_result, tf.float32)
      self.assertAllClose(ref_result, 
                         musa_result_f32.numpy(),
                         rtol=rtol, 
                         atol=atol)
    else:
      self.assertAllClose(ref_result, 
                         musa_result.numpy(),
                         rtol=rtol, 
                         atol=atol)

  def testLayerNormBasic(self):
    """Basic LayerNorm test."""
    x_shape = [2, 3, 4]
    gamma_shape = [4]
    beta_shape = [4]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_layer_norm(x_shape, gamma_shape, beta_shape, dtype=dtype, 
                           rtol=rtol, atol=atol)

  def testLayerNorm2D(self):
    """2D LayerNorm test."""
    x_shape = [10, 5]
    gamma_shape = [5]
    beta_shape = [5]
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_layer_norm(x_shape, gamma_shape, beta_shape, dtype=dtype, 
                           rtol=rtol, atol=atol)

  def testLayerNormDifferentEpsilons(self):
    """LayerNorm with different epsilon values."""
    x_shape = [3, 4]
    gamma_shape = [4]
    beta_shape = [4]
    epsilons = [1e-5, 1e-3, 1e-1]
    for epsilon in epsilons:
      with self.subTest(epsilon=epsilon):
        self._test_layer_norm(x_shape, gamma_shape, beta_shape, epsilon=epsilon)

  def testLayerNormLargeInput(self):
    """LayerNorm with larger input."""
    x_shape = [32, 128]
    gamma_shape = [128]
    beta_shape = [128]
    self._test_layer_norm(x_shape, gamma_shape, beta_shape, dtype=tf.float32)

  def testLayerNormSingleFeature(self):
    """LayerNorm with single feature dimension."""
    x_shape = [5, 1]
    gamma_shape = [1]
    beta_shape = [1]
    self._test_layer_norm(x_shape, gamma_shape, beta_shape, dtype=tf.float32)

  def testLayerNormBatchSize1(self):
    """LayerNorm with batch size 1."""
    x_shape = [1, 10]
    gamma_shape = [10]
    beta_shape = [10]
    self._test_layer_norm(x_shape, gamma_shape, beta_shape, dtype=tf.float32)

  def testLayerNormInvalidShapes(self):
    """Test LayerNorm with invalid shapes."""
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      with tf.device('/device:MUSA:0'):
        layernorm_op = getattr(tf.raw_ops, 'MusaLayerNorm', None)
        if layernorm_op:
          # Mismatched gamma shape
          x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
          gamma = tf.constant([1.0, 2.0], dtype=tf.float32)  # Wrong size
          beta = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
          layernorm_op(x=x, gamma=gamma, beta=beta)


if __name__ == "__main__":
  tf.test.main()