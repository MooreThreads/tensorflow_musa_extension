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

"""Tests for MUSA Softplus operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


dtype_list = [tf.float32]
# dtype_list = [tf.float32, tf.float16, tf.bfloat16, tf.float64]
class SoftplusOpTest(MUSATestCase):
  """Tests for MUSA Softplus operator."""

  
  def _test_softplus(self, shape, dtype, rtol=1e-5, atol=1e-8,
                     value_range=(-10.0, 10.0)):
    """Test softplus operation with given shape and dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    # Generate input tensor
    input_np = np.random.uniform(value_range[0], value_range[1], size=shape).astype(np_dtype)
    input_tf = tf.constant(input_np, dtype=dtype)

    def softplus_wrapper(x):
      return tf.nn.softplus(x)

    self._compare_cpu_musa_results(
        softplus_wrapper, [input_tf], dtype, rtol=rtol, atol=atol)

  def testSoftplusBasic(self):
    """Test Softplus with basic shapes."""
    for dtype in dtype_list:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      self._test_softplus([10], dtype, rtol=rtol, atol=atol)
      self._test_softplus([128], dtype, rtol=rtol, atol=atol)
      self._test_softplus([32, 64], dtype, rtol=rtol, atol=atol)
      self._test_softplus([8, 16, 32], dtype, rtol=rtol, atol=atol)

  def testSoftplusLargeTensor(self):
    """Test Softplus with large tensors."""
    for dtype in dtype_list:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      self._test_softplus([2048, 2048], dtype, rtol=rtol, atol=atol)

  def testSoftplusEmptyTensor(self):
    """Test Softplus with empty tensors."""
    for dtype in dtype_list:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      # Empty tensor with shape [0]
      self._test_softplus([0], dtype, rtol=rtol, atol=atol)
      # Empty tensor with shape [0, 5]
      self._test_softplus([0, 5], dtype, rtol=rtol, atol=atol)

  def testSoftplusZeroValues(self):
    """Test Softplus with all-zero values."""
    for dtype in dtype_list:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
      input_np = np.zeros((256,), dtype=np_dtype)
      input_tf = tf.constant(input_np, dtype=dtype)

      def softplus_wrapper(x):
        return tf.nn.softplus(x)

      self._compare_cpu_musa_results(
          softplus_wrapper, [input_tf], dtype, rtol=rtol, atol=atol)

  def testSoftplusMixedPositiveNegative(self):
    """Test Softplus with mixed positive and negative values."""
    for dtype in dtype_list:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
      input_np = np.random.uniform(-50, 50, size=(512,)).astype(np_dtype)
      input_tf = tf.constant(input_np, dtype=dtype)

      def softplus_wrapper(x):
        return tf.nn.softplus(x)

      self._compare_cpu_musa_results(
          softplus_wrapper, [input_tf], dtype, rtol=rtol, atol=atol)

  def testSoftplusExtremeValues(self):
    """Test Softplus with extreme values for numerical stability."""
    for dtype in dtype_list:
      rtol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 2e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

      # Use a conservative range for fp16/bf16 to reduce overflow-related mismatch
      if dtype == tf.float16:
        vals = np.array([-20.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0],
                        dtype=np_dtype)
      elif dtype == tf.bfloat16:
        vals = np.array([-50.0, -20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0, 50.0],
                        dtype=np_dtype)
      elif dtype == tf.float32:
        vals = np.array([-100.0, -50.0, -20.0, -1.0, 0.0, 1.0, 20.0, 50.0, 100.0],
                        dtype=np_dtype)
      else:  # tf.float64
        vals = np.array([-500.0, -100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0, 500.0],
                        dtype=np_dtype)

      input_tf = tf.constant(vals, dtype=dtype)

      def softplus_wrapper(x):
        return tf.nn.softplus(x)

      self._compare_cpu_musa_results(
          softplus_wrapper, [input_tf], dtype, rtol=rtol, atol=atol)

  def testSoftplusSmallValues(self):
    """Test Softplus with very small values around zero."""
    for dtype in dtype_list:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-6
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

      if dtype == tf.float16:
        eps = np.finfo(np.float16).eps
      elif dtype == tf.float64:
        eps = np.finfo(np.float64).eps
      else:
        eps = np.finfo(np.float32).eps

      vals = np.array([-10 * eps, -eps, 0.0, eps, 10 * eps], dtype=np_dtype)
      input_tf = tf.constant(vals, dtype=dtype)

      def softplus_wrapper(x):
        return tf.nn.softplus(x)

      self._compare_cpu_musa_results(
          softplus_wrapper, [input_tf], dtype, rtol=rtol, atol=atol)

  def testSoftplusScalar(self):
    """Test Softplus with scalar input."""
    for dtype in dtype_list:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
      input_np = np.array(1.2345, dtype=np_dtype)
      input_tf = tf.constant(input_np, dtype=dtype)

      def softplus_wrapper(x):
        return tf.nn.softplus(x)

      self._compare_cpu_musa_results(
          softplus_wrapper, [input_tf], dtype, rtol=rtol, atol=atol)

  def testSoftplusDifferentShapes(self):
    """Test Softplus on tensors with various dimensions."""
    test_shapes = [
        [1],
        [7],
        [1, 1],
        [3, 5],
        [2, 3, 4],
        [2, 2, 2, 2],
    ]
    for dtype in dtype_list:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8

      for shape in test_shapes:
        self._test_softplus(shape, dtype, rtol=rtol, atol=atol)

  def testSoftplusMonotonicitySanity(self):
    """Sanity test: softplus should be monotonic increasing."""
    # This checks both functional correctness and consistency on MUSA.
    for dtype in dtype_list:
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
      x_np = np.linspace(-10.0, 10.0, num=1000).astype(np_dtype)
      x_tf = tf.constant(x_np, dtype=dtype)

      with tf.device('/device:MUSA:0'):
        y = tf.nn.softplus(x_tf)

      y_np = tf.cast(y, tf.float32).numpy() if dtype in [tf.float16, tf.bfloat16] else y.numpy()
      diffs = np.diff(y_np)

      # Allow tiny numerical noise for low precision
      tol = 1e-4 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self.assertTrue(np.all(diffs >= -tol),
                      msg="Softplus output is not monotonic increasing.")

  def testSoftplusInvalidType(self):
    """Test Softplus with invalid (non-floating) dtype should fail."""
    for dtype in [tf.int32, tf.int64]:
      x = tf.constant([1, 2, 3], dtype=dtype)
      with self.assertRaises((tf.errors.InvalidArgumentError, TypeError)):
        with tf.device('/device:MUSA:0'):
          _ = tf.nn.softplus(x)


if __name__ == "__main__":
  tf.test.main()