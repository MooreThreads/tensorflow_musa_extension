#Copyright 2026 The TensorFlow MUSA Authors.All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#== == == == == == == == == == == == == == == == == == == == == == == == == == \

"""Tests for MUSA IsNan operator.

This test assumes:
- TensorFlow core has registered the Op 'IsNan' (math_ops.cc).
- The MUSA plugin registers a DEVICE_MTGPU kernel for 'IsNan'.
"""
import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class IsNanOpTest(MUSATestCase):
  """Tests for MUSA IsNan operator."""

  def _make_input(self, shape, dtype, inject_nan=True, fill_value=None, include_inf=False):
    """Create a numpy input array for a given TF dtype."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    if fill_value is not None:
      x_np = np.full(shape, fill_value, dtype=np_dtype)
    else:
      x_np = np.random.uniform(-1.0, 1.0, size=shape).astype(np_dtype)

    if include_inf and x_np.size > 0:
      x_np.flat[0] = np.inf
      if x_np.size > 1:
        x_np.flat[1] = -np.inf

    if inject_nan and x_np.size > 0:
#Put NaNs in deterministic positions
      x_np.flat[0] = np.nan
      x_np.flat[x_np.size // 2] = np.nan
      x_np.flat[-1] = np.nan

    return x_np

  def _test_isnan(self, shape, dtype, inject_nan=True, fill_value=None, include_inf=False):
    """Test IsNan operation with given shape and dtype."""
    x_np = self._make_input(shape, dtype, inject_nan=inject_nan,
                            fill_value=fill_value, include_inf=include_inf)
    x_tf = tf.constant(x_np, dtype=dtype)

    def isnan_proxy(x):
      return tf.cast(tf.math.is_nan(x), tf.float32)

    self._compare_cpu_musa_results(isnan_proxy, [x_tf], dtype, rtol=0.0, atol=0.0)


    with tf.device("/CPU:0"):
      cpu_bool = tf.math.is_nan(x_tf)
    with tf.device("/device:MUSA:0"):
      musa_bool = tf.math.is_nan(x_tf)

    self.assertEqual(cpu_bool.dtype, tf.bool)
    self.assertEqual(musa_bool.dtype, tf.bool)
    self.assertAllEqual(cpu_bool.shape.as_list(), x_tf.shape.as_list())
    self.assertAllEqual(musa_bool.shape.as_list(), x_tf.shape.as_list())
    self.assertAllEqual(cpu_bool.numpy(), musa_bool.numpy())

  def testIsNanSmall(self):
    """Small tensor correctness."""
    for dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
      self._test_isnan([10], dtype)

  def testIsNanLarge(self):
    """Larger tensor correctness."""
    for dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
      self._test_isnan([256, 256], dtype)

  def testIsNanEmptyTensor(self):
    """Empty tensors should return empty bool tensors with same shape."""
    for dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
      self._test_isnan([0], dtype, inject_nan=False)
      self._test_isnan([0, 5], dtype, inject_nan=False)

  def testIsNanNoNaNs(self):
    """If there are no NaNs, all outputs should be False."""
    for dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
      self._test_isnan([1024], dtype, inject_nan=False, include_inf=False)

  def testIsNanAllNaNs(self):
    """All NaNs should yield all True."""
    for dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
#For float16 / bf16, np.nan is representable; TF will carry NaN.
      self._test_isnan([128], dtype, inject_nan=False, fill_value=np.nan)

  def testIsNanWithInfs(self):
    """Infs are not NaNs; only NaNs should be True."""
    for dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
      self._test_isnan([64], dtype, inject_nan=True, include_inf=True)

  def testIsNanInvalidDType(self):
    """IsNan should reject non-floating types per TF op definition."""
    for dtype in [tf.int32, tf.int64]:
      x = tf.constant([1, 2, 3], dtype=dtype)

      with self.assertRaises((TypeError, tf.errors.InvalidArgumentError)):
        with tf.device("/device:MUSA:0"):
          _ = tf.math.is_nan(x)


if __name__ == "__main__":
  tf.test.main()