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
# =============================================================================
"""Tests for MUSA Round operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class RoundOpTest(MUSATestCase):
  """Tests for MUSA Round operator."""

  def _round_op(self, x):
    """Use rewritten Round op directly."""
    return tf.raw_ops.Round(x=x)

  def _test_round(self, shape, dtype, rtol=1e-5, atol=1e-8):
    """Test round operation with given shape and dtype."""
    if dtype in [tf.int32, tf.int64]:
      np_dtype = dtype.as_numpy_dtype
      x_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)
    else:
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
      x_np = np.random.uniform(-10.0, 10.0, size=shape).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)
    self._compare_cpu_musa_results(self._round_op, [x], dtype, rtol=rtol, atol=atol)

  def testRoundBasicFloat32(self):
    """Test basic round case with float32."""
    self._test_round([4], tf.float32)

  def testRoundMatrixFloatTypes(self):
    """Test matrix inputs for float types."""
    self._test_round([3, 3], tf.float32)
    self._test_round([3, 3], tf.float16, rtol=1e-2, atol=1e-2)
    self._test_round([3, 3], tf.bfloat16, rtol=1e-2, atol=1e-2)

#  def testRoundIntegerTypes(self):
#    """Test integer inputs."""
#    self._test_round([16], tf.int32, rtol=0, atol=0)
#    self._test_round([16], tf.int64, rtol=0, atol=0)

  def testRoundDifferentShapes(self):
    """Test round with different tensor shapes."""
    for shape in [[], [1], [2, 5], [2, 3, 4]]:
      self._test_round(shape, tf.float32)

  def testRoundHalfCases(self):
    """Test key .5 edge inputs to verify consistency."""
    x = tf.constant([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=tf.float32)
    self._compare_cpu_musa_results(self._round_op, [x], tf.float32, rtol=0, atol=0)

  def testRoundLargeTensor(self):
    """Test large tensor with float32."""
    self._test_round([1024, 1024], tf.float32)


if __name__ == "__main__":
  tf.test.main()
