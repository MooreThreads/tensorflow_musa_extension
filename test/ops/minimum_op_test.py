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

"""Tests for MUSA Minimum operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class MinimumOpTest(MUSATestCase):

  def _test_minimum(self, shape, dtype, rtol=1e-5, atol=1e-8):
    np_dtype = dtype.as_numpy_dtype
    if dtype == tf.bfloat16:
      np_dtype = np.float32
    
    if dtype in [tf.int32, tf.int64]:
      x_np = np.random.randint(-1000, 1000, size=shape).astype(np_dtype)
      y_np = np.random.randint(-1000, 1000, size=shape).astype(np_dtype)
    else:
      x_np = np.random.randn(*shape).astype(np_dtype)
      y_np = np.random.randn(*shape).astype(np_dtype)
    
    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.minimum, [x, y], dtype, rtol=rtol, atol=atol)

  def testMinimumFloat32(self):
    self._test_minimum([10, 10], tf.float32)

  def testMinimumInt32(self):
    self._test_minimum([10, 10], tf.int32)

  def testMinimumInt64(self):
    x_np = np.array([2**33, -2**33], dtype=np.int64)
    y_np = np.array([2**34, 0], dtype=np.int64)
    x = tf.constant(x_np, dtype=tf.int64)
    y = tf.constant(y_np, dtype=tf.int64)
    self._compare_cpu_musa_results(tf.minimum, [x, y], tf.int64)

  def testMinimumFloat16(self):
    self._test_minimum([10, 10], tf.float16, rtol=1e-3, atol=1e-3)

  def testMinimumBFloat16(self):
    self._test_minimum([10, 10], tf.bfloat16, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
  tf.test.main()
