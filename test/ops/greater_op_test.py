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

"""Tests for MUSA Greater operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class GreaterOpTest(MUSATestCase):

  def _test_greater(self, shape, dtype):
    np_dtype = dtype.as_numpy_dtype
    if dtype == tf.bfloat16:
        np_dtype = np.float32

    if dtype in [tf.float32, tf.float16, tf.bfloat16]:
      x_np = np.random.randn(*shape).astype(np_dtype)
      y_np = np.random.randn(*shape).astype(np_dtype)
    else:
      x_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)
      y_np = np.random.randint(-100, 100, size=shape).astype(np_dtype)

    x = tf.constant(x_np, dtype=dtype)
    y = tf.constant(y_np, dtype=dtype)

    self._compare_cpu_musa_results(tf.greater, [x, y], dtype)

  def testGreaterFloat32(self):
    self._test_greater([1024, 1024], tf.float32)

  def testGreaterFloat16(self):
    self._test_greater([256, 4096], tf.float16)

  def testGreaterInt32(self):
    self._test_greater([1024, 1024], tf.int32)

  def testGreaterInt64(self):
    self._test_greater([1024, 1024], tf.int64)


if __name__ == "__main__":
  tf.test.main()
