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

"""Tests for MUSA Log1p operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class Log1pOpTest(MUSATestCase):

  def _test_log1p(self, shape, dtype, rtol=1e-5, atol=1e-8):
    np_dtype = dtype.as_numpy_dtype
    if dtype == tf.bfloat16:
      np_dtype = np.float32
    
    if dtype in [tf.float32, tf.float64, tf.float16, tf.bfloat16]:
      x_np = np.random.uniform(-0.9, 3, size=shape).astype(np_dtype)
    else:
      x_np = np.random.randint(-5, 5, size=shape).astype(np_dtype)
      
    x = tf.constant(x_np, dtype=dtype)
    
    self._compare_cpu_musa_results(tf.math.log1p, [x], dtype, rtol=rtol, atol=atol)

  def testLog1pFloat32(self):
    self._test_log1p([1024, 1024], tf.float32)

  def testLog1pFloat16(self):
    self._test_log1p([4, 4], tf.float16, rtol=1e-2, atol=1e-2)

  def testLog1pBFloat16(self):
    self._test_log1p([3, 3], tf.bfloat16, rtol=1e-1, atol=1e-1)

  def testLog1pEdgeCases(self):
    x_np = np.array([-0.5, 0.0, 1.0, 2.0, 10.0], dtype=np.float32)
    x = tf.constant(x_np, dtype=tf.float32)
    self._compare_cpu_musa_results(tf.math.log1p, [x], tf.float32)


if __name__ == "__main__":
  tf.test.main()
