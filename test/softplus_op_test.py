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


class SoftplusOpTest(MUSATestCase):

  def _test_softplus(self, shape, dtype, rtol=1e-5, atol=1e-5):
    np_dtype = dtype.as_numpy_dtype
    if dtype == tf.bfloat16:
      np_dtype = np.float32

    if dtype == tf.float16:
      low, high = -5.0, 5.0
    elif dtype == tf.bfloat16:
      low, high = -3.0, 3.0
    else:
      low, high = -10.0, 10.0

    x_np = np.random.uniform(low, high, size=shape).astype(np_dtype)
    x = tf.constant(x_np, dtype=dtype)

    self._compare_cpu_musa_results(
        tf.math.softplus,
        [x],
        dtype,
        rtol=rtol,
        atol=atol
    )
  def testSoftplusFloat32(self):
    self._test_softplus([10, 10], tf.float32, rtol=1e-4, atol=1e-4)

  def testSoftplusFloat16(self):
    self._test_softplus([2, 3, 4], tf.float16, rtol=1e-2, atol=1e-2)

  def testSoftplusBFloat16(self):
    self._test_softplus([10, 10], tf.bfloat16, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
  tf.test.main()