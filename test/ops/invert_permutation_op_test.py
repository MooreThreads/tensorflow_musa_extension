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

"""Tests for MUSA InvertPermutation operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class InvertPermutationOpTest(MUSATestCase):

  def testInvertPermutationInt32(self):
    x_np = np.array([3, 0, 2, 1], dtype=np.int32)
    
    # 【修复】定义一个 wrapper，显式指定关键字参数 x=...
    def op_wrapper(x):
        return tf.raw_ops.InvertPermutation(x=x)

    self._compare_cpu_musa_results(
        op_wrapper,  # 传入 wrapper 而不是 raw_ops 本身
        [x_np],
        tf.int32
    )

  def testInvertPermutationInt64(self):
    x_np = np.array([4, 2, 1, 3, 0], dtype=np.int64)

    # 【修复】使用 lambda 也可以
    self._compare_cpu_musa_results(
        lambda x: tf.raw_ops.InvertPermutation(x=x),
        [x_np],
        tf.int64
    )

  def testInvertPermutationEmpty(self):
    x_np = np.array([], dtype=np.int32)

    # 【修复】
    def op_wrapper(x):
        return tf.raw_ops.InvertPermutation(x=x)

    self._compare_cpu_musa_results(
        op_wrapper,
        [x_np],
        tf.int32
    )


if __name__ == "__main__":
  tf.test.main()