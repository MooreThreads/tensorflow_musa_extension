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

"""Tests for MUSA SelectV2 (tf.where) operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class SelectOpTest(MUSATestCase):
    """测试 MUSA SelectV2 (tf.where) 算子"""

    def _test_select(self, dtype, np_dtype, tolerance=1e-6):
        """测试指定数据类型的 tf.where"""
        # 准备数据
        # cond: [True, False, True, False]
        cond_np = np.array([True, False, True, False], dtype=np.bool_)
        # x: [1, 2, 3, 4]
        x_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        # y: [10, 20, 30, 40]
        y_np = np.array([10, 20, 30, 40], dtype=np_dtype)

        rtol = tolerance
        atol = tolerance

        # 调用对比方法
        # 注意：tf.where 有三个输入: condition, x, y
        # _compare_cpu_musa_results 会自动在 CPU 和 MUSA 上分别运行并对比结果
        self._compare_cpu_musa_results(
            tf.where,
            [tf.constant(cond_np), tf.constant(x_np, dtype=dtype), tf.constant(y_np, dtype=dtype)],
            dtype,
            rtol=rtol,
            atol=atol
        )

    def testFloat32(self):
        """测试 Float32 类型"""
        self._test_select(tf.float32, np.float32)

    def testInt32(self):
        """测试 Int32 类型"""
        self._test_select(tf.int32, np.int32)

    def testInt64(self):
        """测试 Int64 类型"""
        self._test_select(tf.int64, np.int64)

    def testFloat16(self):
        """测试 Float16 类型 (放宽容差)"""
        self._test_select(tf.float16, np.float16, tolerance=1e-3)

    def testBool(self):
        """测试 Bool 类型"""
        cond = np.array([True, False, True])
        x = np.array([True, True, True])
        y = np.array([False, False, False])
        
        # Bool 类型使用默认容差即可
        # 验证 bool 类型的 Select 操作是否正确
        self._compare_cpu_musa_results(
            tf.where,
            [tf.constant(cond), tf.constant(x), tf.constant(y)],
            tf.bool
        )

    def testBroadcasting(self):
        """测试广播机制 (Broadcasting)"""
        # cond: [False, True] (shape: [2])
        # 将广播匹配 x/y 的最后一维
        cond_np = np.array([False, True])
        # x: [[1, 2], [3, 4]] (shape: [2, 2])
        x_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
        # y: [[5, 6], [7, 8]] (shape: [2, 2])
        y_np = np.array([[5, 6], [7, 8]], dtype=np.float32)

        self._compare_cpu_musa_results(
            tf.where,
            [tf.constant(cond_np), tf.constant(x_np), tf.constant(y_np)],
            tf.float32
        )

    def testScalarBroadcasting(self):
        """测试标量广播"""
        # cond: scalar True
        cond_np = np.array(True)
        # x: [1, 2, 3]
        x_np = np.array([1, 2, 3], dtype=np.float32)
        # y: scalar 100
        # 标量 y 会被广播成 [100, 100, 100] 与 x 形状一致
        y_np = np.array(100, dtype=np.float32)

        self._compare_cpu_musa_results(
            tf.where,
            [tf.constant(cond_np), tf.constant(x_np), tf.constant(y_np)],
            tf.float32
        )

if __name__ == "__main__":
    tf.test.main()