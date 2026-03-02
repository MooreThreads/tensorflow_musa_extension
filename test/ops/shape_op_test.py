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

"""Tests for MUSA Shape operator."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from musa_test_utils import MUSATestCase

class ShapeOpTest(MUSATestCase):
    """测试 MUSA Shape 算子"""

    def testShapeBasic(self):
        """测试基础 Shape 获取"""
        # 测试不同维度的输入
        test_cases = [
            np.random.randn(2),             # 1D
            np.random.randn(2, 3),          # 2D
            np.random.randn(2, 3, 5),       # 3D
            np.random.randn(2, 3, 5, 7),    # 4D
        ]

        for data in test_cases:
            # 默认 int32 输出
            self._compare_cpu_musa_results(
                array_ops.shape,
                [tf.constant(data)],
                tf.int32
            )

    def testShapeInt64(self):
        """测试 out_type=int64 的 Shape 获取"""
        data = np.random.randn(2, 3, 4)
        
        # 使用 lambda 包装以传递 out_type 参数
        def shape_int64(x):
            return array_ops.shape(x, out_type=tf.int64)

        self._compare_cpu_musa_results(
            shape_int64,
            [tf.constant(data)],
            tf.int64
        )

    def testShapeN(self):
        """测试 shape_n (同时获取多个张量的形状)"""
        # shape_n 返回的是 list，需要包装一下取其中一个或全部验证
        # _compare_cpu_musa_results 只能对比单个 tensor 结果
        # 所以这里我们拆开测试，或者验证 shape_n 的第一个结果
        
        x = np.random.randn(2, 3)
        y = np.random.randn(4, 5, 6)
        
        def shape_n_wrapper(t1, t2):
            # shape_n 返回 [shape1, shape2]
            # 为了对比，我们把它们拼接起来或者只返回一个
            # 这里简单起见，返回第一个 shape
            return array_ops.shape_n([t1, t2])[0]

        self._compare_cpu_musa_results(
            shape_n_wrapper,
            [tf.constant(x), tf.constant(y)],
            tf.int32
        )

    def testRank(self):
        """测试 Rank (秩) 算子"""
        # Rank 也是一种特殊的 Shape 信息
        data = np.random.randn(2, 3, 5)
        
        self._compare_cpu_musa_results(
            array_ops.rank,
            [tf.constant(data)],
            tf.int32
        )

    def testSize(self):
        """测试 Size (元素总数) 算子"""
        data = np.random.randn(2, 3, 5) # total size = 30
        
        self._compare_cpu_musa_results(
            array_ops.size,
            [tf.constant(data)],
            tf.int32
        )

if __name__ == "__main__":
    tf.test.main()
