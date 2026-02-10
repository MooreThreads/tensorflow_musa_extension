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

"""Tests for MUSA Size operator using MUSATestCase."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class SizeOpTest(MUSATestCase):
    """
    测试 MUSA Size 算子。
    覆盖：基本形状、不同数据类型输入、不同输出类型(int32/int64)、边界情况。
    """

    def testSizeBasicShapes(self):
        """测试不同形状张量的 Size 计算 (对应原 test_size_basic)"""
        # 测试用例列表：(数据, 描述)
        test_cases = [
            (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), "1D Vector"),
            (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "2D Matrix"),
            (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32), "3D Tensor")
        ]

        for np_data, desc in test_cases:
            print(f"Testing {desc}...")
            self._compare_cpu_musa_results(
                tf.size,
                [tf.constant(np_data)],
                tf.int32
            )

    def testSizeInputDtypes(self):
        """测试不同输入数据类型的 Size 计算 (对应原 test_size_different_dtypes)"""
        # 基础数据形状
        np_data = np.array([[1, 2], [3, 4]])
        
        # 待测类型列表
        dtypes_to_test = [tf.float32, tf.float64, tf.int32, tf.int64, tf.float16]

        for dtype in dtypes_to_test:
            # 构造对应类型的输入 Tensor
            input_tensor = tf.constant(np_data, dtype=dtype)
            
            self._compare_cpu_musa_results(
                tf.size,
                [input_tensor],
                tf.int32
            )

    def testSizeOutputTypes(self):
        """测试不同的输出类型 out_type (对应原 test_size_output_types)"""
        data = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # 1. 测试默认/int32 输出
        # wrapper 用于显式指定 out_type
        def size_int32_wrapper(x):
            return tf.size(x, out_type=tf.int32)

        self._compare_cpu_musa_results(
            size_int32_wrapper,
            [data],
            tf.int32
        )

        # 2. 测试 int64 输出
        def size_int64_wrapper(x):
            return tf.size(x, out_type=tf.int64)

        self._compare_cpu_musa_results(
            size_int64_wrapper,
            [data],
            tf.int64
        )

    def testSizeEdgeCases(self):
        """测试边界情况：标量、空张量等 (对应原 test_size_edge_cases)"""
        
        # 1. 标量 (Scalar) -> Size 应为 1
        self._compare_cpu_musa_results(
            tf.size,
            [tf.constant(42.0)],
            tf.int32
        )

        # 2. 一维空向量 -> Size 应为 0
        self._compare_cpu_musa_results(
            tf.size,
            [tf.constant([], dtype=tf.float32)],
            tf.int32
        )

        # 3. 二维空矩阵 (1x0) -> Size 应为 0
        self._compare_cpu_musa_results(
            tf.size,
            [tf.constant([[]], dtype=tf.float32)],
            tf.int32
        )

        # 4. 高维空张量 -> Size 应为 0
        # tf.zeros 可以在 helper 内部被正确放置到设备上
        self._compare_cpu_musa_results(
            tf.size,
            [tf.zeros((2, 0, 3, 0), dtype=tf.float32)],
            tf.int32
        )

        # 5. 非常规形状 (1x1x1x1x1) -> Size 应为 1
        self._compare_cpu_musa_results(
            tf.size,
            [tf.zeros((1, 1, 1, 1, 1), dtype=tf.float32)],
            tf.int32
        )

if __name__ == "__main__":
    tf.test.main()
