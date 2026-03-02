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

"""Tests for MUSA ShapeN operator."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from musa_test_utils import MUSATestCase

class ShapeNOpTest(MUSATestCase):
    """测试 MUSA ShapeN 算子"""

    def testShapeNBasic(self):
        """测试基础 ShapeN 功能"""
        # 准备多个不同形状的输入
        x = np.random.randn(2, 3)
        y = np.random.randn(4, 5, 6)
        z = np.random.randn(10)
        
        # 定义 Wrapper
        # ShapeN 输入是 list，输出也是 list
        # 为了方便 _compare_cpu_musa_results 对比，我们将输出的 list[Tensor] 
        # 拼接成一个 1D Tensor。
        # 例如: shape(x)=[2,3], shape(y)=[4,5,6] -> concat -> [2, 3, 4, 5, 6]
        def shape_n_concat_wrapper(*inputs):
            # inputs 是一个 tuple，对应传入的参数列表
            shapes = array_ops.shape_n(list(inputs))
            # 将所有形状张量展平并拼接
            flat_shapes = [tf.reshape(s, [-1]) for s in shapes]
            return tf.concat(flat_shapes, axis=0)

        self._compare_cpu_musa_results(
            shape_n_concat_wrapper,
            [tf.constant(x), tf.constant(y), tf.constant(z)],
            tf.int32
        )

    def testShapeNInt64(self):
        """测试 out_type=int64 的 ShapeN"""
        x = np.random.randn(100, 200)
        y = np.random.randn(5, 5)

        def shape_n_int64_wrapper(*inputs):
            shapes = array_ops.shape_n(list(inputs), out_type=tf.int64)
            flat_shapes = [tf.reshape(s, [-1]) for s in shapes]
            return tf.concat(flat_shapes, axis=0)

        self._compare_cpu_musa_results(
            shape_n_int64_wrapper,
            [tf.constant(x), tf.constant(y)],
            tf.int64
        )

    def testShapeNSameInput(self):
        """测试输入包含相同 Tensor 的情况 (对应原测试 _compareShapeN)"""
        x = np.random.randn(2, 3)

        def shape_n_wrapper(t):
            # 模拟原测试: array_ops.shape_n([x, x, x])
            shapes = array_ops.shape_n([t, t, t])
            flat_shapes = [tf.reshape(s, [-1]) for s in shapes]
            return tf.concat(flat_shapes, axis=0)

        self._compare_cpu_musa_results(
            shape_n_wrapper,
            [tf.constant(x)],
            tf.int32
        )

if __name__ == "__main__":
    tf.test.main()
