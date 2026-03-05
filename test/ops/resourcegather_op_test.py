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

"""Tests for MUSA Resource operators using MUSATestCase."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class ResourceOpTest(MUSATestCase):
    """
    测试 MUSA 资源类算子 (Variable, Gather, ScatterAdd).
    利用 _compare_cpu_musa_results 验证 MUSA 与 CPU 的行为一致性。
    """

    def testResourceGatherComparison(self):
        """
        对比 CPU 和 MUSA 的 ResourceGather 结果。
        涵盖: VarHandleOp, AssignVariableOp, ResourceGather
        """
        vocab_size = 100
        embedding_dim = 8
        
        # 准备输入数据 (使用 Tensor 包装以便传递)
        # 词表: 100x8
        h_params = np.array([np.full(embedding_dim, i, dtype=np.float32) for i in range(vocab_size)])
        # 索引
        h_indices = np.array([1, 5, 10, 99], dtype=np.int32)

        # 定义包装函数：在当前设备上创建变量并执行 Gather
        def gather_wrapper(params_val, indices_val):
            # 1. 在当前设备 (CPU 或 MUSA) 创建变量
            # 注意：因为外层有 with tf.device(...), 变量会自动落在这个设备上
            var = tf.Variable(params_val)
            
            # 2. 执行 Gather 操作
            result = tf.gather(var, indices_val)
            return result

        # 执行对比
        self._compare_cpu_musa_results(
            gather_wrapper,
            [tf.constant(h_params), tf.constant(h_indices)],
            tf.float32
        )

    def testResourceScatterAddComparison(self):
        """
        对比 CPU 和 MUSA 的 ResourceScatterAdd 结果。
        涵盖: ResourceScatterAddOp, ReadVariableOp
        """
        vocab_size = 10
        embedding_dim = 4
        
        # 初始参数全为 1.0
        h_params = np.ones((vocab_size, embedding_dim), dtype=np.float32)
        
        # 更新参数: 给索引 1 和 3 分别加上不同的值
        # 索引 1 加 [10, 10...]
        # 索引 3 加 [20, 20...]
        h_updates = np.array([
            [10.0] * embedding_dim,
            [20.0] * embedding_dim
        ], dtype=np.float32)
        h_indices = np.array([1, 3], dtype=np.int32)

        # 定义包装函数：创建变量 -> ScatterAdd -> 读取新值
        def scatter_add_wrapper(params_val, updates_val, indices_val):
            # 1. 创建变量
            var = tf.Variable(params_val)
            
            # 2. 执行 ScatterAdd (这是一个原地更新操作)
            # 构造 IndexedSlices 对象
            ops = var.scatter_add(tf.IndexedSlices(updates_val, indices_val))
            
            # 3. 读取更新后的值返回 (为了对比结果)
            return var.read_value()

        # 执行对比
        self._compare_cpu_musa_results(
            scatter_add_wrapper,
            [tf.constant(h_params), tf.constant(h_updates), tf.constant(h_indices)],
            tf.float32
        )

    def testVariableShapeComparison(self):
        """
        对比 CPU 和 MUSA 的 VariableShape 结果。
        涵盖: MusaVariableShapeOp
        """
        shape = [50, 20]
        h_params = np.zeros(shape, dtype=np.float32)

        def shape_wrapper(params_val):
            var = tf.Variable(params_val)
            # 返回 Shape 张量，这里需要转成 float32 以便 _compare_cpu_musa_results 对比
            # 或者直接对比 int32 (需要基类支持，通常转 float32 比较方便)
            return tf.cast(tf.shape(var), tf.float32)

        # 执行对比
        # 注意：这里我们对比的是 Shape 的数值
        self._compare_cpu_musa_results(
            shape_wrapper,
            [tf.constant(h_params)],
            tf.float32
        )

if __name__ == "__main__":
    tf.test.main()