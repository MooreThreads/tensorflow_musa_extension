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

"""Tests for MUSA ReLU operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class ReluOpTest(MUSATestCase):
    """Tests for MUSA ReLU operator."""

    def _test_relu(self, shape, dtype, rtol=1e-5, atol=1e-8):
        """核心测试逻辑：对比 CPU 和 MUSA 的 ReLU 结果"""
        # 处理 bfloat16 的 numpy 转换问题
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
        
        # 生成包含正数和负数的数据，确保 ReLU 的截断逻辑被触发
        # 范围选择 [-2, 2] 以覆盖 0 点附近的非线性变化
        x_np = np.random.uniform(-2, 2, size=shape).astype(np_dtype)
        x = tf.constant(x_np, dtype=dtype)
        
        # 使用父类的对比方法执行 tf.nn.relu
        self._compare_cpu_musa_results(tf.nn.relu, [x], dtype, rtol=rtol, atol=atol)

    def testReluBasic(self):
        """测试不同数据类型下的 ReLU 精度"""
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            # 为低精度类型设置合理的容忍度
            rtol = 1e-3 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-3 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            
            # 测试中等规模的张量
            self._test_relu([128, 128], dtype, rtol=rtol, atol=atol)

    def testReluShapes(self):
        """测试不同形状（维度）下的 ReLU 逻辑"""
        dtype = tf.float32
        test_shapes = [
            [1024],          # 1D 向量
            [32, 64],        # 2D 矩阵
            [8, 16, 32],     # 3D 张量
            [2, 4, 8, 16],   # 4D 常用卷积输入维度
        ]
        for shape in test_shapes:
            self._test_relu(shape, dtype)

    def testReluEdgeCases(self):
        """测试边界情况"""
        # 1. 测试空张量
        self._test_relu([0], tf.float32)
        # 2. 测试标量
        self._test_relu([], tf.float32)

if __name__ == "__main__":
    tf.test.main()
