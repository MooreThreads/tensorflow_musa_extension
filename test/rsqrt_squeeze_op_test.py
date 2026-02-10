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

"""Tests for MUSA Rsqrt and Squeeze operators."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class MathAndShapeOpTest(MUSATestCase):
    """测试数学运算(Rsqrt)和形状变换(Squeeze)算子"""

    def testRsqrtBasic(self):
        """测试 Rsqrt (Reciprocal Square Root) 算子"""
        # 1. 使用你原始用例中的特定数值进行冒烟测试
        input_data = np.array([1.0, 4.0, 16.0], dtype=np.float32)
        self._compare_cpu_musa_results(
            tf.math.rsqrt, 
            [tf.constant(input_data)], 
            tf.float32
        )

    def testRsqrtRandom(self):
        """使用随机数据测试 Rsqrt 的健壮性"""
        for dtype in [tf.float32, tf.float16]:
            # Rsqrt 对负数会产生 NaN，对 0 会产生 Inf
            # 为了验证计算精度，我们生成正数区间 [0.1, 10.0] 的数据
            shape = [32, 32]
            np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
            data = np.random.uniform(0.1, 10.0, size=shape).astype(np_dtype)
            
            # 适当放宽 float16 的误差容忍度
            rtol = 1e-3 if dtype == tf.float16 else 1e-5
            atol = 1e-3 if dtype == tf.float16 else 1e-8

            self._compare_cpu_musa_results(
                tf.math.rsqrt,
                [tf.constant(data, dtype=dtype)],
                dtype,
                rtol=rtol,
                atol=atol
            )

    def testSqueezeBasic(self):
        """测试 Squeeze (移除维度为1的轴)"""
        # 对应你原始代码中的 x = tf.ones([1, 10, 1])
        shape = [1, 10, 1]
        data = np.ones(shape, dtype=np.float32)
        
        # 默认 Squeeze (移除所有为1的维度)
        self._compare_cpu_musa_results(
            tf.squeeze,
            [tf.constant(data)],
            tf.float32
        )

    def testSqueezeWithAxis(self):
        """测试带 Axis 参数的 Squeeze"""
        # 形状: [1, 5, 1, 3]
        shape = [1, 5, 1, 3]
        data = np.random.randn(*shape).astype(np.float32)

        # 情况 1: 移除 axis=0
        # tf.squeeze(input, axis=[0])
        self._compare_cpu_musa_results(
            lambda x: tf.squeeze(x, axis=[0]),
            [tf.constant(data)],
            tf.float32
        )

        # 情况 2: 移除 axis=2
        self._compare_cpu_musa_results(
            lambda x: tf.squeeze(x, axis=[2]),
            [tf.constant(data)],
            tf.float32
        )

        # 情况 3: 移除 axis=[0, 2]
        self._compare_cpu_musa_results(
            lambda x: tf.squeeze(x, axis=[0, 2]),
            [tf.constant(data)],
            tf.float32
        )

if __name__ == "__main__":
    tf.test.main()