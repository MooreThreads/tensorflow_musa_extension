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

"""Tests for MUSA RestoreV2 operator using MUSATestCase."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.python.framework import dtypes
from musa_test_utils import MUSATestCase

class RestoreV2OpTest(MUSATestCase):
    """
    测试 MUSA RestoreV2 算子。
    继承自 MUSATestCase，自动处理插件加载和设备校验。
    """

    def testRestoreV2Basic(self):
        """
        对应原 SaveTest.testRelativePath
        测试基础的 Save -> Restore 闭环功能。
        """
        # 准备数据
        ckpt_path = os.path.join(self.get_temp_dir(), "basic_ckpt")
        var_name = "x"
        data_np = np.array([100.0], dtype=np.float32)

        # 定义包装函数：
        # 为了对比 RestoreV2 的结果，我们必须先执行 SaveV2。
        # 这里我们在函数内部完成 "写 -> 读" 的过程，确保对比的是 Restore 的输出。
        def save_restore_cycle(prefix, names, slices, data):
            # 1. 保存 (副作用)
            io_ops.save_v2(prefix, names, slices, [data])
            # 2. 恢复并返回
            return io_ops.restore_v2(prefix, names, slices, [tf.float32])[0]

        # 执行对比
        self._compare_cpu_musa_results(
            save_restore_cycle,
            [tf.constant(ckpt_path), tf.constant([var_name]), tf.constant([""]), tf.constant(data_np)],
            tf.float32
        )

    def testRestoreV2WithSliceInput(self):
        """
        对应原 ShapeInferenceTest.testRestoreV2WithSliceInput
        将原本的静态形状推断测试转换为运行时数据对比测试。
        验证 MUSA 是否能正确处理切片说明符 "3 4 0,1:-"。
        """
        ckpt_path = os.path.join(self.get_temp_dir(), "slice_ckpt")
        var_names = ["var1", "var2"]
        
        # 准备符合 slice spec "3 4" 形状的数据
        # var1: 用空 slice 读取，应恢复完整数据
        data1_np = np.random.randn(3, 4).astype(np.float32)
        # var2: 用 slice "3 4 0,1:-" 读取
        # 含义: Dim0 取 [0, 1) 即第0行; Dim1 取 [:, end) 即所有列
        # 预期结果形状: [1, 4]
        data2_np = np.random.randn(3, 4).astype(np.float32)

        # 定义包装函数
        def save_restore_slice_wrapper(prefix, names, slice_spec1, slice_spec2, d1, d2):
            # 1. 保存完整数据 (此时 slice 传空)
            io_ops.save_v2(prefix, names, ["", ""], [d1, d2])
            
            # 2. 使用指定的 slice spec 恢复
            # 拼接 slice 字符串数组
            slices = tf.stack([slice_spec1, slice_spec2])
            outputs = io_ops.restore_v2(prefix, names, slices, [tf.float32, tf.float32])
            
            # 为了方便对比，我们将两个恢复出来的 Tensor (可能有不同形状) 
            # 展平并拼接成一个 1D Tensor 返回
            return tf.concat([tf.reshape(outputs[0], [-1]), tf.reshape(outputs[1], [-1])], axis=0)

        # 构造输入参数
        prefix_t = tf.constant(ckpt_path)
        names_t = tf.constant(var_names)
        
        # 原始测试用例中的 slice spec
        # var1 的 spec: "" (读取全部)
        # var2 的 spec: "3 4 0,1:-" (TensorFlow 扩展切片语法)
        slice1_t = tf.constant("")
        slice2_t = tf.constant("3 4 0,1:-")
        
        d1_t = tf.constant(data1_np)
        d2_t = tf.constant(data2_np)

        # 执行对比
        # 如果 MUSA 能正确解析切片语法，它读出的数据应与 CPU 完全一致
        self._compare_cpu_musa_results(
            save_restore_slice_wrapper,
            [prefix_t, names_t, slice1_t, slice2_t, d1_t, d2_t],
            tf.float32
        )

if __name__ == "__main__":
    tf.test.main()