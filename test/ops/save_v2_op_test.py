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

"""Tests for MUSA SaveV2 operator using MUSATestCase."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import io_ops
# 引入基础测试类
from musa_test_utils import MUSATestCase

class MusaSaveV2Test(MUSATestCase):
    """测试 MUSA SaveV2 算子 (通过 Save+Restore 闭环验证)"""

    def setUp(self):
        # 调用父类初始化
        super(MusaSaveV2Test, self).setUp()
        # 使用 get_temp_dir() 获取安全的临时目录
        self.test_dir = self.get_temp_dir()
        self.prefix = os.path.join(self.test_dir, "ckpt_compare_musa")

    def testSaveV2Comparison(self):
        
        # 1. 准备数据
        var_name = "var_save"
        data_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        
        def save_restore_wrapper(prefix, names, slices, data):
         
            io_ops.save_v2(prefix, names, slices, [data])
            
        
            return io_ops.restore_v2(prefix, names, slices, [tf.float32])[0]

        # 3. 准备输入张量
        prefix_t = tf.constant(self.prefix)
        names_t = tf.constant([var_name])
        slices_t = tf.constant([""]) 
        data_t = tf.constant(data_np)

        # 4. 执行对比
     
        self._compare_cpu_musa_results(
            save_restore_wrapper,
            [prefix_t, names_t, slices_t, data_t],
            tf.float32
        )

if __name__ == "__main__":
    tf.test.main()