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
"""Tests for MUSA FusedBatchNormV3 operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class FusedBatchNormV3OpTest(MUSATestCase):
    """Tests for MUSA FusedBatchNormV3 operator."""

    def _test_fused_batch_norm(self, shape, data_format="NHWC", dtype=tf.float32, rtol=1e-5, atol=1e-8):
        """
        Test FusedBatchNormV3 operation with given parameters.
        """
        np.random.seed(42)  # For reproducibility

        # Determine channel dimension based on data format
        if data_format == "NHWC":
            channel_dim = -1  # Last dimension
        elif data_format == "NCHW":
            channel_dim = 1   # Second dimension
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

        # Extract number of channels
        C = shape[channel_dim]

        # Prepare input and parameters
        # 1. 输入数据 x 跟随测试指定的 dtype (可以是 float16 或 float32)
        x_np = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        
        # 2. 【关键修改】统计量参数必须始终为 float32 (即使 x 是 float16)
        scale_np = np.random.rand(C).astype(np.float32)
        offset_np = np.random.rand(C).astype(np.float32)
        mean_np = np.zeros(C).astype(np.float32)
        var_np = np.ones(C).astype(np.float32)

        # Create TensorFlow constants
        x = tf.constant(x_np, dtype=dtype)
        

        scale = tf.constant(scale_np, dtype=tf.float32)
        offset = tf.constant(offset_np, dtype=tf.float32)
        mean = tf.constant(mean_np, dtype=tf.float32)
        variance = tf.constant(var_np, dtype=tf.float32)

        # Define a wrapper function to call the raw op and extract the output tensor
        def fused_batch_norm_wrapper(x, scale, offset, mean, variance):
            y_raw = tf.raw_ops.FusedBatchNormV3(
                x=x,
                scale=scale,
                offset=offset,
                mean=mean,
                variance=variance,
                epsilon=0.001,
                exponential_avg_factor=1.0,
                data_format=data_format,
                is_training=True
            )
            return y_raw[0]  # Return only the normalized output

        # Compare CPU and MUSA results using the utility method
        self._compare_cpu_musa_results(
            fused_batch_norm_wrapper,
            [x, scale, offset, mean, variance],
            dtype,
            rtol=rtol,
            atol=atol
        )

    # ... 下面的测试用例保持不变 ...
    def testFusedBatchNormNCHW(self):
        shape = [2, 32, 1, 1]
        self._test_fused_batch_norm(shape, data_format="NCHW", dtype=tf.float32, rtol=1e-5, atol=1e-5)

    def testFusedBatchNormNHWC(self):
        shape = [2, 1, 1, 32]
        self._test_fused_batch_norm(shape, data_format="NHWC", dtype=tf.float32, rtol=1e-5, atol=1e-5)

    def testFusedBatchNormNCHWFloat16(self):
        shape = [4, 16, 8, 8]
      
        self._test_fused_batch_norm(shape, data_format="NCHW", dtype=tf.float16, rtol=1e-2, atol=1e-2)

    def testFusedBatchNormDifferentShapes(self):
        test_shapes = [
            [1, 64, 32, 32], 
            [8, 128, 16, 16],
            [2, 3, 224, 224], 
        ]
        for shape in test_shapes:
            with self.subTest(shape=shape):
                self._test_fused_batch_norm(shape, data_format="NCHW", dtype=tf.float32, rtol=1e-4, atol=1e-4)
if __name__ == "__main__":
    tf.test.main()
