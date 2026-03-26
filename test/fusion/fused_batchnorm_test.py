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
"""Coverage tests for the FusedBatchNormV3 op across layouts and gradients."""

import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEST_DIR = os.path.dirname(_CURRENT_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class FusedBatchNormFusionTest(MUSATestCase):
    """Combined FusedBatchNormV3 tests for layouts, dtypes, and gradients."""

    def _build_bn_inputs(self, shape, data_format, dtype):
        np.random.seed(42)
        channel_dim = -1 if data_format == "NHWC" else 1
        channels = shape[channel_dim]

        x_np = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        scale_np = np.random.rand(channels).astype(np.float32)
        offset_np = np.random.rand(channels).astype(np.float32)
        mean_np = np.zeros(channels, dtype=np.float32)
        var_np = np.ones(channels, dtype=np.float32)

        x = tf.constant(x_np, dtype=dtype)
        scale = tf.constant(scale_np, dtype=tf.float32)
        offset = tf.constant(offset_np, dtype=tf.float32)
        mean = tf.constant(mean_np, dtype=tf.float32)
        variance = tf.constant(var_np, dtype=tf.float32)
        return x, scale, offset, mean, variance

    def _compare_forward(self, shape, data_format, dtype, rtol, atol):
        x, scale, offset, mean, variance = self._build_bn_inputs(
            shape, data_format, dtype
        )

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
                is_training=True,
            )
            return y_raw[0]

        self._compare_cpu_musa_results(
            fused_batch_norm_wrapper,
            [x, scale, offset, mean, variance],
            dtype,
            rtol=rtol,
            atol=atol,
        )

    def test_forward_nhwc_small(self):
        self._compare_forward(
            shape=[2, 2, 2, 4],
            data_format="NHWC",
            dtype=tf.float32,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_gradient_dx_nhwc_small(self):
        shape = [2, 2, 2, 4]
        np.random.seed(42)
        x_np = np.random.randn(*shape).astype(np.float32)
        scale_np = np.random.rand(shape[-1]).astype(np.float32)
        offset_np = np.random.rand(shape[-1]).astype(np.float32)
        mean_np = np.zeros(shape[-1], dtype=np.float32)
        var_np = np.ones(shape[-1], dtype=np.float32)

        def grad_dx_op(x, scale, offset, mean, var):
            x = tf.convert_to_tensor(x)
            with tf.GradientTape() as tape:
                tape.watch(x)
                y, _, _, _, _, _ = tf.raw_ops.FusedBatchNormV3(
                    x=x,
                    scale=scale,
                    offset=offset,
                    mean=mean,
                    variance=var,
                    epsilon=0.001,
                    exponential_avg_factor=1.0,
                    data_format="NHWC",
                    is_training=True,
                )
                loss = tf.reduce_sum(y)
            return tape.gradient(loss, x)

        self._compare_cpu_musa_results(
            grad_dx_op,
            [x_np, scale_np, offset_np, mean_np, var_np],
            tf.float32,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_forward_nchw(self):
        self._compare_forward(
            shape=[2, 32, 1, 1],
            data_format="NCHW",
            dtype=tf.float32,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_forward_nhwc(self):
        self._compare_forward(
            shape=[2, 1, 1, 32],
            data_format="NHWC",
            dtype=tf.float32,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_forward_nchw_float16(self):
        self._compare_forward(
            shape=[4, 16, 8, 8],
            data_format="NCHW",
            dtype=tf.float16,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_forward_nchw_various_shapes(self):
        test_shapes = [
            [1, 64, 32, 32],
            [8, 128, 16, 16],
            [2, 3, 224, 224],
        ]
        for shape in test_shapes:
            with self.subTest(shape=shape):
                self._compare_forward(
                    shape=shape,
                    data_format="NCHW",
                    dtype=tf.float32,
                    rtol=1e-4,
                    atol=1e-4,
                )


if __name__ == "__main__":
    tf.test.main()
