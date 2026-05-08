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

"""Tests for MusaReshapeMatMul operator."""

import os
os.environ.setdefault("MUSA_ENABLE_TF32", "0")

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase, load_musa_ops


def is_tf32_enabled():
    return int(os.environ.get("MUSA_ENABLE_TF32", "0")) != 0


def float32_tolerance(default_rtol=1e-5, default_atol=1e-6):
    return (1e-2, 1e-2) if is_tf32_enabled() else (default_rtol, default_atol)


class ReshapeMatMulOpTest(MUSATestCase):
    """Functional tests for MusaReshapeMatMul."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            cls._musa_ops = load_musa_ops()
        except Exception as e:
            print(f"FAILED: Error loading MUSA ops: {e}")
            cls._musa_ops = None

    def _run_graph(self, x_np, w_np, transpose_b=False):
        if self._musa_ops is None:
            self.skipTest("MUSA ops not available")
        x = tf.constant(x_np)
        w = tf.constant(w_np)
        with tf.device("/device:MUSA:0"):
            return self._musa_ops.musa_reshape_mat_mul(
                x=x, w=w, transpose_b=transpose_b
            ).numpy()

    def _run_reference(self, x_np, w_np, transpose_b=False):
        x_shape = list(x_np.shape)
        k = x_shape[-1]
        x2 = x_np.reshape(-1, k)
        w2 = w_np.T if transpose_b else w_np
        y2 = np.matmul(x2, w2)
        out_shape = x_shape[:-1] + [w2.shape[1]]
        return y2.reshape(out_shape)

    def test_basic_rank3_float32(self):
        x_np = np.random.randn(8, 4, 16).astype(np.float32)
        w_np = np.random.randn(16, 32).astype(np.float32)

        expected = self._run_reference(x_np, w_np)
        actual = self._run_graph(x_np, w_np)
        rtol, atol = float32_tolerance()
        self.assertAllClose(expected, actual, rtol=rtol, atol=atol)

    def test_rank4_float16(self):
        x_np = np.random.randn(2, 3, 4, 8).astype(np.float16)
        w_np = np.random.randn(8, 12).astype(np.float16)

        expected = self._run_reference(x_np, w_np).astype(np.float32)
        actual = self._run_graph(x_np, w_np).astype(np.float32)
        self.assertAllClose(expected, actual, rtol=1e-2, atol=1e-2)

    def test_transpose_b(self):
        x_np = np.random.randn(3, 5, 8).astype(np.float32)
        w_np = np.random.randn(12, 8).astype(np.float32)

        expected = self._run_reference(x_np, w_np, transpose_b=True)
        actual = self._run_graph(x_np, w_np, transpose_b=True)
        rtol, atol = float32_tolerance()
        self.assertAllClose(expected, actual, rtol=rtol, atol=atol)

    def test_invalid_dim_mismatch(self):
        x_np = np.random.randn(2, 4, 7).astype(np.float32)
        w_np = np.random.randn(8, 16).astype(np.float32)

        with self.assertRaises(Exception):
            _ = self._run_graph(x_np, w_np)


if __name__ == "__main__":
    tf.test.main()
