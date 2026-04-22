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

import numpy as np
import tensorflow as tf

from musa_test_utils import load_musa_plugin

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_PATH = os.path.normpath(
    os.path.join(_TEST_DIR, "..", "..", "build", "libmusa_plugin.so")
)
PLUGIN_OPS = tf.load_op_library(_PLUGIN_PATH)
tf.compat.v1.disable_eager_execution()


class ReshapeMatMulOpTest(tf.test.TestCase):
    """Functional tests for MusaReshapeMatMul."""

    def _run_graph(self, x_np, w_np, transpose_b=False):
        graph = tf.Graph()
        with graph.as_default():
            x = tf.constant(x_np)
            w = tf.constant(w_np)
            with tf.device("/device:MUSA:0"):
                actual = PLUGIN_OPS.musa_reshape_mat_mul(
                    x=x, w=w, transpose_b=transpose_b
                )
        with tf.compat.v1.Session(graph=graph) as sess:
            return sess.run(actual)

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
        self.assertAllClose(expected, actual, rtol=1e-5, atol=1e-6)

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
        self.assertAllClose(expected, actual, rtol=1e-5, atol=1e-6)

    def test_invalid_dim_mismatch(self):
        x_np = np.random.randn(2, 4, 7).astype(np.float32)
        w_np = np.random.randn(8, 16).astype(np.float32)

        with self.assertRaises(Exception):
            _ = self._run_graph(x_np, w_np)


if __name__ == "__main__":
    tf.test.main()
