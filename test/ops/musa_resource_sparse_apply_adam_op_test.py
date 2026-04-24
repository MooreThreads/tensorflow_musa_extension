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

"""Tests for MUSA ResourceSparseApplyAdam operator."""

import os
import sys

# Add the test directory to path for importing musa_test_utils
test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, test_dir)

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class MusaResourceSparseApplyAdamTest(MUSATestCase):
    """Tests for MUSA fused sparse Adam operator."""

    @classmethod
    def setUpClass(cls):
        """Load the custom op module via tf.load_op_library."""
        super(MusaResourceSparseApplyAdamTest, cls).setUpClass()

        plugin_path = None
        current_dir = os.path.dirname(os.path.abspath(__file__))

        candidate_paths = [
            os.path.join(current_dir, "..", "..", "build", "libmusa_plugin.so"),
            os.path.join(os.path.dirname(current_dir), "..", "build", "libmusa_plugin.so"),
            os.path.join(os.getcwd(), "build", "libmusa_plugin.so"),
        ]

        for path in candidate_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                plugin_path = normalized_path
                break

        if plugin_path and os.path.exists(plugin_path):
            try:
                cls._musa_ops = tf.load_op_library(plugin_path)
            except Exception as e:
                print(f"FAILED: Error loading MUSA ops from {plugin_path}: {e}")
                cls._musa_ops = None
        else:
            searched = [os.path.normpath(p) for p in candidate_paths]
            print("MUSA plugin not found. Searched:\n" + "\n".join(f"  - {loc}" for loc in searched))
            cls._musa_ops = None

        if cls._musa_ops is not None and not hasattr(cls._musa_ops, 'musa_resource_sparse_apply_adam'):
            print("MUSA ops module loaded but musa_resource_sparse_apply_adam not found")
            print("Available ops:", [op for op in dir(cls._musa_ops) if not op.startswith('_')])

    def _compute_expected(self, var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
                         epsilon_np, beta1_power_np, beta2_power_np,
                         grad_np, indices_np):
        """Compute expected Adam update using NumPy."""
        expected_var = var_np.astype(np.float32).copy()
        expected_m = m_np.astype(np.float32).copy()
        expected_v = v_np.astype(np.float32).copy()

        # Compute bias-corrected learning rate
        one_minus_beta1_power = 1.0 - beta1_power_np
        one_minus_beta2_power = 1.0 - beta2_power_np

        if abs(one_minus_beta1_power) < 1e-10:
            lr_t = lr_np  # Initial iteration fallback
        else:
            lr_t = lr_np * np.sqrt(one_minus_beta2_power) / one_minus_beta1_power

        one_minus_beta1 = 1.0 - beta1_np
        one_minus_beta2 = 1.0 - beta2_np

        for i, idx in enumerate(indices_np):
            if idx < 0:
                continue

            g = grad_np[i].astype(np.float32)

            # m_t = beta1 * m + (1 - beta1) * g
            expected_m[idx] = beta1_np * expected_m[idx] + one_minus_beta1 * g

            # v_t = beta2 * v + (1 - beta2) * g^2
            expected_v[idx] = beta2_np * expected_v[idx] + one_minus_beta2 * g * g

            # var = var - lr_t * m_t / (sqrt(v_t) + epsilon)
            v_sqrt = np.sqrt(expected_v[idx])
            expected_var[idx] = expected_var[idx] - lr_t * expected_m[idx] / (v_sqrt + epsilon_np)

        return expected_var, expected_m, expected_v

    def _test_logic(self, var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
                    epsilon_np, beta1_power_np, beta2_power_np,
                    grad_np, indices_np, dtype, index_dtype):
        """Test sparse Adam update on MUSA device."""
        if self._musa_ops is None or not hasattr(self._musa_ops, 'musa_resource_sparse_apply_adam'):
            self.skipTest("MUSA sparse Adam op not available")

        expected_var, expected_m, expected_v = self._compute_expected(
            var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
            epsilon_np, beta1_power_np, beta2_power_np,
            grad_np, indices_np)

        with tf.device("/device:MUSA:0"):
            var = tf.Variable(var_np, dtype=dtype)
            m = tf.Variable(m_np, dtype=dtype)
            v = tf.Variable(v_np, dtype=dtype)

            # Check if variables are actually placed on MUSA device
            # bfloat16 may not be supported on MUSA, causing fallback to CPU
            if 'MUSA' not in var.device:
                self.skipTest(f"{dtype} variables not supported on MUSA device (placed on {var.device})")

            # For bfloat16/half, use float32 values and cast to ensure device placement
            lr = tf.cast(tf.constant(lr_np, dtype=tf.float32), dtype)
            beta1 = tf.cast(tf.constant(beta1_np, dtype=tf.float32), dtype)
            beta2 = tf.cast(tf.constant(beta2_np, dtype=tf.float32), dtype)
            epsilon = tf.cast(tf.constant(epsilon_np, dtype=tf.float32), dtype)
            beta1_power = tf.cast(tf.constant(beta1_power_np, dtype=tf.float32), dtype)
            beta2_power = tf.cast(tf.constant(beta2_power_np, dtype=tf.float32), dtype)

            grad = tf.constant(grad_np, dtype=dtype)
            indices = tf.constant(indices_np, dtype=index_dtype)

            # Call the custom op via the ops module
            self._musa_ops.musa_resource_sparse_apply_adam(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                beta1_power=beta1_power,
                beta2_power=beta2_power,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                grad=grad,
                indices=indices,
                use_locking=False)

            out_var = var.read_value().numpy()
            out_m = m.read_value().numpy()
            out_v = v.read_value().numpy()

        # Using higher tolerance for half precision
        if dtype in [tf.float16, tf.bfloat16]:
            self.assertAllClose(expected_var, out_var, atol=1e-2, rtol=1e-2)
            self.assertAllClose(expected_m, out_m, atol=1e-2, rtol=1e-2)
            self.assertAllClose(expected_v, out_v, atol=1e-2, rtol=1e-2)
        else:
            self.assertAllClose(expected_var, out_var)
            self.assertAllClose(expected_m, out_m)
            self.assertAllClose(expected_v, out_v)

    def testBasic(self):
        """Test basic sparse Adam update."""
        # Note: bfloat16 is tested separately in testBasicBFloat16 as it may not
        # be supported on all MUSA devices
        for dtype in [tf.float32, tf.float16]:
            for index_dtype in [np.int32, np.int64]:
                # Simple 2-row, 2-column embedding
                var_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
                m_np = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
                v_np = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

                lr_np = 0.001
                beta1_np = 0.9
                beta2_np = 0.999
                epsilon_np = 1e-7
                beta1_power_np = 0.9  # First iteration
                beta2_power_np = 0.999

                # Update only row 1
                indices_np = np.array([1], dtype=index_dtype)
                grad_np = np.array([[0.1, 0.2]], dtype=np.float32)

                self._test_logic(
                    var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
                    epsilon_np, beta1_power_np, beta2_power_np,
                    grad_np, indices_np, dtype, index_dtype)

    def testBasicBFloat16(self):
        """Test bfloat16 sparse Adam update (may skip if device doesn't support bfloat16)."""
        for index_dtype in [np.int32, np.int64]:
            var_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            m_np = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
            v_np = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

            lr_np = 0.001
            beta1_np = 0.9
            beta2_np = 0.999
            epsilon_np = 1e-7
            beta1_power_np = 0.9
            beta2_power_np = 0.999

            indices_np = np.array([1], dtype=index_dtype)
            grad_np = np.array([[0.1, 0.2]], dtype=np.float32)

            self._test_logic(
                var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
                epsilon_np, beta1_power_np, beta2_power_np,
                grad_np, indices_np, tf.bfloat16, index_dtype)

    def testMultipleIndices(self):
        """Test updating multiple rows."""
        dtype = tf.float32

        # 5-row, 4-column embedding
        rows = 5
        cols = 4
        var_np = np.random.random([rows, cols]).astype(np.float32)
        m_np = np.zeros([rows, cols], dtype=np.float32)
        v_np = np.zeros([rows, cols], dtype=np.float32)

        lr_np = 0.001
        beta1_np = 0.9
        beta2_np = 0.999
        epsilon_np = 1e-7
        beta1_power_np = 0.81  # Second iteration (0.9^2)
        beta2_power_np = 0.998001  # Second iteration (0.999^2)

        # Update rows 0, 2, 4
        indices_np = np.array([0, 2, 4], dtype=np.int32)
        grad_np = np.random.random([3, cols]).astype(np.float32)

        self._test_logic(
            var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
            epsilon_np, beta1_power_np, beta2_power_np,
            grad_np, indices_np, dtype, np.int32)

    def testEmbeddingScenario(self):
        """Test with larger embedding-like dimensions."""
        for dtype in [tf.float32, tf.float16]:
            # Simulate word embedding: 10000 words, 128 dimensions
            vocab_size = 1000
            embedding_dim = 64
            var_np = np.random.random([vocab_size, embedding_dim]).astype(np.float32)
            m_np = np.zeros([vocab_size, embedding_dim], dtype=np.float32)
            v_np = np.zeros([vocab_size, embedding_dim], dtype=np.float32)

            lr_np = 0.001
            beta1_np = 0.9
            beta2_np = 0.999
            epsilon_np = 1e-7
            beta1_power_np = 0.9
            beta2_power_np = 0.999

            # Batch of 32 tokens
            batch_size = 32
            indices_np = np.random.choice(vocab_size, batch_size, replace=False).astype(np.int32)
            grad_np = np.random.random([batch_size, embedding_dim]).astype(np.float32)

            self._test_logic(
                var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
                epsilon_np, beta1_power_np, beta2_power_np,
                grad_np, indices_np, dtype, np.int32)

    def testEmptyIndices(self):
        """Test with empty indices (no-op)."""
        dtype = tf.float32
        var_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        m_np = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
        v_np = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32)

        lr_np = 0.001
        beta1_np = 0.9
        beta2_np = 0.999
        epsilon_np = 1e-7
        beta1_power_np = 0.9
        beta2_power_np = 0.999

        indices_np = np.array([], dtype=np.int32)
        grad_np = np.zeros([0, 2], dtype=np.float32)

        # Should not change anything
        self._test_logic(
            var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
            epsilon_np, beta1_power_np, beta2_power_np,
            grad_np, indices_np, dtype, np.int32)

    def testLargeRowsInt64Indices(self):
        """Test large rows with int64 indices."""
        dtype = tf.float32

        rows = 500
        cols = 32
        var_np = np.random.random([rows, cols]).astype(np.float32)
        m_np = np.zeros([rows, cols], dtype=np.float32)
        v_np = np.zeros([rows, cols], dtype=np.float32)

        lr_np = 0.001
        beta1_np = 0.9
        beta2_np = 0.999
        epsilon_np = 1e-7
        beta1_power_np = 0.9
        beta2_power_np = 0.999

        # Update first and last rows using int64 indices
        indices_np = np.array([0, rows - 1], dtype=np.int64)
        grad_np = np.random.random([2, cols]).astype(np.float32)

        self._test_logic(
            var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
            epsilon_np, beta1_power_np, beta2_power_np,
            grad_np, indices_np, dtype, np.int64)

    def testInitialIterationEdgeCase(self):
        """Test edge case when beta1_power ≈ 1.0 (initial iteration)."""
        dtype = tf.float32
        var_np = np.array([[1.0]], dtype=np.float32)
        m_np = np.array([[0.0]], dtype=np.float32)
        v_np = np.array([[0.0]], dtype=np.float32)

        lr_np = 0.001
        beta1_np = 0.9
        beta2_np = 0.999
        epsilon_np = 1e-7
        beta1_power_np = 1.0  # Initial iteration edge case
        beta2_power_np = 1.0

        indices_np = np.array([0], dtype=np.int32)
        grad_np = np.array([[0.5]], dtype=np.float32)

        self._test_logic(
            var_np, m_np, v_np, lr_np, beta1_np, beta2_np,
            epsilon_np, beta1_power_np, beta2_power_np,
            grad_np, indices_np, dtype, np.int32)


if __name__ == "__main__":
    tf.test.main()
