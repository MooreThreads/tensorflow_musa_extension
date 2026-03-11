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
"""End-to-end test for TokenMixer fusion optimization.

This test verifies that:
1. The MUSA custom graph optimizer is triggered
2. The Reshape->Transpose->Reshape pattern is correctly matched
3. The fused MusaTokenMixer kernel is called during execution
4. Results are numerically correct compared to standard TF ops on CPU

TokenMixer semantics (Python):
    x = tf.reshape(x, (-1, num_T, num_H, d_k))   # (B,T,D) -> (B,T,H,d_k)
    x = tf.transpose(x, perm=[0,2,1,3])           # (B,T,H,d_k) -> (B,H,T,d_k)
    x = tf.reshape(x, (-1, num_H, num_T * d_k))   # (B,H,T,d_k) -> (B,H,T*d_k)
"""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


def create_config_with_musa_optimizer():
    """Create ConfigProto with MUSA optimizer enabled."""
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rewriter_config = config.graph_options.rewrite_options

    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"

    rewriter_config.min_graph_nodes = -1
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])

    return config


def token_mixer_numpy(x, num_T, num_H, d_k):
    """NumPy reference implementation of TokenMixer."""
    B = x.shape[0]
    x = x.reshape(B, num_T, num_H, d_k)
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(B, num_H, num_T * d_k)
    return x


class TokenMixerFusionE2ETest(MUSATestCase):
    """End-to-end test for TokenMixer fusion."""

    def test_tokenmixer_fusion_basic(self):
        """Test basic TokenMixer fusion with typical dimensions."""
        print("\n" + "=" * 70)
        print("Test: TokenMixer Fusion - Basic")
        print("=" * 70)

        batch_size = 4
        num_T = 128
        num_H = 8
        d_k = 64
        num_D = num_H * d_k  # 512

        np.random.seed(42)
        x_np = np.random.randn(batch_size, num_T, num_D).astype(np.float32)

        print(f"\n  Input shape: {x_np.shape}")
        print(f"  num_T={num_T}, num_H={num_H}, d_k={d_k}")

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, num_T, num_D],
                    name="input"
                )

                # 前置计算节点：element-wise 乘以一个常量缩放因子
                scale = tf.constant(0.5, dtype=tf.float32, name="scale")
                pre_computed = tf.multiply(x, scale, name="pre_compute")

                # TokenMixer pattern: Reshape -> Transpose -> Reshape
                reshaped1 = tf.reshape(pre_computed, [-1, num_T, num_H, d_k],
                                    name="reshape1")
                transposed = tf.transpose(reshaped1, perm=[0, 2, 1, 3],
                                        name="transpose")
                reshaped2 = tf.reshape(transposed, [-1, num_H, num_T * d_k],
                                    name="reshape2")

                # 后置计算节点：加一个 bias
                bias = tf.constant(1.0, dtype=tf.float32, name="bias")
                output = tf.add(reshaped2, bias, name="post_compute")

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np})

        # NumPy reference
        x_scaled = x_np * 0.5
        expected = token_mixer_numpy(x_scaled, num_T, num_H, d_k)
        expected = expected + 1.0

        print(f"  Output shape: {result.shape}")
        print(f"  Expected shape: {expected.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        print("\n" + "=" * 70)
        print("✓ TokenMixer basic fusion test passed")
        print("=" * 70 + "\n")

    def test_tokenmixer_fusion_small(self):
        """Test TokenMixer fusion with small dimensions for easy debugging."""
        print("\n" + "=" * 70)
        print("Test: TokenMixer Fusion - Small (Debug)")
        print("=" * 70)

        batch_size = 2
        num_T = 4
        num_H = 2
        d_k = 3
        num_D = num_H * d_k  # 6

        np.random.seed(0)
        x_np = np.arange(batch_size * num_T * num_D,
                         dtype=np.float32).reshape(batch_size, num_T, num_D)

        print(f"\n  Input shape: {x_np.shape}")
        print(f"  num_T={num_T}, num_H={num_H}, d_k={d_k}")

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, num_T, num_D],
                    name="input"
                )

                # 前置计算节点：element-wise 乘以一个常量缩放因子
                scale = tf.constant(0.5, dtype=tf.float32, name="scale")
                pre_computed = tf.multiply(x, scale, name="pre_compute")

                # TokenMixer pattern: Reshape -> Transpose -> Reshape
                reshaped1 = tf.reshape(pre_computed, [-1, num_T, num_H, d_k],
                                    name="reshape1")
                transposed = tf.transpose(reshaped1, perm=[0, 2, 1, 3],
                                        name="transpose")
                reshaped2 = tf.reshape(transposed, [-1, num_H, num_T * d_k],
                                    name="reshape2")

                # 后置计算节点：加一个 bias
                bias = tf.constant(1.0, dtype=tf.float32, name="bias")
                output = tf.add(reshaped2, bias, name="post_compute")

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np})

        x_scaled = x_np * 0.5
        expected = token_mixer_numpy(x_scaled, num_T, num_H, d_k)
        expected = expected + 1.0

        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        print("\n" + "=" * 70)
        print("✓ TokenMixer small fusion test passed")
        print("=" * 70 + "\n")

    def test_tokenmixer_fusion_batch1(self):
        """Test TokenMixer fusion with batch_size=1."""
        print("\n" + "=" * 70)
        print("Test: TokenMixer Fusion - Batch Size 1")
        print("=" * 70)

        batch_size = 1
        num_T = 64
        num_H = 4
        d_k = 32
        num_D = num_H * d_k

        np.random.seed(123)
        x_np = np.random.randn(batch_size, num_T, num_D).astype(np.float32)

        print(f"\n  Input shape: {x_np.shape}")

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, num_T, num_D],
                    name="input"
                )

                # 前置计算节点：element-wise 乘以一个常量缩放因子
                scale = tf.constant(0.5, dtype=tf.float32, name="scale")
                pre_computed = tf.multiply(x, scale, name="pre_compute")

                # TokenMixer pattern: Reshape -> Transpose -> Reshape
                reshaped1 = tf.reshape(pre_computed, [-1, num_T, num_H, d_k],
                                    name="reshape1")
                transposed = tf.transpose(reshaped1, perm=[0, 2, 1, 3],
                                        name="transpose")
                reshaped2 = tf.reshape(transposed, [-1, num_H, num_T * d_k],
                                    name="reshape2")

                # 后置计算节点：加一个 bias
                bias = tf.constant(1.0, dtype=tf.float32, name="bias")
                output = tf.add(reshaped2, bias, name="post_compute")
        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np})

        x_scaled = x_np * 0.5
        expected = token_mixer_numpy(x_scaled, num_T, num_H, d_k)
        expected = expected + 1.0

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        print("\n" + "=" * 70)
        print("✓ TokenMixer batch=1 fusion test passed")
        print("=" * 70 + "\n")

    def test_tokenmixer_fusion_is_applied(self):
        """Verify that the fusion IS applied: MusaTokenMixer node exists in optimized graph."""
        print("\n" + "=" * 70)
        print("Test: TokenMixer Fusion - Verify Fusion Applied")
        print("=" * 70)

        batch_size = 2
        num_T = 8
        num_H = 2
        d_k = 4
        num_D = num_H * d_k

        np.random.seed(42)
        x_np = np.random.randn(batch_size, num_T, num_D).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, num_T, num_D],
                    name="input"
                )

                # 前置计算节点：element-wise 乘以一个常量缩放因子
                scale = tf.constant(0.5, dtype=tf.float32, name="scale")
                pre_computed = tf.multiply(x, scale, name="pre_compute")

                # TokenMixer pattern: Reshape -> Transpose -> Reshape
                reshaped1 = tf.reshape(pre_computed, [-1, num_T, num_H, d_k],
                                    name="reshape1")
                transposed = tf.transpose(reshaped1, perm=[0, 2, 1, 3],
                                        name="transpose")
                reshaped2 = tf.reshape(transposed, [-1, num_H, num_T * d_k],
                                    name="reshape2")

                # 后置计算节点：加一个 bias
                bias = tf.constant(1.0, dtype=tf.float32, name="bias")
                output = tf.add(reshaped2, bias, name="post_compute")

        config = create_config_with_musa_optimizer()

        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np},
                              options=run_options,
                              run_metadata=run_metadata)

        # Check the OPTIMIZED graph (partition_graphs), not sess.graph_def
        has_fused_node = False
        fused_node_name = None
        all_ops = set()

        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                all_ops.add(node.op)
                if node.op == "MusaTokenMixer":
                    has_fused_node = True
                    fused_node_name = node.name

        print(f"\n  Op types in optimized graph: {sorted(all_ops)}")
        print(f"  MusaTokenMixer node found: {has_fused_node}")
        if fused_node_name:
            print(f"  Fused node name: {fused_node_name}")

        x_scaled = x_np * 0.5
        expected = token_mixer_numpy(x_scaled, num_T, num_H, d_k)
        expected = expected + 1.0
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(has_fused_node,
                        "MusaTokenMixer fusion was NOT applied to the graph")

        print("\n" + "=" * 70)
        print("✓ TokenMixer fusion-applied verification passed")
        print("=" * 70 + "\n")

    def test_tokenmixer_no_fusion_wrong_perm(self):
        """Verify that wrong transpose perm does NOT trigger fusion."""
        print("\n" + "=" * 70)
        print("Test: TokenMixer - No Fusion With Wrong Perm")
        print("=" * 70)

        batch_size = 2
        num_T = 8
        num_H = 2
        d_k = 4
        num_D = num_H * d_k

        np.random.seed(99)
        x_np = np.random.randn(batch_size, num_T, num_D).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, num_T, num_D],
                    name="input"
                )

                # 前置计算节点：element-wise 乘以一个常量缩放因子
                scale = tf.constant(0.5, dtype=tf.float32, name="scale")
                pre_computed = tf.multiply(x, scale, name="pre_compute")

                # TokenMixer pattern: Reshape -> Transpose -> Reshape
                reshaped1 = tf.reshape(pre_computed, [-1, num_T, num_H, d_k],
                                    name="reshape1")
                transposed = tf.transpose(reshaped1, perm=[0, 1, 3, 2],    # 错误 perm
                                        name="transpose")
                reshaped2 = tf.reshape(transposed, [-1, num_T, d_k * num_H],  # 对应错误 perm 的输出 shape
                                    name="reshape2")

                # 后置计算节点：加一个 bias
                bias = tf.constant(1.0, dtype=tf.float32, name="bias")
                output = tf.add(reshaped2, bias, name="post_compute")

        config = create_config_with_musa_optimizer()

        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np},
                              options=run_options,
                              run_metadata=run_metadata)

        has_fused_node = any(
            node.op == "MusaTokenMixer"
            for pg in run_metadata.partition_graphs
            for node in pg.node
        )

        print(f"\n  MusaTokenMixer node found: {has_fused_node}")

        B = x_np.shape[0]
        x_scaled = x_np * 0.5
        expected = x_scaled.reshape(B, num_T, num_H, d_k)
        expected = expected.transpose(0, 1, 3, 2)         # 和图中保持一致
        expected = expected.reshape(B, num_T, d_k * num_H)
        expected = expected + 1.0

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        self.assertFalse(has_fused_node,
                         "MusaTokenMixer fusion should NOT have been applied "
                         "for perm=[0,1,3,2]")

        print("\n" + "=" * 70)
        print("✓ TokenMixer wrong-perm test passed (no fusion, correct result)")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    tf.test.main()