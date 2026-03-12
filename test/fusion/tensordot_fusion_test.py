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
"""End-to-end tests for TensorDot fusion optimization.

Verifies:
1. Graph fusion: tf.tensordot subgraph is replaced by MusaTensorDot node.
2. Numerical correctness: fused result matches np.tensordot reference.
3. Various axes combinations and shapes.
4. Negative test: non-Tensordot Reshape should not be fused.

IMPORTANT: tf.tensordot(name=...) generates internal nodes named
  <name>/Reshape, <name>/MatMul, <name>/Reshape_1, etc.
The C++ fusion pass (FindTensorDotPrefix) requires the output Reshape node's
name to contain "/Tensordot" (with a slash). So the `name` parameter MUST
be in the form "scope/Tensordot" to ensure correct prefix extraction.
"""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

# MUSA device float32 matmul accumulation differs from CPU double-precision
# numpy. For large reductions (hidden_size >= 512), max abs diff can reach
# ~0.1, so we use generous tolerances.
_RTOL = 5e-3
_ATOL = 5e-3

_RTOL_LARGE = 1e-2
_ATOL_LARGE = 1e-2


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


def _run_tensordot_graph(a_np, w_np, axes, scope="test"):
    """Build a tf.tensordot graph, run it with MUSA optimizer, return result
    and the optimized partition graphs for inspection.

    The tensordot op is named "<scope>/Tensordot" so that the generated
    internal nodes have the "/Tensordot" prefix required by the fusion pass.
    """
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            a = tf.compat.v1.placeholder(
                tf.float32, shape=a_np.shape, name=f"{scope}/input"
            )
            w = tf.constant(w_np, dtype=tf.float32, name=f"{scope}/weight")
            output = tf.tensordot(a, w, axes=axes, name=f"{scope}/Tensordot")

    config = create_config_with_musa_optimizer()

    run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
    run_metadata = tf.compat.v1.RunMetadata()

    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        result = sess.run(
            output,
            feed_dict={a: a_np},
            options=run_options,
            run_metadata=run_metadata,
        )

    return result, run_metadata.partition_graphs


def _has_fused_op(partition_graphs, op_name="MusaTensorDot"):
    """Check whether a fused op exists in any partition graph."""
    for pg in partition_graphs:
        for node in pg.node:
            if node.op == op_name:
                return True
    return False


def _count_fused_ops(partition_graphs, op_name="MusaTensorDot"):
    """Count fused ops across all partition graphs."""
    return sum(
        1
        for pg in partition_graphs
        for node in pg.node
        if node.op == op_name
    )


# =============================================================================
# Test class
# =============================================================================

class TensorDotFusionTest(MUSATestCase):
    """End-to-end tests for MusaTensorDot fusion."""

    # -----------------------------------------------------------------
    # 1. Basic: axes=([-1],[0]) — 最常见的 tensordot 用法
    # -----------------------------------------------------------------
    def test_basic_last_first_axes(self):
        """axes=([-1],[0]): contract last dim of A with first dim of B."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — basic axes=([-1],[0])")
        print("=" * 70)

        np.random.seed(42)
        a_np = np.random.randn(4, 128, 768).astype(np.float32)
        w_np = np.random.randn(768, 64).astype(np.float32)

        result, pgs = _run_tensordot_graph(a_np, w_np, axes=([-1], [0]), scope="basic")

        expected = np.tensordot(a_np, w_np, axes=([-1], [0]))
        print(f"  A shape: {a_np.shape}, W shape: {w_np.shape}")
        print(f"  Output shape: {result.shape}, Expected shape: {expected.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)

        fused = _has_fused_op(pgs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertTrue(fused, "MusaTensorDot node not found in optimized graph")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 2. 2-D inputs: 矩阵乘法退化形式
    # -----------------------------------------------------------------
    def test_2d_matmul_degenerate(self):
        """axes=([1],[0]) on 2-D tensors — equivalent to matmul."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — 2D matmul degenerate")
        print("=" * 70)

        np.random.seed(7)
        a_np = np.random.randn(32, 64).astype(np.float32)
        w_np = np.random.randn(64, 16).astype(np.float32)

        result, pgs = _run_tensordot_graph(a_np, w_np, axes=([1], [0]), scope="mm")

        expected = np.tensordot(a_np, w_np, axes=([1], [0]))
        print(f"  A shape: {a_np.shape}, W shape: {w_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)

        fused = _has_fused_op(pgs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertTrue(fused, "MusaTensorDot node not found in optimized graph")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 3. 非最后维收缩: axes=([1],[0]) on 3-D — 需要内部 transpose
    # -----------------------------------------------------------------
    def test_contract_middle_axis(self):
        """axes=([1],[0]) on 3-D A: contracts middle dim, requires internal transpose."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — contract middle axis")
        print("=" * 70)

        np.random.seed(11)
        a_np = np.random.randn(4, 3, 256).astype(np.float32)
        w_np = np.random.randn(3, 32).astype(np.float32)

        result, pgs = _run_tensordot_graph(a_np, w_np, axes=([1], [0]), scope="mid")

        expected = np.tensordot(a_np, w_np, axes=([1], [0]))
        print(f"  A shape: {a_np.shape}, W shape: {w_np.shape}")
        print(f"  Output shape: {result.shape}, Expected: {expected.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)

        fused = _has_fused_op(pgs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertTrue(fused, "MusaTensorDot node not found in optimized graph")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 4. batch=1 小尺寸，方便 debug
    # -----------------------------------------------------------------
    def test_small_batch1(self):
        """Small dimensions with batch_size=1 for easy debugging."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — small batch=1")
        print("=" * 70)

        np.random.seed(0)
        a_np = np.arange(1 * 4 * 6, dtype=np.float32).reshape(1, 4, 6)
        w_np = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)

        result, pgs = _run_tensordot_graph(a_np, w_np, axes=([-1], [0]), scope="s1")

        expected = np.tensordot(a_np, w_np, axes=([-1], [0]))
        print(f"  A shape: {a_np.shape}, W shape: {w_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)

        fused = _has_fused_op(pgs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertTrue(fused, "MusaTensorDot node not found in optimized graph")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 5. 大 batch + 大 hidden — 贴近真实模型
    # -----------------------------------------------------------------
    def test_large_realistic(self):
        """Realistic dimensions close to production model shapes."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — large realistic")
        print("=" * 70)

        np.random.seed(99)
        a_np = np.random.randn(16, 64, 512).astype(np.float32)
        w_np = np.random.randn(512, 256).astype(np.float32)

        result, pgs = _run_tensordot_graph(a_np, w_np, axes=([-1], [0]), scope="lg")

        expected = np.tensordot(a_np, w_np, axes=([-1], [0]))
        print(f"  A shape: {a_np.shape}, W shape: {w_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)

        fused = _has_fused_op(pgs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertTrue(fused, "MusaTensorDot node not found in optimized graph")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 6. 多个 tensordot 串联 — 验证多次融合
    # -----------------------------------------------------------------
    def test_chained_tensordots(self):
        """Two consecutive tensordots — both should be fused."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — chained tensordots")
        print("=" * 70)

        np.random.seed(55)
        a_np = np.random.randn(4, 32, 128).astype(np.float32)
        w1_np = np.random.randn(128, 64).astype(np.float32)
        w2_np = np.random.randn(64, 16).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                a = tf.compat.v1.placeholder(
                    tf.float32, shape=a_np.shape, name="chain/input")
                w1 = tf.constant(w1_np, dtype=tf.float32, name="chain/w1")
                w2 = tf.constant(w2_np, dtype=tf.float32, name="chain/w2")

                mid = tf.tensordot(a, w1, axes=([-1], [0]), name="chain/Tensordot_1")
                out = tf.tensordot(mid, w2, axes=([-1], [0]), name="chain/Tensordot_2")

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(out, feed_dict={a: a_np},
                              options=run_options, run_metadata=run_metadata)

        expected = np.tensordot(
            np.tensordot(a_np, w1_np, axes=([-1], [0])),
            w2_np, axes=([-1], [0]),
        )
        print(f"  A shape: {a_np.shape}")
        print(f"  W1: {w1_np.shape}, W2: {w2_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)

        fused_count = _count_fused_ops(run_metadata.partition_graphs)
        print(f"  MusaTensorDot count: {fused_count}")
        self.assertGreaterEqual(fused_count, 2,
                                f"Expected >=2 MusaTensorDot nodes, got {fused_count}")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 7. 有前置/后置运算 — 确保融合不影响上下游
    # -----------------------------------------------------------------
    def test_with_pre_and_post_ops(self):
        """TensorDot surrounded by pre/post element-wise ops."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — with pre & post ops")
        print("=" * 70)

        np.random.seed(77)
        a_np = np.random.randn(4, 16, 64).astype(np.float32)
        w_np = np.random.randn(64, 32).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                a = tf.compat.v1.placeholder(
                    tf.float32, shape=a_np.shape, name="pp/input")
                w = tf.constant(w_np, dtype=tf.float32, name="pp/weight")

                scaled = tf.multiply(a, 0.5, name="pp/pre_scale")
                td = tf.tensordot(scaled, w, axes=([-1], [0]), name="pp/Tensordot")
                output = tf.add(td, 1.0, name="pp/post_bias")

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={a: a_np},
                              options=run_options, run_metadata=run_metadata)

        expected = np.tensordot(a_np * 0.5, w_np, axes=([-1], [0])) + 1.0
        print(f"  A shape: {a_np.shape}, W shape: {w_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)

        fused = _has_fused_op(run_metadata.partition_graphs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertTrue(fused, "MusaTensorDot node not found in optimized graph")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 8. 负例: 普通 Reshape + MatMul 不应该触发融合
    # -----------------------------------------------------------------
    def test_no_fusion_plain_matmul(self):
        """Plain Reshape+MatMul (not from tf.tensordot) should NOT be fused."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — negative: plain matmul no fusion")
        print("=" * 70)

        np.random.seed(33)
        a_np = np.random.randn(4, 16, 64).astype(np.float32)
        w_np = np.random.randn(64, 32).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                a = tf.compat.v1.placeholder(
                    tf.float32, shape=a_np.shape, name="neg_input")
                w = tf.constant(w_np, dtype=tf.float32, name="neg_weight")

                flat = tf.reshape(a, [-1, 64], name="neg_flatten")
                mm = tf.matmul(flat, w, name="neg_matmul")
                output = tf.reshape(mm, [4, 16, 32], name="neg_unflatten")

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={a: a_np},
                              options=run_options, run_metadata=run_metadata)

        expected = (a_np.reshape(-1, 64) @ w_np).reshape(4, 16, 32)
        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)

        fused = _has_fused_op(run_metadata.partition_graphs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertFalse(fused,
                         "MusaTensorDot should NOT appear for plain Reshape+MatMul")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 9. 4-D tensor: axes=([3],[0]) — 高维输入
    # -----------------------------------------------------------------
    def test_4d_input(self):
        """4-D input tensor with axes=([3],[0])."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — 4D input")
        print("=" * 70)

        np.random.seed(88)
        a_np = np.random.randn(2, 4, 8, 16).astype(np.float32)
        w_np = np.random.randn(16, 6).astype(np.float32)

        result, pgs = _run_tensordot_graph(a_np, w_np, axes=([3], [0]), scope="d4")

        expected = np.tensordot(a_np, w_np, axes=([3], [0]))
        print(f"  A shape: {a_np.shape}, W shape: {w_np.shape}")
        print(f"  Output shape: {result.shape}, Expected: {expected.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)

        fused = _has_fused_op(pgs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertTrue(fused, "MusaTensorDot node not found in optimized graph")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 10. 验证融合节点属性: axes_a / axes_b 值正确
    # -----------------------------------------------------------------
    def test_fusion_attrs_correct(self):
        """Check that the fused MusaTensorDot node carries correct axes attrs."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — verify axes attrs")
        print("=" * 70)

        np.random.seed(42)
        a_np = np.random.randn(4, 8, 16).astype(np.float32)
        w_np = np.random.randn(16, 6).astype(np.float32)

        _, pgs = _run_tensordot_graph(a_np, w_np, axes=([-1], [0]), scope="attr")

        fused_nodes = [
            node
            for pg in pgs
            for node in pg.node
            if node.op == "MusaTensorDot"
        ]

        self.assertTrue(fused_nodes, "No MusaTensorDot node found")

        node = fused_nodes[0]
        axes_a = list(node.attr["axes_a"].list.i)
        axes_b = list(node.attr["axes_b"].list.i)

        print(f"  Fused node: {node.name}")
        print(f"  axes_a = {axes_a}")
        print(f"  axes_b = {axes_b}")

        self.assertEqual(len(axes_a), 1, f"Expected 1 axis_a, got {axes_a}")
        self.assertIn(axes_a[0], [2, -1],
                      f"axes_a should be [2] or [-1], got {axes_a}")

        self.assertEqual(axes_b, [0], f"Expected axes_b=[0], got {axes_b}")

        print("  PASSED")


if __name__ == "__main__":
    tf.test.main()
