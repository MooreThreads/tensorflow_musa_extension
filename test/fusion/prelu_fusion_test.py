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
"""End-to-end test for PRelu fusion optimization.

This test verifies that:
1. The MUSA custom graph optimizer is triggered
2. The PRelu pattern is correctly matched:
   AddV2
   ├── Relu1 -> Select
   └── Mul
       ├── Const (alpha)
       └── Relu2 -> Neg2 -> Select (same as Relu1's input)
3. The fused MusaPRelu kernel is called during execution
4. Results are numerically correct compared to standard TF ops on CPU

PRelu semantics (from the pattern):
    output = AddV2(Relu(Select), Mul(Const, Relu(Neg(Select))))
           = relu(x) + alpha * relu(-x)
           = max(0, x) + alpha * max(0, -x)
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


def prelu_pattern_numpy(x, alpha):
    """NumPy reference implementation matching the fusion pattern.

    Pattern output: relu(x) + alpha * relu(-x)
    This equals: max(0, x) + alpha * max(0, -x)

    Args:
        x: input tensor of any shape
        alpha: alpha tensor with same shape as x (for channel-wise PRelu)
    """
    relu_x = np.maximum(0, x)       # Relu1(Select) = relu(x)
    relu_neg_x = np.maximum(0, -x)  # Relu2(Neg2(Select)) = relu(-x)
    return relu_x + alpha * relu_neg_x


def build_prelu_fusion_graph(input_tensor, alpha_array):
    """Build the exact graph pattern that matches PRelu fusion.

    Pattern structure:
        Select
        ├──→ Relu1 ──→ AddV2
        └──→ Neg2 → Relu2 → Mul (with Const alpha) ──→ AddV2

    The Select node acts as identity (passes input through).

    Args:
        input_tensor: TF placeholder or tensor
        alpha_array: numpy array with same shape as input_tensor (excluding batch dim)
                     or same full shape as input_tensor
    """
    # alpha as a Const node (input 0 of Mul)
    alpha = tf.constant(alpha_array, dtype=tf.float32, name="alpha")

    # Select node: identity-like behavior
    # Select(cond, then, else) returns then where cond is true, else otherwise
    zeros = tf.zeros_like(input_tensor, name="zeros")
    cond = tf.greater(input_tensor, zeros, name="cond")
    select = tf.where(cond, input_tensor, input_tensor, name="select")

    # Branch 1: Relu1(Select) -> goes to AddV2 input 0
    relu1 = tf.nn.relu(select, name="relu1")

    # Branch 2: Neg2(Select) -> Relu2 -> Mul with alpha
    neg2 = tf.negative(select, name="neg2")
    relu2 = tf.nn.relu(neg2, name="relu2")

    # Mul: Const(alpha) * Relu2
    mul = tf.multiply(alpha, relu2, name="mul")

    # AddV2: Relu1 + Mul
    addv2 = tf.add(relu1, mul, name="addv2")

    return addv2


class PReluFusionE2ETest(MUSATestCase):
    """End-to-end test for PRelu fusion."""

    def test_prelu_fusion_basic(self):
        """Test basic PRelu fusion with typical dimensions."""
        print("\n" + "=" * 70)
        print("Test: PRelu Fusion - Basic")
        print("=" * 70)

        batch_size = 4
        input_dim = 128

        np.random.seed(42)
        x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
        # alpha is a 1D tensor (channel-wise PRelu)
        alpha_np = np.full(input_dim, 0.25, dtype=np.float32)

        print(f"\n  Input shape: {x_np.shape}")
        print(f"  Alpha shape: {alpha_np.shape}")

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, input_dim],
                    name="input"
                )

                # Pre-compute node
                scale = tf.constant(0.5, dtype=tf.float32, name="scale")
                pre_computed = tf.multiply(x, scale, name="pre_compute")

                # PRelu fusion pattern
                prelu_out = build_prelu_fusion_graph(pre_computed, alpha_np)

                # Post-compute node
                bias = tf.constant(1.0, dtype=tf.float32, name="bias")
                output = tf.add(prelu_out, bias, name="post_compute")

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np})

        # NumPy reference
        x_scaled = x_np * 0.5
        expected = prelu_pattern_numpy(x_scaled, alpha_np) + 1.0

        print(f"  Output shape: {result.shape}")
        print(f"  Expected shape: {expected.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        print("\n" + "=" * 70)
        print("✓ PRelu basic fusion test passed")
        print("=" * 70 + "\n")

    def test_prelu_fusion_small(self):
        """Test PRelu fusion with small dimensions for easy debugging."""
        print("\n" + "=" * 70)
        print("Test: PRelu Fusion - Small (Debug)")
        print("=" * 70)

        batch_size = 2
        input_dim = 4

        np.random.seed(0)
        # Create input with both positive and negative values
        x_np = np.array([[-2.0, -1.0, 0.0, 1.0],
                         [2.0, -0.5, 0.5, -2.0]], dtype=np.float32)
        # alpha is a 1D tensor
        alpha_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        print(f"\n  Input:\n{x_np}")
        print(f"  Alpha: {alpha_np}")

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, input_dim],
                    name="input"
                )

                # PRelu fusion pattern directly
                output = build_prelu_fusion_graph(x, alpha_np)

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np})

        expected = prelu_pattern_numpy(x_np, alpha_np)

        print(f"\n  Output:\n{result}")
        print(f"  Expected:\n{expected}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        print("\n" + "=" * 70)
        print("✓ PRelu small fusion test passed")
        print("=" * 70 + "\n")

    def test_prelu_fusion_is_applied(self):
        """Verify that the fusion IS applied: MusaPRelu node exists in optimized graph."""
        print("\n" + "=" * 70)
        print("Test: PRelu Fusion - Verify Fusion Applied")
        print("=" * 70)

        batch_size = 4
        input_dim = 32

        np.random.seed(42)
        x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
        alpha_np = np.full(input_dim, 0.15, dtype=np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, input_dim],
                    name="input"
                )

                output = build_prelu_fusion_graph(x, alpha_np)

        config = create_config_with_musa_optimizer()

        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np},
                              options=run_options,
                              run_metadata=run_metadata)

        # Check the OPTIMIZED graph (partition_graphs)
        has_fused_node = False
        fused_node_name = None
        all_ops = set()

        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                all_ops.add(node.op)
                if node.op == "MusaPRelu":
                    has_fused_node = True
                    fused_node_name = node.name

        print(f"\n  Op types in optimized graph: {sorted(all_ops)}")
        print(f"  MusaPRelu node found: {has_fused_node}")
        if fused_node_name:
            print(f"  Fused node name: {fused_node_name}")

        expected = prelu_pattern_numpy(x_np, alpha_np)
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(has_fused_node,
                        "MusaPRelu fusion was NOT applied to the graph")

        print("\n" + "=" * 70)
        print("✓ PRelu fusion-applied verification passed")
        print("=" * 70 + "\n")


    def test_prelu_fusion_with_negative_values(self):
        """Test PRelu fusion with predominantly negative input values."""
        print("\n" + "=" * 70)
        print("Test: PRelu Fusion - Negative Input Values")
        print("=" * 70)

        batch_size = 4
        input_dim = 32

        np.random.seed(123)
        # Generate mostly negative values
        x_np = (np.random.randn(batch_size, input_dim) - 2.0).astype(np.float32)
        alpha_np = np.full(input_dim, 0.3, dtype=np.float32)

        print(f"\n  Input range: [{x_np.min():.2f}, {x_np.max():.2f}]")
        print(f"  Alpha shape: {alpha_np.shape}")

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, input_dim],
                    name="input"
                )

                output = build_prelu_fusion_graph(x, alpha_np)

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np})

        expected = prelu_pattern_numpy(x_np, alpha_np)

        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        print("\n" + "=" * 70)
        print("✓ PRelu negative values test passed")
        print("=" * 70 + "\n")

    def test_prelu_fusion_random_alpha(self):
        """Test PRelu fusion with random alpha values per channel."""
        print("\n" + "=" * 70)
        print("Test: PRelu Fusion - Random Alpha Per Channel")
        print("=" * 70)

        batch_size = 8
        input_dim = 64

        np.random.seed(456)
        x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
        # Random alpha values between 0.1 and 0.5
        alpha_np = np.random.uniform(0.1, 0.5, size=input_dim).astype(np.float32)

        print(f"\n  Input shape: {x_np.shape}")
        print(f"  Alpha range: [{alpha_np.min():.3f}, {alpha_np.max():.3f}]")

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, input_dim],
                    name="input"
                )

                output = build_prelu_fusion_graph(x, alpha_np)

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np})

        expected = prelu_pattern_numpy(x_np, alpha_np)

        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        print("\n" + "=" * 70)
        print("✓ PRelu random alpha test passed")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    tf.test.main()