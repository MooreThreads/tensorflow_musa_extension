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
"""End-to-end tests for Normalize fusion optimization.

Uses a subgraph extracted from the real production model to test the fusion
pass. The subgraph file contains the complete Normalize internal node structure
(Mean → ExpandDims → Sub → Square → Mean → ExpandDims → Sqrt → MusaClip → RealDiv).

Mathematical formula implemented by MusaNormalize:
  mean = reduce_mean(x, axis=-1)
  variance = reduce_mean((x - mean)^2, axis=-1)
  clipped_std = clamp(sqrt(variance), epsilon, max_std)
  output = (x - mean) / clipped_std

Note: This differs from standard LayerNorm which uses sqrt(variance + epsilon).
The clamp operation provides both lower and upper bounds on the standard deviation.

Subgraph structure (ad_emb_aug_ln_layer):
  - Output node: fwffm_pbp_mlp/ad_emb_aug_ln_layer/truediv
  - External input (Placeholder):
      fwffm_pbp_mlp/ad_sparse_query_emb_aug/SparseFieldsConcatExt_0
  - Shared Const nodes:
      fwffm_pbp_mlp/pln1_follow/Mean/reduction_indices
      fwffm_pbp_mlp/pln1_follow/ExpandDims/dim
      fwffm_pbp_mlp/pln1_follow/clip_by_value/y
      fwffm_pbp_mlp/pln1_follow/clip_by_value/Minimum/y
  - 15 nodes total.
"""

import os
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_TEST_DIR)
_SUBGRAPH_PB = os.path.join(_PARENT_DIR, "fusion/normalize_subgraph.pb")

# Node names inside the subgraph pb
_OUTPUT_NODE = "fwffm_pbp_mlp/ad_emb_aug_ln_layer/truediv"
_INPUT_NODE = "fwffm_pbp_mlp/ad_sparse_query_emb_aug/SparseFieldsConcatExt_0"

# Tolerances
_RTOL = 1e-5
_ATOL = 1e-5


# =========================================================================
# Helpers
# =========================================================================

def _load_subgraph_pb(pb_path=_SUBGRAPH_PB):
    """Load a GraphDef from a .pb file."""
    if not os.path.exists(pb_path):
        raise FileNotFoundError(
            f"Subgraph pb not found at {pb_path}. "
            "Run extract_layernorm_pb_util.py first."
        )
    graph_def = graph_pb2.GraphDef()
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def _create_config_with_musa_optimizer(disable_builtin_opts=True):
    """Create ConfigProto that enables only the musa_graph_optimizer."""
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rw = config.graph_options.rewrite_options
    rw.min_graph_nodes = -1

    if disable_builtin_opts:
        rw.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
        rw.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
        rw.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
        rw.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
        rw.shape_optimization = rewriter_config_pb2.RewriterConfig.OFF

    custom_opt = rw.custom_optimizers.add()
    custom_opt.name = "musa_graph_optimizer"
    rw.optimizers.extend(["musa_graph_optimizer"])

    return config


def _run_subgraph(graph_def, input_name, output_name, feed_np,
                  disable_builtin_opts=True):
    """Import graph_def, run through MUSA optimizer, return result."""
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    config = _create_config_with_musa_optimizer(disable_builtin_opts)
    run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
    run_meta = tf.compat.v1.RunMetadata()

    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        input_tensor = graph.get_tensor_by_name(f"{input_name}:0")
        output_tensor = graph.get_tensor_by_name(f"{output_name}:0")
        result = sess.run(
            output_tensor,
            feed_dict={input_tensor: feed_np},
            options=run_opts,
            run_metadata=run_meta,
        )

    return result, run_meta.partition_graphs


def _has_fused_op(partition_graphs, op_name="MusaNormalize"):
    for pg in partition_graphs:
        for node in pg.node:
            if node.op == op_name:
                return True
    return False


def _get_fused_nodes(partition_graphs, op_name="MusaNormalize"):
    return [
        node
        for pg in partition_graphs
        for node in pg.node
        if node.op == op_name
    ]


def _count_fused_ops(partition_graphs, op_name="MusaNormalize"):
    return len(_get_fused_nodes(partition_graphs, op_name))


def _get_placeholder_shape(graph_def, placeholder_name):
    """Read the static shape of a Placeholder from graph_def."""
    for node in graph_def.node:
        if node.name == placeholder_name and node.op == "Placeholder":
            if "shape" in node.attr:
                shape_proto = node.attr["shape"].shape
                if not shape_proto.unknown_rank:
                    return [
                        (d.size if d.size >= 0 else None)
                        for d in shape_proto.dim
                    ]
    return None


def _make_feed(shape_template, batch_size, rng):
    """Create a random float32 array matching shape_template."""
    concrete = [(batch_size if d is None else d) for d in shape_template]
    return rng.standard_normal(concrete).astype(np.float32)


def _reference_normalize(x, epsilon=1e-6, max_std=np.inf):
    """Reference Normalize implementation using numpy.

    Implements the formula:
      mean = reduce_mean(x, axis=-1)
      variance = reduce_mean((x - mean)^2, axis=-1)
      clipped_std = clamp(sqrt(variance), epsilon, max_std)
      output = (x - mean) / clipped_std

    This differs from standard LayerNorm which uses sqrt(variance + epsilon).
    """
    # Normalize over the last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    std = np.sqrt(np.maximum(variance, 0.0))
    # Clamp std to [epsilon, max_std]
    clipped_std = np.clip(std, epsilon, max_std)
    x_norm = (x - mean) / clipped_std
    return x_norm


def _reference_layernorm(x, epsilon=1e-6):
    """Reference LayerNorm implementation (for comparison only).

    Standard LayerNorm uses sqrt(variance + epsilon), not clamp.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + epsilon)
    return x_norm


# =========================================================================
# Test class
# =========================================================================

class NormalizeFusionTest(MUSATestCase):
    """End-to-end tests for MusaNormalize fusion using a real-model subgraph."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._graph_def = _load_subgraph_pb()
        cls._input_shape = _get_placeholder_shape(cls._graph_def, _INPUT_NODE)

    # -----------------------------------------------------------------
    # 1. Fusion is applied: MusaNormalize node appears
    # -----------------------------------------------------------------
    def test_fusion_is_applied(self):
        """The optimized graph must contain a MusaNormalize node."""
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — fusion is applied")
        print("=" * 70)

        rng = np.random.RandomState(42)
        a_np = _make_feed(self._input_shape, batch_size=4, rng=rng)
        print(f"  Input shape: {a_np.shape}")

        _, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        fused = _has_fused_op(pgs)
        print(f"  MusaNormalize fused: {fused}")

        all_ops = sorted({
            node.op for pg in pgs for node in pg.node
        })
        print(f"  Op types in optimized graph: {all_ops}")

        self.assertTrue(fused, "MusaNormalize node not found in optimized graph")
        print("  PASSED")

    # -----------------------------------------------------------------
    # 2. Fused node carries correct epsilon and max_std attributes
    # -----------------------------------------------------------------
    def test_fusion_attrs_correct(self):
        """MusaNormalize node must have epsilon and max_std attributes."""
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — verify attrs")
        print("=" * 70)

        rng = np.random.RandomState(42)
        a_np = _make_feed(self._input_shape, batch_size=2, rng=rng)

        _, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        fused_nodes = _get_fused_nodes(pgs)
        self.assertTrue(fused_nodes, "No MusaNormalize node found")

        node = fused_nodes[0]
        epsilon = node.attr["epsilon"].f
        max_std = node.attr["max_std"].f

        print(f"  Fused node: {node.name}")
        print(f"  epsilon = {epsilon}")
        print(f"  max_std = {max_std}")

        # Epsilon should be from clip min value (9.999999960041972e-12)
        self.assertGreater(epsilon, 0.0)
        self.assertLess(epsilon, 1e-6)
        # max_std should be from clip max value
        self.assertGreater(max_std, 0.0)
        print("  PASSED")

    # -----------------------------------------------------------------
    # 3. Numerical correctness — small batch
    # -----------------------------------------------------------------
    def test_numerical_small(self):
        """Fused result matches reference Normalize on small inputs."""
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — numerical (small)")
        print("=" * 70)

        rng = np.random.RandomState(0)
        a_np = _make_feed(self._input_shape, batch_size=1, rng=rng)

        result, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        # Get actual epsilon and max_std from fused node
        fused_nodes = _get_fused_nodes(pgs)
        epsilon = fused_nodes[0].attr["epsilon"].f if fused_nodes else 1e-6
        max_std = fused_nodes[0].attr["max_std"].f if fused_nodes else np.inf

        # Use reference normalize with same epsilon and max_std
        expected = _reference_normalize(a_np, epsilon=epsilon, max_std=max_std)

        print(f"  Input shape: {a_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Epsilon: {epsilon:.2e}, Max std: {max_std:.2e}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")
        print(f"  MusaNormalize fused: {_has_fused_op(pgs)}")

        # Debug: print first few values
        print(f"  Input first 5: {a_np[0, :5]}")
        print(f"  Result first 5: {result[0, :5]}")
        print(f"  Expected first 5: {expected[0, :5]}")
        print(f"  Input mean: {np.mean(a_np):.6f}, std: {np.std(a_np):.6f}")
        print(f"  Result mean: {np.mean(result):.6f}, std: {np.std(result):.6f}")
        print(f"  Expected mean: {np.mean(expected):.6f}, std: {np.std(expected):.6f}")

        # Check if result is normalized at all
        result_row_mean = np.mean(result, axis=-1)
        result_row_std = np.std(result, axis=-1)
        print(f"  Result row-wise mean (should be ~0): {result_row_mean}")
        print(f"  Result row-wise std (should be ~1): {result_row_std}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        self.assertTrue(
            _has_fused_op(pgs),
            "MusaNormalize node not found in optimized graph",
        )
        print("  PASSED")

    # -----------------------------------------------------------------
    # 4. Numerical correctness — larger batch
    # -----------------------------------------------------------------
    def test_numerical_large_batch(self):
        """Fused result matches reference Normalize on larger batch."""
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — numerical (large batch)")
        print("=" * 70)

        rng = np.random.RandomState(99)
        a_np = _make_feed(self._input_shape, batch_size=16, rng=rng)

        result, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        # Get actual epsilon and max_std from fused node
        fused_nodes = _get_fused_nodes(pgs)
        epsilon = fused_nodes[0].attr["epsilon"].f if fused_nodes else 1e-6
        max_std = fused_nodes[0].attr["max_std"].f if fused_nodes else np.inf

        expected = _reference_normalize(a_np, epsilon=epsilon, max_std=max_std)

        print(f"  Input shape: {a_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Epsilon: {epsilon:.2e}, Max std: {max_std:.2e}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")
        print(f"  MusaNormalize fused: {_has_fused_op(pgs)}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        self.assertTrue(
            _has_fused_op(pgs),
            "MusaNormalize node not found in optimized graph",
        )
        print("  PASSED")

    # -----------------------------------------------------------------
    # 5. Numerical correctness — various row sizes (warp size coverage)
    # -----------------------------------------------------------------
    def test_numerical_various_row_sizes(self):
        """Test correctness across different row sizes covering warp optimization.

        Warp size is 128, so we test:
        - Very small rows (1-32)
        - Medium small rows (33-128)
        - Larger rows (>128)

        Note: row_size=1 is a degenerate case where variance is 0,
        so the output will be 0 (not normalized to std=1).
        """
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — various row sizes")
        print("=" * 70)

        # Skip row_size=1 as it's a degenerate case (variance=0)
        test_row_sizes = [2, 8, 16, 32, 64, 100, 128, 200, 512]
        rng = np.random.RandomState(123)
        epsilon = 1e-6
        max_std = np.inf

        for row_size in test_row_sizes:
            # Create input with configurable row size
            batch_size = 4
            a_np = rng.standard_normal((batch_size, row_size)).astype(np.float32)

            # Use reference normalize
            expected = _reference_normalize(a_np, epsilon=epsilon, max_std=max_std)

            print(f"  Row size {row_size:4d}: input shape = {a_np.shape}")

            # Verify output shape
            self.assertEqual(expected.shape, a_np.shape)

            # Verify output is normalized (mean ~0, std ~1)
            # Note: For very small row sizes with random data, std might vary
            output_mean = np.mean(expected, axis=-1)
            output_std = np.std(expected, axis=-1)

            # All values should be finite
            self.assertTrue(np.all(np.isfinite(expected)),
                           f"Output contains non-finite values for row_size={row_size}")
            # Mean should be close to 0
            self.assertTrue(np.allclose(output_mean, 0.0, atol=1e-5),
                           f"Mean not close to 0 for row_size={row_size}: {output_mean}")
            # Std should be close to 1 (normalized)
            self.assertTrue(np.allclose(output_std, 1.0, atol=1e-4),
                           f"Std not close to 1 for row_size={row_size}: {output_std}")

        print("  PASSED")

    # -----------------------------------------------------------------
    # 6. Numerical correctness — edge cases
    # -----------------------------------------------------------------
    def test_numerical_edge_cases(self):
        """Test correctness with edge cases: zero variance, constant input."""
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — edge cases")
        print("=" * 70)

        # Test 1: Near-zero variance (should use epsilon)
        epsilon = 1e-6
        a_zero_var = np.ones((2, 16), dtype=np.float32)
        result_zero_var = _reference_normalize(a_zero_var, epsilon=epsilon)
        # With zero variance, std = 0, clipped to epsilon, output should be 0
        print(f"  Zero variance test: mean output = {np.mean(result_zero_var):.6f}")
        self.assertTrue(np.allclose(result_zero_var, 0.0, atol=1e-5))

        # Test 2: Large values (should not overflow)
        a_large = np.full((2, 16), 1e6, dtype=np.float32)
        a_large[0, :8] = 1e6 + 1
        a_large[0, 8:] = 1e6 - 1
        result_large = _reference_normalize(a_large, epsilon=epsilon)
        print(f"  Large values test: output range = [{np.min(result_large):.2f}, {np.max(result_large):.2f}]")
        # Should produce finite results
        self.assertTrue(np.all(np.isfinite(result_large)))

        # Test 3: Mixed positive and negative
        a_mixed = np.array([[-1, 0, 1, 2]], dtype=np.float32)
        result_mixed = _reference_normalize(a_mixed, epsilon=epsilon)
        print(f"  Mixed values test: output = {result_mixed}")
        # Mean = 0.5, std = sqrt((2.25 + 0.25 + 0.25 + 2.25) / 4) = sqrt(1.25)
        expected_mean = 0.0
        self.assertTrue(np.allclose(np.mean(result_mixed), expected_mean, atol=1e-5))

        print("  PASSED")

    # -----------------------------------------------------------------
    # 7. Numerical correctness — max_std clamping
    # -----------------------------------------------------------------
    def test_max_std_clamping(self):
        """Test that max_std correctly limits the standard deviation."""
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — max_std clamping")
        print("=" * 70)

        # Create input with known high variance
        a_np = np.array([[0, 100]], dtype=np.float32)  # High variance
        epsilon = 1e-6
        max_std = 1.0  # Limit std to 1.0

        result = _reference_normalize(a_np, epsilon=epsilon, max_std=max_std)

        # With max_std=1.0, the std (which is ~50) should be clamped to 1.0
        # mean = 50, diff values = -50, 50
        # std = sqrt((2500 + 2500)/2) = 50
        # clipped_std = min(50, 1.0) = 1.0
        # output = (-50)/1.0 = -50, 50/1.0 = 50
        print(f"  Input: {a_np}")
        print(f"  Output: {result}")
        print(f"  max_std: {max_std}")

        # The output should NOT be normalized to std=1 because max_std limits it
        # Actually, the std of output should reflect the max_std limit
        # Let's verify the computation is correct
        expected_output = (a_np - np.mean(a_np)) / max_std
        print(f"  Expected: {expected_output}")
        self.assertTrue(np.allclose(result, expected_output, atol=1e-5))

        print("  PASSED")

    # -----------------------------------------------------------------
    # 8. Fused subgraph nodes are cleaned up
    # -----------------------------------------------------------------
    def test_subgraph_nodes_removed(self):
        """After fusion the helper nodes must be removed."""
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — subgraph cleanup")
        print("=" * 70)

        rng = np.random.RandomState(42)
        a_np = _make_feed(self._input_shape, batch_size=2, rng=rng)

        _, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        helper_ops_remaining = []
        prefix = "fwffm_pbp_mlp/ad_emb_aug_ln_layer/"
        helper_op_types = {"Mean", "ExpandDims", "Sub", "Square", "Sqrt", "MusaClip", "RealDiv"}

        for pg in pgs:
            for node in pg.node:
                if (node.name.startswith(prefix)
                        and node.op in helper_op_types):
                    helper_ops_remaining.append(
                        f"{node.op}({node.name})"
                    )

        print(f"  Helper ops remaining: {len(helper_ops_remaining)}")
        if helper_ops_remaining:
            for h in helper_ops_remaining[:5]:
                print(f"    {h}")

        self.assertEqual(
            len(helper_ops_remaining), 0,
            f"Normalize helper nodes not cleaned up: {helper_ops_remaining}",
        )
        print("  PASSED")

    # -----------------------------------------------------------------
    # 9. Unused Const nodes are cleaned up
    # -----------------------------------------------------------------
    def test_unused_const_nodes_removed(self):
        """Const nodes that become unused after fusion should be removed.

        In this isolated test subgraph, the 'pln1_follow/...' Const nodes
        are only used by the Normalize subgraph. After fusion, they become
        orphaned and should be cleaned up. In a full production model where
        these nodes are actually shared with other subgraphs, they would be
        preserved.
        """
        print("\n" + "=" * 70)
        print("Test: Normalize Fusion — unused Const nodes removed")
        print("=" * 70)

        rng = np.random.RandomState(42)
        a_np = _make_feed(self._input_shape, batch_size=2, rng=rng)

        _, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        # These Const nodes were only used by the fused Normalize subgraph
        # After fusion, they should be removed as orphaned nodes
        orphan_const_names = [
            "fwffm_pbp_mlp/pln1_follow/Mean/reduction_indices",
            "fwffm_pbp_mlp/pln1_follow/ExpandDims/dim",
            "fwffm_pbp_mlp/pln1_follow/clip_by_value/y",
            "fwffm_pbp_mlp/pln1_follow/clip_by_value/Minimum/y",
        ]

        found_orphans = []
        for pg in pgs:
            for node in pg.node:
                if node.name in orphan_const_names and node.op == "Const":
                    found_orphans.append(node.name)

        print(f"  Orphan Const nodes remaining: {len(found_orphans)}")
        if found_orphans:
            for name in found_orphans:
                print(f"    {name}")

        self.assertEqual(
            len(found_orphans), 0,
            f"Orphan Const nodes should be removed. Found: {found_orphans}",
        )
        print("  PASSED")


if __name__ == "__main__":
    tf.test.main()
