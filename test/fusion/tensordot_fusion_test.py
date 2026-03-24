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

Uses a subgraph extracted from the real production model (graph_def.pb) to
test the fusion pass.  The subgraph file ``tensordot_subgraph.pb`` is
produced by ``extract_tensordot_pb_util.py`` and placed next to the test
directory.  It contains the *complete* tf.tensordot internal node structure
(Shape → GatherV2 → Prod → Pack → Reshape → Transpose → Reshape → MatMul
→ Reshape + ConcatV2) which is what the C++ fusion pass expects.

Why not use ``tf.tensordot()`` directly?
  When there is no real MUSA device, ``allow_soft_placement`` causes
  everything to fall back to CPU.  TF's built-in Grappler constant-folding
  pass then aggressively folds the shape-computation helper nodes, leaving
  only Reshape→MatMul→Reshape by the time our custom optimizer runs.
  The simplified graph no longer matches the fusion pattern.  Loading a
  pre-built subgraph from ``.pb`` bypasses this issue.

Subgraph structure (``longseq_mixer_ad_hard/Tensordot``):
  - Output Reshape: ``fwffm_pbp_mlp/longseq_mixer_ad_hard/Tensordot``
  - External input (Placeholder):
      ``fwffm_pbp_mlp/user_sparse_sequence_ads/SparseFieldsConcatExt_0``
  - Weight (Const → Identity):
      ``fwffm_pbp_mlp/longseq_mixer_ad_hard/Tensordot/ReadVariableOp/resource``
  - 23 nodes total.
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
_SUBGRAPH_PB = os.path.join(_PARENT_DIR, "fusion/tensordot_subgraph.pb")

# Node names inside the subgraph pb
_OUTPUT_NODE = "fwffm_pbp_mlp/longseq_mixer_ad_hard/Tensordot"
_INPUT_NODE = (
    "fwffm_pbp_mlp/user_sparse_sequence_ads/SparseFieldsConcatExt_0"
)

# Tolerances — CPU oneDNN float32 vs numpy float64 can diverge on large
# matmuls, so we use generous values.
_RTOL = 5e-3
_ATOL = 5e-3
_RTOL_LARGE = 1e-2
_ATOL_LARGE = 1e-2


# =========================================================================
# Helpers
# =========================================================================

def _load_subgraph_pb(pb_path=_SUBGRAPH_PB):
    """Load a GraphDef from a .pb file."""
    if not os.path.exists(pb_path):
        raise FileNotFoundError(
            f"Subgraph pb not found at {pb_path}. "
            "Run extract_tensordot_pb_util.py first."
        )
    graph_def = graph_pb2.GraphDef()
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def _create_config_with_musa_optimizer(disable_builtin_opts=True):
    """Create ConfigProto that enables only the musa_graph_optimizer.

    When *disable_builtin_opts* is True the built-in Grappler passes
    (constant folding, arithmetic optimization, …) are turned off so that
    the tensordot helper nodes survive until our custom optimizer runs.
    """
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
    """Import *graph_def*, run the graph through the MUSA optimizer, and
    return (result_np, partition_graphs)."""
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


def _has_fused_op(partition_graphs, op_name="MusaTensorDot"):
    for pg in partition_graphs:
        for node in pg.node:
            if node.op == op_name:
                return True
    return False


def _get_fused_nodes(partition_graphs, op_name="MusaTensorDot"):
    return [
        node
        for pg in partition_graphs
        for node in pg.node
        if node.op == op_name
    ]


def _count_fused_ops(partition_graphs, op_name="MusaTensorDot"):
    return len(_get_fused_nodes(partition_graphs, op_name))


def _extract_weight_from_pb(graph_def, weight_node_name):
    """Read the weight Const tensor from graph_def so we can compute a
    numpy reference result."""
    for node in graph_def.node:
        if node.name == weight_node_name and node.op == "Const":
            tensor_proto = node.attr["value"].tensor
            return tf.make_ndarray(tensor_proto)
    return None


def _get_placeholder_shape(graph_def, placeholder_name):
    """Read the static shape of a Placeholder from graph_def.

    Returns a list where ``None`` represents a dynamic dimension (e.g. batch).
    """
    for node in graph_def.node:
        if node.name == placeholder_name and node.op == "Placeholder":
            if "shape" in node.attr:
                shape_proto = node.attr["shape"].shape
                if not shape_proto.unknown_rank:
                    return [
                        (d.size if d.size >= 0 else None)
                        for d in shape_proto.dim
                    ]
            if "_output_shapes" in node.attr:
                shapes = node.attr["_output_shapes"].list.shape
                if shapes:
                    return [
                        (d.size if d.size >= 0 else None)
                        for d in shapes[0].dim
                    ]
    return None


def _make_feed(shape_template, batch_size, rng):
    """Create a random float32 array matching *shape_template*.

    ``None`` dimensions are replaced with *batch_size*.
    """
    concrete = [(batch_size if d is None else d) for d in shape_template]
    return rng.standard_normal(concrete).astype(np.float32)


# =========================================================================
# Test class
# =========================================================================

class TensorDotFusionTest(MUSATestCase):
    """End-to-end tests for MusaTensorDot fusion using a real-model subgraph."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._graph_def = _load_subgraph_pb()
        cls._weight_np = _extract_weight_from_pb(
            cls._graph_def,
            "fwffm_pbp_mlp/longseq_mixer_ad_hard/Tensordot/"
            "ReadVariableOp/resource",
        )
        cls._input_shape = _get_placeholder_shape(cls._graph_def, _INPUT_NODE)

    # -----------------------------------------------------------------
    # 1. Fusion is applied: MusaTensorDot node appears after optimization
    # -----------------------------------------------------------------
    def test_fusion_is_applied(self):
        """The optimized graph must contain a MusaTensorDot node."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — fusion is applied")
        print("=" * 70)

        rng = np.random.RandomState(42)
        a_np = _make_feed(self._input_shape, batch_size=4, rng=rng)
        print(f"  Input shape: {a_np.shape}")

        _, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        fused = _has_fused_op(pgs)
        print(f"  MusaTensorDot fused: {fused}")

        all_ops = sorted({
            node.op for pg in pgs for node in pg.node
        })
        print(f"  Op types in optimized graph: {all_ops}")

        self.assertTrue(fused, "MusaTensorDot node not found in optimized graph")
        print("  PASSED")

    # -----------------------------------------------------------------
    # 2. Fused node carries correct axes attributes
    # -----------------------------------------------------------------
    def test_fusion_attrs_correct(self):
        """MusaTensorDot node must have axes_a and axes_b attributes."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — verify axes attrs")
        print("=" * 70)

        rng = np.random.RandomState(42)
        a_np = _make_feed(self._input_shape, batch_size=2, rng=rng)

        _, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        fused_nodes = _get_fused_nodes(pgs)
        self.assertTrue(fused_nodes, "No MusaTensorDot node found")

        node = fused_nodes[0]
        axes_a = list(node.attr["axes_a"].list.i)
        axes_b = list(node.attr["axes_b"].list.i)

        print(f"  Fused node: {node.name}")
        print(f"  axes_a = {axes_a}")
        print(f"  axes_b = {axes_b}")

        self.assertEqual(len(axes_a), 1)
        self.assertEqual(axes_b, [0])
        print("  PASSED")

    # -----------------------------------------------------------------
    # 3. Numerical correctness — small batch
    # -----------------------------------------------------------------
    def test_numerical_small(self):
        """Fused result matches np.tensordot on small inputs."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — numerical (small)")
        print("=" * 70)

        if self._weight_np is None:
            self.skipTest("Could not extract weight from pb")

        rng = np.random.RandomState(0)
        a_np = _make_feed(self._input_shape, batch_size=1, rng=rng)

        result, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )
        expected = np.tensordot(
            a_np, self._weight_np, axes=([-1], [0])
        )

        print(f"  A shape: {a_np.shape}, W shape: {self._weight_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")
        print(f"  MusaTensorDot fused: {_has_fused_op(pgs)}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        self.assertTrue(
            _has_fused_op(pgs),
            "MusaTensorDot node not found in optimized graph",
        )
        print("  PASSED")

    # -----------------------------------------------------------------
    # 4. Numerical correctness — larger batch
    # -----------------------------------------------------------------
    def test_numerical_large_batch(self):
        """Fused result matches np.tensordot on a larger batch."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — numerical (large batch)")
        print("=" * 70)

        if self._weight_np is None:
            self.skipTest("Could not extract weight from pb")

        rng = np.random.RandomState(99)
        a_np = _make_feed(self._input_shape, batch_size=16, rng=rng)

        result, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )
        expected = np.tensordot(
            a_np, self._weight_np, axes=([-1], [0])
        )

        print(f"  A shape: {a_np.shape}, W shape: {self._weight_np.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")
        print(f"  MusaTensorDot fused: {_has_fused_op(pgs)}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)
        self.assertTrue(
            _has_fused_op(pgs),
            "MusaTensorDot node not found in optimized graph",
        )
        print("  PASSED")

    # -----------------------------------------------------------------
    # 5. Fused subgraph nodes are cleaned up
    # -----------------------------------------------------------------
    def test_subgraph_nodes_removed(self):
        """After fusion the helper nodes (GatherV2, Prod, Pack, ConcatV2,
        Shape) must be removed from the graph."""
        print("\n" + "=" * 70)
        print("Test: TensorDot Fusion — subgraph cleanup")
        print("=" * 70)

        rng = np.random.RandomState(42)
        a_np = _make_feed(self._input_shape, batch_size=2, rng=rng)

        _, pgs = _run_subgraph(
            self._graph_def, _INPUT_NODE, _OUTPUT_NODE, a_np,
        )

        helper_ops_remaining = []
        prefix = "fwffm_pbp_mlp/longseq_mixer_ad_hard/Tensordot/"
        helper_op_types = {"GatherV2", "Prod", "Pack", "ConcatV2", "Shape"}
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
            f"Tensordot helper nodes not cleaned up: {helper_ops_remaining}",
        )
        print("  PASSED")

    # -----------------------------------------------------------------
    # 6. Negative: plain Reshape+MatMul must NOT trigger fusion
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

        config = _create_config_with_musa_optimizer(disable_builtin_opts=False)
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output, feed_dict={a: a_np},
                options=run_opts, run_metadata=run_meta,
            )

        expected = (a_np.reshape(-1, 64) @ w_np).reshape(4, 16, 32)
        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL_LARGE, atol=_ATOL_LARGE)

        fused = _has_fused_op(run_meta.partition_graphs)
        print(f"  MusaTensorDot fused: {fused}")
        self.assertFalse(
            fused,
            "MusaTensorDot should NOT appear for plain Reshape+MatMul",
        )
        print("  PASSED")


if __name__ == "__main__":
    tf.test.main()
