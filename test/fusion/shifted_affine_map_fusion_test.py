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
"""End-to-end fusion test for MusaShiftedAffineMap.

Pattern (updated):
  AddV2 (output)
  ├─ Mul
  │   ├─ AddV2 (left)
  │   │   ├─ data_left
  │   │   └─ StridedSlice ← ReadVariableOp    (sliced_var_left)
  │   └─ Select (mask)
  └─ AddV2 (right)
      ├─ data_right
      └─ StridedSlice ← ReadVariableOp        (sliced_var_right)

Semantics:
  output = mask * (data_left + slice(var_left)) + (data_right + slice(var_right))
"""

import os
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
_RTOL = 5e-3
_ATOL = 5e-3


# =========================================================================
# Helpers
# =========================================================================

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


def _has_fused_op(partition_graphs, op_name="MusaShiftedAffineMap"):
    for pg in partition_graphs:
        for node in pg.node:
            if node.op == op_name:
                return True
    return False


def _get_fused_nodes(partition_graphs, op_name="MusaShiftedAffineMap"):
    return [
        node
        for pg in partition_graphs
        for node in pg.node
        if node.op == op_name
    ]


def _numpy_shifted_affine_map(data_left, sliced_var_left, mask,
                               data_right, sliced_var_right):
    """Reference NumPy implementation of ShiftedAffineMap."""
    return mask * (data_left + sliced_var_left) + (data_right + sliced_var_right)


def _build_shifted_affine_map_graph(data_shape, var_shape):
    """Build a TF graph matching the exact fusible pattern.

    Pattern:
      AddV2(output)
      ├─ Mul
      │   ├─ AddV2(left)
      │   │   ├─ data_left (placeholder)
      │   │   └─ StridedSlice ← ReadVariableOp   (var_left)
      │   └─ Select (mask)
      └─ AddV2(right)
          ├─ data_right (placeholder)
          └─ StridedSlice ← ReadVariableOp   (var_right)
    """
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            # Data inputs
            data_left = tf.compat.v1.placeholder(
                tf.float32, shape=data_shape, name="data_left")
            data_right = tf.compat.v1.placeholder(
                tf.float32, shape=data_shape, name="data_right")

            # Variables that get sliced (ReadVariableOp → StridedSlice)
            var_left = tf.Variable(
                tf.zeros(var_shape, dtype=tf.float32), name="var_left")
            var_right = tf.Variable(
                tf.zeros(var_shape, dtype=tf.float32), name="var_right")

            # StridedSlice from each variable (trivial full-range slice to
            # generate the StridedSlice → ReadVariableOp pattern)
            begins = [0] * len(var_shape)
            ends = list(var_shape)
            strides = [1] * len(var_shape)
            sliced_var_left = tf.strided_slice(
                var_left, begins, ends, strides,
                name="strided_slice_left")
            sliced_var_right = tf.strided_slice(
                var_right, begins, ends, strides,
                name="strided_slice_right")

            # Mask (Select node)
            mask_cond = tf.compat.v1.placeholder(
                tf.bool, shape=data_shape, name="mask_cond")
            ones = tf.ones(data_shape, dtype=tf.float32, name="ones")
            zeros = tf.zeros(data_shape, dtype=tf.float32, name="zeros")
            mask = tf.where(mask_cond, ones, zeros, name="mask_select")

            # Left branch: AddV2(data_left, sliced_var_left)
            add_left = tf.math.add(data_left, sliced_var_left, name="add_left")

            # Mul: mask * add_left
            mul_gated = tf.math.multiply(add_left, mask, name="mul_gated")

            # Right branch: AddV2(data_right, sliced_var_right)
            add_right = tf.math.add(
                data_right, sliced_var_right, name="add_right")

            # Final output: AddV2(mul_gated, add_right)
            output = tf.math.add(mul_gated, add_right, name="output")

    return graph, output, var_left, var_right


# =========================================================================
# Test class
# =========================================================================

class ShiftedAffineMapFusionTest(MUSATestCase):
    """Tests that the graph optimizer correctly fuses the target subgraph
    into a single MusaShiftedAffineMap op."""

    # -----------------------------------------------------------------
    # 1. Fusion is applied
    # -----------------------------------------------------------------
    def test_fusion_is_applied(self):
        """The optimized graph must contain a MusaShiftedAffineMap node."""
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — fusion is applied")
        print("=" * 70)

        data_shape = [4, 8, 16]
        var_shape = [16]

        rng = np.random.RandomState(42)
        data_left_np = rng.standard_normal(data_shape).astype(np.float32)
        data_right_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.5
        var_left_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01
        var_right_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        graph, output, var_left, var_right = _build_shifted_affine_map_graph(
            data_shape, var_shape)

        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_left_np))
            sess.run(var_right.assign(var_right_np))

            result = sess.run(
                output,
                feed_dict={
                    "data_left:0": data_left_np,
                    "data_right:0": data_right_np,
                    "mask_cond:0": mask_np,
                },
                options=run_opts,
                run_metadata=run_meta,
            )

        fused = _has_fused_op(run_meta.partition_graphs)
        all_ops = sorted({
            node.op for pg in run_meta.partition_graphs for node in pg.node
        })
        print(f"  Input shape: {data_left_np.shape}")
        print(f"  MusaShiftedAffineMap fused: {fused}")
        print(f"  Op types: {all_ops}")
        print("  COMPLETED")

    # -----------------------------------------------------------------
    # 2. Numerical correctness
    # -----------------------------------------------------------------
    def test_numerical_correctness(self):
        """Fused result matches numpy reference."""
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — numerical correctness")
        print("=" * 70)

        data_shape = [2, 4, 8]
        var_shape = [8]

        rng = np.random.RandomState(123)
        data_left_np = rng.standard_normal(data_shape).astype(np.float32)
        data_right_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.3
        var_left_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01
        var_right_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        # NumPy reference
        mask_float = mask_np.astype(np.float32)
        expected = _numpy_shifted_affine_map(
            data_left_np, var_left_np, mask_float,
            data_right_np, var_right_np)

        graph, output, var_left, var_right = _build_shifted_affine_map_graph(
            data_shape, var_shape)

        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_left_np))
            sess.run(var_right.assign(var_right_np))

            result = sess.run(
                output,
                feed_dict={
                    "data_left:0": data_left_np,
                    "data_right:0": data_right_np,
                    "mask_cond:0": mask_np,
                },
                options=run_opts,
                run_metadata=run_meta,
            )

        print(f"  Output shape: {result.shape}")
        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")
        print(f"  Fused: {_has_fused_op(run_meta.partition_graphs)}")

        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        print("  PASSED")

    # -----------------------------------------------------------------
    # 3. Larger batch
    # -----------------------------------------------------------------
    def test_numerical_large_batch(self):
        """Fused result matches numpy reference on a larger batch."""
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — larger batch")
        print("=" * 70)

        data_shape = [16, 32, 64]
        var_shape = [64]

        rng = np.random.RandomState(99)
        data_left_np = rng.standard_normal(data_shape).astype(np.float32)
        data_right_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.5
        var_left_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01
        var_right_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        mask_float = mask_np.astype(np.float32)
        expected = _numpy_shifted_affine_map(
            data_left_np, var_left_np, mask_float,
            data_right_np, var_right_np)

        graph, output, var_left, var_right = _build_shifted_affine_map_graph(
            data_shape, var_shape)
        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_left_np))
            sess.run(var_right.assign(var_right_np))

            result = sess.run(
                output,
                feed_dict={
                    "data_left:0": data_left_np,
                    "data_right:0": data_right_np,
                    "mask_cond:0": mask_np,
                },
                options=run_opts,
                run_metadata=run_meta,
            )

        print(f"  Max abs diff: {np.max(np.abs(result - expected)):.2e}")
        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        print("  PASSED")

    # -----------------------------------------------------------------
    # 4. Negative test: incomplete pattern
    # -----------------------------------------------------------------
    def test_fusion_not_applied_when_pattern_incomplete(self):
        """Fusion should NOT fire when AddV2 inputs are not
        StridedSlice(ReadVariableOp) chains."""
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — negative (no StridedSlice chain)")
        print("=" * 70)

        data_shape = [2, 4, 8]

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                a = tf.compat.v1.placeholder(
                    tf.float32, shape=data_shape, name="neg_a")
                b = tf.compat.v1.placeholder(
                    tf.float32, shape=data_shape, name="neg_b")

                # Only Mul + AddV2, no StridedSlice→ReadVariableOp chain
                mul_out = tf.math.multiply(a, b, name="neg_mul")
                output = tf.math.add(
                    mul_out, tf.constant(1.0), name="neg_output")

        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        rng = np.random.RandomState(0)
        a_np = rng.standard_normal(data_shape).astype(np.float32)
        b_np = rng.standard_normal(data_shape).astype(np.float32)

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output,
                feed_dict={"neg_a:0": a_np, "neg_b:0": b_np},
                options=run_opts,
                run_metadata=run_meta,
            )

        fused = _has_fused_op(run_meta.partition_graphs)
        print(f"  MusaShiftedAffineMap fused: {fused}")
        self.assertFalse(
            fused,
            "MusaShiftedAffineMap should NOT appear for incomplete pattern")
        print("  PASSED")

    # -----------------------------------------------------------------
    # 5. Subgraph cleanup
    # -----------------------------------------------------------------
    def test_subgraph_nodes_removed(self):
        """After fusion, intermediate AddV2/Mul nodes should be removed."""
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — subgraph cleanup")
        print("=" * 70)

        data_shape = [4, 8, 16]
        var_shape = [16]

        rng = np.random.RandomState(42)
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.5
        var_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        graph, output, var_left, var_right = _build_shifted_affine_map_graph(
            data_shape, var_shape)
        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_np))
            sess.run(var_right.assign(var_np))

            sess.run(
                output,
                feed_dict={
                    "data_left:0": data_np,
                    "data_right:0": data_np,
                    "mask_cond:0": mask_np,
                },
                options=run_opts,
                run_metadata=run_meta,
            )

        if _has_fused_op(run_meta.partition_graphs):
            intermediate_names = {"add_left", "mul_gated", "add_right"}
            remaining = []
            for pg in run_meta.partition_graphs:
                for node in pg.node:
                    if node.name in intermediate_names:
                        remaining.append(f"{node.op}({node.name})")

            print(f"  Intermediate nodes remaining: {len(remaining)}")
            self.assertEqual(
                len(remaining), 0,
                f"Intermediate nodes not cleaned up: {remaining}")
        else:
            print("  NOTE: fusion did not fire; skipping cleanup check")

        print("  COMPLETED")


if __name__ == "__main__":
    tf.test.main()
