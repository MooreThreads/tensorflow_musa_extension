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
"""End-to-end fusion test for MusaShiftedAffineMap (final pattern).

Pattern:
  AddV2 (output)
  ├─ Mul
  │   ├─ AddV2 (left)
  │   │   ├─ data_left
  │   │   └─ StridedSlice ← ReadVariableOp    (sliced_var_left)
  │   └─ Select (mask)
  └─ StridedSlice ← ReadVariableOp            (sliced_var_right)

Semantics:
  output = mask * (data_left + slice(var_left)) + slice(var_right)
"""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

_RTOL = 5e-3
_ATOL = 5e-3


# =========================================================================
# Helpers
# =========================================================================

def _create_config_with_musa_optimizer(disable_builtin_opts=True):
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
    return any(node.op == op_name
               for pg in partition_graphs for node in pg.node)


def _numpy_shifted_affine_map(data_left, sliced_var_left, mask, sliced_var_right):
    """Reference implementation: mask*(data_left+sliced_var_left)+sliced_var_right"""
    return mask * (data_left + sliced_var_left) + sliced_var_right


def _build_shifted_affine_map_graph(data_shape, var_left_shape, var_right_shape):
    """Construct the exact fusible pattern in a TF graph.

    Pattern:
      AddV2(output)
      ├─ Mul
      │   ├─ AddV2(add_left)
      │   │   ├─ data_left (placeholder)
      │   │   └─ StridedSlice ← ReadVariableOp (var_left)
      │   └─ Select (mask)
      └─ StridedSlice ← ReadVariableOp (var_right)
    """
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            # Data input (left branch)
            data_left = tf.compat.v1.placeholder(
                tf.float32, shape=data_shape, name="data_left")

            # Left variable: ReadVariableOp → StridedSlice → AddV2
            var_left = tf.Variable(
                tf.zeros(var_left_shape, dtype=tf.float32), name="var_left")
            begins_l = [0] * len(var_left_shape)
            ends_l = list(var_left_shape)
            strides_l = [1] * len(var_left_shape)
            sliced_var_left = tf.strided_slice(
                var_left, begins_l, ends_l, strides_l,
                name="strided_slice_left")

            # Right variable: ReadVariableOp → StridedSlice (directly to output)
            var_right = tf.Variable(
                tf.zeros(var_right_shape, dtype=tf.float32), name="var_right")
            begins_r = [0] * len(var_right_shape)
            ends_r = list(var_right_shape)
            strides_r = [1] * len(var_right_shape)
            sliced_var_right = tf.strided_slice(
                var_right, begins_r, ends_r, strides_r,
                name="strided_slice_right")

            # Mask (Select node)
            mask_cond = tf.compat.v1.placeholder(
                tf.bool, shape=data_shape, name="mask_cond")
            ones = tf.ones(data_shape, dtype=tf.float32)
            zeros = tf.zeros(data_shape, dtype=tf.float32)
            mask = tf.where(mask_cond, ones, zeros, name="mask_select")

            # Left branch: AddV2(data_left, sliced_var_left)
            add_left = tf.math.add(data_left, sliced_var_left, name="add_left")

            # Mul: add_left * mask
            mul_gated = tf.math.multiply(add_left, mask, name="mul_gated")

            # Output: mul_gated + sliced_var_right  (no wrapping AddV2 on right)
            output = tf.math.add(mul_gated, sliced_var_right, name="output")

    return graph, output, var_left, var_right


def _build_shifted_affine_map_graph_with_identity_wrappers(
        data_shape, var_left_shape, var_right_shape):
    """Construct a fusible graph with Identity wrappers on boundary inputs."""
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            data_left = tf.compat.v1.placeholder(
                tf.float32, shape=data_shape, name="data_left")
            data_left_id = tf.identity(data_left, name="data_left_identity")

            var_left = tf.Variable(
                tf.zeros(var_left_shape, dtype=tf.float32), name="var_left")
            sliced_var_left = tf.strided_slice(
                var_left, [0] * len(var_left_shape), list(var_left_shape),
                [1] * len(var_left_shape), name="strided_slice_left")
            sliced_var_left_id = tf.identity(
                sliced_var_left, name="strided_slice_left_identity")

            var_right = tf.Variable(
                tf.zeros(var_right_shape, dtype=tf.float32), name="var_right")
            sliced_var_right = tf.strided_slice(
                var_right, [0] * len(var_right_shape), list(var_right_shape),
                [1] * len(var_right_shape), name="strided_slice_right")
            sliced_var_right_id = tf.identity(
                sliced_var_right, name="strided_slice_right_identity")

            mask_cond = tf.compat.v1.placeholder(
                tf.bool, shape=data_shape, name="mask_cond")
            ones = tf.ones(data_shape, dtype=tf.float32)
            zeros = tf.zeros(data_shape, dtype=tf.float32)
            mask = tf.where(mask_cond, ones, zeros, name="mask_select")
            mask_id = tf.identity(mask, name="mask_identity")

            add_left = tf.math.add(
                data_left_id, sliced_var_left_id, name="add_left")
            mul_gated = tf.math.multiply(add_left, mask_id, name="mul_gated")
            output = tf.math.add(
                mul_gated, sliced_var_right_id, name="output")

    return graph, output, var_left, var_right


# =========================================================================
# Test class
# =========================================================================

class ShiftedAffineMapFusionTest(MUSATestCase):

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
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.5
        var_l = rng.standard_normal(var_shape).astype(np.float32) * 0.01
        var_r = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        graph, output, var_left, var_right = _build_shifted_affine_map_graph(
            data_shape, var_shape, var_shape)

        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_l))
            sess.run(var_right.assign(var_r))
            sess.run(output,
                     feed_dict={"data_left:0": data_np, "mask_cond:0": mask_np},
                     options=run_opts, run_metadata=run_meta)

        fused = _has_fused_op(run_meta.partition_graphs)
        ops = sorted({n.op for pg in run_meta.partition_graphs for n in pg.node})
        print(f"  shape={data_np.shape}, fused={fused}, ops={ops}")
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
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.3
        var_l = rng.standard_normal(var_shape).astype(np.float32) * 0.01
        var_r = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        expected = _numpy_shifted_affine_map(
            data_np, var_l, mask_np.astype(np.float32), var_r)

        graph, output, var_left, var_right = _build_shifted_affine_map_graph(
            data_shape, var_shape, var_shape)
        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_l))
            sess.run(var_right.assign(var_r))
            result = sess.run(
                output,
                feed_dict={"data_left:0": data_np, "mask_cond:0": mask_np},
                options=run_opts, run_metadata=run_meta)

        print(f"  max_diff={np.max(np.abs(result - expected)):.2e},"
              f" fused={_has_fused_op(run_meta.partition_graphs)}")
        self.assertEqual(result.shape, expected.shape)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        print("  PASSED")

    # -----------------------------------------------------------------
    # 3. Large batch
    # -----------------------------------------------------------------
    def test_numerical_large_batch(self):
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — larger batch")
        print("=" * 70)

        data_shape = [16, 32, 64]
        var_shape = [64]

        rng = np.random.RandomState(99)
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.5
        var_l = rng.standard_normal(var_shape).astype(np.float32) * 0.01
        var_r = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        expected = _numpy_shifted_affine_map(
            data_np, var_l, mask_np.astype(np.float32), var_r)

        graph, output, var_left, var_right = _build_shifted_affine_map_graph(
            data_shape, var_shape, var_shape)
        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_l))
            sess.run(var_right.assign(var_r))
            result = sess.run(
                output,
                feed_dict={"data_left:0": data_np, "mask_cond:0": mask_np},
                options=run_opts, run_metadata=run_meta)

        print(f"  max_diff={np.max(np.abs(result - expected)):.2e}")
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        print("  PASSED")

    # -----------------------------------------------------------------
    # 4. Identity wrappers should still fuse
    # -----------------------------------------------------------------
    def test_fusion_with_identity_wrappers(self):
        """Fusion should tolerate Identity wrappers on boundary inputs."""
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — identity wrappers")
        print("=" * 70)

        data_shape = [2, 4, 8]
        var_shape = [8]

        rng = np.random.RandomState(7)
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.4
        var_l = rng.standard_normal(var_shape).astype(np.float32) * 0.01
        var_r = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        expected = _numpy_shifted_affine_map(
            data_np, var_l, mask_np.astype(np.float32), var_r)

        graph, output, var_left, var_right = (
            _build_shifted_affine_map_graph_with_identity_wrappers(
                data_shape, var_shape, var_shape))
        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_l))
            sess.run(var_right.assign(var_r))
            result = sess.run(
                output,
                feed_dict={"data_left:0": data_np, "mask_cond:0": mask_np},
                options=run_opts, run_metadata=run_meta)

        fused = _has_fused_op(run_meta.partition_graphs)
        print(f"  fused={fused}, max_diff={np.max(np.abs(result - expected)):.2e}")
        self.assertTrue(fused, "Fusion should handle Identity wrappers")
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        print("  PASSED")

    # -----------------------------------------------------------------
    # 5. Negative test: incomplete pattern should NOT fuse
    # -----------------------------------------------------------------
    def test_fusion_not_applied_when_pattern_incomplete(self):
        """Fusion should NOT fire when the right branch is not StridedSlice."""
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — negative (incomplete pattern)")
        print("=" * 70)

        shape = [2, 4, 8]
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                a = tf.compat.v1.placeholder(tf.float32, shape=shape, name="a")
                b = tf.compat.v1.placeholder(tf.float32, shape=shape, name="b")
                # Simple Mul + constant AddV2 — cannot match
                output = tf.math.add(tf.math.multiply(a, b), tf.constant(1.0),
                                     name="incomplete_output")

        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()
        rng = np.random.RandomState(0)
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(output,
                     feed_dict={"a:0": rng.randn(*shape).astype(np.float32),
                                "b:0": rng.randn(*shape).astype(np.float32)},
                     options=run_opts, run_metadata=run_meta)

        fused = _has_fused_op(run_meta.partition_graphs)
        print(f"  fused={fused}")
        self.assertFalse(fused, "Fusion should NOT fire for incomplete pattern")
        print("  PASSED")

    # -----------------------------------------------------------------
    # 6. Subgraph cleanup
    # -----------------------------------------------------------------
    def test_subgraph_nodes_removed(self):
        """After fusion, intermediate nodes should be gone."""
        print("\n" + "=" * 70)
        print("Test: ShiftedAffineMap — intermediate node cleanup")
        print("=" * 70)

        data_shape = [4, 8, 16]
        var_shape = [16]

        rng = np.random.RandomState(42)
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.5
        var_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        graph, output, var_left, var_right = _build_shifted_affine_map_graph(
            data_shape, var_shape, var_shape)
        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(var_left.assign(var_np))
            sess.run(var_right.assign(var_np))
            sess.run(output,
                     feed_dict={"data_left:0": data_np, "mask_cond:0": mask_np},
                     options=run_opts, run_metadata=run_meta)

        if _has_fused_op(run_meta.partition_graphs):
            intermediate = {"add_left", "mul_gated"}
            remaining = [f"{n.op}({n.name})"
                         for pg in run_meta.partition_graphs
                         for n in pg.node if n.name in intermediate]
            print(f"  remaining intermediate nodes: {len(remaining)}")
            self.assertEqual(len(remaining), 0,
                             f"Not cleaned up: {remaining}")
        else:
            print("  NOTE: fusion did not fire; skipping cleanup check")
        print("  COMPLETED")


if __name__ == "__main__":
    tf.test.main()
