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
"""End-to-end fusion test for MusaShiftedAffineMap (Comprehensive Suite)."""

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


def _numpy_shifted_affine_map(data_left, mask, sliced_var_right):
    """Reference implementation: mask * data_left + sliced_var_right"""
    return mask * data_left + sliced_var_right


def _build_exact_pb_match_graph(data_shape, right_shape, data_np, right_np):
    """Construct a graph that exactly mimics the Constant-Folded topology."""
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            mask_cond = tf.compat.v1.placeholder(tf.bool, shape=data_shape, name="mask_cond")
            ones = tf.ones(data_shape, dtype=tf.float32)
            zeros = tf.zeros(data_shape, dtype=tf.float32)
            mask = tf.where(mask_cond, ones, zeros, name="Select_mask")

            # Left Branch is fully folded constant
            const_left = tf.constant(data_np, dtype=tf.float32, name="folded_const_left")
            id_left = tf.identity(const_left, name="id_folded_left")

            # Right Branch is also a folded constant
            const_right = tf.constant(right_np, dtype=tf.float32, name="folded_const_right")
            id_right = tf.identity(const_right, name="id_folded_right")

            # Main Computation
            mul_gated = tf.math.multiply(id_left, mask, name="mul_gated")
            output = tf.math.add(mul_gated, id_right, name="output_add")

    return graph, output

def _build_shifted_affine_map_graph(data_shape, right_shape, data_np, right_np):
    """Construct graph with standard const ops for numerical tests."""
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            mask_cond = tf.compat.v1.placeholder(tf.bool, shape=data_shape, name="mask_cond")
            ones = tf.ones(data_shape, dtype=tf.float32)
            zeros = tf.zeros(data_shape, dtype=tf.float32)
            mask = tf.where(mask_cond, ones, zeros, name="mask_select")

            const_left = tf.constant(data_np, dtype=tf.float32, name="data_left")
            const_right = tf.constant(right_np, dtype=tf.float32, name="data_right")

            mul_gated = tf.math.multiply(const_left, mask, name="mul_gated")
            output = tf.math.add(mul_gated, const_right, name="output")

    return graph, output


# =========================================================================
# Test class
# =========================================================================

class ShiftedAffineMapFusionTest(MUSATestCase):

    # -----------------------------------------------------------------
    # Test 1: Exact PB Topology (Folded Const)
    # -----------------------------------------------------------------
    def test_fusion_with_frozen_pb_topology(self):
        """Test if fusion fires on the exact topology."""
        print("\n" + "=" * 70)
        print("Test 1: ShiftedAffineMap — Strict PB Topology Match")
        print("=" * 70)

        data_shape = [4, 8, 16]
        var_shape = [16]

        rng = np.random.RandomState(42)
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.5
        var_r_np = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        expected = _numpy_shifted_affine_map(data_np, mask_np.astype(np.float32), var_r_np)
        graph, output = _build_exact_pb_match_graph(data_shape, var_shape, data_np, var_r_np)

        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output,
                feed_dict={"mask_cond:0": mask_np},
                options=run_opts, run_metadata=run_meta)

        fused = _has_fused_op(run_meta.partition_graphs)
        print(f"  max_diff={np.max(np.abs(result - expected)):.2e}, fused={fused}")
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        self.assertTrue(fused, "MusaShiftedAffineMap node not found in optimized graph!")
        print("  PASSED")

    # -----------------------------------------------------------------
    # Test 2: Numerical Correctness (Standard Constants)
    # -----------------------------------------------------------------
    def test_numerical_correctness(self):
        """Fused result matches numpy reference."""
        print("\n" + "=" * 70)
        print("Test 2: ShiftedAffineMap — numerical correctness")
        print("=" * 70)

        data_shape = [2, 4, 8]
        var_shape = [8]

        rng = np.random.RandomState(123)
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.3
        var_r = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        expected = _numpy_shifted_affine_map(data_np, mask_np.astype(np.float32), var_r)
        graph, output = _build_shifted_affine_map_graph(data_shape, var_shape, data_np, var_r)
        
        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output,
                feed_dict={"mask_cond:0": mask_np},
                options=run_opts, run_metadata=run_meta)

        fused = _has_fused_op(run_meta.partition_graphs)
        self.assertTrue(fused)
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        print("  PASSED")

    # -----------------------------------------------------------------
    # Test 3: Large Batch
    # -----------------------------------------------------------------
    def test_numerical_large_batch(self):
        print("\n" + "=" * 70)
        print("Test 3: ShiftedAffineMap — larger batch")
        print("=" * 70)

        data_shape = [16, 32, 64]
        var_shape = [64]

        rng = np.random.RandomState(99)
        data_np = rng.standard_normal(data_shape).astype(np.float32)
        mask_np = rng.random(data_shape) > 0.5
        var_r = rng.standard_normal(var_shape).astype(np.float32) * 0.01

        expected = _numpy_shifted_affine_map(data_np, mask_np.astype(np.float32), var_r)
        graph, output = _build_shifted_affine_map_graph(data_shape, var_shape, data_np, var_r)
        
        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output,
                feed_dict={"mask_cond:0": mask_np},
                options=run_opts, run_metadata=run_meta)

        self.assertTrue(_has_fused_op(run_meta.partition_graphs))
        self.assertAllClose(result, expected, rtol=_RTOL, atol=_ATOL)
        print("  PASSED")

    # -----------------------------------------------------------------
    # Test 4: Negative test (incomplete pattern should NOT fuse)
    # -----------------------------------------------------------------
    def test_fusion_not_applied_when_pattern_incomplete(self):
        """Fusion should NOT fire when the graph doesn't match the strict topology."""
        print("\n" + "=" * 70)
        print("Test 4: ShiftedAffineMap — negative (incomplete pattern)")
        print("=" * 70)

        shape = [2, 4, 8]
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                a = tf.compat.v1.placeholder(tf.float32, shape=shape, name="a")
                b = tf.compat.v1.placeholder(tf.float32, shape=shape, name="b")
                # Simple Mul + constant AddV2 — missing StridedSlice/Mask etc.
                output = tf.math.add(tf.math.multiply(a, b), tf.constant(1.0), name="incomplete_output")

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
        self.assertFalse(fused, "Fusion should NOT fire for incomplete pattern")
        print("  PASSED")


if __name__ == "__main__":
    tf.test.main()