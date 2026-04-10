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

"""Tests for ConcatV2+MatMul fusion."""

import os
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2

def create_config_with_musa_optimizer():
    """Create ConfigProto with MUSA optimizer enabled."""
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"

    rewriter_config.min_graph_nodes = -1
    return config

class ConcatMatMulFusionTest(MUSATestCase):
    """Tests for ConcatV2+MatMul fusion."""

    def _test_concat_matmul_fusion(self, dtype=tf.float32, rtol=1e-5, atol=1e-5):
        """Helper to test ConcatV2 + MatMul pattern fusion with different dtypes."""
        # Set seeds for reproducibility
        np.random.seed(42)
        tf.compat.v1.set_random_seed(42)

        # Define shapes
        shape1 = [2, 16]
        shape2 = [2, 16]
        weight_shape = [32, 8]

        # Data for inputs
        if dtype == tf.bfloat16:
            np_a = np.random.randn(*shape1).astype(np.float32)
            np_b = np.random.randn(*shape2).astype(np.float32)
            np_w = np.random.randn(*weight_shape).astype(np.float32)
        else:
            np_a = np.random.randn(*shape1).astype(dtype.as_numpy_dtype)
            np_b = np.random.randn(*shape2).astype(dtype.as_numpy_dtype)
            np_w = np.random.randn(*weight_shape).astype(dtype.as_numpy_dtype)

        # Reference implementation (CPU)
        with tf.device('/CPU:0'):
            a_tf = tf.constant(np_a, dtype=dtype)
            b_tf = tf.constant(np_b, dtype=dtype)
            w_tf = tf.constant(np_w, dtype=dtype)

            concat_cpu = tf.concat([a_tf, b_tf], axis=1)
            expected_out = tf.matmul(concat_cpu, w_tf)

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                a = tf.compat.v1.placeholder(dtype, shape=shape1, name="input_a")
                b = tf.compat.v1.placeholder(dtype, shape=shape2, name="input_b")
                w = tf.constant(np_w, dtype=dtype, name="weight")

                # Concat + MatMul pattern
                concat_node = tf.concat([a, b], axis=1, name="concat")
                matmul_node = tf.matmul(concat_node, w, name="matmul")
                # Add a consumer to ensure it's not pruned
                output = matmul_node * 1.0

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual_out = sess.run(output, feed_dict={a: np_a, b: np_b})

        # Verification
        self.assertAllClose(actual_out, expected_out.numpy(), rtol=rtol, atol=atol)
        print(f"Successfully ran ConcatMatMul fusion test for {dtype.name} and verified results")

    def test_concat_matmul_fusion_float32(self):
        self._test_concat_matmul_fusion(dtype=tf.float32, rtol=1e-5, atol=1e-5)

    def test_concat_matmul_fusion_float16(self):
        self._test_concat_matmul_fusion(dtype=tf.float16, rtol=1e-2, atol=1e-2)

    def test_concat_matmul_fusion_bfloat16(self):
        self._test_concat_matmul_fusion(dtype=tf.bfloat16, rtol=1e-2, atol=1e-2)

    def test_concat_matmul_fusion_applied(self):
        """Verify that ConcatV2+MatMul fusion is applied: MusaConcatMatMul node exists in optimized graph."""
        # Define shapes
        shape1 = [2, 16]
        shape2 = [2, 16]
        weight_shape = [32, 8]

        # Data for inputs
        np_a = np.random.randn(*shape1).astype(np.float32)
        np_b = np.random.randn(*shape2).astype(np.float32)
        np_w = np.random.randn(*weight_shape).astype(np.float32)

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                a = tf.compat.v1.placeholder(tf.float32, shape=shape1, name="input_a")
                b = tf.compat.v1.placeholder(tf.float32, shape=shape2, name="input_b")
                w = tf.constant(np_w, dtype=tf.float32, name="weight")

                # Concat + MatMul pattern
                concat_node = tf.concat([a, b], axis=1, name="concat")
                matmul_node = tf.matmul(concat_node, w, name="matmul")
                # Add a consumer to ensure it's not pruned
                output = matmul_node * 1.0

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(output, feed_dict={a: np_a, b: np_b},
                     options=run_options, run_metadata=run_metadata)

        # Check for MusaConcatMatMul node in partitioned graphs
        has_fused_node = False
        fused_node_name = ""
        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                if "MusaConcatMatMul" in node.op:
                    has_fused_node = True
                    fused_node_name = node.name
                    break

        self.assertTrue(has_fused_node, "MusaConcatMatMul fusion was NOT applied to the graph")
        print(f"Verified: Found fused node '{fused_node_name}' with op 'MusaConcatMatMul'")

    def _run_partitioned_graph(self, graph, output, feed_dict):
        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual_out = sess.run(
                output,
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata,
            )

        return actual_out, run_metadata.partition_graphs

    def _find_concat_fused_node(self, partition_graphs):
        for partition_graph in partition_graphs:
            for node in partition_graph.node:
                if node.op == "MusaConcatMatMul" or "MusaConcatMatMul" in node.op:
                    return node
        return None

    def _find_two_layer_concat_fused_node(self, partition_graphs):
        for partition_graph in partition_graphs:
            for node in partition_graph.node:
                if node.op == "MusaTwoLayerConcatMatMul":
                    return node
        return None

    def _assert_concat_fused_node(
        self, partition_graphs, expected_fused_ops=None, expected_alpha=None
    ):
        fused_node = self._find_concat_fused_node(partition_graphs)
        self.assertIsNotNone(
            fused_node, "MusaConcatMatMul fusion was NOT applied to the graph"
        )
        if expected_fused_ops is not None:
            fused_ops = [x.decode("utf-8") for x in fused_node.attr["fused_ops"].list.s]
            self.assertEqual(fused_ops, expected_fused_ops)
        if expected_alpha is not None:
            self.assertAlmostEqual(
                fused_node.attr["activation_alpha"].f, expected_alpha, places=6
            )
        return fused_node

    def _test_concat_matmul_bias_activation_fusion(
        self,
        activation=None,
        alpha=0.2,
        dtype=tf.float32,
        rtol=1e-5,
        atol=1e-5,
    ):
        np.random.seed(42)
        tf.compat.v1.set_random_seed(42)

        shape1 = [2, 16]
        shape2 = [2, 16]
        weight_shape = [32, 8]
        bias_shape = [8]

        np_a = np.random.randn(*shape1).astype(dtype.as_numpy_dtype)
        np_b = np.random.randn(*shape2).astype(dtype.as_numpy_dtype)
        np_w = np.random.randn(*weight_shape).astype(dtype.as_numpy_dtype)
        np_bias = np.random.randn(*bias_shape).astype(dtype.as_numpy_dtype)

        expected = np.matmul(np.concatenate([np_a, np_b], axis=1), np_w) + np_bias
        if activation == "relu":
            expected = np.maximum(expected, 0)
        elif activation == "leakyrelu":
            expected = np.where(expected >= 0, expected, expected * alpha)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                a = tf.compat.v1.placeholder(dtype, shape=shape1, name="input_a")
                b = tf.compat.v1.placeholder(dtype, shape=shape2, name="input_b")
                w = tf.constant(np_w, dtype=dtype, name="weight")
                bias = tf.constant(np_bias, dtype=dtype, name="bias")

                concat_node = tf.concat([a, b], axis=1, name="concat")
                fused = tf.matmul(concat_node, w, name="matmul")
                fused = tf.nn.bias_add(fused, bias, name="bias_add")
                if activation == "relu":
                    fused = tf.nn.relu(fused, name="relu")
                elif activation == "leakyrelu":
                    fused = tf.nn.leaky_relu(fused, alpha=alpha, name="leaky_relu")
                output = fused * 1.0

        actual, partition_graphs = self._run_partitioned_graph(
            graph, output, {a: np_a, b: np_b}
        )
        self.assertAllClose(actual, expected, rtol=rtol, atol=atol)

        expected_fused_ops = ["BiasAdd"]
        expected_alpha = None
        if activation == "relu":
            expected_fused_ops = ["BiasAdd", "Relu"]
        elif activation == "leakyrelu":
            expected_fused_ops = ["BiasAdd", "LeakyRelu"]
            expected_alpha = alpha

        fused_node = self._assert_concat_fused_node(
            partition_graphs,
            expected_fused_ops=expected_fused_ops,
            expected_alpha=expected_alpha,
        )
        print(
            f"Verified concat fusion for activation={activation or 'none'} with node '{fused_node.name}'"
        )

    def _test_two_layer_concat_matmul_fusion(
        self, activation="relu", alpha=0.2, dtype=tf.float32, rtol=1e-5, atol=1e-5
    ):
        np.random.seed(123)
        tf.compat.v1.set_random_seed(123)

        shape1 = [3, 8]
        shape2 = [3, 8]
        w0_shape = [16, 10]
        b0_shape = [10]
        w1_shape = [10, 6]
        b1_shape = [6]

        np_a = np.random.randn(*shape1).astype(dtype.as_numpy_dtype)
        np_b = np.random.randn(*shape2).astype(dtype.as_numpy_dtype)
        np_w0 = np.random.randn(*w0_shape).astype(dtype.as_numpy_dtype)
        np_b0 = np.random.randn(*b0_shape).astype(dtype.as_numpy_dtype)
        np_w1 = np.random.randn(*w1_shape).astype(dtype.as_numpy_dtype)
        np_b1 = np.random.randn(*b1_shape).astype(dtype.as_numpy_dtype)

        hidden = np.matmul(np.concatenate([np_a, np_b], axis=1), np_w0) + np_b0
        if activation == "relu":
            hidden = np.maximum(hidden, 0)
        else:
            hidden = np.where(hidden >= 0, hidden, hidden * alpha)
        expected = np.matmul(hidden, np_w1) + np_b1

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                a = tf.compat.v1.placeholder(dtype, shape=shape1, name="input_a")
                b = tf.compat.v1.placeholder(dtype, shape=shape2, name="input_b")
                w0 = tf.constant(np_w0, dtype=dtype, name="w0")
                b0 = tf.constant(np_b0, dtype=dtype, name="b0")
                w1 = tf.constant(np_w1, dtype=dtype, name="w1")
                b1 = tf.constant(np_b1, dtype=dtype, name="b1")

                hidden = tf.concat([a, b], axis=1, name="concat")
                hidden = tf.matmul(hidden, w0, name="matmul0")
                hidden = tf.nn.bias_add(hidden, b0, name="bias_add0")
                if activation == "relu":
                    hidden = tf.nn.relu(hidden, name="relu")
                else:
                    hidden = tf.nn.leaky_relu(hidden, alpha=alpha, name="leaky_relu")
                out = tf.matmul(hidden, w1, name="matmul1")
                out = tf.nn.bias_add(out, b1, name="bias_add1")
                output = out * 1.0

        actual, partition_graphs = self._run_partitioned_graph(
            graph, output, {a: np_a, b: np_b}
        )
        self.assertAllClose(actual, expected, rtol=rtol, atol=atol)

        fused_node = self._find_two_layer_concat_fused_node(partition_graphs)
        self.assertIsNotNone(
            fused_node,
            "MusaTwoLayerConcatMatMul fusion was NOT applied to the graph",
        )
        self.assertEqual(
            fused_node.attr["activation_type"].s.decode("utf-8"),
            "Relu" if activation == "relu" else "LeakyRelu",
        )
        if activation == "leakyrelu":
            self.assertAlmostEqual(
                fused_node.attr["activation_alpha"].f, alpha, places=6
            )
        print(
            f"Verified two-layer concat fusion for activation={activation} with node '{fused_node.name}'"
        )

    def test_concat_matmul_bias_fusion_float32(self):
        self._test_concat_matmul_bias_activation_fusion(
            activation=None, dtype=tf.float32, rtol=1e-5, atol=1e-5
        )

    def test_concat_matmul_bias_relu_fusion_float32(self):
        self._test_concat_matmul_bias_activation_fusion(
            activation="relu", dtype=tf.float32, rtol=1e-5, atol=1e-5
        )

    def test_concat_matmul_bias_leakyrelu_fusion_float32(self):
        self._test_concat_matmul_bias_activation_fusion(
            activation="leakyrelu",
            alpha=0.15,
            dtype=tf.float32,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_two_layer_concat_matmul_relu_fusion_float32(self):
        self._test_two_layer_concat_matmul_fusion(
            activation="relu", dtype=tf.float32, rtol=1e-5, atol=1e-5
        )

    def test_two_layer_concat_matmul_leakyrelu_fusion_float32(self):
        self._test_two_layer_concat_matmul_fusion(
            activation="leakyrelu",
            alpha=0.15,
            dtype=tf.float32,
            rtol=1e-5,
            atol=1e-5,
        )

if __name__ == "__main__":
    tf.test.main()
