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

"""Tests for MatMul + BiasAdd/Add/AddV2 fusion."""

import os

os.environ.setdefault("MUSA_ENABLE_TF32", "0")

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase
from tensorflow.core.protobuf import config_pb2


def create_config_with_musa_optimizer():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    rewriter_config.min_graph_nodes = -1
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])
    return config


def has_fused_matmul_biasadd(run_metadata):
    for partition_graph in run_metadata.partition_graphs:
        for node in partition_graph.node:
            if node.op == "MusaMatMulBiasAdd":
                return True
    return False


class MatMulBiasAddFusionTest(MUSATestCase):

    def test_add_rank2_row_bias_fusion_applied_and_correct(self):
        np.random.seed(2026)
        m, k, n = 4, 8, 16
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(1, n).astype(np.float32)

        with tf.device("/CPU:0"):
            expected = tf.matmul(tf.constant(x_np), tf.constant(w_np)) + tf.constant(b_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="row_bias")
                out = tf.add(tf.matmul(x, w), b, name="matmul_add_row_bias")
                output = out * 1.0

        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()
        with tf.compat.v1.Session(graph=graph, config=create_config_with_musa_optimizer()) as sess:
            actual = sess.run(
                output,
                feed_dict={x: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        self.assertTrue(
            has_fused_matmul_biasadd(run_metadata),
            "MusaMatMulBiasAdd fusion was not applied for Add [1, N] bias",
        )
        self.assertAllClose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)

    def test_addv2_rank2_row_bias_fusion_applied_when_bias_first(self):
        np.random.seed(2027)
        m, k, n = 3, 5, 7
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(1, n).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="row_bias")
                mm = tf.matmul(x, w, name="matmul")
                out = tf.raw_ops.AddV2(x=b, y=mm, name="row_bias_addv2")
                output = out * 1.0

        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()
        with tf.compat.v1.Session(graph=graph, config=create_config_with_musa_optimizer()) as sess:
            sess.run(
                output,
                feed_dict={x: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        self.assertTrue(
            has_fused_matmul_biasadd(run_metadata),
            "MusaMatMulBiasAdd fusion was not applied for AddV2 [1, N] bias",
        )

    def test_add_rank2_column_broadcast_not_fused(self):
        np.random.seed(2028)
        m, k, n = 4, 8, 16
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(m, 1).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[m, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="column_bias")
                out = tf.add(tf.matmul(x, w), b, name="matmul_add_column_bias")
                output = out * 1.0

        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()
        with tf.compat.v1.Session(graph=graph, config=create_config_with_musa_optimizer()) as sess:
            sess.run(
                output,
                feed_dict={x: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        self.assertFalse(
            has_fused_matmul_biasadd(run_metadata),
            "MusaMatMulBiasAdd fusion should not be applied for [M, 1] Add broadcast",
        )


if __name__ == "__main__":
    tf.test.main()
