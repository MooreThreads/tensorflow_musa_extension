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

"""Tests for MatMul+Bias fusion (MatMulBiasFusion pattern)."""

import os

os.environ.setdefault("MUSA_ENABLE_TF32", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


def create_config_with_musa_optimizer(disabled_patterns=None):
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    if disabled_patterns:
        custom_optimizer.parameter_map["disabled_fusion_patterns"].s = ",".join(
            disabled_patterns
        ).encode("utf-8")
    rewriter_config.min_graph_nodes = -1
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])
    return config


def is_tf32_enabled():
    return int(os.environ.get("MUSA_ENABLE_TF32", "0")) != 0


def float32_tolerance(default_rtol=2e-2, default_atol=2e-2):
    return (1e-2, 1e-2) if is_tf32_enabled() else (default_rtol, default_atol)


class MatMulBiasFusionTest(MUSATestCase):

    def test_matmul_bias_fusion_applied(self):
        np.random.seed(42)
        tf.random.set_seed(42)
        m, k, n = 4, 8, 16
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="b")
                mm = tf.matmul(x, w)
                bias = tf.nn.bias_add(mm, b)
                output = bias * 2.0

        config = create_config_with_musa_optimizer(
            disabled_patterns=["MatMulBiasAddFusion"]
        )
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual = sess.run(
                output,
                feed_dict={x: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        has_fused_node = False
        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                if node.op == "MusaFusedMatMul":
                    has_fused_node = True
                    break

        self.assertTrue(
            has_fused_node,
            "MusaFusedMatMul fusion was NOT applied to the graph",
        )

        with tf.device("/CPU:0"):
            expected = (
                tf.nn.bias_add(
                    tf.matmul(tf.constant(x_np), tf.constant(w_np)),
                    tf.constant(b_np),
                )
                * 2.0
            )

        rtol, atol = float32_tolerance()
        self.assertAllClose(actual, expected.numpy(), rtol=rtol, atol=atol)

    def test_matmul_bias_fusion_numerical(self):
        np.random.seed(99)
        tf.random.set_seed(99)
        m, k, n = 6, 12, 10
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        with tf.device("/CPU:0"):
            expected = tf.nn.bias_add(
                tf.matmul(tf.constant(x_np), tf.constant(w_np)),
                tf.constant(b_np),
            )

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="b")
                mm = tf.matmul(x, w)
                bias = tf.nn.bias_add(mm, b)
                output = bias * 1.5

        config = create_config_with_musa_optimizer(
            disabled_patterns=["MatMulBiasAddFusion"]
        )
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual = sess.run(output, feed_dict={x: x_np})

        rtol, atol = float32_tolerance()
        self.assertAllClose(
            actual, (expected * 1.5).numpy(), rtol=rtol, atol=atol
        )

    def test_matmul_bias_fusion_not_applied_multi_consumer(self):
        np.random.seed(7)
        m, k, n = 4, 8, 16
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="b")
                mm = tf.matmul(x, w)
                bias = tf.nn.bias_add(mm, b)
                mm_consumer2 = tf.identity(mm, name="mm_extra_consumer")
                output = bias + mm_consumer2

        config = create_config_with_musa_optimizer(
            disabled_patterns=["MatMulBiasAddFusion"]
        )
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(
                output,
                feed_dict={x: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        has_fused_node = False
        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                if node.op == "MusaFusedMatMul":
                    has_fused_node = True
                    break

        self.assertFalse(
            has_fused_node,
            "MusaFusedMatMul fusion should NOT be applied when MatMul has multiple consumers",
        )


if __name__ == "__main__":
    tf.test.main()
