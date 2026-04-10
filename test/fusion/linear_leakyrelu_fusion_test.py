# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2


def create_config_with_musa_optimizer():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    rewriter_config.min_graph_nodes = -1
    return config


class LinearLeakyReluFusionTest(MUSATestCase):
    def test_linear_leakyrelu_fusion_basic(self):
        np.random.seed(42)
        tf.random.set_seed(42)

        m, k, n = 4, 8, 16
        alpha = 0.15
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        with tf.device("/CPU:0"):
            x_tf = tf.constant(x_np)
            w_tf = tf.constant(w_np)
            b_tf = tf.constant(b_np)
            mm = tf.matmul(x_tf, w_tf)
            bias = tf.nn.bias_add(mm, b_tf)
            expected_out = tf.nn.leaky_relu(bias, alpha=alpha) * 2.0

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="b")

                mm_musa = tf.matmul(x, w)
                bias_musa = tf.nn.bias_add(mm_musa, b)
                leakyrelu_out = tf.nn.leaky_relu(bias_musa, alpha=alpha)
                output = leakyrelu_out * 2.0

        with tf.compat.v1.Session(graph=graph, config=create_config_with_musa_optimizer()) as sess:
            actual_out = sess.run(output, feed_dict={x: x_np})

        self.assertAllClose(actual_out, expected_out.numpy(), rtol=1e-5, atol=1e-5)

    def test_linear_leakyrelu_fusion_applied(self):
        m, k, n = 4, 8, 16
        alpha = 0.25
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
                leakyrelu_out = tf.nn.leaky_relu(bias, alpha=alpha)
                output = leakyrelu_out * 1.5

        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()
        with tf.compat.v1.Session(graph=graph, config=create_config_with_musa_optimizer()) as sess:
            sess.run(output, feed_dict={x: x_np}, options=run_options, run_metadata=run_metadata)

        fused_node = None
        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                if node.op == "MusaLinearLeakyRelu":
                    fused_node = node
                    break

        self.assertIsNotNone(fused_node, "MusaLinearLeakyRelu fusion was NOT applied")
        self.assertAlmostEqual(fused_node.attr["alpha"].f, alpha, places=6)


if __name__ == "__main__":
    tf.test.main()
