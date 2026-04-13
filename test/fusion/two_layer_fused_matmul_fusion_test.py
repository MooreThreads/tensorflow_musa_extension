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


class TwoLayerFusedMatMulFusionTest(MUSATestCase):
    def _run_two_layer_test(self, activation="relu", alpha=0.2):
        np.random.seed(42)
        tf.random.set_seed(42)

        m, k0, n0, n1 = 4, 8, 16, 6
        x_np = np.random.randn(m, k0).astype(np.float32)
        w0_np = np.random.randn(k0, n0).astype(np.float32)
        b0_np = np.random.randn(n0).astype(np.float32)
        w1_np = np.random.randn(n0, n1).astype(np.float32)
        b1_np = np.random.randn(n1).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k0], name="x")
                w0 = tf.constant(w0_np, dtype=tf.float32, name="w0")
                b0 = tf.constant(b0_np, dtype=tf.float32, name="b0")
                w1 = tf.constant(w1_np, dtype=tf.float32, name="w1")
                b1 = tf.constant(b1_np, dtype=tf.float32, name="b1")

                hidden = tf.matmul(x, w0)
                hidden = tf.nn.bias_add(hidden, b0)
                if activation == "relu":
                    hidden = tf.nn.relu(hidden, name="relu")
                else:
                    hidden = tf.nn.leaky_relu(hidden, alpha=alpha, name="leaky_relu")
                out = tf.matmul(hidden, w1)
                out = tf.nn.bias_add(out, b1)
                output = out * 1.0

        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()
        with tf.compat.v1.Session(graph=graph, config=create_config_with_musa_optimizer()) as sess:
            sess.run(output, feed_dict={x: x_np}, options=run_options, run_metadata=run_metadata)

        fused_node = None
        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                if node.op == "MusaTwoLayerFusedMatMul":
                    fused_node = node
                    break

        self.assertIsNotNone(
            fused_node, "MusaTwoLayerFusedMatMul fusion was NOT applied"
        )
        self.assertEqual(
            fused_node.attr["activation_type"].s.decode("utf-8"),
            "Relu" if activation == "relu" else "LeakyRelu",
        )
        if activation == "leakyrelu":
            self.assertAlmostEqual(
                fused_node.attr["activation_alpha"].f, alpha, places=6
            )

    def test_two_layer_fused_matmul_relu_fusion_applied(self):
        self._run_two_layer_test(activation="relu")

    def test_two_layer_fused_matmul_leakyrelu_fusion_applied(self):
        self._run_two_layer_test(activation="leakyrelu", alpha=0.15)

if __name__ == "__main__":
    tf.test.main()
