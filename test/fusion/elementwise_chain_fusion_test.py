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
"""End-to-end tests for generic elementwise-chain fusion."""

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
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])

    return config


def get_musa_fused_elementwise_nodes(run_metadata):
    return [
        node
        for partition_graph in run_metadata.partition_graphs
        for node in partition_graph.node
        if node.op == "MusaFusedElementwise"
    ]


class ElementwiseChainFusionE2ETest(MUSATestCase):
    def _run_graph(self, graph, fetches, feed_dict):
        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                fetches,
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata,
            )

        return result, get_musa_fused_elementwise_nodes(run_metadata)

    def test_linear_elementwise_chain_is_fused(self):
        x_np = np.array(
            [[0.2, 0.4, 0.6, 0.8], [1.0, 1.2, 1.4, 1.6]], dtype=np.float32
        )
        bias_np = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        scale_np = np.float32(0.5)
        expected = np.exp((x_np + bias_np) * scale_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 4], name="x"
                )
                bias = tf.constant(bias_np, dtype=tf.float32, name="bias")
                scale = tf.constant(scale_np, dtype=tf.float32, name="scale")
                output = tf.exp(
                    tf.multiply(
                        tf.add(x, bias, name="chain_add"),
                        scale,
                        name="chain_mul",
                    ),
                    name="chain_exp",
                )

        result, fused_nodes = self._run_graph(graph, output, {x: x_np})

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(
            fused_nodes,
            "Expected Add -> Mul -> Exp chain to be fused into MusaFusedElementwise",
        )

    def test_select_pow_dag_is_fused(self):
        x_np = np.array(
            [[1.2, 0.8, 1.5, 0.7], [0.9, 1.1, 1.3, 0.6]], dtype=np.float32
        )
        y_np = np.array(
            [[0.7, 1.0, 0.9, 1.4], [1.2, 0.8, 0.6, 1.5]], dtype=np.float32
        )
        z_np = np.array(
            [[0.2, 0.4, 0.1, 0.3], [0.6, 0.5, 0.4, 0.2]], dtype=np.float32
        )
        mask_np = np.array(
            [[True, False, True, False], [False, True, False, True]]
        )
        bias_np = np.array([0.05, 0.10, 0.15, 0.20], dtype=np.float32)

        expected = np.power(np.where(mask_np, x_np, y_np), z_np + bias_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[2, 4], name="x"
                )
                y = tf.compat.v1.placeholder(
                    tf.float32, shape=[2, 4], name="y"
                )
                z = tf.compat.v1.placeholder(
                    tf.float32, shape=[2, 4], name="z"
                )
                mask = tf.constant(mask_np, dtype=tf.bool, name="mask")
                bias = tf.constant(bias_np, dtype=tf.float32, name="bias")
                selected = tf.where(mask, x, y, name="chain_select")
                shifted = tf.add(z, bias, name="chain_add")
                output = tf.pow(selected, shifted, name="chain_pow")

        result, fused_nodes = self._run_graph(
            graph, output, {x: x_np, y: y_np, z: z_np}
        )

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(
            fused_nodes,
            "Expected Select + Add -> Pow DAG to be fused into MusaFusedElementwise",
        )

    def test_two_step_light_chain_is_not_fused(self):
        x_np = np.array(
            [[0.2, 0.4, 0.6, 0.8], [1.0, 1.2, 1.4, 1.6]], dtype=np.float32
        )
        bias_np = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        scale_np = np.float32(0.5)
        expected = (x_np + bias_np) * scale_np

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 4], name="x"
                )
                bias = tf.constant(bias_np, dtype=tf.float32, name="bias")
                scale = tf.constant(scale_np, dtype=tf.float32, name="scale")
                output = tf.multiply(
                    tf.add(x, bias, name="cheap_add"),
                    scale,
                    name="cheap_mul",
                )

        result, fused_nodes = self._run_graph(graph, output, {x: x_np})

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(
            fused_nodes,
            "Did not expect a cheap two-step Add -> Mul chain to be fused",
        )


if __name__ == "__main__":
    tf.test.main()
