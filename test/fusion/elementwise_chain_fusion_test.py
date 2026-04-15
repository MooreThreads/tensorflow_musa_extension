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
"""End-to-end tests for linear elementwise-chain fusion."""

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

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output,
                feed_dict={x: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        fused_nodes = get_musa_fused_elementwise_nodes(run_metadata)

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(
            fused_nodes,
            "Expected Add -> Mul -> Exp chain to be fused into MusaFusedElementwise",
        )


if __name__ == "__main__":
    tf.test.main()
