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
"""End-to-end tests for shifted affine map fusion."""

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


def get_shifted_affine_fused_nodes(run_metadata):
    return [
        node
        for partition_graph in run_metadata.partition_graphs
        for node in partition_graph.node
        if node.op == "MusaShiftedAffineMap"
    ]


class ShiftedAffineMapFusionE2ETest(MUSATestCase):
    """Graph-level tests for Select <- Add(Mul(x, const1), const2) fusion."""

    def test_shifted_affine_map_fusion_is_applied(self):
        x_np = np.array(
            [[1.0, -2.0, 3.0, 4.0], [0.5, 2.0, -1.5, 8.0]], dtype=np.float32
        )
        cond_np = np.array(
            [[True, False, True, False], [False, True, False, True]], dtype=np.bool_
        )
        scale_np = np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32)
        bias_np = np.array([1.0, 0.5, -3.0, 2.0], dtype=np.float32)
        expected = np.where(cond_np, x_np * scale_np + bias_np, x_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name="x")
                cond = tf.compat.v1.placeholder(tf.bool, shape=[None, 4], name="cond")
                scale = tf.constant(scale_np, dtype=tf.float32, name="scale_const")
                bias = tf.constant(bias_np, dtype=tf.float32, name="bias_const")
                mul = tf.multiply(x, scale, name="branch_mul")
                shifted = tf.add(mul, bias, name="branch_add")
                output = tf.where(cond, shifted, x, name="select_out")

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output,
                feed_dict={x: x_np, cond: cond_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        fused_nodes = get_shifted_affine_fused_nodes(run_metadata)

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(fused_nodes, "Expected shifted affine branch to be fused")
        self.assertEqual(len(fused_nodes[0].input), 3)

    def test_shifted_affine_map_fusion_is_not_applied_with_non_const_scale(self):
        x_np = np.array(
            [[1.0, -2.0, 3.0, 4.0], [0.5, 2.0, -1.5, 8.0]], dtype=np.float32
        )
        cond_np = np.array(
            [[True, False, True, False], [False, True, False, True]], dtype=np.bool_
        )
        scale_np = np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32)
        bias_np = np.array([1.0, 0.5, -3.0, 2.0], dtype=np.float32)
        expected = np.where(cond_np, x_np * scale_np + bias_np, x_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name="x")
                cond = tf.compat.v1.placeholder(tf.bool, shape=[None, 4], name="cond")
                scale = tf.compat.v1.placeholder(tf.float32, shape=[4], name="scale_input")
                bias = tf.constant(bias_np, dtype=tf.float32, name="bias_const")
                mul = tf.multiply(x, scale, name="branch_mul")
                shifted = tf.add(mul, bias, name="branch_add")
                output = tf.where(cond, shifted, x, name="select_out")

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output,
                feed_dict={x: x_np, cond: cond_np, scale: scale_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        fused_nodes = get_shifted_affine_fused_nodes(run_metadata)

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(
            fused_nodes,
            "Did not expect fusion when scale comes from a non-const placeholder",
        )


if __name__ == "__main__":
    tf.test.main()
