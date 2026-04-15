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
"""End-to-end tests for MusaConcatMatMulSplit fusion."""

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


def get_concat_matmul_split_nodes(run_metadata):
    return [
        node
        for partition_graph in run_metadata.partition_graphs
        for node in partition_graph.node
        if node.op == "MusaConcatMatMulSplit"
    ]


class ConcatMatMulSplitFusionE2ETest(MUSATestCase):
    def _run_and_assert_fused(self, graph, fetches, feed_dict):
        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            results = sess.run(
                fetches,
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata,
            )

        fused_nodes = get_concat_matmul_split_nodes(run_metadata)
        self.assertTrue(
            fused_nodes,
            "Expected ConcatV2 -> MatMul -> many Slice to be fused into MusaConcatMatMulSplit",
        )
        return results

    def test_concat_matmul_then_col_slice_is_fused(self):
        x0_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        x1_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        weight_np = np.array(
            [
                [1.0, 0.0, 2.0, 1.0],
                [0.0, 1.0, 1.0, 2.0],
                [1.0, 1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        concat_np = np.concatenate([x0_np, x1_np], axis=1)
        matmul_np = np.matmul(concat_np, weight_np)
        expected0 = matmul_np[:, :2]
        expected1 = matmul_np[:, 2:]

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x0 = tf.compat.v1.placeholder(
                    tf.float32, shape=[2, 2], name="x0"
                )
                x1 = tf.compat.v1.placeholder(
                    tf.float32, shape=[2, 2], name="x1"
                )
                weight = tf.constant(weight_np, dtype=tf.float32, name="weight")
                concat = tf.concat([x0, x1], axis=1, name="concat_inputs")
                matmul = tf.matmul(concat, weight, name="concat_matmul")
                out0 = tf.raw_ops.Slice(
                    input=matmul,
                    begin=tf.constant([0, 0], dtype=tf.int32, name="slice0_begin"),
                    size=tf.constant([2, 2], dtype=tf.int32, name="slice0_size"),
                    name="slice0",
                )
                out1 = tf.raw_ops.Slice(
                    input=matmul,
                    begin=tf.constant([0, 2], dtype=tf.int32, name="slice1_begin"),
                    size=tf.constant([2, 2], dtype=tf.int32, name="slice1_size"),
                    name="slice1",
                )

        result0, result1 = self._run_and_assert_fused(
            graph, [out0, out1], {x0: x0_np, x1: x1_np}
        )

        self.assertAllClose(result0, expected0, rtol=1e-5, atol=1e-6)
        self.assertAllClose(result1, expected1, rtol=1e-5, atol=1e-6)

    def test_concat_matmul_then_row_slice_is_fused(self):
        x0_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        x1_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        weight_np = np.array(
            [
                [1.0, 0.0, 2.0, 1.0],
                [0.0, 1.0, 1.0, 2.0],
                [1.0, 1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        concat_np = np.concatenate([x0_np, x1_np], axis=1)
        matmul_np = np.matmul(concat_np, weight_np)
        expected0 = matmul_np[:1, :]
        expected1 = matmul_np[1:, :]

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x0 = tf.compat.v1.placeholder(
                    tf.float32, shape=[2, 2], name="x0"
                )
                x1 = tf.compat.v1.placeholder(
                    tf.float32, shape=[2, 2], name="x1"
                )
                weight = tf.constant(weight_np, dtype=tf.float32, name="weight")
                concat = tf.concat([x0, x1], axis=1, name="concat_inputs")
                matmul = tf.matmul(concat, weight, name="concat_matmul")
                out0 = tf.raw_ops.Slice(
                    input=matmul,
                    begin=tf.constant([0, 0], dtype=tf.int32, name="row0_begin"),
                    size=tf.constant([1, 4], dtype=tf.int32, name="row0_size"),
                    name="row0",
                )
                out1 = tf.raw_ops.Slice(
                    input=matmul,
                    begin=tf.constant([1, 0], dtype=tf.int32, name="row1_begin"),
                    size=tf.constant([1, 4], dtype=tf.int32, name="row1_size"),
                    name="row1",
                )

        result0, result1 = self._run_and_assert_fused(
            graph, [out0, out1], {x0: x0_np, x1: x1_np}
        )

        self.assertAllClose(result0, expected0, rtol=1e-5, atol=1e-6)
        self.assertAllClose(result1, expected1, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()
