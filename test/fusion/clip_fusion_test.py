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
"""End-to-end tests for clip-pattern -> MusaClip fusion."""

import time

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


def get_musa_clip_fused_nodes(run_metadata):
    return [
        node
        for partition_graph in run_metadata.partition_graphs
        for node in partition_graph.node
        if node.op == "MusaClip"
    ]


class ClipFusionE2ETest(MUSATestCase):
    """Functional tests for graph-level clip fusion."""

    def test_clip_fusion_minimum_then_maximum_is_applied(self):
        x_np = np.array(
            [[-3.0, -1.0, 2.0, 8.0], [0.5, 1.5, 7.0, 9.0]],
            dtype=np.float32,
        )
        lo_np = np.float32(0.0)
        hi_np = np.float32(6.0)
        expected = np.maximum(np.minimum(x_np, hi_np), lo_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 4], name="x"
                )
                lo = tf.constant(lo_np, dtype=tf.float32, name="lo")
                hi = tf.constant(hi_np, dtype=tf.float32, name="hi")

                output = tf.maximum(
                    tf.minimum(x, hi, name="clip_min_first"),
                    lo,
                    name="clip_max_second",
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

        fused_nodes = get_musa_clip_fused_nodes(run_metadata)

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(
            fused_nodes,
            "Expected Maximum(Minimum(x, hi), lo) chain to be fused into MusaClip",
        )

    def test_clip_fusion_maximum_then_minimum_is_not_applied(self):
        x_np = np.array(
            [[-3.0, -1.0, 2.0, 8.0], [0.5, 1.5, 7.0, 9.0]],
            dtype=np.float32,
        )
        lo_np = np.float32(0.0)
        hi_np = np.float32(6.0)
        expected = np.minimum(np.maximum(x_np, lo_np), hi_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 4], name="x"
                )
                lo = tf.constant(lo_np, dtype=tf.float32, name="lo")
                hi = tf.constant(hi_np, dtype=tf.float32, name="hi")

                output = tf.minimum(
                    tf.maximum(x, lo, name="clip_max_first"),
                    hi,
                    name="clip_min_second",
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

        fused_nodes = get_musa_clip_fused_nodes(run_metadata)

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(
            fused_nodes,
            "Did not expect Minimum(Maximum(x, lo), hi) chain to match the simplified fusion rule",
        )


class ClipFusionPerfE2ETest(MUSATestCase):
    """Performance test for a minimal [?, 1] clip chain."""

    def test_clip_chain_perf_with_column_vector_input(self):
        rows = 4096
        warmup_rounds = 10
        timed_rounds = 200

        x_np = np.linspace(-8.0, 8.0, num=rows, dtype=np.float32).reshape(-1, 1)
        lo_np = np.float32(0.0)
        hi_np = np.float32(6.0)
        expected = np.maximum(np.minimum(x_np, hi_np), lo_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, 1],
                    name="clip_input",
                )
                lo = tf.constant(lo_np, dtype=tf.float32, name="clip_lo")
                hi = tf.constant(hi_np, dtype=tf.float32, name="clip_hi")
                output = tf.maximum(
                    tf.minimum(x, hi, name="clip_by_value/Minimum"),
                    lo,
                    name="clip_by_value",
                )

        graph_def = graph.as_graph_def()
        nodes_by_name = {node.name: node for node in graph_def.node}
        self.assertEqual(nodes_by_name["clip_by_value"].op, "Maximum")
        self.assertEqual(nodes_by_name["clip_by_value/Minimum"].op, "Minimum")

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            warmup_result = sess.run(
                output,
                feed_dict={x: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )
            self.assertAllClose(warmup_result, expected, rtol=1e-5, atol=1e-6)

            for _ in range(max(0, warmup_rounds - 1)):
                sess.run(output, feed_dict={x: x_np})

            times = []
            result = None
            for _ in range(timed_rounds):
                start_time = time.perf_counter()
                result = sess.run(output, feed_dict={x: x_np})
                end_time = time.perf_counter()
                times.append(end_time - start_time)

        fused_nodes = get_musa_clip_fused_nodes(run_metadata)

        self.assertIsNotNone(result)
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(
            "clip perf: "
            f"rows={rows}, warmup_rounds={warmup_rounds}, timed_rounds={timed_rounds}, "
            f"fused_nodes={len(fused_nodes)}, avg_time={avg_time:.8f}s, "
            f"min_time={min_time:.8f}s, max_time={max_time:.8f}s"
        )


if __name__ == "__main__":
    tf.test.main()
