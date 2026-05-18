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

"""Fusion tests for MusaPlnCascade and MusaPlnCascadeBlock."""

from collections import Counter

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


def _create_config_with_musa_optimizer(disable_builtin_opts=True):
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rw = config.graph_options.rewrite_options
    rw.min_graph_nodes = -1
    if disable_builtin_opts:
        rw.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
        rw.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
        rw.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
        rw.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
        rw.shape_optimization = rewriter_config_pb2.RewriterConfig.OFF

    custom_opt = rw.custom_optimizers.add()
    custom_opt.name = "musa_graph_optimizer"
    rw.optimizers.extend(["musa_graph_optimizer"])
    return config


def _partition_op_counter(partition_graphs):
    return Counter(node.op for graph_def in partition_graphs for node in graph_def.node)


def _find_first_op_node(partition_graphs, op_name):
    for graph_def in partition_graphs:
        for node in graph_def.node:
            if node.op == op_name:
                return node
    return None


def _build_single_step_pending_affine_graph(batch, width):
    add_vec_np = np.linspace(0.7, 1.3, width).astype(np.float32)
    bias_vec_np = np.linspace(-0.2, 0.2, width).astype(np.float32)

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            x = tf.compat.v1.placeholder(tf.float32, shape=[batch, width], name="x")
            ids = tf.compat.v1.placeholder(tf.int64, shape=[batch, width], name="ids")
            with tf.name_scope("test_scope/pln1_0"):
                cond = tf.equal(ids, tf.constant(11, dtype=tf.int64), name="eq")

                add_vec = tf.constant(
                    add_vec_np,
                    dtype=tf.float32,
                    name="add_vec",
                )
                bias_vec = tf.constant(
                    bias_vec_np,
                    dtype=tf.float32,
                    name="bias_vec",
                )

                candidate = tf.math.add(
                    tf.math.multiply(add_vec, x, name="mul"),
                    bias_vec,
                    name="add",
                )
                out = tf.raw_ops.Select(condition=cond, x=candidate, y=x, name="select")

    return graph, out, add_vec_np, bias_vec_np


def _build_two_step_chain_graph(batch, width):
    add1_np = np.linspace(0.8, 1.2, width).astype(np.float32)
    bias1_np = np.linspace(-0.1, 0.1, width).astype(np.float32)
    add2_np = np.linspace(1.1, 0.9, width).astype(np.float32)
    bias2_np = np.linspace(0.2, -0.2, width).astype(np.float32)

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            x = tf.compat.v1.placeholder(tf.float32, shape=[batch, width], name="x")
            ids = tf.compat.v1.placeholder(tf.int64, shape=[batch, width], name="ids")
            with tf.name_scope("test_scope/pln1_0"):
                cond1 = tf.equal(ids, tf.constant(11, dtype=tf.int64), name="eq1")
                cond2 = tf.equal(ids, tf.constant(22, dtype=tf.int64), name="eq2")

                add1 = tf.constant(
                    add1_np,
                    dtype=tf.float32,
                    name="add1",
                )
                bias1 = tf.constant(
                    bias1_np,
                    dtype=tf.float32,
                    name="bias1",
                )

                add2 = tf.constant(
                    add2_np,
                    dtype=tf.float32,
                    name="add2",
                )
                bias2 = tf.constant(
                    bias2_np,
                    dtype=tf.float32,
                    name="bias2",
                )

                cand1 = tf.math.add(
                    tf.math.multiply(add1, x, name="mul1"),
                    bias1,
                    name="addv1",
                )
                s1 = tf.raw_ops.Select(condition=cond1, x=cand1, y=x, name="select1")

                cand2 = tf.math.add(
                    tf.math.multiply(add2, s1, name="mul2"),
                    bias2,
                    name="addv2",
                )
                # Second step intentionally uses select_on_true=false path.
                out = tf.raw_ops.Select(condition=cond2, x=s1, y=cand2, name="select2")

    return graph, out, add1_np, bias1_np, add2_np, bias2_np


class PlnCascadeBlockFusionTest(MUSATestCase):
    def test_single_step_pending_affine_fuses_to_pln_cascade(self):
        batch, width = 4, 8
        graph, out, add_vec, bias_vec = _build_single_step_pending_affine_graph(
            batch, width
        )

        rng = np.random.RandomState(7)
        x_np = rng.standard_normal((batch, width)).astype(np.float32)
        ids_np = rng.randint(9, 14, size=(batch, width), dtype=np.int64)

        expected = np.where(
            ids_np == 11,
            x_np * add_vec.reshape(1, -1) + bias_vec.reshape(1, -1),
            x_np,
        )

        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                out,
                feed_dict={"x:0": x_np, "ids:0": ids_np},
                options=run_opts,
                run_metadata=run_meta,
            )

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        op_counter = _partition_op_counter(run_meta.partition_graphs)
        self.assertGreaterEqual(op_counter.get("MusaPlnCascade", 0), 1)
        self.assertEqual(op_counter.get("MusaPlnCascadeBlock", 0), 0)

    def test_two_step_chain_fuses_to_single_block(self):
        batch, width = 4, 8
        graph, out, add1, bias1, add2, bias2 = _build_two_step_chain_graph(batch, width)

        rng = np.random.RandomState(11)
        x_np = rng.standard_normal((batch, width)).astype(np.float32)
        ids_np = rng.randint(10, 25, size=(batch, width), dtype=np.int64)

        s1 = np.where(
            ids_np == 11,
            x_np * add1.reshape(1, -1) + bias1.reshape(1, -1),
            x_np,
        )
        expected = np.where(
            ids_np == 22,
            s1,
            s1 * add2.reshape(1, -1) + bias2.reshape(1, -1),
        )

        config = _create_config_with_musa_optimizer()
        run_opts = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_meta = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                out,
                feed_dict={"x:0": x_np, "ids:0": ids_np},
                options=run_opts,
                run_metadata=run_meta,
            )

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

        op_counter = _partition_op_counter(run_meta.partition_graphs)
        self.assertEqual(op_counter.get("MusaPlnCascade", 0), 0)
        self.assertGreaterEqual(op_counter.get("MusaPlnCascadeBlock", 0), 1)

        block_node = _find_first_op_node(
            run_meta.partition_graphs, "MusaPlnCascadeBlock"
        )
        assert block_node is not None, "MusaPlnCascadeBlock node not found"
        self.assertIn("N", block_node.attr)
        self.assertEqual(block_node.attr["N"].i, 2)


if __name__ == "__main__":
    tf.test.main()
