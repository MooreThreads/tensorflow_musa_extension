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

"""Graph-level tests for MusaReshapeMatMul fusion."""

import unittest
import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import op_def_registry

from musa_test_utils import load_musa_plugin

load_musa_plugin()


def create_config_with_musa_optimizer():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rw = config.graph_options.rewrite_options
    rw.min_graph_nodes = -1
    rw.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
    rw.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.shape_optimization = rewriter_config_pb2.RewriterConfig.OFF

    custom_opt = rw.custom_optimizers.add()
    custom_opt.name = "musa_graph_optimizer"
    rw.optimizers.extend(["musa_graph_optimizer"])
    return config


def get_dumped_fused_nodes(dump_dir):
    after_fusion = [
        os.path.join(dump_dir, name)
        for name in os.listdir(dump_dir)
        if "_after_fusion." in name
    ]
    if not after_fusion:
        return []

    path = sorted(after_fusion)[-1]
    with tf.io.gfile.GFile(path, "rb" if path.endswith(".pb") else "r") as f:
        content = f.read()
    needle = b'MusaReshapeMatMul' if isinstance(content, bytes) else "MusaReshapeMatMul"
    return [path] if needle in content else []


class ReshapeMatMulFusionTest(tf.test.TestCase):
    """Tests for Reshape->MatMul(Const)->Reshape fusion."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if op_def_registry.get("MusaReshapeMatMul") is None:
            raise unittest.SkipTest(
                "MusaReshapeMatMul is not registered. Rebuild plugin first."
            )

    def test_fusion_applied(self):
        x_np = np.random.randn(8, 4, 16).astype(np.float32)
        w_np = np.random.randn(16, 32).astype(np.float32)

        expected = np.matmul(x_np.reshape(-1, 16), w_np).reshape(8, 4, 32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4, 16], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                flat = tf.reshape(x, [-1, 16], name="flat")
                mm = tf.matmul(flat, w, name="matmul")
                d0, d1, _ = tf.unstack(tf.shape(x), name="shape_unpack")
                output = tf.reshape(mm, tf.stack([d0, d1, 32], name="restore_shape"),
                                    name="restore")

        config = create_config_with_musa_optimizer()
        with tempfile.TemporaryDirectory() as dump_dir:
            old_dump = os.environ.get("MUSA_DUMP_GRAPHDEF")
            old_dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR")
            os.environ["MUSA_DUMP_GRAPHDEF"] = "1"
            os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = dump_dir
            try:
                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    result = sess.run(output, feed_dict={x: x_np})
            finally:
                if old_dump is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF"] = old_dump
                if old_dump_dir is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF_DIR", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = old_dump_dir

            fused_nodes = get_dumped_fused_nodes(dump_dir)
        self.assertAllClose(result, expected, rtol=1e-2, atol=1e-2)
        self.assertTrue(
            fused_nodes,
            "Expected Reshape->MatMul(Const)->Reshape to be fused",
        )

    def test_fusion_not_applied_for_non_restore_shape(self):
        x_np = np.random.randn(8, 4, 16).astype(np.float32)
        w_np = np.random.randn(16, 32).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4, 16], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                flat = tf.reshape(x, [-1, 16], name="flat")
                mm = tf.matmul(flat, w, name="matmul")
                d0, d1, _ = tf.unstack(tf.shape(x), name="shape_unpack")
                output = tf.reshape(mm, tf.stack([d0, 32, d1], name="restore_shape"),
                                    name="restore")

        config = create_config_with_musa_optimizer()
        with tempfile.TemporaryDirectory() as dump_dir:
            old_dump = os.environ.get("MUSA_DUMP_GRAPHDEF")
            old_dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR")
            os.environ["MUSA_DUMP_GRAPHDEF"] = "1"
            os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = dump_dir
            try:
                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    _ = sess.run(output, feed_dict={x: x_np})
            finally:
                if old_dump is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF"] = old_dump
                if old_dump_dir is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF_DIR", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = old_dump_dir

            fused_nodes = get_dumped_fused_nodes(dump_dir)
        self.assertFalse(
            fused_nodes,
            "Did not expect invalid restore shape pattern to be fused",
        )

    def test_fusion_applied_for_unpack_const_prefix_shape(self):
        x_np = np.random.randn(8, 13, 16).astype(np.float32)
        w_np = np.random.randn(16, 32).astype(np.float32)

        expected = np.matmul(x_np.reshape(-1, 16), w_np).reshape(8, 13, 32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13, 16], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                flat = tf.reshape(x, [-1, 16], name="flat")
                mm = tf.matmul(flat, w, name="matmul")
                d0, _, _ = tf.unstack(tf.shape(x), name="shape_unpack")
                output = tf.reshape(
                    mm,
                    tf.stack([d0, 13, 32], name="restore_shape"),
                    name="restore",
                )

        config = create_config_with_musa_optimizer()
        with tempfile.TemporaryDirectory() as dump_dir:
            old_dump = os.environ.get("MUSA_DUMP_GRAPHDEF")
            old_dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR")
            os.environ["MUSA_DUMP_GRAPHDEF"] = "1"
            os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = dump_dir
            try:
                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    result = sess.run(output, feed_dict={x: x_np})
            finally:
                if old_dump is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF"] = old_dump
                if old_dump_dir is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF_DIR", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = old_dump_dir

            fused_nodes = get_dumped_fused_nodes(dump_dir)
        self.assertAllClose(result, expected, rtol=1e-2, atol=1e-2)
        self.assertTrue(
            fused_nodes,
            "Expected mixed Unpack/Const restore shape pattern to be fused",
        )


if __name__ == "__main__":
    tf.test.main()
