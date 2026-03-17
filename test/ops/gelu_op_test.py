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

"""Tests for the direct MusaGelu fused operator."""

import os
import sys
import tempfile

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEST_DIR = os.path.dirname(_CURRENT_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from musa_test_utils import MUSATestCase


class GeluOpTest(MUSATestCase):
    """Tests for the direct MusaGelu custom op."""

    @classmethod
    def setUpClass(cls):
        super(GeluOpTest, cls).setUpClass()

        plugin_path = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(current_dir, "..", "..", "build", "libmusa_plugin.so"),
            os.path.join(os.path.dirname(current_dir), "..", "build", "libmusa_plugin.so"),
            os.path.join(os.getcwd(), "build", "libmusa_plugin.so"),
            os.path.join(os.getcwd(), "..", "build", "libmusa_plugin.so"),
        ]

        for path in candidate_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                plugin_path = normalized_path
                break

        if plugin_path and os.path.exists(plugin_path):
            cls._musa_ops = tf.load_op_library(plugin_path)
        else:
            searched_locations = [os.path.normpath(path) for path in candidate_paths]
            raise FileNotFoundError(
                "MUSA plugin not found. Searched locations:\n"
                + "\n".join(f"  - {loc}" for loc in searched_locations)
            )

    def _dump_graph(self, graph, tag):
        """Dump GraphDef to a temporary pbtxt file and print a node summary."""
        graph_def = graph.as_graph_def()
        graph_text = text_format.MessageToString(graph_def)

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"musa_gelu_{tag}_",
            suffix=".pbtxt",
            delete=False,
        ) as handle:
            handle.write(graph_text)
            dump_path = handle.name

        print("\n" + "=" * 70)
        print(f"MusaGelu {tag} Graph")
        print("=" * 70)
        print(f"Graph dumped to: {dump_path}")
        print("Nodes:")
        for node in graph_def.node:
            print(f"  {node.name:24s} op={node.op:12s} inputs={list(node.input)}")
        print("\nGraphDef:")
        print(graph_text)

        return graph_def, dump_path

    def _build_musa_gelu_graph(self, approximate):
        """Build a graph with a direct MusaGelu op."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 8], name="input"
                )
                y = self._musa_ops.musa_gelu(
                    x=x,
                    approximate=approximate,
                    name="musa_gelu_output",
                )
        return graph, x, y

    def _run_and_check(self, approximate):
        """Build, dump, and run one direct MusaGelu graph."""
        graph, x, y = self._build_musa_gelu_graph(approximate=approximate)
        graph_def, _ = self._dump_graph(
            graph, "approx" if approximate else "exact"
        )

        gelu_nodes = [node for node in graph_def.node if node.op == "MusaGelu"]
        self.assertEqual(len(gelu_nodes), 1, "Expected exactly one MusaGelu node")
        self.assertEqual(gelu_nodes[0].attr["approximate"].b, approximate)

        np.random.seed(123 if approximate else 42)
        x_np = np.random.randn(4, 8).astype(np.float32)
        expected = tf.nn.gelu(tf.constant(x_np), approximate=approximate).numpy()

        with tf.compat.v1.Session(graph=graph) as sess:
            result = sess.run(y, feed_dict={x: x_np})

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

    def test_musa_gelu_exact_graph(self):
        """Build and run the exact MusaGelu graph."""
        self._run_and_check(approximate=False)

    def test_musa_gelu_approximate_graph(self):
        """Build and run the approximate MusaGelu graph."""
        self._run_and_check(approximate=True)


if __name__ == "__main__":
    tf.test.main()
