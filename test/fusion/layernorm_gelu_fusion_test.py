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
"""End-to-end tests for LayerNorm and GELU fusion optimizations."""

import glob
import os
import sys
import tempfile
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")
os.environ.setdefault(
    "TF_CPP_VMODULE",
    "musa_graph_optimizer=1,gelu_fusion=1,musa_gelu_op=1",
)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEST_DIR = os.path.dirname(_CURRENT_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2


def create_config_with_musa_optimizer():
    """Create ConfigProto with the MUSA optimizer enabled."""
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"

    rewriter_config.min_graph_nodes = -1
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])

    return config


class LayerNormGeluFusionE2ETest(MUSATestCase):
    """End-to-end test for LayerNorm and GELU fusion."""

    def _run_with_timing(self, sess, output_tensor, feed_dict, tag, timed_runs=5):
        """Run a graph, printing warmup and average execution time."""
        start = time.perf_counter()
        result = sess.run(output_tensor, feed_dict=feed_dict)
        warmup_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        for _ in range(timed_runs):
            sess.run(output_tensor, feed_dict=feed_dict)
        avg_ms = ((time.perf_counter() - start) * 1000.0) / timed_runs

        print(f"  {tag} warmup: {warmup_ms:.3f} ms")
        print(f"  {tag} avg({timed_runs} runs): {avg_ms:.3f} ms")
        return result

    def _load_after_fusion_dump(self, dump_dir):
        """Load the last after_fusion dump as both text and GraphDef."""
        dump_files = sorted(glob.glob(os.path.join(dump_dir, "*_after_fusion.pbtxt")))
        self.assertTrue(dump_files, f"No after_fusion dump found in {dump_dir}")

        with open(dump_files[-1], "r", encoding="utf-8") as handle:
            dump_text = handle.read()

        graph_def = graph_pb2.GraphDef()
        text_format.Parse(dump_text, graph_def)
        return dump_text, graph_def

    def _build_exact_gelu_graph(self, input_shape):
        """Build the exact-erf GELU graph shape seen in the large model dump."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=input_shape, name="gelu_input"
                )
                rsqrt_two = tf.constant(
                    np.array([0.70710678118], dtype=np.float32), name="rsqrt_two"
                )
                div = tf.math.multiply(rsqrt_two, x, name="div_sqrt2")
                erf = tf.math.erf(div, name="erf")
                one_plus_erf = tf.math.add(
                    erf, tf.constant(1.0, dtype=tf.float32), name="one_plus_erf"
                )
                half_factor = tf.math.multiply(
                    tf.constant(0.5, dtype=tf.float32),
                    one_plus_erf,
                    name="half_factor",
                )
                y = tf.math.multiply(x, half_factor, name="gelu_output")
        return graph, x, y

    def _build_approx_gelu_graph(self, input_shape):
        """Build the tanh-approximate GELU graph."""
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=input_shape, name="gelu_input"
                )
                half_x = tf.math.multiply(
                    x, tf.constant(0.5, dtype=tf.float32), name="half_x"
                )
                x_pow3 = tf.math.pow(
                    x, tf.constant(3.0, dtype=tf.float32), name="x_pow3"
                )
                cubic_term = tf.math.multiply(
                    tf.constant(0.044715, dtype=tf.float32),
                    x_pow3,
                    name="cubic_term",
                )
                inner = tf.math.add(x, cubic_term, name="approx_inner")
                scaled_inner = tf.math.multiply(
                    tf.constant(0.7978845608, dtype=tf.float32),
                    inner,
                    name="approx_scaled_inner",
                )
                tanh = tf.math.tanh(scaled_inner, name="approx_tanh")
                one_plus_tanh = tf.math.add(
                    tf.constant(1.0, dtype=tf.float32),
                    tanh,
                    name="one_plus_tanh",
                )
                y = tf.math.multiply(one_plus_tanh, half_x, name="gelu_output")
        return graph, x, y

    def _run_gelu_case(self, approximate):
        """Run one GELU fusion case and assert the fused op is present."""
        batch_size = 8
        hidden_size = 256
        np.random.seed(123 if approximate else 42)
        x_np = np.random.randn(batch_size, hidden_size).astype(np.float32)

        if approximate:
            graph, x, output = self._build_approx_gelu_graph([None, hidden_size])
            case_name = "Approx GELU"
        else:
            graph, x, output = self._build_exact_gelu_graph([None, hidden_size])
            case_name = "Exact GELU"

        expected = tf.nn.gelu(
            tf.constant(x_np), approximate=approximate
        ).numpy()

        config = create_config_with_musa_optimizer()
        old_dump = os.environ.get("MUSA_DUMP_GRAPHDEF")
        old_dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR")

        with tempfile.TemporaryDirectory(prefix="musa_gelu_fusion_") as dump_dir:
            os.environ["MUSA_DUMP_GRAPHDEF"] = "1"
            os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = dump_dir

            try:
                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    result = self._run_with_timing(
                        sess,
                        output,
                        feed_dict={x: x_np},
                        tag=case_name,
                    )
            finally:
                if old_dump is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF"] = old_dump

                if old_dump_dir is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF_DIR", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = old_dump_dir

            dump_text, graph_def = self._load_after_fusion_dump(dump_dir)
            self.assertIn('op: "MusaGelu"', dump_text)

            fused_nodes = [node for node in graph_def.node if node.op == "MusaGelu"]
            self.assertTrue(fused_nodes, "No MusaGelu node found in after_fusion dump")
            self.assertEqual(len(fused_nodes), 1, "Expected exactly one MusaGelu node")
            self.assertEqual(
                fused_nodes[-1].attr["approximate"].b,
                approximate,
                "MusaGelu approximate attr mismatch in fused graph",
            )
            residual_original_nodes = [
                node.name for node in graph_def.node if node.name.endswith("_original")
            ]
            self.assertFalse(
                residual_original_nodes,
                f"Residual original nodes were not removed: {residual_original_nodes}",
            )
            if approximate:
                residual_ops = [node.op for node in graph_def.node if node.op in ("Tanh", "Pow")]
            else:
                residual_ops = [
                    node.op
                    for node in graph_def.node
                    if node.op in ("Erf", "Erfc", "RealDiv", "Div")
                ]
            self.assertFalse(
                residual_ops,
                f"Residual unfused GELU ops remained in graph: {residual_ops}",
            )

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)

    def test_layernorm_fusion_with_musa_device(self):
        """Test LayerNorm fusion with explicit MUSA device placement."""
        print("\n" + "=" * 70)
        print("Test: LayerNorm Fusion with MUSA Device")
        print("=" * 70)

        batch_size = 4
        seq_len = 128
        hidden_size = 768
        epsilon = 1e-12

        np.random.seed(42)
        x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        gamma_np = np.ones(hidden_size, dtype=np.float32)
        beta_np = np.zeros(hidden_size, dtype=np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, seq_len, hidden_size], name="input"
                )

                mean, var = tf.nn.moments(x, axes=[-1], keepdims=True, name="moments")
                normalized = (x - mean) / tf.sqrt(var + epsilon)

                gamma = tf.constant(gamma_np, name="gamma")
                beta = tf.constant(beta_np, name="beta")

                scaled = tf.multiply(normalized, gamma, name="mul_gamma")
                output = tf.add(scaled, beta, name="add_beta")

        config = create_config_with_musa_optimizer()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = self._run_with_timing(
                sess, output, feed_dict={x: x_np}, tag="LayerNorm"
            )

        print(f"  Output shape: {result.shape}")
        print(f"  Output mean: {result.mean():.6f}")
        print(f"  Output std: {result.std():.6f}")

    def test_exact_gelu_fusion_with_musa_device(self):
        """Test exact GELU fusion using the erf-based graph pattern."""
        print("\n" + "=" * 70)
        print("Test: Exact GELU Fusion with MUSA Device")
        print("=" * 70)
        self._run_gelu_case(approximate=False)

    def test_approximate_gelu_fusion_with_musa_device(self):
        """Test tanh-approximate GELU fusion."""
        print("\n" + "=" * 70)
        print("Test: Approximate GELU Fusion with MUSA Device")
        print("=" * 70)
        self._run_gelu_case(approximate=True)


if __name__ == "__main__":
    tf.test.main()
