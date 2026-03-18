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
"""End-to-end tests for LayerNorm-core fusion optimization."""

import glob
import os
import sys
import tempfile
import time

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEST_DIR = os.path.dirname(_CURRENT_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from musa_test_utils import MUSATestCase
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


_FEATURE_DIM = 768
_HALF_DIM = _FEATURE_DIM // 2
_CLIP_UPPER = np.float32(10000000.0)
_CLIP_LOWER = np.float32(9.99999996e-12)
_LAYERNORM_KERNEL_ENV = "MUSA_ENABLE_LAYERNORM_FUSION_KERNEL"
_RUN_PERF_TESTS_ENV = "MUSA_RUN_PERF_TESTS"


def create_config_with_musa_optimizer():
    """Create ConfigProto with only the MUSA optimizer enabled."""
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


def normalize_core_numpy(left_np, right_np, scale_np):
    """Reference implementation for the fused normalize-core pattern."""
    x = np.concatenate([left_np, right_np], axis=1)
    mu = np.mean(x, axis=1, keepdims=True)
    xc = x - mu
    var = np.mean(xc * xc, axis=1, keepdims=True)
    std = np.clip(np.sqrt(var), _CLIP_LOWER, _CLIP_UPPER)
    gamma = 1.0 + scale_np.reshape(1, -1)
    return (xc / std) * gamma


class LayerNormFusionE2ETest(MUSATestCase):
    """End-to-end tests for MusaLayerNorm fusion."""

    def _load_after_fusion_dump(self, dump_dir):
        dump_files = sorted(glob.glob(os.path.join(dump_dir, "*_after_fusion.pbtxt")))
        self.assertTrue(dump_files, f"No after_fusion dump found in {dump_dir}")

        with open(dump_files[-1], "r", encoding="utf-8") as handle:
            dump_text = handle.read()

        graph_def = graph_pb2.GraphDef()
        text_format.Parse(dump_text, graph_def)
        return dump_text, graph_def

    def _run_case(
        self,
        left_np,
        right_np,
        scale_np,
        enable_kernel,
        dump_graph=False,
        warmup_iters=0,
        measure_iters=0,
    ):
        graph, left, right, scale, output = self._build_graph()
        config = create_config_with_musa_optimizer()
        feed_dict = {left: left_np, right: right_np, scale: scale_np}

        old_kernel_flag = os.environ.get(_LAYERNORM_KERNEL_ENV)
        old_dump = os.environ.get("MUSA_DUMP_GRAPHDEF")
        old_dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR")

        dump_text = None
        graph_def = None
        avg_latency_ms = None
        result = None

        with tempfile.TemporaryDirectory(prefix="musa_layernorm_fusion_") as dump_dir:
            os.environ[_LAYERNORM_KERNEL_ENV] = "1" if enable_kernel else "0"
            if dump_graph:
                os.environ["MUSA_DUMP_GRAPHDEF"] = "1"
                os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = dump_dir

            try:
                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    for _ in range(warmup_iters):
                        sess.run(output, feed_dict=feed_dict)

                    if measure_iters > 0:
                        start = time.perf_counter()
                        for _ in range(measure_iters):
                            result = sess.run(output, feed_dict=feed_dict)
                        elapsed = time.perf_counter() - start
                        avg_latency_ms = elapsed * 1000.0 / measure_iters
                    else:
                        result = sess.run(output, feed_dict=feed_dict)
            finally:
                if old_kernel_flag is None:
                    os.environ.pop(_LAYERNORM_KERNEL_ENV, None)
                else:
                    os.environ[_LAYERNORM_KERNEL_ENV] = old_kernel_flag

                if old_dump is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF"] = old_dump

                if old_dump_dir is None:
                    os.environ.pop("MUSA_DUMP_GRAPHDEF_DIR", None)
                else:
                    os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = old_dump_dir

            if dump_graph:
                dump_text, graph_def = self._load_after_fusion_dump(dump_dir)

        return result, dump_text, graph_def, avg_latency_ms

    def _build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                left = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, _HALF_DIM], name="left_input"
                )
                right = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, _HALF_DIM], name="right_input"
                )
                scale = tf.compat.v1.placeholder(
                    tf.float32, shape=[_FEATURE_DIM], name="scale_input"
                )

                concat_axis = tf.constant(1, dtype=tf.int32, name="concat_axis")
                concat = tf.raw_ops.ConcatV2(
                    values=[left, right], axis=concat_axis, name="concat"
                )

                reduce_axis = tf.constant(1, dtype=tf.int32, name="reduce_axis")
                expand_dim = tf.constant(-1, dtype=tf.int32, name="expand_dim")
                mean = tf.raw_ops.Mean(
                    input=concat,
                    axis=reduce_axis,
                    keep_dims=False,
                    name="mean",
                )
                mean_expand = tf.raw_ops.ExpandDims(
                    input=mean, axis=expand_dim, name="mean_expand"
                )

                sub_a = tf.subtract(concat, mean_expand, name="sub_a")
                sub_b = tf.subtract(concat, mean_expand, name="sub_b")
                sub_c = tf.subtract(concat, mean_expand, name="sub_c")

                var_mul = tf.multiply(sub_b, sub_c, name="var_mul")
                var_mean = tf.raw_ops.Mean(
                    input=var_mul,
                    axis=reduce_axis,
                    keep_dims=False,
                    name="var_mean",
                )
                var_expand = tf.raw_ops.ExpandDims(
                    input=var_mean, axis=expand_dim, name="var_expand"
                )

                std = tf.sqrt(var_expand, name="std")
                std_min = tf.minimum(
                    std,
                    tf.constant(_CLIP_UPPER, dtype=tf.float32, name="clip_upper"),
                    name="std_min",
                )
                std_max = tf.maximum(
                    std_min,
                    tf.constant(_CLIP_LOWER, dtype=tf.float32, name="clip_lower"),
                    name="std_max",
                )
                normalized = tf.raw_ops.RealDiv(
                    x=sub_a, y=std_max, name="normalized"
                )

                scale_expand_dim = tf.constant(
                    0, dtype=tf.int32, name="scale_expand_dim"
                )
                scale_expand = tf.raw_ops.ExpandDims(
                    input=scale, axis=scale_expand_dim, name="scale_expand"
                )
                gamma = tf.raw_ops.AddV2(
                    x=tf.constant(1.0, dtype=tf.float32, name="gamma_one"),
                    y=scale_expand,
                    name="gamma",
                )
                output = tf.multiply(normalized, gamma, name="layernorm_output")

        return graph, left, right, scale, output

    def test_layernorm_core_fusion(self):
        rng = np.random.RandomState(42)
        batch_size = 8

        left_np = rng.standard_normal((batch_size, _HALF_DIM)).astype(np.float32)
        right_np = rng.standard_normal((batch_size, _HALF_DIM)).astype(np.float32)
        scale_np = rng.uniform(-0.1, 0.1, size=(_FEATURE_DIM,)).astype(np.float32)

        expected = normalize_core_numpy(left_np, right_np, scale_np)
        result, dump_text, graph_def, _ = self._run_case(
            left_np, right_np, scale_np, enable_kernel=True, dump_graph=True
        )

        self.assertIn('op: "MusaLayerNorm"', dump_text)

        fused_nodes = [node for node in graph_def.node if node.op == "MusaLayerNorm"]
        self.assertEqual(len(fused_nodes), 1, "Expected exactly one fused LayerNorm node")
        self.assertEqual(len(fused_nodes[0].input), 3)
        self.assertAllClose(
            fused_nodes[0].attr["epsilon"].f,
            np.float32(_CLIP_LOWER * _CLIP_LOWER),
            rtol=1e-5,
            atol=1e-28,
        )

        residual_ops = {
            node.op
            for node in graph_def.node
            if node.op in ("RealDiv", "Div", "Mean", "Sqrt", "Minimum", "Maximum")
        }
        self.assertFalse(
            residual_ops,
            f"Residual normalize-core ops remained in graph: {sorted(residual_ops)}",
        )

        self.assertTrue(
            any(node.op == "ZerosLike" for node in graph_def.node),
            "Expected fused graph to synthesize a beta=ZerosLike input",
        )

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

    def test_layernorm_core_kernel_gate(self):
        rng = np.random.RandomState(123)
        batch_size = 8

        left_np = rng.standard_normal((batch_size, _HALF_DIM)).astype(np.float32)
        right_np = rng.standard_normal((batch_size, _HALF_DIM)).astype(np.float32)
        scale_np = rng.uniform(-0.1, 0.1, size=(_FEATURE_DIM,)).astype(np.float32)
        expected = normalize_core_numpy(left_np, right_np, scale_np)

        fused_result, fused_dump, fused_graph, _ = self._run_case(
            left_np, right_np, scale_np, enable_kernel=True, dump_graph=True
        )
        fallback_result, fallback_dump, fallback_graph, _ = self._run_case(
            left_np, right_np, scale_np, enable_kernel=False, dump_graph=True
        )

        self.assertIn('op: "MusaLayerNorm"', fused_dump)
        self.assertNotIn('op: "MusaLayerNorm"', fallback_dump)
        self.assertTrue(
            any(node.op == "MusaLayerNorm" for node in fused_graph.node),
            "Expected fused graph to contain MusaLayerNorm",
        )
        self.assertFalse(
            any(node.op == "MusaLayerNorm" for node in fallback_graph.node),
            "Expected fallback graph to keep the original normalize-core ops",
        )

        self.assertAllClose(fused_result, expected, rtol=1e-5, atol=1e-5)
        self.assertAllClose(fallback_result, expected, rtol=1e-5, atol=1e-5)
        self.assertAllClose(fused_result, fallback_result, rtol=1e-5, atol=1e-5)

    def test_layernorm_core_perf_compare(self):
        if os.environ.get(_RUN_PERF_TESTS_ENV) != "1":
            self.skipTest(
                f"Set {_RUN_PERF_TESTS_ENV}=1 to run LayerNorm fusion performance comparison"
            )

        rng = np.random.RandomState(2026)
        batch_size = 2048
        warmup_iters = 20
        measure_iters = 50

        left_np = rng.standard_normal((batch_size, _HALF_DIM)).astype(np.float32)
        right_np = rng.standard_normal((batch_size, _HALF_DIM)).astype(np.float32)
        scale_np = rng.uniform(-0.1, 0.1, size=(_FEATURE_DIM,)).astype(np.float32)

        fused_result, fused_dump, _, _ = self._run_case(
            left_np,
            right_np,
            scale_np,
            enable_kernel=True,
            dump_graph=True,
        )
        fallback_result, fallback_dump, _, _ = self._run_case(
            left_np,
            right_np,
            scale_np,
            enable_kernel=False,
            dump_graph=True,
        )
        _, _, _, fused_ms = self._run_case(
            left_np,
            right_np,
            scale_np,
            enable_kernel=True,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
        )
        _, _, _, fallback_ms = self._run_case(
            left_np,
            right_np,
            scale_np,
            enable_kernel=False,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
        )

        self.assertIn('op: "MusaLayerNorm"', fused_dump)
        self.assertNotIn('op: "MusaLayerNorm"', fallback_dump)
        self.assertIsNotNone(fused_ms)
        self.assertIsNotNone(fallback_ms)
        self.assertGreater(fused_ms, 0.0)
        self.assertGreater(fallback_ms, 0.0)
        self.assertAllClose(fused_result, fallback_result, rtol=1e-5, atol=1e-5)

        speedup_pct = (fallback_ms - fused_ms) / fallback_ms * 100.0
        print("\n" + "=" * 70)
        print("LayerNorm Fusion Performance Comparison")
        print("=" * 70)
        print(f"  Entry shape: ConcatV2 -> [{batch_size}, {_FEATURE_DIM}]")
        print(f"  Fused avg latency:    {fused_ms:.3f} ms")
        print(f"  Unfused avg latency:  {fallback_ms:.3f} ms")
        print(f"  Relative uplift:      {speedup_pct:.2f}%")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    tf.test.main()
