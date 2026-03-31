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
"""End-to-end tests for FuseLayerNormV2 fusion optimization."""

import glob
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
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


_D0 = 4
_D1 = 8
_D2 = 16
_EPSILON = 1e-5


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


def layernorm_numpy(x_np, gamma_np, beta_np, epsilon):
    mean = np.mean(x_np, axis=-1, keepdims=True)
    var = np.mean((x_np - mean) * (x_np - mean), axis=-1, keepdims=True)
    normalized = (x_np - mean) / np.sqrt(var + epsilon)
    return normalized * gamma_np.reshape(1, 1, -1) + beta_np.reshape(1, 1, -1)


class FuseLayerNormV2FusionE2ETest(MUSATestCase):
    """End-to-end tests for MusaFuseLayerNormV2 fusion."""

    def _load_after_fusion_dump(self, dump_dir):
        dump_files = sorted(glob.glob(os.path.join(dump_dir, "*_after_fusion.pbtxt")))
        self.assertTrue(dump_files, f"No after_fusion dump found in {dump_dir}")

        with open(dump_files[-1], "r", encoding="utf-8") as handle:
            dump_text = handle.read()

        graph_def = graph_pb2.GraphDef()
        text_format.Parse(dump_text, graph_def)
        return dump_text, graph_def

    def _build_graph(
        self,
        epsilon,
        use_equivalent_split_dims=False,
        use_mismatched_dims=False,
        use_invalid_reshape_shape=False,
    ):
        if use_equivalent_split_dims and use_mismatched_dims:
            raise ValueError(
                "use_equivalent_split_dims and use_mismatched_dims cannot both be True"
            )

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[_D0, _D1, _D2], name="x_input"
                )
                gamma = tf.compat.v1.placeholder(
                    tf.float32, shape=[_D2], name="gamma_input"
                )
                beta = tf.compat.v1.placeholder(
                    tf.float32, shape=[_D2], name="beta_input"
                )

                shape_x = tf.raw_ops.Shape(input=x, out_type=tf.int32, name="shape_x")

                d0 = tf.raw_ops.Slice(
                    input=shape_x,
                    begin=tf.constant([0], dtype=tf.int32, name="d0_begin"),
                    size=tf.constant([1], dtype=tf.int32, name="d0_size"),
                    name="d0",
                )
                d1 = tf.raw_ops.Slice(
                    input=shape_x,
                    begin=tf.constant([1], dtype=tf.int32, name="d1_begin"),
                    size=tf.constant([1], dtype=tf.int32, name="d1_size"),
                    name="d1",
                )
                d2 = tf.raw_ops.Slice(
                    input=shape_x,
                    begin=tf.constant([2], dtype=tf.int32, name="d2_begin"),
                    size=tf.constant([1], dtype=tf.int32, name="d2_size"),
                    name="d2",
                )

                scalar_shape = tf.constant([], dtype=tf.int32, name="scalar_shape")
                d0_scalar = tf.raw_ops.Reshape(
                    tensor=d0, shape=scalar_shape, name="d0_scalar"
                )
                d1_scalar = tf.raw_ops.Reshape(
                    tensor=d1, shape=scalar_shape, name="d1_scalar"
                )
                d2_scalar = tf.raw_ops.Reshape(
                    tensor=d2, shape=scalar_shape, name="d2_scalar"
                )

                d0d1_scalar = tf.multiply(d0_scalar, d1_scalar, name="d0d1_scalar")

                if use_invalid_reshape_shape:
                    reshape_n = tf.constant(2, dtype=tf.int32, name="reshape_n_invalid")
                    reshape_channel = tf.raw_ops.FloorDiv(
                        x=d0d1_scalar,
                        y=tf.constant(2, dtype=tf.int32, name="reshape_channel_divisor"),
                        name="reshape_channel_invalid",
                    )
                else:
                    reshape_n = tf.constant(1, dtype=tf.int32, name="reshape_n")
                    reshape_channel = d0d1_scalar

                reshape_w = tf.constant(1, dtype=tf.int32, name="reshape_w")
                reshape4_shape = tf.raw_ops.Pack(
                    values=[reshape_n, reshape_channel, d2_scalar, reshape_w],
                    axis=0,
                    name="reshape4_shape",
                )

                x4 = tf.raw_ops.Reshape(tensor=x, shape=reshape4_shape, name="reshape_to_4d")

                one_f = tf.constant(1.0, dtype=tf.float32, name="one_f")
                zero_f = tf.constant(0.0, dtype=tf.float32, name="zero_f")
                if use_equivalent_split_dims:
                    scale_dims = tf.raw_ops.Pack(
                        values=[reshape_channel], axis=0, name="scale_dims_pack"
                    )
                    channel_alias = tf.identity(
                        reshape_channel, name="reshape_channel_alias"
                    )
                    offset_dims = tf.raw_ops.Pack(
                        values=[channel_alias], axis=0, name="offset_dims_pack"
                    )
                elif use_mismatched_dims:
                    scale_dims = tf.raw_ops.Pack(
                        values=[reshape_channel], axis=0, name="scale_dims_pack"
                    )
                    offset_dims = tf.raw_ops.Pack(
                        values=[
                            tf.constant(
                                _D0 * _D1, dtype=tf.int32, name="offset_dims_const"
                            )
                        ],
                        axis=0,
                        name="offset_dims_pack",
                    )
                else:
                    shared_dims = tf.raw_ops.Pack(
                        values=[reshape_channel], axis=0, name="shared_dims_pack"
                    )
                    scale_dims = shared_dims
                    offset_dims = shared_dims

                scale = tf.raw_ops.Fill(dims=scale_dims, value=one_f, name="scale_fill")
                offset = tf.raw_ops.Fill(dims=offset_dims, value=zero_f, name="offset_fill")
                mean = tf.raw_ops.Fill(dims=scale_dims, value=zero_f, name="mean_fill")
                variance = tf.raw_ops.Fill(dims=scale_dims, value=one_f, name="var_fill")

                y4, _, _, _, _, _ = tf.raw_ops.FusedBatchNormV3(
                    x=x4,
                    scale=scale,
                    offset=offset,
                    mean=mean,
                    variance=variance,
                    epsilon=epsilon,
                    exponential_avg_factor=1.0,
                    data_format="NCHW",
                    is_training=True,
                    name="bn_v3",
                )

                y = tf.raw_ops.Reshape(
                    tensor=y4,
                    shape=tf.raw_ops.Shape(input=x, out_type=tf.int32, name="shape_for_restore"),
                    name="reshape_back_3d",
                )
                y_scaled = tf.multiply(y, gamma, name="mul_gamma")
                out = tf.raw_ops.AddV2(x=y_scaled, y=beta, name="fuselayernormv2_out")

        return graph, x, gamma, beta, out

    def _run_case(
        self,
        x_np,
        gamma_np,
        beta_np,
        epsilon,
        dump_graph,
        use_equivalent_split_dims=False,
        use_mismatched_dims=False,
        use_invalid_reshape_shape=False,
    ):
        graph, x, gamma, beta, output = self._build_graph(
            epsilon,
            use_equivalent_split_dims=use_equivalent_split_dims,
            use_mismatched_dims=use_mismatched_dims,
            use_invalid_reshape_shape=use_invalid_reshape_shape,
        )
        config = create_config_with_musa_optimizer()
        feed_dict = {x: x_np, gamma: gamma_np, beta: beta_np}

        old_dump = os.environ.get("MUSA_DUMP_GRAPHDEF")
        old_dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR")

        dump_text = None
        graph_def = None
        result = None

        with tempfile.TemporaryDirectory(prefix="musa_fuselayernormv2_") as dump_dir:
            if dump_graph:
                os.environ["MUSA_DUMP_GRAPHDEF"] = "1"
                os.environ["MUSA_DUMP_GRAPHDEF_DIR"] = dump_dir

            try:
                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    result = sess.run(output, feed_dict=feed_dict)
            finally:
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

        return result, dump_text, graph_def

    def _assert_fusion_state(self, dump_text, graph_def, expect_fused):
        fused_nodes = [node for node in graph_def.node if node.op == "MusaLayerNorm"]
        residual_bn = [node.name for node in graph_def.node if node.op == "FusedBatchNormV3"]
        if expect_fused:
            self.assertIn('op: "MusaLayerNorm"', dump_text)
            self.assertEqual(
                len(fused_nodes), 1, "Expected exactly one MusaLayerNorm node"
            )
            self.assertFalse(
                residual_bn,
                f"Residual FusedBatchNormV3 nodes remained in fused graph: {residual_bn}",
            )
        else:
            self.assertNotIn('op: "MusaLayerNorm"', dump_text)
            self.assertEqual(
                len(fused_nodes), 0, "MusaLayerNorm should not be created for this pattern"
            )
            self.assertTrue(
                residual_bn,
                "Expected FusedBatchNormV3 to remain when fusion should not trigger",
            )

    def test_fuselayernormv2_fusion_is_applied(self):
        rng = np.random.RandomState(2026)
        x_np = rng.standard_normal((_D0, _D1, _D2)).astype(np.float32)
        gamma_np = rng.uniform(0.8, 1.2, size=(_D2,)).astype(np.float32)
        beta_np = rng.uniform(-0.1, 0.1, size=(_D2,)).astype(np.float32)

        expected = layernorm_numpy(x_np, gamma_np, beta_np, _EPSILON)
        result, dump_text, graph_def = self._run_case(
            x_np=x_np,
            gamma_np=gamma_np,
            beta_np=beta_np,
            epsilon=_EPSILON,
            dump_graph=True,
        )

        self.assertIsNotNone(dump_text)
        self.assertIsNotNone(graph_def)
        self._assert_fusion_state(dump_text, graph_def, expect_fused=True)

        fused_nodes = [node for node in graph_def.node if node.op == "MusaLayerNorm"]
        self.assertEqual(len(fused_nodes[0].input), 3)
        self.assertAllClose(
            fused_nodes[0].attr["epsilon"].f,
            np.float32(_EPSILON),
            rtol=1e-6,
            atol=1e-8,
        )

        self.assertAllClose(result, expected, rtol=1e-4, atol=1e-4)

    def test_fuselayernormv2_fusion_with_equivalent_split_dims(self):
        rng = np.random.RandomState(2027)
        x_np = rng.standard_normal((_D0, _D1, _D2)).astype(np.float32)
        gamma_np = rng.uniform(0.8, 1.2, size=(_D2,)).astype(np.float32)
        beta_np = rng.uniform(-0.1, 0.1, size=(_D2,)).astype(np.float32)

        _, dump_text, graph_def = self._run_case(
            x_np=x_np,
            gamma_np=gamma_np,
            beta_np=beta_np,
            epsilon=_EPSILON,
            dump_graph=True,
            use_equivalent_split_dims=True,
        )

        self.assertIsNotNone(dump_text)
        self.assertIsNotNone(graph_def)
        self._assert_fusion_state(dump_text, graph_def, expect_fused=True)

    def test_fuselayernormv2_no_fusion_for_mismatched_dims(self):
        rng = np.random.RandomState(2028)
        x_np = rng.standard_normal((_D0, _D1, _D2)).astype(np.float32)
        gamma_np = rng.uniform(0.8, 1.2, size=(_D2,)).astype(np.float32)
        beta_np = rng.uniform(-0.1, 0.1, size=(_D2,)).astype(np.float32)

        _, dump_text, graph_def = self._run_case(
            x_np=x_np,
            gamma_np=gamma_np,
            beta_np=beta_np,
            epsilon=_EPSILON,
            dump_graph=True,
            use_mismatched_dims=True,
        )

        self.assertIsNotNone(dump_text)
        self.assertIsNotNone(graph_def)
        self._assert_fusion_state(dump_text, graph_def, expect_fused=False)

    def test_fuselayernormv2_no_fusion_for_invalid_reshape_shape(self):
        rng = np.random.RandomState(2029)
        x_np = rng.standard_normal((_D0, _D1, _D2)).astype(np.float32)
        gamma_np = rng.uniform(0.8, 1.2, size=(_D2,)).astype(np.float32)
        beta_np = rng.uniform(-0.1, 0.1, size=(_D2,)).astype(np.float32)

        _, dump_text, graph_def = self._run_case(
            x_np=x_np,
            gamma_np=gamma_np,
            beta_np=beta_np,
            epsilon=_EPSILON,
            dump_graph=True,
            use_invalid_reshape_shape=True,
        )

        self.assertIsNotNone(dump_text)
        self.assertIsNotNone(graph_def)
        self._assert_fusion_state(dump_text, graph_def, expect_fused=False)


if __name__ == "__main__":
    tf.test.main()
