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
"""End-to-end tests for safe-clip-pattern -> MusaSafeClip fusion."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase
from tensorflow.core.protobuf import config_pb2

tf.compat.v1.disable_eager_execution()

def create_config_with_musa_optimizer():
    """Create ConfigProto with MUSA optimizer enabled."""
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    rewriter_config.min_graph_nodes = -1
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])
    return config


def get_musa_safe_clip_fused_nodes(run_metadata):
    """Helper to extract MusaSafeClip nodes from partition graphs."""
    return [
        node
        for partition_graph in run_metadata.partition_graphs
        for node in partition_graph.node
        if node.op == "MusaSafeClip"
    ]


class SafeClipFusionE2ETest(MUSATestCase):
    """Functional tests for graph-level safe clip fusion."""

    def _run_safe_clip_fusion_test(self, x_np, lo_np, hi_np, dtype):
        # Expected behavior of SafeClip:
        # nan -> 0.0
        # values < lo -> lo
        # values > hi -> hi
        # lo <= values <= hi -> values
        
        # Calculate expected using numpy
        clip_val = np.maximum(np.minimum(x_np, hi_np), lo_np)
        expected = np.where(np.isnan(x_np), np.zeros_like(x_np), clip_val)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(dtype, shape=x_np.shape, name="x")
                lo = tf.constant(lo_np, dtype=dtype, name="lo")
                hi = tf.constant(hi_np, dtype=dtype, name="hi")

                # Pattern: Select(IsNan(x), 0, Maximum(Minimum(x, hi), lo))
                clip_op = tf.maximum(tf.minimum(x, hi), lo)
                isnan = tf.math.is_nan(x)
                # Use a constant zero that's compatible with Select's expectations.
                # In some cases, tf.where might be mapped to Select or SelectV2.
                # The fusion pattern specifically checks for 'Select' or 'SelectV2'.
                zero = tf.constant(0.0, dtype=dtype, name="zero")
                output = tf.where(isnan, zero, clip_op)

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

        fused_nodes = get_musa_safe_clip_fused_nodes(run_metadata)
        self.assertAllClose(result, expected, atol=1e-5)
        return fused_nodes

    def test_safe_clip_fusion_is_applied(self):
        """Specifically check if the fusion happens."""
        x_np = np.array([-1.0, 0.0, 1.0, np.nan, 2.0, 3.0], dtype=np.float32)
        lo_np = np.float32(0.5)
        hi_np = np.float32(2.5)
        
        fused_nodes = self._run_safe_clip_fusion_test(x_np, lo_np, hi_np, tf.float32)
        
        # Currently, the fusion might be failing in the current environment due to 
        # how tf.where/tf.zeros are lowered. This test will help identify that.
        self.assertTrue(
            len(fused_nodes) > 0,
            "Expected Select(IsNan(x), 0, Maximum(Minimum(x, hi), lo)) to be fused into MusaSafeClip",
        )

    def test_safe_clip_fusion_fp32(self):
        x_np = np.array([-1.0, 0.0, 1.0, np.nan, 2.0, 3.0, np.inf, -np.inf], dtype=np.float32)
        lo_np = np.float32(0.5)
        hi_np = np.float32(2.5)
        self._run_safe_clip_fusion_test(x_np, lo_np, hi_np, tf.float32)

    def test_safe_clip_fusion_fp16(self):
        x_np = np.array([-1.0, 0.5, 1.0, np.nan, 2.0, 5.0], dtype=np.float16)
        lo_np = np.float16(0.2)
        hi_np = np.float16(3.0)
        self._run_safe_clip_fusion_test(x_np, lo_np, hi_np, tf.float16)

    def test_safe_clip_fusion_large_shape(self):
        shape = [2, 1024]
        x_np = np.random.uniform(-10, 10, size=shape).astype(np.float32)
        # Add some NaNs
        x_np.flat[np.random.choice(x_np.size, 10, replace=False)] = np.nan
        lo_np = np.float32(-2.0)
        hi_np = np.float32(2.0)
        self._run_safe_clip_fusion_test(x_np, lo_np, hi_np, tf.float32)

    def test_safe_clip_fusion_all_nan(self):
        x_np = np.full((10,), np.nan, dtype=np.float32)
        lo_np = np.float32(0.0)
        hi_np = np.float32(1.0)
        self._run_safe_clip_fusion_test(x_np, lo_np, hi_np, tf.float32)

    def test_safe_clip_fusion_no_nan(self):
        x_np = np.array([-1.0, 2.0, 5.0], dtype=np.float32)
        lo_np = np.float32(0.0)
        hi_np = np.float32(3.0)
        self._run_safe_clip_fusion_test(x_np, lo_np, hi_np, tf.float32)


if __name__ == "__main__":
    tf.test.main()

