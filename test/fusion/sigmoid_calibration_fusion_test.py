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
"""End-to-end test for SigmoidCalibration fusion optimization.

This test verifies that:
1. The MUSA custom graph optimizer is triggered
2. The S / (S + Scale * (1 - S)) pattern is correctly matched
3. The fused MusaSigmoidCalibration kernel is called during execution
4. Results are numerically correct compared to standard TF ops on CPU
"""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


def create_config_with_musa_optimizer():
    """Create ConfigProto with MUSA optimizer enabled."""
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rewriter_config = config.graph_options.rewrite_options

    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"

    rewriter_config.min_graph_nodes = -1
    # rewriter_config.optimizers.extend(["musa_graph_optimizer"])

    return config


def sigmoid_calibration_numpy(x, scale):
    """NumPy reference implementation of SigmoidCalibration."""
    s = 1.0 / (1.0 + np.exp(-x))
    return s / (s + scale * (1.0 - s))


class SigmoidCalibrationFusionE2ETest(MUSATestCase):
    """End-to-end test for SigmoidCalibration fusion."""

    def test_sigmoid_calibration_fusion_basic(self):
        """Test basic SigmoidCalibration fusion with typical dimensions."""
        batch_size = 4
        height = 128
        width = 128
        channels = 3

        np.random.seed(42)
        x_np = np.random.randn(batch_size, height, width, channels).astype(np.float32)
        scale_val = 2.0

        print(f"\n  Input shape: {x_np.shape}")
        print(f"  Scale value: {scale_val}")

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, height, width, channels],
                    name="input"
                )
                
                # SigmoidCalibration pattern: S / (S + Scale * (1 - S))
                s = tf.sigmoid(x, name="sigmoid")
                one = tf.constant(1.0, dtype=tf.float32, name="one")
                one_minus_s = tf.subtract(one, s, name="sub")
                scale = tf.constant(scale_val, dtype=tf.float32, name="scale")
                scaled_one_minus_s = tf.multiply(scale, one_minus_s, name="mul")
                denom = tf.add(s, scaled_one_minus_s, name="add")
                output = tf.divide(s, denom, name="div")

        # Run on MUSA with optimizer enabled
        config = create_config_with_musa_optimizer()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            # Check if fusion happened (this depends on your implementation of optimizer logging/VLOG)
            # In some cases we might just verify numerical correctness
            musa_result = sess.run(output, feed_dict={x: x_np})

        # Reference result from NumPy
        ref_result = sigmoid_calibration_numpy(x_np, scale_val)

        # Numerical verification
        self.assertAllClose(musa_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_sigmoid_calibration_fusion_applied(self):
        """Test if SigmoidCalibration fusion is actually applied by checking the graph."""
        batch_size = 1
        height = 16
        width = 16
        channels = 1

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, height, width, channels], name="input")
                s = tf.sigmoid(x, name="sigmoid")
                one = tf.constant(1.0, dtype=tf.float32, name="one")
                one_minus_s = tf.subtract(one, s, name="sub")
                scale = tf.constant(2.0, dtype=tf.float32, name="scale")
                scaled_one_minus_s = tf.multiply(scale, one_minus_s, name="mul")
                denom = tf.add(s, scaled_one_minus_s, name="add")
                output = tf.divide(s, denom, name="div")

        config = create_config_with_musa_optimizer()
        # Enable Grappler debug output to capture the optimized graph
        config.graph_options.rewrite_options.min_graph_nodes = -1
        
        metadata = config_pb2.RunMetadata()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            # First run to trigger optimization
            sess.run(output, feed_dict={x: np.random.randn(batch_size, height, width, channels).astype(np.float32)})
            
            # The optimized graph can be retrieved from the session if needed, 
            # but usually we check the VLOG output or rely on the existence of the fused node in the callable
            
            # Access the partitioned graphs (this is one way to see if the op changed)
            opts = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            sess.run(output, feed_dict={x: np.random.randn(batch_size, height, width, channels).astype(np.float32)},
                     options=opts, run_metadata=metadata)
            
            # Search for MusaSigmoidCalibration in the executed graph partitions
            found_fused_op = False
            for dev_stats in metadata.step_stats.dev_stats:
                for node_stats in dev_stats.node_stats:
                    if "MusaSigmoidCalibration" in node_stats.node_name or \
                       "fused_sigmoid_calibration" in node_stats.node_name:
                        found_fused_op = True
                        print(f"  Found fused op: {node_stats.node_name}")
                        break
                if found_fused_op: break
            
            self.assertTrue(found_fused_op, "SigmoidCalibration fusion was NOT applied!")

if __name__ == "__main__":
    tf.test.main()
