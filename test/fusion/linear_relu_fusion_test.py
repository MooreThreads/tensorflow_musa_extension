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

"""Tests for Linear+Relu fusion."""

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
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])

    return config


class LinearReluFusionTest(MUSATestCase):
    """Tests for Linear+Relu fusion."""

    def test_linear_relu_fusion_basic(self):
        """Test Linear+Relu pattern fusion."""
        # Set seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Define shapes
        m, k, n = 4, 8, 16
        
        # Input data
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        # Reference implementation (CPU)
        with tf.device('/CPU:0'):
            x_tf = tf.constant(x_np)
            w_tf = tf.constant(w_np)
            b_tf = tf.constant(b_np)
            
            mm = tf.matmul(x_tf, w_tf)
            bias = tf.nn.bias_add(mm, b_tf)
            expected_out = tf.nn.relu(bias)
            # Add a consumer to ensure it's not pruned and has someone to redirect to
            expected_out = expected_out * 2.0

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="b")
                
                # This pattern should be matched by LinearReluFusion
                mm_musa = tf.matmul(x, w)
                bias_musa = tf.nn.bias_add(mm_musa, b)
                relu_out = tf.nn.relu(bias_musa)
                # Add a consumer node
                output = relu_out * 2.0

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual_out = sess.run(output, feed_dict={x: x_np})

        # Verification
        self.assertAllClose(actual_out, expected_out.numpy(), rtol=1e-5, atol=1e-5)

    def test_linear_relu_fusion_applied(self):
        """Verify that Linear+Relu fusion is applied: MusaLinearRelu node exists in optimized graph."""
        m, k, n = 4, 8, 16
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x")
                w = tf.constant(w_np, dtype=tf.float32, name="w")
                b = tf.constant(b_np, dtype=tf.float32, name="b")
                
                mm_musa = tf.matmul(x, w)
                bias_musa = tf.nn.bias_add(mm_musa, b)
                relu_out = tf.nn.relu(bias_musa)
                # Add a consumer node
                output = relu_out * 2.0

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(output, feed_dict={x: x_np}, options=run_options, run_metadata=run_metadata)

        # Check for fused node
        has_fused_node = False
        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                if node.op == "MusaLinearRelu":
                    has_fused_node = True
                    break

        self.assertTrue(has_fused_node, "MusaLinearRelu fusion was NOT applied to the graph")


if __name__ == "__main__":
    tf.test.main()
