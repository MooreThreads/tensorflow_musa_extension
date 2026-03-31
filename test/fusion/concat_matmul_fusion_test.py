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

"""Tests for ConcatV2+MatMul fusion."""

import os
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

from tensorflow.core.protobuf import config_pb2

def create_config_with_musa_optimizer():
    """Create ConfigProto with MUSA optimizer enabled."""
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"

    rewriter_config.min_graph_nodes = -1
    return config

class ConcatMatMulFusionTest(MUSATestCase):
    """Tests for ConcatV2+MatMul fusion."""

    def test_concat_matmul_fusion_basic(self):
        """Test ConcatV2 + MatMul pattern fusion."""
        # Set seeds for reproducibility
        np.random.seed(42)
        tf.compat.v1.set_random_seed(42)

        # Define shapes
        shape1 = [2, 16]
        shape2 = [2, 16]
        weight_shape = [32, 8]
        
        # Data for inputs
        np_a = np.random.randn(*shape1).astype(np.float32)
        np_b = np.random.randn(*shape2).astype(np.float32)
        np_w = np.random.randn(*weight_shape).astype(np.float32)

        # Reference implementation (CPU)
        with tf.device('/CPU:0'):
            a_tf = tf.constant(np_a)
            b_tf = tf.constant(np_b)
            w_tf = tf.constant(np_w)
            
            concat_cpu = tf.concat([a_tf, b_tf], axis=1)
            expected_out = tf.matmul(concat_cpu, w_tf)

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                a = tf.compat.v1.placeholder(tf.float32, shape=shape1, name="input_a")
                b = tf.compat.v1.placeholder(tf.float32, shape=shape2, name="input_b")
                w = tf.constant(np_w, dtype=tf.float32, name="weight")
                
                # Concat + MatMul pattern
                concat_node = tf.concat([a, b], axis=1, name="concat")
                matmul_node = tf.matmul(concat_node, w, name="matmul")
                # Add a consumer to ensure it's not pruned
                output = matmul_node * 1.0

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual_out = sess.run(output, feed_dict={a: np_a, b: np_b})

        # Verification
        self.assertAllClose(actual_out, expected_out.numpy(), rtol=1e-5, atol=1e-5)
        print("Successfully ran ConcatMatMul fusion test and verified results")

    def test_concat_matmul_fusion_applied(self):
        """Verify that ConcatV2+MatMul fusion is applied: MusaConcatMatMul node exists in optimized graph."""
        # Define shapes
        shape1 = [2, 16]
        shape2 = [2, 16]
        weight_shape = [32, 8]
        
        # Data for inputs
        np_a = np.random.randn(*shape1).astype(np.float32)
        np_b = np.random.randn(*shape2).astype(np.float32)
        np_w = np.random.randn(*weight_shape).astype(np.float32)

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                a = tf.compat.v1.placeholder(tf.float32, shape=shape1, name="input_a")
                b = tf.compat.v1.placeholder(tf.float32, shape=shape2, name="input_b")
                w = tf.constant(np_w, dtype=tf.float32, name="weight")
                
                # Concat + MatMul pattern
                concat_node = tf.concat([a, b], axis=1, name="concat")
                matmul_node = tf.matmul(concat_node, w, name="matmul")
                # Add a consumer to ensure it's not pruned
                output = matmul_node * 1.0

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(output, feed_dict={a: np_a, b: np_b}, 
                     options=run_options, run_metadata=run_metadata)

        # Check for MusaConcatMatMul node in partitioned graphs
        has_fused_node = False
        fused_node_name = ""
        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                if "MusaConcatMatMul" in node.op:
                    has_fused_node = True
                    fused_node_name = node.name
                    break
        
        self.assertTrue(has_fused_node, "MusaConcatMatMul fusion was NOT applied to the graph")
        print(f"Verified: Found fused node '{fused_node_name}' with op 'MusaConcatMatMul'")

if __name__ == "__main__":
    tf.test.main()

