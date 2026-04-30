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

"""Tests for Linear+Activation fusion."""

import os
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


class LinearActivationFusionTest(MUSATestCase):
    """Tests for Linear+Activation fusion."""

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

                # This pattern should be matched by LinearActivationFusion
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
        """Verify that Linear+Relu fusion is applied as MusaLinearActivation."""
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
                if node.op == "MusaLinearActivation":
                    has_fused_node = True
                    break

        self.assertTrue(has_fused_node, "MusaLinearActivation fusion was NOT applied to the graph")

    def test_linear_relu_fusion_various_batch_sizes(self):
        """Test fusion correctness across several batch sizes."""
        np.random.seed(7)
        tf.random.set_seed(7)

        k, n = 6, 10
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        for m in (1, 3, 8):
            x_np = np.random.randn(m, k).astype(np.float32)

            # Reference on CPU
            with tf.device('/CPU:0'):
                x_tf = tf.constant(x_np)
                w_tf = tf.constant(w_np)
                b_tf = tf.constant(b_np)
                expected = tf.nn.relu(tf.nn.bias_add(tf.matmul(x_tf, w_tf), b_tf)) * 1.5

            # MUSA graph
            graph = tf.Graph()
            with graph.as_default():
                with tf.device('/device:MUSA:0'):
                    x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x_bs")
                    w = tf.constant(w_np, dtype=tf.float32, name="w_bs")
                    b = tf.constant(b_np, dtype=tf.float32, name="b_bs")

                    mm = tf.matmul(x, w)
                    bias = tf.nn.bias_add(mm, b)
                    out = tf.nn.relu(bias)
                    # extra consumer
                    out = out * 1.5

            config = create_config_with_musa_optimizer()
            with tf.compat.v1.Session(graph=graph, config=config) as sess:
                actual = sess.run(out, feed_dict={x: x_np})

            self.assertAllClose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)

    def test_linear_relu_fusion_not_applied_with_intervening_op(self):
        """If an extra op exists between MatMul and BiasAdd, fusion should not occur."""
        m, k, n = 2, 5, 7
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x_int")
                w = tf.constant(w_np, dtype=tf.float32, name="w_int")
                b = tf.constant(b_np, dtype=tf.float32, name="b_int")

                mm = tf.matmul(x, w)
                # Insert an identity (or any intervening op) to block fusion
                mid = tf.identity(mm, name="intervening_identity")
                bias = tf.nn.bias_add(mid, b)
                relu_out = tf.nn.relu(bias)
                output = relu_out * 2.0

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(output, feed_dict={x: x_np}, options=run_options, run_metadata=run_metadata)

        # Ensure fused node is NOT present when intervening op exists
        has_fused_node = False
        for partition_graph in run_metadata.partition_graphs:
            for node in partition_graph.node:
                if node.op == "MusaLinearActivation":
                    has_fused_node = True
                    break

        self.assertFalse(has_fused_node, "MusaLinearActivation fusion should NOT be applied when an intervening op exists")

    def test_linear_relu_fusion_dtypes(self):
        """Test fusion correctness across multiple dtypes: float32, float16, bfloat16."""
        np.random.seed(21)
        tf.random.set_seed(21)

        m, k, n = 3, 6, 8
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        dtypes = [tf.float32, tf.float16, tf.bfloat16]

        for dtype in dtypes:
            x_np = np.random.randn(m, k).astype(np.float32)

            # Reference computed in float32
            with tf.device('/CPU:0'):
                x_tf = tf.constant(x_np, dtype=tf.float32)
                w_tf = tf.constant(w_np, dtype=tf.float32)
                b_tf = tf.constant(b_np, dtype=tf.float32)
                expected = tf.nn.relu(tf.nn.bias_add(tf.matmul(x_tf, w_tf), b_tf)) * 0.75
                expected_f32 = expected.numpy()

            # Build MUSA graph: accept float32 feeds then cast to target dtype inside graph
            graph = tf.Graph()
            with graph.as_default():
                with tf.device('/device:MUSA:0'):
                    x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x_dt")
                    x = tf.cast(x_ph, dtype)
                    w = tf.constant(w_np, dtype=dtype, name="w_dt")
                    b = tf.constant(b_np, dtype=dtype, name="b_dt")

                    mm = tf.matmul(x, w)
                    bias = tf.nn.bias_add(mm, b)
                    out = tf.nn.relu(bias)
                    out = out * tf.constant(0.75, dtype=dtype)
                    # cast back to float32 for stable comparison
                    out_f32 = tf.cast(out, tf.float32)

            config = create_config_with_musa_optimizer()
            with tf.compat.v1.Session(graph=graph, config=config) as sess:
                actual = sess.run(out_f32, feed_dict={x_ph: x_np})

            # Tolerances adjusted for reduced-precision dtypes
            if dtype == tf.float32:
                rtol, atol = 1e-5, 1e-5
            elif dtype == tf.float16:
                rtol, atol = 1e-2, 1e-2
            else:  # bfloat16
                rtol, atol = 2e-2, 2e-2

            self.assertAllClose(actual, expected_f32, rtol=rtol, atol=atol)

    def test_linear_relu_fusion_large_features(self):
        """Optional large-feature test. Enable by setting MUSA_RUN_LARGE_TESTS=1."""
        if not os.environ.get("MUSA_RUN_LARGE_TESTS"):
            self.skipTest("Large tests disabled; set MUSA_RUN_LARGE_TESTS=1 to run")

        np.random.seed(321)
        tf.random.set_seed(321)

        # Larger feature dims but smaller batch to balance memory
        m, k, n = 128, 2048, 1024
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        # Reference on CPU
        with tf.device('/CPU:0'):
            expected = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.constant(x_np), tf.constant(w_np)), tf.constant(b_np))) * 0.9

        # MUSA graph
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x_large_feat")
                w = tf.constant(w_np, dtype=tf.float32, name="w_large_feat")
                b = tf.constant(b_np, dtype=tf.float32, name="b_large_feat")

                mm = tf.matmul(x, w)
                bias = tf.nn.bias_add(mm, b)
                out = tf.nn.relu(bias) * 0.9

        config = create_config_with_musa_optimizer()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual = sess.run(out, feed_dict={x: x_np})

        self.assertAllClose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)

    def test_linear_relu_fusion_large_batch(self):
        """Optional large-batch test. Enable by setting MUSA_RUN_LARGE_TESTS=1."""
        if not os.environ.get("MUSA_RUN_LARGE_TESTS"):
            self.skipTest("Large tests disabled; set MUSA_RUN_LARGE_TESTS=1 to run")

        np.random.seed(123)
        tf.random.set_seed(123)

        # Larger but reasonable sizes to exercise throughput without OOM on typical test machines
        m, k, n = 2048, 512, 512
        x_np = np.random.randn(m, k).astype(np.float32)
        w_np = np.random.randn(k, n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        # Reference on CPU
        with tf.device('/CPU:0'):
            expected = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.constant(x_np), tf.constant(w_np)), tf.constant(b_np))) * 1.0

        # MUSA graph
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, k], name="x_large_batch")
                w = tf.constant(w_np, dtype=tf.float32, name="w_large_batch")
                b = tf.constant(b_np, dtype=tf.float32, name="b_large_batch")

                mm = tf.matmul(x, w)
                bias = tf.nn.bias_add(mm, b)
                out = tf.nn.relu(bias)

        config = create_config_with_musa_optimizer()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            actual = sess.run(out, feed_dict={x: x_np})

        self.assertAllClose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tf.test.main()
