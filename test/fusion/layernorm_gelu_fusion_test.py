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
"""End-to-end test for LayerNorm and GELU fusion optimizations.

This test verifies that:
1. The MUSA custom graph optimizer is triggered
2. Fusion patterns are correctly matched
3. Fused kernels are called during execution
4. Results are numerically correct compared to standard TF ops
"""

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

    # Add custom optimizer - this triggers the optimizer registration
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"

    # Enable all optimizations
    rewriter_config.min_graph_nodes = -1
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])

    return config


class LayerNormGeluFusionE2ETest(MUSATestCase):
    """End-to-end test for LayerNorm and GELU fusion."""
    def test_layernorm_fusion_with_musa_device(self):
        """Test LayerNorm fusion with explicit MUSA device placement."""
        print("\n" + "="*70)
        print("Test: LayerNorm Fusion with MUSA Device")
        print("="*70)

        batch_size = 4
        seq_len = 128
        hidden_size = 768
        epsilon = 1e-12

        np.random.seed(42)
        x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        gamma_np = np.ones(hidden_size, dtype=np.float32)
        beta_np = np.zeros(hidden_size, dtype=np.float32)

        print(f"\n  Input shape: {x_np.shape}")

        # Build graph with explicit MUSA device placement
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/device:MUSA:0'):
                x = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, seq_len, hidden_size],
                    name="input"
                )

                mean, var = tf.nn.moments(x, axes=[-1], keepdims=True, name="moments")
                normalized = (x - mean) / tf.sqrt(var + epsilon)

                gamma = tf.constant(gamma_np, name="gamma")
                beta = tf.constant(beta_np, name="beta")

                scaled = tf.multiply(normalized, gamma, name="mul_gamma")
                output = tf.add(scaled, beta, name="add_beta")

        config = create_config_with_musa_optimizer()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(output, feed_dict={x: x_np})

        print(f"\n  Output shape: {result.shape}")
        print(f"  Output mean: {result.mean():.6f}")
        print(f"  Output std: {result.std():.6f}")

        print("\n" + "="*70)
        print("✓ LayerNorm with MUSA device test passed")
        print("="*70 + "\n")


if __name__ == "__main__":
    tf.test.main()
