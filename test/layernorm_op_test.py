# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Tests for MUSA LayerNorm operator."""

import os
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


def layernorm_ref(x, gamma, beta, eps=1e-5):
    """Reference implementation of LayerNorm."""
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
    inv = tf.math.rsqrt(var + eps)
    y = (x - mean) * inv
    return y * gamma + beta


class LayerNormOpTest(MUSATestCase):
    """Tests for MUSA LayerNorm operator."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class by loading the MUSA plugin using load_op_library."""
        super(LayerNormOpTest, cls).setUpClass()
        
        # Use the same path resolution logic as in musa_test_utils
        plugin_path = None
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Common plugin locations to try
        candidate_paths = [
            # Relative to test directory (most common case)
            os.path.join(current_dir, "..", "build", "libmusa_plugin.so"),
            # Relative to project root (when running from project root)
            os.path.join(os.path.dirname(current_dir), "build", "libmusa_plugin.so"),
            # Current working directory build
            os.path.join(os.getcwd(), "build", "libmusa_plugin.so"),
        ]

        for path in candidate_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                plugin_path = normalized_path
                break

        if plugin_path and os.path.exists(plugin_path):
            try:
                # Use tf.load_op_library instead of tf.load_library
                cls._musa_ops = tf.load_op_library(plugin_path)
                print(f"SUCCESS: MUSA LayerNorm ops loaded from {plugin_path}!")
            except Exception as e:
                print(f"FAILED: Error loading MUSA ops from {plugin_path}: {e}")
                cls._musa_ops = None
        else:
            # Provide helpful error message
            searched_locations = [os.path.normpath(path) for path in candidate_paths]
            print(f"MUSA plugin not found. Searched locations:\n" +
                  "\n".join(f"  - {loc}" for loc in searched_locations))
            cls._musa_ops = None

    def _test_layernorm(self, x_shape, dtype, eps=1e-5):
        """Test LayerNorm with given shape and dtype."""
        # Skip if MUSA ops are not available
        if self._musa_ops is None:
            self.skipTest("MUSA LayerNorm ops module not available")
        
        # Handle numpy dtype compatibility
        if dtype == tf.bfloat16:
            np_dtype = np.float32
        else:
            np_dtype = dtype.as_numpy_dtype
        
        # Generate random input data
        x_np = np.random.uniform(-10, 10, size=x_shape).astype(np_dtype)
        gamma_np = np.ones((x_shape[-1],), dtype=np_dtype)
        beta_np = np.zeros((x_shape[-1],), dtype=np_dtype)
        
        # Create TensorFlow constants
        x = tf.constant(x_np, dtype=dtype)
        gamma = tf.constant(gamma_np, dtype=dtype)
        beta = tf.constant(beta_np, dtype=dtype)
        
        # Test on CPU using reference implementation
        with tf.device('/CPU:0'):
            cpu_result = layernorm_ref(x, gamma, beta, eps)
        
        # Test on MUSA using custom op
        with tf.device('/device:MUSA:0'):
            musa_result = self._musa_ops.musa_layer_norm(x=x, gamma=gamma, beta=beta, epsilon=eps)
        
        # Compare results
        if dtype in [tf.float16, tf.bfloat16]:
            cpu_result_f32 = tf.cast(cpu_result, tf.float32)
            musa_result_f32 = tf.cast(musa_result, tf.float32)
            rtol = 1e-2 if dtype == tf.float16 else 1e-3
            atol = 1e-2 if dtype == tf.float16 else 1e-3
            self.assertAllClose(cpu_result_f32.numpy(), musa_result_f32.numpy(), 
                               rtol=rtol, atol=atol)
        else:
            rtol = 1e-5
            atol = 1e-5
            self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), 
                               rtol=rtol, atol=atol)

    def testLayerNormBasic(self):
        """Test basic LayerNorm operation with standard shapes."""
        for dtype in [tf.float32, tf.float16]:
            with self.subTest(dtype=dtype):
                self._test_layernorm([1024, 1024], dtype)

    def testLayerNormDifferentShapes(self):
        """Test LayerNorm with various different shapes."""
        test_shapes = [
            [1, 64],           # Small vector
            [32, 128],         # Medium matrix  
            [256, 512],        # Large matrix
            [8, 16, 32],       # 3D tensor
            [2, 4, 8, 16],     # 4D tensor
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                self._test_layernorm(shape, tf.float32)

    def testLayerNormCornerCases(self):
        """Test LayerNorm with corner cases like small dimensions."""
        # Test with very small last dimension
        self._test_layernorm([100, 1], tf.float32)
        self._test_layernorm([100, 2], tf.float32)
        
        # Test with single element
        self._test_layernorm([1, 1], tf.float32)
        
        # Test with large last dimension
        self._test_layernorm([10, 8192], tf.float32)

    def testLayerNormDifferentEpsilons(self):
        """Test LayerNorm with different epsilon values."""
        shape = [256, 512]
        x_np = np.random.uniform(-10, 10, size=shape).astype(np.float32)
        gamma_np = np.ones((shape[-1],), dtype=np.float32)
        beta_np = np.zeros((shape[-1],), dtype=np.float32)
        
        x = tf.constant(x_np, dtype=tf.float32)
        gamma = tf.constant(gamma_np, dtype=tf.float32)
        beta = tf.constant(beta_np, dtype=tf.float32)
        
        test_epsilons = [1e-5, 1e-3, 1e-1]
        
        for eps in test_epsilons:
            with self.subTest(epsilon=eps):
                # Test on CPU using reference implementation
                with tf.device('/CPU:0'):
                    cpu_result = layernorm_ref(x, gamma, beta, eps)
                
                # Skip if MUSA ops are not available
                if self._musa_ops is None:
                    self.skipTest("MUSA LayerNorm ops module not available")
                
                # Test on MUSA using custom op
                with tf.device('/device:MUSA:0'):
                    musa_result = self._musa_ops.musa_layer_norm(x=x, gamma=gamma, beta=beta, epsilon=eps)
                
                self.assertAllClose(cpu_result.numpy(), musa_result.numpy(), 
                                   rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tf.test.main()