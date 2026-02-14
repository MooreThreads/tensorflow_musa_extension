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

"""Tests for MUSA Interact operator."""

import os
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


def interact_ref(input_tensor):
    """Reference implementation of Interact (Dot) operation."""
    # Interaction is essentially Batch MatMul (X, X^T)
    return tf.matmul(input_tensor, input_tensor, transpose_b=True)


class InteractOpTest(MUSATestCase):
    """Tests for MUSA Interact operator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class by loading the MUSA plugin using load_op_library."""
        super(InteractOpTest, cls).setUpClass()
        
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
                print(f"SUCCESS: MUSA Interact ops loaded from {plugin_path}!")
            except Exception as e:
                print(f"FAILED: Error loading MUSA ops from {plugin_path}: {e}")
                cls._musa_ops = None
        else:
            # Provide helpful error message
            searched_locations = [os.path.normpath(path) for path in candidate_paths]
            print(f"MUSA plugin not found. Searched locations:\n" +
                  "\n".join(f"  - {loc}" for loc in searched_locations))
            cls._musa_ops = None

    def _test_interact(self, input_shape, dtype):
        """Test Interact operation with given shape and dtype."""
        # Skip if MUSA ops are not available
        if self._musa_ops is None:
            self.skipTest("MUSA Interact ops module not available")
        
        # Handle numpy dtype compatibility
        if dtype == tf.bfloat16:
            np_dtype = np.float32
        else:
            np_dtype = dtype.as_numpy_dtype
        
        # Generate random input data
        input_np = np.random.uniform(-1, 1, size=input_shape).astype(np_dtype)
        
        # Create TensorFlow constant
        input_tensor = tf.constant(input_np, dtype=dtype)
        
        # Test on CPU using reference implementation
        with tf.device('/CPU:0'):
            cpu_result = interact_ref(input_tensor)
        
        # Test on MUSA using custom op
        with tf.device('/device:MUSA:0'):
            musa_result = self._musa_ops.musa_interact(input=input_tensor)
        
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

    def testInteractBasic(self):
        """Test basic Interact operation with standard DLRM shapes."""
        # Standard DLRM interaction shape: [Batch, NumFeatures, EmbeddingDim]
        shape = [128, 27, 128]
        # Only test float32 as MusaInteract only supports float type
        self._test_interact(shape, tf.float32)

    def testInteractDifferentShapes(self):
        """Test Interact with various different shapes."""
        test_shapes = [
            [32, 10, 64],      # Small DLRM
            [64, 20, 128],     # Medium DLRM  
            [256, 30, 256],    # Large DLRM
            [16, 5, 32],       # Very small
            [512, 50, 64],     # Large batch, many features
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                self._test_interact(shape, tf.float32)

    def testInteractCornerCases(self):
        """Test Interact with corner cases like small dimensions."""
        # Test with minimal dimensions
        self._test_interact([1, 2, 1], tf.float32)
        self._test_interact([2, 1, 2], tf.float32)
        
        # Test with single element
        self._test_interact([1, 1, 1], tf.float32)
        
        # Test with large embedding dimension
        self._test_interact([32, 10, 512], tf.float32)

    def testInteractFloat32Only(self):
        """Test Interact with float32 data type (primary supported type)."""
        shape = [64, 16, 128]
        self._test_interact(shape, tf.float32)


if __name__ == "__main__":
    tf.test.main()