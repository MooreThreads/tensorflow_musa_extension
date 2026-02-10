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

"""Tests for MUSA Softmax and LogSoftmax operators using MUSATestCase."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class SoftmaxOpTest(MUSATestCase):
    """Test class for Softmax and LogSoftmax on MUSA."""

    def _test_softmax_common(self, shape, dtype, axis=-1, log=False, rtol=1e-5, atol=1e-8):
        """Helper to test softmax/log_softmax using _compare_cpu_musa_results."""
        print(f"Testing shape={shape}, dtype={dtype}, axis={axis}, log={log}")
        
        # Prepare input data
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
        input_np = np.random.randn(*shape).astype(np_dtype)
        
        # Determine which op to test
        op_func = tf.nn.log_softmax if log else tf.nn.softmax

        # Create a wrapper to pass the 'axis' parameter
        def op_wrapper(x):
            return op_func(x, axis=axis)

        # Prepare input tensor
        if dtype == tf.bfloat16:
            input_tensor = tf.cast(tf.constant(input_np), tf.bfloat16)
        else:
            input_tensor = tf.constant(input_np, dtype=dtype)

        # Run comparison
        self._compare_cpu_musa_results(
            op_wrapper,
            [input_tensor],
            dtype=dtype,
            rtol=rtol,
            atol=atol
        )

    def testSoftmaxBasic(self):
        """Test basic softmax with common data types."""
        shape = [2, 8]
        # For float16/bfloat16, use relaxed tolerance
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            self._test_softmax_common(shape, dtype, log=False, rtol=rtol, atol=atol)

    def testLogSoftmaxBasic(self):
        """Test basic log_softmax with common data types."""
        shape = [2, 8]
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            self._test_softmax_common(shape, dtype, log=True, rtol=rtol, atol=atol)

    def testVariousAxes(self):
        """Test softmax with various different shapes and axes."""
        test_shapes = [
            [10],               # 1D
            [10, 10],           # 2D Square
            [3, 4, 5],          # 3D
            [2, 3, 4, 5],       # 4D
        ]
        # Use float32 for these tests to ensure high precision
        for shape in test_shapes:
            for axis in range(len(shape)):
                # Test Softmax
                self._test_softmax_common(shape, tf.float32, axis=axis, log=False)
                # Test LogSoftmax
                self._test_softmax_common(shape, tf.float32, axis=axis, log=True)

    def testLargeInput(self):
        """Test with a larger input size to check stability/performance."""
        shape = [4, 1024]
        self._test_softmax_common(shape, tf.float32, axis=-1, log=False)

if __name__ == "__main__":
    tf.test.main()
