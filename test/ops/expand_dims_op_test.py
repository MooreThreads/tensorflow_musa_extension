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
# =============================================================================
"""Tests for MUSA ExpandDims operator."""

import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase


class ExpandDimsOpTest(MUSATestCase):
    """Tests for MUSA ExpandDims operator."""

    def _test_expand_dims(self, input_shape, axis, dtype):
        """Test expand_dims operation with given parameters."""
        # Generate random input data
        if dtype in [np.int32, np.int64]:
            x_np = np.random.randint(-10, 10, size=input_shape).astype(dtype)
        elif dtype == np.bool_:
            x_np = np.random.choice([True, False], size=input_shape)
        else:  # float types
            if len(input_shape) == 0:
                # Handle scalar input
                x_np = np.array(np.random.randn()).astype(dtype)
            else:
                x_np = np.random.randn(*input_shape).astype(dtype)

        # Handle bfloat16 dtype conversion
        tf_dtype = tf.bfloat16 if dtype == tf.bfloat16.as_numpy_dtype else dtype
        x = tf.constant(x_np, dtype=tf_dtype)

        # Define the operation as a lambda to pass to the utility method
        def expand_dims_op(tensor):
            return tf.expand_dims(tensor, axis=axis)

        # Compare CPU and MUSA results
        # expand_dims is a shape/identity op, so we can use very strict tolerances
        self._compare_cpu_musa_results(expand_dims_op, [x], tf_dtype, rtol=0, atol=0)

    def testExpandDimsBasicCases(self):
        """Test various basic cases for expand_dims."""
        basic_cases = [
            ([10], 0, "Float32 (Wide 侧常用)"),
            ([10], 1, "末尾增加维度"),
            ([10], -1, "负索引末尾"),
            ([3, 5], 1, "中间增加维度"),
            ([2, 3, 4], 0, "开头增加维度"),
            ([2, 3, 4], 3, "4D 扩展"),
            ([100, 256], -2, "Deep 侧 Embedding 常用"),
        ]

        for shape, axis, desc in basic_cases:
            with self.subTest(description=desc, shape=shape, axis=axis):
                self._test_expand_dims(shape, axis, np.float32)

    def testExpandDimsDataTypes(self):
        """Test expand_dims with different data types."""
        dtypes_to_test = {
            "int32": np.int32,
            "int64": np.int64,
            "float32": np.float32,
            "bool": np.bool_,
            "bfloat16": tf.bfloat16.as_numpy_dtype,
        }

        for name, dtype in dtypes_to_test.items():
            with self.subTest(data_type=name):
                self._test_expand_dims([5, 5], 0, dtype)

    def testExpandDimsEdgeCases(self):
        """Test expand_dims with edge cases."""
        # Scalar input (0-D tensor)
        with self.subTest(description="Scalar input, axis=0"):
            self._test_expand_dims([], 0, np.float32)

        # Large tensor
        with self.subTest(description="Large tensor 128x128"):
            self._test_expand_dims([128, 128], 0, np.float32)

        with self.subTest(description="Large tensor, axis=-1"):
            self._test_expand_dims([128, 128], -1, np.float32)

        # 5D tensor
        with self.subTest(description="5D tensor"):
            self._test_expand_dims([2, 3, 4, 5, 6], 2, np.float32)

        # uint8 type (common in image processing)
        with self.subTest(description="uint8 type"):
            self._test_expand_dims([10, 10], 1, np.uint8)

        # float64 type
        with self.subTest(description="float64 type"):
            self._test_expand_dims([5, 5], 0, np.float64)

    def testExpandDimsNegativeAxes(self):
        """Test expand_dims with various negative axis values."""
        negative_axis_cases = [
            ([2, 3, 4], -1, "3D, axis=-1"),
            ([2, 3, 4], -2, "3D, axis=-2"),
            ([2, 3, 4], -3, "3D, axis=-3"),
            ([2, 3, 4], -4, "3D, axis=-4 (start)"),
            ([10, 20, 30, 40], -2, "4D, axis=-2"),
        ]

        for shape, axis, desc in negative_axis_cases:
            with self.subTest(description=desc):
                self._test_expand_dims(shape, axis, np.float32)


if __name__ == "__main__":
    tf.test.main()
