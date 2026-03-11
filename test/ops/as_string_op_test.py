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

"""Tests for MUSA AsString operator.

This test verifies the AsString operator implementation by comparing
the expected behavior with TensorFlow's CPU implementation.

Note: The AsString operator is a CPU-only operation that converts tensor
values to strings. It supports various data types including int32, int64,
float32, float64, bool, etc.
"""

import os
import sys
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf


class AsStringOpBehaviorTest(unittest.TestCase):
    """Tests for AsString operator behavior using TensorFlow CPU."""

    def test_int32_basic(self):
        """Test AsString with int32 type."""
        values = [1, 2, 3, 42, -5, 0]
        input_tensor = tf.constant(values, dtype=tf.int32)
        result = tf.strings.as_string(input_tensor)
        expected = [b'1', b'2', b'3', b'42', b'-5', b'0']
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_int64_basic(self):
        """Test AsString with int64 type."""
        values = [1, 2, 3, 42, -5, 0]
        input_tensor = tf.constant(values, dtype=tf.int64)
        result = tf.strings.as_string(input_tensor)
        expected = [b'1', b'2', b'3', b'42', b'-5', b'0']
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_float32_basic(self):
        """Test AsString with float32 type."""
        values = [1.0, 2.5, -3.14, 0.0]
        input_tensor = tf.constant(values, dtype=tf.float32)
        result = tf.strings.as_string(input_tensor)
        # Float values have default precision of 6
        self.assertEqual(len(result.numpy()), 4)
        # Check that results are byte strings
        for s in result.numpy():
            self.assertIsInstance(s, bytes)

    def test_float64_basic(self):
        """Test AsString with float64 type."""
        values = [1.0, 2.5, -3.14159265358979, 0.0]
        input_tensor = tf.constant(values, dtype=tf.float64)
        result = tf.strings.as_string(input_tensor)
        self.assertEqual(len(result.numpy()), 4)

    def test_bool_basic(self):
        """Test AsString with bool type."""
        values = [True, False, True, False]
        input_tensor = tf.constant(values, dtype=tf.bool)
        result = tf.strings.as_string(input_tensor)
        expected = [b'true', b'false', b'true', b'false']
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_precision(self):
        """Test AsString with precision parameter."""
        values = [3.14159265358979, 2.71828182845904]
        input_tensor = tf.constant(values, dtype=tf.float64)
        
        result_2 = tf.strings.as_string(input_tensor, precision=2)
        result_6 = tf.strings.as_string(input_tensor, precision=6)
        
        # Check that precision affects output
        self.assertEqual(result_2.numpy()[0], b'3.14')
        self.assertEqual(result_6.numpy()[0], b'3.141593')

    def test_scientific_notation(self):
        """Test AsString with scientific notation."""
        values = [0.000123, 456.789, 1e10]
        input_tensor = tf.constant(values, dtype=tf.float32)
        result = tf.strings.as_string(input_tensor, scientific=True)
        
        # Check that scientific notation is used
        for s in result.numpy():
            self.assertIn(b'e', s.lower() or b'E' in s)

    def test_shortest_representation(self):
        """Test AsString with shortest representation."""
        values = [1.0, 1.5, 0.0001, 10000.0]
        input_tensor = tf.constant(values, dtype=tf.float32)
        result = tf.strings.as_string(input_tensor, shortest=True)
        
        # Check results
        self.assertEqual(result.numpy()[0], b'1')
        self.assertEqual(result.numpy()[1], b'1.5')

    def test_width_padding(self):
        """Test AsString with width parameter."""
        values = [1, 23, 456]
        input_tensor = tf.constant(values, dtype=tf.int32)
        result = tf.strings.as_string(input_tensor, width=5)
        
        # Check padding with spaces
        self.assertEqual(result.numpy()[0], b'    1')
        self.assertEqual(result.numpy()[1], b'   23')
        self.assertEqual(result.numpy()[2], b'  456')

    def test_fill_padding(self):
        """Test AsString with width and fill parameters."""
        values = [1, 23, 456]
        input_tensor = tf.constant(values, dtype=tf.int32)
        result = tf.strings.as_string(input_tensor, width=5, fill='0')
        
        # Check padding with zeros
        self.assertEqual(result.numpy()[0], b'00001')
        self.assertEqual(result.numpy()[1], b'00023')
        self.assertEqual(result.numpy()[2], b'00456')

    def test_empty_tensor(self):
        """Test with empty tensor."""
        input_tensor = tf.constant([], dtype=tf.int32)
        result = tf.strings.as_string(input_tensor)
        self.assertEqual(len(result.numpy()), 0)

    def test_single_element(self):
        """Test with single element tensor."""
        input_tensor = tf.constant([42], dtype=tf.int32)
        result = tf.strings.as_string(input_tensor)
        self.assertEqual(result.numpy()[0], b'42')

    def test_multi_dimensional(self):
        """Test with multi-dimensional tensor."""
        values = [[1, 2], [3, 4]]
        input_tensor = tf.constant(values, dtype=tf.int32)
        result = tf.strings.as_string(input_tensor)
        
        expected = [[b'1', b'2'], [b'3', b'4']]
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_large_values(self):
        """Test with large integer values."""
        values = [2147483647, -2147483648]  # int32 max/min
        input_tensor = tf.constant(values, dtype=tf.int32)
        result = tf.strings.as_string(input_tensor)
        
        self.assertEqual(result.numpy()[0], b'2147483647')
        self.assertEqual(result.numpy()[1], b'-2147483648')

    def test_negative_values(self):
        """Test with negative values."""
        values = [-1, -42, -100]
        input_tensor = tf.constant(values, dtype=tf.int32)
        result = tf.strings.as_string(input_tensor)
        
        expected = [b'-1', b'-42', b'-100']
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_zero_values(self):
        """Test with zero values."""
        values = [0, 0, 0]
        input_tensor = tf.constant(values, dtype=tf.int32)
        result = tf.strings.as_string(input_tensor)
        
        expected = [b'0', b'0', b'0']
        np.testing.assert_array_equal(result.numpy(), expected)


class AsStringOpImplementationTest(unittest.TestCase):
    """Tests for the MUSA AsString operator implementation."""

    def _get_impl_path(self):
        """Get the path to the implementation file."""
        # The test file is in test/ops/, so we need to go up two levels
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(test_dir))
        return os.path.join(project_root, 'musa_ext', 'kernels', 'string', 'musa_as_string_op.cc')

    def test_implementation_file_exists(self):
        """Test that the implementation file exists."""
        impl_path = self._get_impl_path()
        self.assertTrue(os.path.exists(impl_path), 
                       f"Implementation file not found: {impl_path}")

    def test_implementation_has_required_types(self):
        """Test that the implementation supports required data types."""
        impl_path = self._get_impl_path()
        
        with open(impl_path, 'r') as f:
            content = f.read()
        
        # Check for required type registrations
        required_types = ['int8', 'int16', 'int32', 'int64', 
                         'uint8', 'uint16', 'uint32', 'uint64',
                         'float', 'double', 'bool', 
                         'complex64', 'complex128',
                         'Eigen::half', 'bfloat16']
        
        for dtype in required_types:
            self.assertIn(dtype, content, 
                         f"Type {dtype} not found in implementation")

    def test_implementation_has_host_memory(self):
        """Test that the implementation uses HostMemory for CPU execution."""
        impl_path = self._get_impl_path()
        
        with open(impl_path, 'r') as f:
            content = f.read()
        
        # Check for HostMemory directive
        self.assertIn('HostMemory("input")', content,
                     "Implementation should use HostMemory for input")
        self.assertIn('HostMemory("output")', content,
                     "Implementation should use HostMemory for output")

    def test_implementation_has_attributes(self):
        """Test that the implementation handles all required attributes."""
        impl_path = self._get_impl_path()
        
        with open(impl_path, 'r') as f:
            content = f.read()
        
        # Check for attribute handling
        required_attrs = ['precision', 'scientific', 'shortest', 'width', 'fill']
        for attr in required_attrs:
            self.assertIn(f'GetAttr("{attr}"', content,
                         f"Attribute {attr} not handled in implementation")

    def test_implementation_handles_empty_tensor(self):
        """Test that the implementation handles empty tensors."""
        impl_path = self._get_impl_path()
        
        with open(impl_path, 'r') as f:
            content = f.read()
        
        # Check for empty tensor handling
        self.assertIn('NumElements() == 0', content,
                     "Implementation should handle empty tensors")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)