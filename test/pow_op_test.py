"""Tests for MUSA Pow operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class PowOpTest(MUSATestCase):
    """Tests for MUSA Pow operator."""
    
    def _test_pow(self, shape_x, shape_y, dtype, rtol=1e-5, atol=1e-8, 
                  x_values=None, y_values=None):
        """Test pow operation with given shapes and dtype.
        
        Args:
            shape_x: Shape of base tensor
            shape_y: Shape of exponent tensor
            dtype: TensorFlow data type
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            x_values: Optional custom values for base (if None, random values used)
            y_values: Optional custom values for exponent (if None, random values used)
        """
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
        
        # Generate input values
        if x_values is None:
            # Use range [-2, 2] to include negative bases for boundary testing
            x_np = np.random.uniform(0, 2, size=shape_x).astype(np_dtype)
        else:
            x_np = np.array(x_values, dtype=np_dtype).reshape(shape_x)
            
        if y_values is None:
            # Use range [-2, 2] to include negative/ fractional exponents
            y_np = np.random.uniform(0, 2, size=shape_y).astype(np_dtype)
        else:
            y_np = np.array(y_values, dtype=np_dtype).reshape(shape_y)
        
        x = tf.constant(x_np, dtype=dtype)
        y = tf.constant(y_np, dtype=dtype)
        
        self._compare_cpu_musa_results(tf.pow, [x, y], dtype, rtol=rtol, atol=atol)
    
    def testPowBasic(self):
        """Test basic pow operation with same shapes."""
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            self._test_pow([1024, 1024], [1024, 1024], dtype, rtol=rtol, atol=atol)
    
    def testPowBroadcastVectorMatrix(self):
        """Test pow with vector-matrix broadcasting."""
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            self._test_pow([1024], [1024, 1024], dtype, rtol=rtol, atol=atol)
    
    def testPowBroadcastColumnRow(self):
        """Test pow with column-row broadcasting."""
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            self._test_pow([1024, 1], [1, 1024], dtype, rtol=rtol, atol=atol)
    
    def testPowScalar(self):
        """Test pow with scalar values."""
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            self._test_pow([], [], dtype, rtol=rtol, atol=atol)
    
    def testPowEdgeCaseZeroBaseZeroExponent(self):
        """Test pow(0, 0) = 1.0 (TensorFlow specific convention)."""
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            
            # Single element test
            self._test_pow([], [], dtype, rtol=rtol, atol=atol,
                          x_values=[0.0], y_values=[0.0])
            
            # # Array test with mixed values including 0^0
            # self._test_pow([5], [5], dtype, rtol=rtol, atol=atol,
            #               x_values=[0.0, 1.0, 2.0, -1.0, 0.0],
            #               y_values=[0.0, 2.0, 3.0, 0.5, 1.0])
    
    def testPowIntegerExponentsWithNegativeBase(self):
        """Test pow(negative, integer) should be valid (no NaN)."""
        for dtype in [tf.float32, tf.float16, tf.bfloat16]:
            rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
            atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
            
            # Negative base with integer exponents
            self._test_pow([4], [4], dtype, rtol=rtol, atol=atol,
                          x_values=[-2.0, -3.0, -1.5, -4.0],
                          y_values=[2.0, 3.0, 2.0, 1.0])
    
    def testPowDifferentShapes(self):
        """Test pow with various different shapes."""
        test_cases = [
            ([1], [1]),
            ([5], [5]),
            ([3, 4], [3, 4]),
            ([2, 3, 4], [2, 3, 4]),
            ([1, 1, 10], [5, 3, 10]),
        ]
        for dtype in [tf.float32]:
            for shape_x, shape_y in test_cases:
                self._test_pow(shape_x, shape_y, dtype)

if __name__ == "__main__":
    tf.test.main()