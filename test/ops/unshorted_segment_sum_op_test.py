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

"""Tests for MUSA UnsortedSegmentSum operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class UnsortedSegmentSumOpTest(MUSATestCase):
  """Tests for MUSA UnsortedSegmentSum operator."""

  def _test_unsorted_segment_sum(self, data, segment_ids, num_segments, dtype, 
                                 index_type=tf.int32, rtol=1e-5, atol=1e-8):
    """Test unsorted_segment_sum operation."""
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np_dtype)
    else:
        data = data.astype(np_dtype)
        
    segment_ids = np.array(segment_ids, dtype=index_type.as_numpy_dtype)

    x = tf.constant(data, dtype=dtype)
    ids = tf.constant(segment_ids, dtype=index_type)

    num_seg_tensor = tf.constant(num_segments, dtype=index_type)

    def op_func(data_in, ids_in):
        return tf.math.unsorted_segment_sum(data_in, ids_in, num_seg_tensor)

    self._compare_cpu_musa_results(op_func, [x, ids], dtype, rtol=rtol, atol=atol)

  def testBasic(self):
    """Test basic functionality with various types."""
    inputs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    segment_ids = [0, 1, 0]
    num_segments = 2
    
    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
        rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
        atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
        
        self._test_unsorted_segment_sum(inputs, segment_ids, num_segments, dtype, rtol=rtol, atol=atol)

  def testIndicesTypes(self):
    """Test int32 and int64 indices."""
    inputs = [[1, 2], [3, 4]]
    segment_ids = [1, 0]
    num_segments = 2

    for index_type in [tf.int32, tf.int64]:
        self._test_unsorted_segment_sum(inputs, segment_ids, num_segments, tf.float32, index_type=index_type)

  def testEmpty(self):
    """Test empty input and empty indices."""
    inputs = []
    segment_ids = []
    num_segments = 3
    
    self._test_unsorted_segment_sum(inputs, segment_ids, num_segments, tf.float32)

  def testDropIndices(self):
    """Test that negative indices are ignored (dropped)."""
    inputs = [[1, 2], [3, 4], [5, 6], [7, 8]]
    segment_ids = [0, -1, 1, -1]
    num_segments = 2
    
    self._test_unsorted_segment_sum(inputs, segment_ids, num_segments, tf.float32)

  def testHighDim(self):
    """Test higher dimensional data."""
    shape = [4, 2, 2]
    inputs = np.random.uniform(0, 10, size=shape)
    segment_ids = [0, 1, 0, 1]
    num_segments = 2
    
    self._test_unsorted_segment_sum(inputs, segment_ids, num_segments, tf.float32)


if __name__ == "__main__":
  tf.test.main()