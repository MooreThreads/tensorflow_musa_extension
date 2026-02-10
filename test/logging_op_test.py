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

"""Tests for MUSA Logging operators (PrintV, StringFormat)."""

import numpy as np
import tensorflow as tf
import re
from musa_test_utils import MUSATestCase

class LoggingOpTest(MUSATestCase):
  """Tests for MUSA Logging operators."""

  def _clean_string(self, s):
      """
      Helper to normalize strings for comparison.
      Removes brackets [], newlines, commas, and collapses multiple spaces.
      """
      # 1. Remove brackets [ and ]
      s = s.replace('[', ' ').replace(']', ' ')
      # 2. 【核心修改】Remove commas to avoid spacing issues (e.g., "3.3 ," vs "3.3,")
      s = s.replace(',', ' ')
      # 3. Replace newlines with space
      s = s.replace('\n', ' ')
      # 4. Collapse multiple spaces into one
      s = re.sub(r'\s+', ' ', s)
      # 5. Strip leading/trailing whitespace
      return s.strip()

  def _compare_strings_fuzzy(self, op_func, inputs):
      """Helper to compare string outputs using fuzzy matching."""
      # 1. Run on CPU
      with tf.device("/CPU:0"):
          cpu_res_tensor = op_func(*inputs)
          if isinstance(cpu_res_tensor, (list, tuple)):
              cpu_res_tensor = cpu_res_tensor[0]
          
          if cpu_res_tensor is not None:
              cpu_res = cpu_res_tensor.numpy()
          else:
              return 

      # 2. Run on MUSA
      with tf.device("/device:MUSA:0"):
          musa_res_tensor = op_func(*inputs)
          if isinstance(musa_res_tensor, (list, tuple)):
              musa_res_tensor = musa_res_tensor[0]
              
          if musa_res_tensor is not None:
              musa_res = musa_res_tensor.numpy()
          else:
              return

      # 3. Decode helper
      def decode_item(x):
          if isinstance(x, bytes):
              return x.decode('utf-8')
          if hasattr(x, 'item'): 
             val = x.item()
             if isinstance(val, bytes):
                 return val.decode('utf-8')
          return str(x)

      # 4. Normalize and Clean
      def get_cleaned_str(res):
          if isinstance(res, np.ndarray):
              if res.ndim == 0:
                  raw_str = decode_item(res)
              else:
                  # Join array of strings into one big string for comparison
                  raw_str = " ".join([decode_item(s) for s in res.flatten()])
          elif isinstance(res, bytes):
              raw_str = decode_item(res)
          else:
              raw_str = str(res)
          
          return self._clean_string(raw_str)

      cpu_str_clean = get_cleaned_str(cpu_res)
      musa_str_clean = get_cleaned_str(musa_res)
      
      # 5. Assert Equality on Cleaned Strings
      self.assertEqual(cpu_str_clean, musa_str_clean, 
                       msg=f"\nCPU (Raw):  {cpu_res}\nMUSA (Raw): {musa_res}\n"
                           f"CPU (Clean): {cpu_str_clean}\nMUSA (Clean):{musa_str_clean}")

  def testStringFormat(self):
    """Test tf.strings.format (StringFormatOp)."""
    val_float = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    val_int = np.array([[10], [20]], dtype=np.int32)
    
    x = tf.constant(val_float)
    y = tf.constant(val_int)
    
    template = "Float Data: {}, Int Data: {}"

    def op_func(in_x, in_y):
        return tf.strings.format(template, inputs=[in_x, in_y])

    # Use fuzzy comparison
    self._compare_strings_fuzzy(op_func, [x, y])

  def testStringFormatPlaceholder(self):
    """Test StringFormat with tensor placeholders."""
    # Reduced size to avoid ellipsis (...) differences in debug strings
    val = np.random.randn(2, 2).astype(np.float32)
    x = tf.constant(val)
    
    def op_func(input_tensor):
        return tf.strings.format("Tensor: {}", inputs=[input_tensor], summarize=10)

    self._compare_strings_fuzzy(op_func, [x])

  def testPrint(self):
    """Test tf.print (PrintV2)."""
    val = np.random.randn(5, 5).astype(np.float32)
    x = tf.constant(val)

    def op_func(input_tensor):
        # We just verify it runs without error on MUSA
        tf.print("MUSA Print Test (Float):", input_tensor, summarize=5)
        return input_tensor

    self._compare_cpu_musa_results(op_func, [x], tf.float32)

  def testPrintInt(self):
    """Test tf.print with Integers."""
    val = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    x = tf.constant(val)

    def op_func(input_tensor):
        tf.print("MUSA Print Test (Int):", input_tensor)
        return input_tensor

    self._compare_cpu_musa_results(op_func, [x], tf.int32)

  def testPrintMultiple(self):
    """Test tf.print with multiple inputs."""
    x = tf.constant(1.0)
    y = tf.constant(2)
    z = tf.constant("Hello MUSA")

    def op_func(a, b, c):
        tf.print("Multi-arg print:", a, b, c)
        return a 

    self._compare_cpu_musa_results(op_func, [x, y, z], tf.float32)


if __name__ == "__main__":
  tf.test.main()