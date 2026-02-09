import tensorflow as tf
import numpy as np
import time
import os 
# 确保加载了你的 MUSA 插件
# try:
#     import tensorflow_musa
# except ImportError:
#     print("Warning: tensorflow_musa plugin not found, ensure it is installed.")

class MusaExpandDimsTest(tf.test.TestCase):

    def _run_expand_dims(self, input_shape, axis, dtype=np.float32):
        """在 MUSA 和 CPU 上分别运行并对比结果"""
        
        # 构造输入数据
        np_input = np.random.randn(*input_shape).astype(dtype)
        
        with tf.device('/CPU:0'):
            t_input_cpu = tf.constant(np_input)
            output_cpu = tf.expand_dims(t_input_cpu, axis=axis)

        # 切换到 MUSA 设备
        try:
            with tf.device('/device:MUSA:0'):
                t_input_musa = tf.constant(np_input)
                output_musa = tf.expand_dims(t_input_musa, axis=axis)
                
                # 检查结果是否一致
                self.assertAllClose(output_cpu, output_musa, atol=1e-5)
                return True
        except Exception as e:
            print(f"\n[FAIL] Test failed for shape {input_shape} at axis {axis}")
            print(f"Error: {e}")
            return False

    def test_all_cases(self):
        print("\n" + "="*50)
        print("开始测试 MUSA ExpandDims 算子")
        print("="*50)

        test_cases = [
            # (input_shape, axis, dtype_name)
            ([10], 0, "Float32 (Wide侧常用)"),
            ([10], 1, "末尾增加维度"),
            ([10], -1, "负索引末尾"),
            ([3, 5], 1, "中间增加维度"),
            ([2, 3, 4], 0, "开头增加维度"),
            ([2, 3, 4], 3, "4D 扩展"),
            ([100, 256], -2, "Deep侧 Embedding 常用"),
        ]

        for shape, axis, desc in test_cases:
            start_time = time.time()
            success = self._run_expand_dims(shape, axis)
            end_time = time.time()
            
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} {desc:25} | Shape: {str(shape):10} | Axis: {axis:2} | Time: {(end_time-start_time)*1000:.2f}ms")

        # 特殊类型测试
        print("-" * 50)
        print("测试不同数据类型:")
        for dtype in [np.float32, np.int32, np.int64, np.bool_]:
            success = self._run_expand_dims([5, 5], 0, dtype=dtype)
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} DataType: {np.dtype(dtype).name:10}")

        print("="*50)

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"

    if os.path.exists(plugin_path):
        print(f"DEBUG: Found MUSA plugin at {plugin_path}")
        try:
            tf.load_library(plugin_path)
            print("SUCCESS: MUSA plugin loaded successfully!")
        except Exception as e:
            print(f"FAILED: Error loading plugin: {e}")
    else:
        print(f"ERROR: Plugin not found at {plugin_path}. Did you run build.sh?")
if __name__ == '__main__':
    # 强制不使用 GPU 模拟环境，确保走 MUSA 路径
     
    load_musa_plugin()
    tf.test.main()
