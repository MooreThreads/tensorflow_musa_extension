import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
import os

# 关闭动态图模式，兼容插件开发调试
tf.compat.v1.disable_eager_execution()

def load_musa_plugin():
    plugin_path = os.path.join(os.getcwd(), "../build/libmusa_plugin.so")
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)
        print(f"DEBUG: Successfully loaded MUSA plugin")

class MusaSubIsolationTest(test.TestCase):
    def _run_sub_with_check(self, x_np, y_np, dtype):
        # 强制配置只使用一张卡，方便看日志
        config = tf.compat.v1.ConfigProto(log_device_placement=True)
        
        with tf.compat.v1.Session(config=config) as sess:
            x_tf = tf.compat.v1.placeholder(dtype, shape=x_np.shape)
            y_tf = tf.compat.v1.placeholder(dtype, shape=y_np.shape)
            
            with tf.device("/device:MUSA:0"):
                res_op = tf.math.subtract(x_tf, y_tf)
            
            print(f"  [EXEC] Running session.run for {dtype.name}...")
            # 运行并获取结果。如果算子有“内伤”，sess.run 结束后可能不会立刻报错，
            # 但硬件状态已经坏了。
            musa_res = sess.run(res_op, feed_dict={x_tf: x_np, y_tf: y_np})
            
            cpu_res = x_np.astype(dtype.as_numpy_dtype) - y_np.astype(dtype.as_numpy_dtype)
            self.assertAllClose(musa_res, cpu_res, atol=1e-3)
            print(f"  [CHECK] Numerical check passed for {dtype.name}")

    def test_isolation(self):
        load_musa_plugin()
        
        # 定义要测试的类型
        # 你可以通过注释掉列表里的元素来“单独测试”或“排除测试”
        # 例如：只留 dtypes.float32 看看还会不会有 stream error
        test_types = [
            dtypes.float32, 
            # dtypes.int32,     # 暂时排除
            # dtypes.bfloat16,  # 暂时排除
        ]

        # 准备数据：3个元素（非对齐风险）
        x_np = np.array([10, 20, 30])
        y_np = np.array([1, 2, 3])

        for dtype in test_types:
            print(f"\n" + "="*40)
            print(f"Testing DataType: {dtype.name}")
            try:
                self._run_sub_with_check(x_np, y_np, dtype)
                print(f"DONE: {dtype.name} finished without immediate crash.")
            except Exception as e:
                print(f"FAILED: {dtype.name} encountered error: {e}")
            print("="*40)

if __name__ == "__main__":
    # 调试建议：运行前 export MUSA_DEVICE_BLOCKING=1
    test.main()
