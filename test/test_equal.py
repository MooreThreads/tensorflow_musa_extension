import tensorflow as tf
import numpy as np
import os

def load_musa_plugin():
    # 请根据你的实际 build 路径修改
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        print(f"DEBUG: Loading MUSA plugin from {plugin_path}")
        tf.load_library(plugin_path)
    else:
        print(f"ERROR: Plugin not found at {plugin_path}")

class MusaEqualTest(tf.test.TestCase):

    def _run_equal(self, shape0, shape1, dtype):
        """对比 CPU 和 MUSA 的 Equal 结果"""
        # 构造一些包含相同元素的随机数据
        if dtype in [np.int32, np.int64]:
            # 生成 0-2 之间的整数，增加“相等”的概率
            data0 = np.random.randint(0, 3, size=shape0).astype(dtype)
            data1 = np.random.randint(0, 3, size=shape1).astype(dtype)
        else:
            # 浮点数：通过截断来制造相等
            data0 = np.round(np.random.randn(*shape0)).astype(dtype)
            data1 = np.round(np.random.randn(*shape1)).astype(dtype)

        # CPU 运行
        with tf.device('/CPU:0'):
            t0_cpu = tf.constant(data0)
            t1_cpu = tf.constant(data1)
            res_cpu = tf.equal(t0_cpu, t1_cpu)

        # MUSA 运行
        try:
            with tf.device('/device:MUSA:0'):
                t0_musa = tf.constant(data0)
                t1_musa = tf.constant(data1)
                res_musa = tf.equal(t0_musa, t1_musa)
                
                # 核心校验
                self.assertAllEqual(res_cpu, res_musa)
                return True
        except Exception as e:
            print(f"\n[FAIL] Type: {dtype} | Shape: {shape0} vs {shape1}")
            print(f"Error: {e}")
            return False

    def test_basic_equality(self):
        print("\n" + "="*60)
        print("开始测试 MUSA Equal 算子")
        print("="*60)

        # 测试用例：(shape0, shape1, 类型描述)
        test_cases = [
            ([1024], [1024], "基础向量比较"),
            ([32, 128], [32, 128], "矩阵比较 (Wide&Deep 常用)"),
            ([64, 1], [64, 512], "行广播比较"),
            ([1], [100], "标量广播比较"),
        ]

        for s0, s1, desc in test_cases:
            success = self._run_equal(s0, s1, np.float32)
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} {desc:25} | {str(s0):10} vs {str(s1)}")

    def test_data_types(self):
        print("-" * 60)
        print("测试多数据类型支持:")
        # 重点测试你刚加的 bf16 和 fp16
        dtypes = {
            "int32": np.int32,
            "int64": np.int64,
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": tf.bfloat16.as_numpy_dtype
        }

        for name, dt in dtypes.items():
            success = self._run_equal([128], [128], dt)
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} DataType: {name:10}")

        print("="*60)

if __name__ == '__main__':
    load_musa_plugin()
    # 诊断：打印可见设备
    print("Visible Devices:", tf.config.list_physical_devices())
    tf.test.main()

