import tensorflow as tf
import numpy as np
import os

# --- 1. 加载 MUSA 插件 ---
plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
if os.path.exists(plugin_path):
    tf.load_library(plugin_path)

# --- 2. 强制 MUSA 执行 ---
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(True)

def run_case(name, np_a, np_b):
    print(f"\n>> 正在测试: {name}")
    print(f"   Shape A: {np_a.shape}, Shape B: {np_b.shape}")

    # --- MUSA 运行 ---
    try:
        with tf.device('/device:MUSA:0'):
            # 显式转 Tensor，防止 Sync Failed
            t_a = tf.constant(np_a)
            t_b = tf.constant(np_b)
            # 调用算子
            musa_out = tf.maximum(t_a, t_b)
            musa_res = musa_out.numpy()
        print("   ✅ [MUSA] 运行成功")
    except Exception as e:
        print(f"   ❌ [MUSA] 失败: {e}")
        return

    # --- CPU 对比 ---
    with tf.device('/CPU:0'):
        cpu_res = tf.maximum(tf.constant(np_a), tf.constant(np_b)).numpy()

    # --- 验证 ---
    # Maximum 是逐元素的，结果应该完全一致
    diff = np.abs(musa_res - cpu_res).max()
    if diff == 0:
        print("   ✅ [通过] 结果完全一致 (0 误差)！")
    else:
        print(f"   ❌ [失败] 存在误差: {diff}")

if __name__ == "__main__":
    print("="*40)
    print("🚀 MUSA Maximum 算子验证")
    print("="*40)

    # Case 1: 相同形状 (基础功能)
    # 模拟数据：A=[1, 5, -2], B=[4, 2, 3] -> Expect=[4, 5, 3]
    a1 = np.array([1.0, 5.0, -2.0], dtype=np.float32)
    b1 = np.array([4.0, 2.0, 3.0], dtype=np.float32)
    run_case("基础逐元素比对", a1, b1)

    # Case 2: 广播机制 (进阶功能)
    # A是标量，B是向量 -> A 会被广播去和 B 的每一个元素比
    a2 = np.array([3.0], dtype=np.float32)
    b2 = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    run_case("标量广播测试", a2, b2)

    # Case 3: 复杂广播
    # A=(2,1), B=(3,) -> Result=(2,3)
    a3 = np.random.randn(2, 1).astype(np.float32)
    b3 = np.random.randn(3).astype(np.float32)
    run_case("矩阵广播测试", a3, b3)
