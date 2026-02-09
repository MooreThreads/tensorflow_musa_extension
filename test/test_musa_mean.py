import tensorflow as tf
import numpy as np
import os

# 1. 加载插件
plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
if os.path.exists(plugin_path):
    tf.load_library(plugin_path)

# 2. 强制报错，不许回退 CPU
tf.config.set_soft_device_placement(False)

def verify_mean():
    print("\n" + "="*40)
    print("🚀 MUSA Mean 算子最终验证")
    print("="*40)

    shape = (2, 512, 1024)
    axis = -1
    # 准备数据
    np_data = np.random.randn(*shape).astype(np.float32)

    # --- MUSA 运行 ---
    print(f"1. MUSA 运行中...")
    try:
        with tf.device('/device:MUSA:0'):
            # 【关键修正】先转成 Tensor，确保数据已安全抵达显存
            musa_input = tf.constant(np_data)
            musa_out = tf.reduce_mean(musa_input, axis=axis)
            musa_res = musa_out.numpy()
        print("✅ [MUSA] 运行成功！")
    except Exception as e:
        print(f"❌ [MUSA] 运行失败: {e}")
        return

    # --- CPU 运行 ---
    print("2. CPU 对比中...")
    with tf.device('/CPU:0'):
        cpu_input = tf.constant(np_data)
        cpu_res = tf.reduce_mean(cpu_input, axis=axis).numpy()
    
    # --- 结果对比 ---
    diff = np.abs(musa_res - cpu_res).max()
    print(f"3. 最大误差: {diff:.6e}")
    
    if diff < 1e-4:
        print("✅ [通过] 结果完全一致！🎉")
    else:
        print("❌ [警告] 精度可能有问题")

if __name__ == "__main__":
    verify_mean()
