import tensorflow as tf
import numpy as np
import os

# --- 1. 加载你的插件 ---
SO_PATH = '/workspace/tensorflow_musa/build/libmusa_plugin.so'
if not os.path.exists(SO_PATH):
    print(f"❌ 找不到 .so 文件: {SO_PATH}")
    exit(1)

tf.load_op_library(SO_PATH)
print("✅ MUSA 插件加载成功")

def verify_and_print(name, tensor):
    """验证张量属性并打印采样数据"""
    val = tensor.numpy()
    print(f"[{name}] Device: {tensor.device} | Dtype: {tensor.dtype.name}")
    print(f"采样数据 (前4个): {val.flatten()[:4]}")
    
    # 验证非全零（随机性初步检查）
    if np.any(val != 0):
        print(f"✅ {name} 验证通过：存在有效数值。")
    else:
        print(f"⚠️ {name} 告警：结果全为0，请检查内核逻辑。")

def test_random_ops():
    # 强制不使用软放置，确保算子一定在 MUSA 上运行
    tf.config.set_soft_device_placement(False)
    
    test_shape = [4, 4]
    
    with tf.device('/device:MUSA:0'):
        print("\n" + "="*50)
        print("🚀 开始 MUSA 随机数算子全集测试")
        print("="*50)

        # --- 1. 测试 RandomUniform (Float32) ---
        print("\n测试 1: RandomUniform (Float32)")
        u_float = tf.random.uniform(test_shape, minval=0, maxval=1.0, dtype=tf.float32)
        verify_and_print("UniformFloat", u_float)

        # --- 2. 测试 RandomStandardNormal (Float32) ---
        print("\n测试 2: RandomStandardNormal (Float32)")
        n_float = tf.random.normal(test_shape, mean=0.0, stddev=1.0, dtype=tf.float32)
        verify_and_print("NormalFloat", n_float)

        # --- 3. 测试新添加的 RandomUniformInt (Int32) ---
        print("\n测试 3: RandomUniformInt (Int32)")
        # 注意：TF 的 RandomUniformInt 必须明确指定 minval 和 maxval
        u_int = tf.random.uniform(test_shape, minval=0, maxval=100, dtype=tf.int32)
        verify_and_print("UniformInt32", u_int)
        # 验证范围
        if np.all((u_int.numpy() >= 0) & (u_int.numpy() < 100)):
            print("✅ 范围验证通过 [0, 100)")

        # --- 4. 测试无状态随机数 (StatelessRandomUniformV2) ---
        print("\n测试 4: StatelessRandomUniformV2 (Keras常用)")
        # seed 是 [2] 形状的张量
        u_stateless = tf.random.stateless_uniform(test_shape, seed=[1, 2], dtype=tf.float32)
        verify_and_print("StatelessUniform", u_stateless)

        # --- 5. 测试随机性 (对比两次运行) ---
        print("\n测试 5: 随机性验证 (多次调用结果应不同)")
        u1 = tf.random.uniform([2, 2]).numpy()
        u2 = tf.random.uniform([2, 2]).numpy()
        if not np.array_equal(u1, u2):
            print("✅ 随机性验证通过：连续两次生成的数值不相同。")
        else:
            print("❌ 随机性验证失败：连续两次生成了完全相同的数值！")

if __name__ == "__main__":
    # 开启日志以便观察算子执行（可选）
    # tf.debugging.set_log_device_placement(True)
    
    try:
        test_random_ops()
        print("\n" + "="*50)
        print("🎉 所有 MUSA 随机数算子基础测试完成！")
        print("="*50)
    except Exception as e:
        print(f"\n💥 测试过程中发生崩溃!")
        print(f"错误详情: {e}")
