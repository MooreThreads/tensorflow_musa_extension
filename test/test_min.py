import tensorflow as tf
import numpy as np
import os

# 配置：设置库路径
LIB_PATH = "/workspace/tensorflow_musa/build/libmusa_plugin.so"

def load_plugin():
    if not os.path.exists(LIB_PATH):
        print(f"❌ 错误: 找不到插件文件: {LIB_PATH}")
        print("请先执行: cd ../build && make -j8")
        exit(1)
    try:
        _ = tf.load_library(LIB_PATH)
        print(f"✅ 成功加载插件: {LIB_PATH}")
    except Exception as e:
        print(f"❌ 加载插件失败: {e}")
        exit(1)

def run_test_case(name, tf_dtype, np_dtype, shape=(10, 10), tolerance=1e-5):
    print(f"\n🧪 测试场景: [{name}] | 类型: {tf_dtype} | 形状: {shape}")
    
    # 1. 生成随机数据 (NumPy)
    if np_dtype in [np.int32, np.int64]:
        # 整数生成，包含负数和大数
        data_a = np.random.randint(-1000, 1000, size=shape).astype(np_dtype)
        data_b = np.random.randint(-1000, 1000, size=shape).astype(np_dtype)
    else:
        # 浮点生成
        data_a = np.random.randn(*shape).astype(np_dtype)
        data_b = np.random.randn(*shape).astype(np_dtype)

    # 2. 计算预期结果 (Ground Truth using CPU/NumPy)
    expected = np.minimum(data_a, data_b)

    # 3. MUSA 运行 (TensorFlow)
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        with tf.device("/device:MUSA:0"):
            # 对于 bfloat16，NumPy 不支持，需要先转 tf.float32 再转 bfloat16
            if name == "BFloat16":
                t_a = tf.cast(tf.constant(data_a), dtype=tf.bfloat16)
                t_b = tf.cast(tf.constant(data_b), dtype=tf.bfloat16)
            else:
                t_a = tf.constant(data_a, dtype=tf_dtype)
                t_b = tf.constant(data_b, dtype=tf_dtype)
            
            # 核心算子调用
            output_op = tf.minimum(t_a, t_b)
            
            # 如果是 BF16/FP16，转回 float32 以便和 numpy 比较
            if name in ["BFloat16", "Half"]:
                output_op = tf.cast(output_op, tf.float32)

            try:
                # 执行
                result = sess.run(output_op)
                
                # 4. 验证结果
                # 处理 BF16 的精度损失问题，适当放宽 tolerance
                diff = np.abs(result - expected)
                max_diff = np.max(diff)
                
                if max_diff <= tolerance:
                    print(f"   ✅ 通过! 最大误差: {max_diff:.8f}")
                else:
                    print(f"   ❌ 失败! 最大误差: {max_diff:.8f} (阈值: {tolerance})")
                    print(f"   前3个预期值: {expected.flatten()[:3]}")
                    print(f"   前3个实际值: {result.flatten()[:3]}")
                    
            except Exception as e:
                print(f"   💥 运行时崩溃: {e}")

def main():
    print("========================================")
    print("      MUSA Minimum 算子全类型测试       ")
    print("========================================")
    
    # 禁用 Eager 以模拟真实训练图模式
    tf.compat.v1.disable_eager_execution()
    
    load_plugin()
    
    # 1. 基础 Float32
    run_test_case("Float32", tf.float32, np.float32)
    
    # 2. 基础 Int32
    run_test_case("Int32", tf.int32, np.int32, tolerance=0)
    
    # 3. 关键 Int64 (Wide & Deep 索引必备)
    # 使用大数测试 int64 是否截断
    print("\n🧪 测试场景: [Int64 Large Number Check]")
    data_a = np.array([2**33, -2**33], dtype=np.int64) 
    data_b = np.array([2**34, 0], dtype=np.int64)
    expected = np.minimum(data_a, data_b)
    
    with tf.compat.v1.Session() as sess:
        with tf.device("/device:MUSA:0"):
            res = sess.run(tf.minimum(tf.constant(data_a), tf.constant(data_b)))
            if np.array_equal(res, expected):
                print(f"   ✅ Int64 大数测试通过! 结果: {res}")
            else:
                print(f"   ❌ Int64 失败! 预期 {expected}, 实际 {res}")

    # 4. 半精度 FP16 (Half)
    # 注意：FP16 精度较低，容差设为 1e-3
    run_test_case("Half", tf.half, np.float16, tolerance=1e-3)
    
    # 5. 关键 BFloat16 (刚刚修复的)
    # BF16 尾数只有 7 位，精度比 FP16 还低，容差设为 1e-2
    # 我们用 float32 模拟 numpy 输入
    run_test_case("BFloat16", tf.bfloat16, np.float32, tolerance=2e-2)

    print("\n========================================")
    print("🎉 所有测试结束")

if __name__ == "__main__":
    main()

