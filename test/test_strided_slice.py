import tensorflow as tf
import numpy as np
import os

# 强制显示设备日志，确认是否跑在 MUSA 上
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def check_result(musa_val, cpu_val, name, rtol=1e-5, atol=1e-5):
    """通用结果比对函数"""
    try:
        np.testing.assert_allclose(musa_val, cpu_val, rtol=rtol, atol=atol)
        print(f"  [PASS] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}\n错误详情: {e}")

def test_cast_op():
    print("\n" + "="*50)
    print("开始测试 MUSA Cast 算子 (包含拦截逻辑验证)")
    print("="*50)

    test_cases = [
        # (源类型, 目标类型, 测试数据)
        (tf.float32, tf.int32, [1.5, 2.9, -3.1, 0.0]),
        (tf.int32, tf.float32, [10, 20, -5, 0]),
        (tf.float32, tf.int64, [1e9, 2e9, -1e9]),
        # 重点：测试您的 bool 拦截器逻辑
        (tf.bool, tf.float32, [True, False, True, True]),
        (tf.bool, tf.int32, [True, False, False, True]),
        # 重点：测试混合精度 bfloat16
        (tf.float32, tf.bfloat16, [1.0, 65504.0, 1e10]),
    ]

    for src_dt, dst_dt, data in test_cases:
        name = f"{src_dt.name} -> {dst_dt.name}"
        
        # CPU 结果 (参考基准)
        with tf.device('/CPU:0'):
            cpu_input = tf.constant(data, dtype=src_dt)
            cpu_output = tf.cast(cpu_input, dtype=dst_dt)

        # MUSA 结果
        with tf.device('/device:MUSA:0'):
            musa_input = tf.constant(data, dtype=src_dt)
            musa_output = tf.cast(musa_input, dtype=dst_dt)

        # 比对（BF16 允许较大误差）
        tol = 1e-2 if dst_dt == tf.bfloat16 else 1e-5
        check_result(musa_output.numpy(), cpu_output.numpy(), name, rtol=tol, atol=tol)

def test_strided_slice_op():
    print("\n" + "="*50)
    print("开始测试 MUSA StridedSlice 算子 (复杂切片验证)")
    print("="*50)

    # 模拟 Wide & Deep 常见的特征矩阵 [Batch=3, Features=5]
    raw_data = np.arange(15).reshape(3, 5).astype(np.float32)
    
    test_cases = [
        # (描述, 切片语法)
        ("基础全量切片 [:]", lambda x: x[:]),
        ("提取前两列 (Wide侧常用)", lambda x: x[:, 0:2]),
        ("带步长的切片 (隔行取样)", lambda x: x[::2, ::2]),
        #("逆序切片 (步长为-1)", lambda x: x[:, ::-1]),
        ("收缩维度 (Shrink Axis)", lambda x: x[0, :]),
        ("空张量切片", lambda x: x[0:0, 0:0]),
    ]

    for desc, slice_fn in test_cases:
        # CPU
        with tf.device('/CPU:0'):
            cpu_input = tf.constant(raw_data)
            cpu_res = slice_fn(cpu_input)

        # MUSA
        with tf.device('/device:MUSA:0'):
            musa_input = tf.constant(raw_data)
            musa_res = slice_fn(musa_input)

        check_result(musa_res.numpy(), cpu_res.numpy(), desc)
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

if __name__ == "__main__":
    import os
    load_musa_plugin()
    # 检查 MUSA 是否可用
    musa_devices = tf.config.list_physical_devices('MUSA')
    if not musa_devices:
        print("错误: 未检测到 MUSA 设备，请检查插件是否加载成功。")
    else:
        try:
            test_cast_op()
            test_strided_slice_op()
            print("\n" + "="*50)
            print("所有测试运行完毕！")
            print("="*50)
        except Exception as e:
            print(f"\n测试运行过程中发生异常: {e}")
