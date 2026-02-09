import os
import numpy as np
import tensorflow as tf

# 加载插件
PLUGIN_PATH = '/workspace/tensorflow_musa/build/libmusa_plugin.so'
if os.path.exists(PLUGIN_PATH):
    _ = tf.load_op_library(PLUGIN_PATH)
else:
    print(f"❌ 找不到插件文件 {PLUGIN_PATH}")
    exit(1)

def test_raw_ops_bn_nchw():
    print("\n" + "="*60)
    print("🚀 测试 MUSA FusedBatchNormV3 [模式: NCHW]")
    print("="*60)

    # 1. 准备数据 (NCHW 格式: Channel 在第二维)
    # Shape: [Batch, Channel, Height, Width]
    N, C, H, W = 2, 32, 1, 1
    shape = [N, C, H, W]  # <--- 修改点 1: 形状变为 [2, 32, 1, 1]
    
    np.random.seed(42)
    x_val = np.random.randn(*shape).astype(np.float32)
    scale_val = np.random.rand(C).astype(np.float32)
    offset_val = np.random.rand(C).astype(np.float32)
    mean_val = np.zeros(C).astype(np.float32) 
    var_val = np.ones(C).astype(np.float32)

    # 2. 运行测试
    # ---------------------------------------------------------
    print(f"输入形状: {shape}, Data Format: NCHW")
    
    try:
        # MUSA 运行
        with tf.device("/device:MUSA:0"):
            x_musa = tf.constant(x_val)
            scale_musa = tf.constant(scale_val)
            offset_musa = tf.constant(offset_val)
            mean_musa = tf.constant(mean_val)
            var_musa = tf.constant(var_val)

            y_musa_raw = tf.raw_ops.FusedBatchNormV3(
                x=x_musa,
                scale=scale_musa,
                offset=offset_musa,
                mean=mean_musa,
                variance=var_musa,
                epsilon=0.001,
                exponential_avg_factor=1.0,
                data_format="NCHW",  # <--- 修改点 2: 显式指定 NCHW
                is_training=True
            )
            y_musa = y_musa_raw[0]

        # CPU 基准 (用于比对)
        with tf.device("/CPU:0"):
            y_cpu_raw = tf.raw_ops.FusedBatchNormV3(
                x=tf.constant(x_val),
                scale=tf.constant(scale_val),
                offset=tf.constant(offset_val),
                mean=tf.constant(mean_val),
                variance=tf.constant(var_val),
                epsilon=0.001,
                exponential_avg_factor=1.0,
                data_format="NCHW",  # <--- 修改点 2
                is_training=True
            )
            y_cpu = y_cpu_raw[0]

        # 精度比对
        diff = np.abs(y_cpu.numpy() - y_musa.numpy()).max()
        print(f"\nForward Output Diff (Y): {diff:.6e}")
        
        if diff < 1e-4:
            print("✅ NCHW 测试通过")
        else:
            print("❌ NCHW 测试失败")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")

if __name__ == "__main__":
    test_raw_ops_bn_nchw()
