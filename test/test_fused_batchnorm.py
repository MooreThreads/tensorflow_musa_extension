import os
import numpy as np
import tensorflow as tf

# 1. 加载插件
PLUGIN_PATH = '/workspace/tensorflow_musa/build/libmusa_plugin.so'
if os.path.exists(PLUGIN_PATH):
    _ = tf.load_op_library(PLUGIN_PATH)
    print(f"✅ 成功加载 MUSA 插件: {PLUGIN_PATH}")
else:
    print(f"❌ 找不到插件文件 {PLUGIN_PATH}")
    exit(1)

def test_raw_ops_bn():
    print("\n" + "="*60)
    print("🚀 使用 tf.raw_ops 直接测试 MUSA FusedBatchNormV3 (绕过 Keras)")
    print("="*60)

    # 1. 准备数据 (强制 float32, NHWC)
    # Shape: [Batch=2, Height=2, Width=2, Channel=4] (用小一点的数据方便调试)
    N, H, W, C = 2, 1, 1, 32
    shape = [N, H, W, C]
    
    np.random.seed(42)
    x_val = np.random.randn(*shape).astype(np.float32)
    scale_val = np.random.rand(C).astype(np.float32)
    offset_val = np.random.rand(C).astype(np.float32)
    # 训练模式下，输入的 mean/var 即使为空也是有意义的输入位，这里给初始值
    mean_val = np.zeros(C).astype(np.float32) 
    var_val = np.ones(C).astype(np.float32)

    # ---------------------------------------------------------
    # 2. CPU 运行基准 (Golden Result)
    # ---------------------------------------------------------
    print("正在运行 CPU 基准...")
    with tf.device("/CPU:0"):
        x_cpu = tf.constant(x_val)
        scale_cpu = tf.constant(scale_val)
        offset_cpu = tf.constant(offset_val)
        mean_cpu = tf.constant(mean_val)
        var_cpu = tf.constant(var_val)

        # 显式调用 Raw Op
        # FusedBatchNormV3 返回 6 个输出: [y, batch_mean, batch_var, reserve_1, reserve_2, reserve_3]
        y_cpu_raw = tf.raw_ops.FusedBatchNormV3(
            x=x_cpu,
            scale=scale_cpu,
            offset=offset_cpu,
            mean=mean_cpu, 
            variance=var_cpu,
            epsilon=0.001,
            exponential_avg_factor=1.0,
            data_format="NHWC",
            is_training=True  # 【关键】先测 True，因为它是最难的
        )
        y_cpu = y_cpu_raw[0] # 第一个是输出结果
        
        # 为了验证反向，我们需要 GradientTape
        with tf.GradientTape() as tape:
            tape.watch(x_cpu)
            tape.watch(scale_cpu)
            tape.watch(offset_cpu)
            out_cpu = tf.raw_ops.FusedBatchNormV3(
                x=x_cpu, scale=scale_cpu, offset=offset_cpu, mean=mean_cpu, variance=var_cpu,
                epsilon=0.001, exponential_avg_factor=1.0, data_format="NHWC", is_training=True
            )
            loss_cpu = tf.reduce_sum(out_cpu[0])
        
        # 计算 x, scale, offset 的梯度
        grads_cpu = tape.gradient(loss_cpu, [x_cpu, scale_cpu, offset_cpu])

    # ---------------------------------------------------------
    # 3. MUSA 运行测试 (Target Result)
    # ---------------------------------------------------------
    print("正在运行 MUSA 测试...")
    try:
        with tf.device("/device:MUSA:0"):
            x_musa = tf.constant(x_val)
            scale_musa = tf.constant(scale_val)
            offset_musa = tf.constant(offset_val)
            mean_musa = tf.constant(mean_val) # dummy for training=True
            var_musa = tf.constant(var_val)   # dummy for training=True

            # 【重点】：这里会强制 TensorFlow 查找注册在 MUSA 上的 FusedBatchNormV3 Kernel
            # 此时你的 C++ 代码里的 ">>>>> [DEBUG]..." 必须出现！
            y_musa_raw = tf.raw_ops.FusedBatchNormV3(
                x=x_musa,
                scale=scale_musa,
                offset=offset_musa,
                mean=mean_musa,
                variance=var_musa,
                epsilon=0.001,
                exponential_avg_factor=1.0,
                data_format="NHWC",
                is_training=True
            )
            y_musa = y_musa_raw[0]

            # 反向测试
            with tf.GradientTape() as tape:
                tape.watch(x_musa)
                tape.watch(scale_musa)
                tape.watch(offset_musa)
                out_musa = tf.raw_ops.FusedBatchNormV3(
                    x=x_musa, scale=scale_musa, offset=offset_musa, mean=mean_musa, variance=var_musa,
                    epsilon=0.001, exponential_avg_factor=1.0, data_format="NHWC", is_training=True
                )
                loss_musa = tf.reduce_sum(out_musa[0])
            
            grads_musa = tape.gradient(loss_musa, [x_musa, scale_musa, offset_musa])

    except Exception as e:
        print(f"\n❌ MUSA 运行崩溃: {e}")
        return

    # ---------------------------------------------------------
    # 4. 结果比对
    # ---------------------------------------------------------
    print("\n" + "-"*30)
    print("📊 精度比对结果")
    print("-" * 30)

    # 前向 Y
    diff_y = np.abs(y_cpu.numpy() - y_musa.numpy()).max()
    print(f"Forward Output Diff (Y) : {diff_y:.6e}")

    # 反向 DX
    diff_dx = np.abs(grads_cpu[0].numpy() - grads_musa[0].numpy()).max()
    print(f"Backward Grad Diff (dX): {diff_dx:.6e}")
    
    # 反向 dScale
    diff_dscale = np.abs(grads_cpu[1].numpy() - grads_musa[1].numpy()).max()
    print(f"Backward Grad Diff (dS): {diff_dscale:.6e}")

    # 反向 dOffset
    diff_doffset = np.abs(grads_cpu[2].numpy() - grads_musa[2].numpy()).max()
    print(f"Backward Grad Diff (dB): {diff_doffset:.6e}")

    if diff_y < 1e-4 and diff_dx < 1e-4:
        print("\n✅ [SUCCESS] 算子精度验证通过！")
    else:
        print("\n❌ [FAIL] 精度误差过大！")

if __name__ == "__main__":
    test_raw_ops_bn()
