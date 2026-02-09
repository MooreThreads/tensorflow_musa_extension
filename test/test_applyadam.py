import tensorflow as tf
import numpy as np
import os

so_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
print(f"--- Loading Plugin from: {so_path} ---")
try:
    _plugin = tf.load_op_library(so_path)
    print("Plugin loaded successfully")
except Exception as e:
    print(f"Load failed: {e}")
    exit(1)

print("\n--- Checking MUSA Device ---")
musa_devices = tf.config.list_physical_devices('MUSA')
print(f"Found MUSA Devices: {musa_devices}")

print("\n--- Testing Memory Operations ---")
try:
    with tf.device('/device:MUSA:0'):
        a = tf.identity(tf.constant([1.0, 2.0, 3.0]))
        print(f"Memory check value: {a.numpy()}")
        print("Memory Test Passed")
except Exception as e:
    print(f"Memory Test Failed: {e}")

print("\n" + "="*60)
print("ApplyAdam Operator Test Suite (Resource Version)")
print("="*60)

def apply_adam_numpy(var, m, v, grad, lr, beta1, beta2, epsilon, beta1_power, beta2_power):
    """仅计算更新后的 var"""
    # === 修复点：将 List 转换为 Numpy Array 以支持数学运算 ===
    var = np.array(var)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m_new / (1 - beta1_power)
    v_hat = v_new / (1 - beta2_power)
    update = m_hat / (np.sqrt(v_hat) + epsilon)
    var_new = var - lr * update
    return var_new

def test_applyadam_basic():
    """使用 ResourceApplyAdam 测试"""
    print("\n--- 测试基本 ApplyAdam 功能（使用 Variable）---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):

            lr = tf.constant(0.01, dtype=tf.float32)
            beta1 = tf.constant(0.9, dtype=tf.float32)
            beta2 = tf.constant(0.999, dtype=tf.float32)
            epsilon = tf.constant(1e-7, dtype=tf.float32)
            beta1_power = tf.constant(0.9 ** 10, dtype=tf.float32)
            beta2_power = tf.constant(0.999 ** 10, dtype=tf.float32)

            # 使用 tf.Variable
            var = tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32)
            m = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float32)
            v = tf.Variable([0.01, 0.02, 0.03], dtype=tf.float32)
            grad = tf.constant([0.5, -0.5, 1.0], dtype=tf.float32)

            # 强制同步初始化变量
            _ = var.read_value().numpy()
            _ = m.read_value().numpy()
            _ = v.read_value().numpy()

            print(f"var.shape = {var.shape}")
            print(f"grad.shape = {grad.shape}")

            # 使用 ResourceApplyAdam
            tf.raw_ops.ResourceApplyAdam(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                beta1_power=beta1_power,
                beta2_power=beta2_power,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                grad=grad,
                use_locking=False,
                use_nesterov=False
            )

            # 获取结果
            var_updated = var.numpy()
            
            # NumPy 参考计算
            var_expected = apply_adam_numpy(
                [1.0, 2.0, 3.0],
                [0.1, 0.2, 0.3],
                [0.01, 0.02, 0.03],
                [0.5, -0.5, 1.0],
                lr.numpy(),
                beta1.numpy(),
                beta2.numpy(),
                epsilon.numpy(),
                beta1_power.numpy(),
                beta2_power.numpy()
            )

            print(f"MUSA Result: {var_updated}")
            print(f"NumPy Result: {var_expected}")

            if np.allclose(var_updated, var_expected, rtol=1e-4):
                print("✓ Basic ApplyAdam Test Passed")
            else:
                print("✗ Basic ApplyAdam Test Failed (Values Mismatch)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✗ Basic ApplyAdam Test Failed: {e}")

def main():
    test_applyadam_basic()

if __name__ == "__main__":
    main()
