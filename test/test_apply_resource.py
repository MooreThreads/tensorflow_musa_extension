import tensorflow as tf
import os

# --- 1. 加载 MUSA 插件 ---
try:
    # 请确保路径指向你编译好的 so 文件
    tf.load_op_library('/workspace/tensorflow_musa/build/libmusa_plugin.so')
    print(">>>> SUCCESS: MUSA plugin loaded. <<<<")
except Exception as e:
    print(f"Plugin Load Failed: {e}")

# 强制使用 MUSA
with tf.device("/device:MUSA:0"):
    var = tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32)
    m   = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32)
    v   = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32)

    grad = tf.constant([0.1, 0.1, 0.1], dtype=tf.float32)

    lr = tf.constant(0.01, dtype=tf.float32)
    beta1_power = tf.constant(0.9, dtype=tf.float32)
    beta2_power = tf.constant(0.999, dtype=tf.float32)
    beta1 = tf.constant(0.9, dtype=tf.float32)
    beta2 = tf.constant(0.999, dtype=tf.float32)
    epsilon = tf.constant(1e-8, dtype=tf.float32)

    tf.raw_ops.ResourceApplyAdam(
        var=var.handle,   # ✅ resource
        m=m.handle,       # ✅ resource
        v=v.handle,       # ✅ resource
        lr=lr,
        beta1_power=beta1_power,
        beta2_power=beta2_power,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        grad=grad,
        use_locking=False
    )

    print("var =", var.numpy())

