import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

# 加载插件
try:
    tf.load_op_library('/workspace/tensorflow_musa/build/libmusa_plugin.so')
    print(">>>> SUCCESS: MUSA plugin loaded. <<<<")
except Exception as e:
    print(f"Plugin Load Failed: {e}")

def test_op(shape=(256, 100)):
    batch, classes = shape
    print(f"\n--- Testing Shape {shape} ---")
    
    # 构造数据
    logits_np = np.random.randn(batch, classes).astype(np.float32)
    labels_np = np.random.randint(0, classes, size=batch).astype(np.int32)

    # CPU 结果
    with tf.device('/CPU:0'):
        c_loss, c_grad = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
            tf.constant(logits_np), tf.constant(labels_np)
        )

    # MUSA 结果
    try:
        with tf.device('/device:MUSA:0'):
            m_loss_t, m_grad_t = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
                tf.constant(logits_np), tf.constant(labels_np)
            )
            m_loss, m_grad = m_loss_t.numpy(), m_grad_t.numpy()
    except Exception as e:
        print(f"MUSA Failed: {e}")
        return

    # 验证精度
    l_match = np.allclose(c_loss.numpy(), m_loss, atol=1e-4)
    g_match = np.allclose(c_grad.numpy(), m_grad, atol=1e-4)

    print(f"Loss Match: {l_match}")
    print(f"Grad Match: {g_match}")
    
    if not (l_match and g_match):
        print(f"Max Loss Diff: {np.max(np.abs(c_loss.numpy() - m_loss))}")
        print(f"Max Grad Diff: {np.max(np.abs(c_grad.numpy() - m_grad))}")
        print("❌ FAILED")
    else:
        print("✅ PASSED")

if __name__ == "__main__":
    # 测试小规模
    test_op(shape=(4, 8))
    # 测试大规模 (Wide & Deep 典型规模)
    test_op(shape=(1024, 1000))
