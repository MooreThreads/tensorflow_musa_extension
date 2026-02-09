import tensorflow as tf
import os
import numpy as np

# 1. 加载插件
plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
if os.path.exists(plugin_path):
    tf.load_library(plugin_path)
    print("SUCCESS: MUSA plugin loaded.")

def test_musa_logging():
    print("\n--- Testing MUSA Logging Ops (PrintV & StringFormat) ---")

    with tf.device('/device:MUSA:0'):
        # 创建 MUSA 上的数据
        a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32, name="tensor_a")
        b = tf.constant([[10.0], [20.0]], dtype=tf.float32, name="tensor_b")

        # --- 测试 1: PrintV ---
        # 在 Eager 模式下直接调用，或者在 Graph 模式下作为控制流
        print("\n[Action] Calling tf.print (Triggering MusaPrintVOp)...")
        tf.print("Values on MUSA:", a, b, summarize=3)

        # --- 测试 2: StringFormat ---
        # tf.strings.format 会在后端调用 StringFormat 算子
        print("\n[Action] Calling tf.strings.format (Triggering MusaStringFormatOp)...")
        formatted_str = tf.strings.format("Data A: {}, Data B: {}", (a, b))
        
        # 打印生成的字符串 Tensor 内容
        print("Resulting String Tensor:", formatted_str.numpy().decode('utf-8'))

    print("\nLOGGING TESTS COMPLETED!")

if __name__ == "__main__":
    test_musa_logging()

