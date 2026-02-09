import tensorflow as tf
import numpy as np
import os

# 强制使用 MUSA 设备
device_name = "/device:MUSA:0"

def load_musa_plugin():
    # 请确保路径指向你编译出来的 .so 文件
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        try:
            tf.load_library(plugin_path)
            print("✅ SUCCESS: MUSA plugin loaded!")
        except Exception as e:
            print(f"❌ FAILED: Error loading plugin: {e}")
    else:
        print(f"⚠️ ERROR: Plugin not found at {plugin_path}")

def test_resource_ops():
    print(f"🚀 Starting MUSA Resource Operators Comprehensive Test on {device_name}...")

    # 1. 准备模拟数据
    vocab_size = 100
    embedding_dim = 8
    # 词表数据: [[0,0...], [1,1...], ..., [99,99...]]
    h_params = np.array([np.full(embedding_dim, i, dtype=np.float32) for i in range(vocab_size)])
    h_indices = np.array([1, 5, 10, 99], dtype=np.int32)

    try:
        with tf.device(device_name):
            # --- 测试 1: VarHandleOp & AssignVariableOp ---
            params_var = tf.Variable(h_params, name="test_embedding_table")
            indices = tf.constant(h_indices, dtype=tf.int32)
            print("✅ Step 1: Variable initialized and assigned on MUSA.")

            # --- 测试 2: ResourceGather ---
            output = tf.gather(params_var, indices)
            print("📊 Step 2: ResourceGather executed.")
            print("🔢 Gather Output Shape:", output.shape)
            
            # 验证 Gather 数值
            expected_gather = h_params[h_indices]
            if np.allclose(output.numpy(), expected_gather):
                print("⭐ SUCCESS: ResourceGather output matches CPU reference!")
            else:
                print("❌ ERROR: ResourceGather output mismatch!")

            # --- 测试 3: VariableShape ---
            # 触发 MusaVariableShapeOp
            var_shape = tf.shape(params_var)
            print(f"📏 Step 3: VariableShape Result: {var_shape.numpy()}")
            if np.array_equal(var_shape.numpy(), [vocab_size, embedding_dim]):
                print("⭐ SUCCESS: VariableShape is correct!")
            else:
                print("❌ ERROR: VariableShape mismatch!")

            # --- 测试 4: ResourceScatterAdd ---
            print("➕ Step 4: Testing ResourceScatterAdd...")
            # 给索引 1 的位置加上 10.0 (原本是 1.0，加完应该是 11.0)
            update_val = 10.0
            h_updates = np.full((1, embedding_dim), update_val, dtype=np.float32)
            h_scatter_indices = np.array([1], dtype=np.int32)
            
            # 触发 MusaResourceScatterAddOp
            params_var.scatter_add(tf.IndexedSlices(h_updates, h_scatter_indices))
            
            # 读取索引 1 的新值进行验证
            new_val_at_1 = tf.gather(params_var, [1]).numpy()
            expected_val_at_1 = h_params[1] + update_val
            
            print(f"🔄 Value at index 1 after ScatterAdd: {new_val_at_1}")
            if np.allclose(new_val_at_1, expected_val_at_1):
                print("⭐ SUCCESS: ResourceScatterAdd calculation is correct!")
            else:
                print(f"❌ ERROR: ScatterAdd result mismatch! Expected {expected_val_at_1}")

    except Exception as e:
        print(f"\n💥 Test Failed! Unexpected Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    test_resource_ops()
