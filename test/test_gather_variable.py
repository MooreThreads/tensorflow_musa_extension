import numpy as np
import tensorflow as tf
import os

# 确保加载插件
def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)
        print("SUCCESS: MUSA plugin loaded.")
    else:
        print("ERROR: Plugin not found.")

def test_variable_and_gather_on_musa():
    print("\n--- Testing ResourceVariable & Gather on MUSA ---")
    
    # 1. 准备数据
    # 模拟一个 Embedding 矩阵 (4行, 3列)
    embedding_matrix = np.array([
        [0.1, 0.1, 0.1], # index 0
        [0.2, 0.2, 0.2], # index 1
        [0.3, 0.3, 0.3], # index 2
        [0.4, 0.4, 0.4]  # index 3
    ], dtype=np.float32)
    
    # 想要查表的索引
    ids_to_lookup = np.array([1, 3], dtype=np.int32)
    # 预期结果应该是 embedding_matrix 的第 1 行和第 3 行
    expected_gather_result = embedding_matrix[ids_to_lookup]

    try:
        with tf.device('/device:MUSA:0'):
            # --- STEP 1: 测试变量初始化 (AssignVariableOp) ---
            musa_embedding = tf.Variable(embedding_matrix, name="musa_emb")
            print("STEP 1: Variable initialized on MUSA.")

            # --- STEP 2: 测试查表 (ResourceGather) ---
            # tf.gather 在处理 Resource Variable 时会触发 MusaResourceGatherOp
            lookup_indices = tf.constant(ids_to_lookup, dtype=tf.int32)
            gather_result = tf.gather(musa_embedding, lookup_indices)
            
            print(f"Gather Result from MUSA:\n{gather_result.numpy()}")
            
            # 验证结果
            np.testing.assert_allclose(gather_result.numpy(), expected_gather_result, atol=1e-5)
            print("STEP 2 (ResourceGather): SUCCESS")

            # --- STEP 3: 测试变量更新 (AssignAdd) ---
            # 模拟梯度更新：给整张表加上一个微小的偏移
            update_val = np.ones_like(embedding_matrix, dtype=np.float32) * 0.1
            musa_embedding.assign_add(update_val)
            
            updated_val = musa_embedding.read_value()
            print(f"Updated Matrix (First Row): {updated_val.numpy()[0]}")
            
            # 验证更新后的第一行：[0.1, 0.1, 0.1] + 0.1 = [0.2, 0.2, 0.2]
            np.testing.assert_allclose(updated_val.numpy()[0], [0.2, 0.2, 0.2], atol=1e-5)
            print("STEP 3 (AssignAdd): SUCCESS")

        print("\nALL MUSA RESOURCE OPS TESTS PASSED!")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")

if __name__ == "__main__":
    load_musa_plugin()
    test_variable_and_gather_on_musa()
