import tensorflow as tf
import numpy as np
import os

so_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
print(f"--- 1. Loading Plugin from: {so_path} ---")
try:
    _plugin = tf.load_op_library(so_path)
    print("Plugin loaded successfully")
except Exception as e:
    print(f"Load failed: {e}")
    exit(1)

print("\n--- 2. Checking MUSA Device ---")
musa_devices = tf.config.list_physical_devices('MUSA')
print(f"Found MUSA Devices: {musa_devices}")

print("\n--- 3. Testing Memory Operations ---")
try:
    with tf.device('/device:MUSA:0'):
        a = tf.identity(tf.constant([1, 2, 3], dtype=tf.int32))
        print(f"Memory check value: {a.numpy()}")
        print("Memory Test Passed")
except Exception as e:
    print(f"Memory Test Failed: {e}")

print("\n" + "="*60)
print("InvertPermutation Operator Test Suite")
print("="*60)


def test_invertpermutation_basic():
    """测试基本 InvertPermutation 功能"""
    print("\n--- 测试基本 InvertPermutation ---")
    try:
        tf.config.set_soft_device_placement(True)
        with tf.device('/device:MUSA:0'):
            # 测试用例：每个元素是 [0, n) 的一个排列
            test_cases = [
                (tf.constant([0, 1, 2, 3], dtype=tf.int32), "Identity permutation (int32)"),
                (tf.constant([3, 0, 2, 1], dtype=tf.int32), "Random permutation (int32)"),
                (tf.constant([2, 0, 1], dtype=tf.int64), "Small permutation (int64)"),
                (tf.constant([4, 2, 1, 3, 0], dtype=tf.int64), "Larger permutation (int64)"),
            ]

            for perm_tensor, description in test_cases:
                print(f"\n测试: {description}")
                print(f" 输入: {perm_tensor.numpy()}")
                print(f" 输入类型: {perm_tensor.dtype}")

                # 调用底层 InvertPermutation 算子
                inv_perm = tf.raw_ops.InvertPermutation(x=perm_tensor)
                result = inv_perm.numpy()
                print(f" 输出: {result}")
                print(f" 输出类型: {inv_perm.dtype}")
                print(f" 输出设备: {inv_perm.device}")

                # 验证：inv_perm[perm[i]] == i
                perm_np = perm_tensor.numpy()
                valid = True
                for i in range(len(perm_np)):
                    if result[perm_np[i]] != i:
                        valid = False
                        break

                if valid:
                    print(f" ✓ {description} Test Passed")
                else:
                    print(f" ✗ {description} Test Failed: Inversion incorrect")

    except Exception as e:
        print(f"✗ Basic InvertPermutation Test Failed: {e}")


def test_invertpermutation_edge_cases():
    """测试边界情况"""
    print("\n--- 测试边界情况 ---")
    edge_cases = [
        (tf.constant([], dtype=tf.int32), "Empty permutation (int32)"),
        (tf.constant([0], dtype=tf.int64), "Single-element permutation (int64)"),
    ]

    try:
        tf.config.set_soft_device_placement(True)
        with tf.device('/device:MUSA:0'):
            for perm_tensor, description in edge_cases:
                print(f"\n测试: {description}")
                print(f" 输入形状: {perm_tensor.shape}")
                print(f" 输入: {perm_tensor.numpy()}")

                inv_perm = tf.raw_ops.InvertPermutation(x=perm_tensor)
                result = inv_perm.numpy()
                print(f" 输出: {result}")

                # 验证空或单元素情况
                if len(perm_tensor) == 0:
                    assert result.size == 0
                elif len(perm_tensor) == 1:
                    assert result[0] == 0

                print(f" ✓ {description} Test Passed")

    except Exception as e:
        print(f"✗ Edge Case Test Failed: {e}")


def main():
    """主测试函数"""
    test_invertpermutation_basic()
    test_invertpermutation_edge_cases()

    print("\n" + "="*60)
    print("All InvertPermutation Operator Tests Completed!")
    print("="*60)


if __name__ == "__main__":
    main()
