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
        a = tf.identity(tf.constant([1.0, 2.0, 3.0]))
        print(f"Memory check value: {a.numpy()}")
        print("Memory Test Passed")
except Exception as e:
    print(f"Memory Test Failed: {e}")

print("\n" + "="*60)
print("Size Operator Test Suite")
print("="*60)

def test_size_basic():
    """测试基本Size操作"""
    print("\n--- 测试基本Size操作 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 测试不同形状的张量
            test_cases = [
                (tf.constant([1.0, 2.0, 3.0, 4.0]), "1D Vector", 4),
                (tf.constant([[1.0, 2.0], [3.0, 4.0]]), "2D Matrix", 4),
                (tf.constant([[[1.0, 2.0], [3.0, 4.0]], 
                               [[5.0, 6.0], [7.0, 8.0]]]), "3D Tensor", 8),
                (tf.constant(42.0), "Scalar", 1),
                (tf.constant([], dtype=tf.float32), "Empty Vector", 0),
                (tf.constant([[]], dtype=tf.float32), "Empty Matrix", 0),
            ]
            
            for tensor, description, expected_size in test_cases:
                print(f"\n测试: {description}")
                print(f"  形状: {tensor.shape}")
                print(f"  数据类型: {tensor.dtype}")
                
                # 计算大小
                size_tensor = tf.size(tensor)
                actual_size = size_tensor.numpy()
                
                print(f"  计算大小: {actual_size}")
                print(f"  期望大小: {expected_size}")
                print(f"  Size结果类型: {size_tensor.dtype}")
                print(f"  Size结果设备: {size_tensor.device}")
                
                if actual_size == expected_size:
                    print(f"  ✓ {description} Size Test Passed")
                else:
                    print(f"  ✗ {description} Size Test Failed")
                    
    except Exception as e:
        print(f"✗ Basic Size Test Failed: {e}")

def test_size_different_dtypes():
    """测试不同数据类型的Size操作"""
    print("\n--- 测试不同数据类型 ---")
    
    test_cases = [
        (tf.float32, "float32"),
        (tf.float64, "float64"),
        (tf.int32, "int32"),
        (tf.int64, "int64"),
    ]
   
    for dtype, dtype_name in test_cases:
        print(f"\n测试数据类型: {dtype_name}")
        try:
            tf.config.set_soft_device_placement(True)

            with tf.device('/device:MUSA:0'):
                # 创建测试数据
                if dtype in [tf.float32, tf.float64]:
                    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
                elif dtype == tf.half:
                    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float16)
                else:
                    tensor = tf.constant([[1, 2], [3, 4]], dtype=dtype)
                
                print(f"  输入张量形状: {tensor.shape}")
                print(f"  输入张量类型: {tensor.dtype}")
                
                # 计算大小
                size_result = tf.size(tensor)
                actual_size = size_result.numpy()
                
                print(f"  Size结果: {actual_size}")
                print(f"  Size结果类型: {size_result.dtype}")
                
                if actual_size == 4:  # 2x2矩阵有4个元素
                    print(f"  ✓ {dtype_name} Size Test Passed")
                else:
                    print(f"  ✗ {dtype_name} Size Test Failed: Expected 4, got {actual_size}")
                    
        except Exception as e:
            print(f"  ✗ {dtype_name} Size Test Failed: {e}")

def test_size_output_types():
    """测试不同的输出类型"""
    print("\n--- 测试不同输出类型 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 创建测试张量
            tensor = tf.constant([[1.0, 2.0, 3.0], 
                                   [4.0, 5.0, 6.0]], dtype=tf.float32)
            print(f"输入张量形状: {tensor.shape}")
            print(f"元素总数: {tensor.shape.num_elements()}")
            
            # 测试int32输出
            print("\n1. 测试int32输出类型:")
            # 使用tf.size指定输出类型
            size_int32 = tf.size(tensor, out_type=tf.int32)
            print(f"  Size结果: {size_int32.numpy()}")
            print(f"  Size结果类型: {size_int32.dtype}")
            
            # 测试int64输出
            print("\n2. 测试int64输出类型:")
            size_int64 = tf.size(tensor, out_type=tf.int64)
            print(f"  Size结果: {size_int64.numpy()}")
            print(f"  Size结果类型: {size_int64.dtype}")
            
            # 验证结果一致性
            if size_int32.numpy() == size_int64.numpy() == 6:
                print("✓ Output Types Test Passed: int32和int64输出一致")
            else:
                print("✗ Output Types Test Failed: 输出不一致")
                
    except Exception as e:
        print(f"✗ Output Types Test Failed: {e}")

def test_size_edge_cases():
    """测试边界情况"""
    print("\n--- 测试边界情况 ---")
    
    edge_cases = [
        ("零维标量", tf.constant(42.0), 1),
        ("一维空向量", tf.constant([], dtype=tf.float32), 0),
        ("二维空矩阵", tf.constant([[]], dtype=tf.float32), 0),
        ("高维空张量", tf.zeros((2, 0, 3, 0), dtype=tf.float32), 0),
        ("单元素向量", tf.constant([99.0]), 1),
        ("非常规形状", tf.zeros((1, 1, 1, 1, 1), dtype=tf.float32), 1),
    ]
    
    for description, tensor, expected_size in edge_cases:
        print(f"\n测试: {description}")
        try:
            tf.config.set_soft_device_placement(True)

            with tf.device('/device:MUSA:0'):
                print(f"  张量形状: {tensor.shape}")
                print(f"  张量类型: {tensor.dtype}")
                
                size_result = tf.size(tensor)
                actual_size = size_result.numpy()
                
                print(f"  实际大小: {actual_size}")
                print(f"  期望大小: {expected_size}")
                
                if actual_size == expected_size:
                    print(f"  ✓ {description} 测试通过")
                else:
                    print(f"  ✗ {description} 测试失败")
                    
        except Exception as e:
            print(f"  ✗ {description} 测试失败: {e}")


def main():
    """主测试函数"""
    # 运行所有测试
    test_size_basic()
    test_size_different_dtypes()
    test_size_output_types()
    test_size_edge_cases()

    print("\n" + "="*60)
    print("All Size Operator Tests Completed!")
    print("="*60)

if __name__ == "__main__":
    main()
