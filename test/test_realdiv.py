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
print("RealDiv Operator Test Suite")
print("="*60)

def test_realdiv_basic():
    """测试基本除法操作"""
    print("\n--- 测试基本除法 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 创建测试数据
            dividend = tf.constant([10.0, 20.0, 30.0, 40.0], dtype=tf.float32)
            divisor = tf.constant([2.0, 4.0, 5.0, 8.0], dtype=tf.float32)
            
            print(f"Dividend: {dividend.numpy()}")
            print(f"Divisor: {divisor.numpy()}")
            
            # 使用tf.divide进行除法
            result = tf.divide(dividend, divisor)
            
            print(f"Division result: {result.numpy()}")
            print(f"Result shape: {result.shape}")
            print(f"Result dtype: {result.dtype}")
            print(f"Result device: {result.device}")
            
            # 验证结果
            expected = np.array([5.0, 5.0, 6.0, 5.0], dtype=np.float32)
            if np.allclose(result.numpy(), expected, rtol=1e-5):
                print("✓ Basic RealDiv Test Passed")
            else:
                print(f"✗ Basic RealDiv Test Failed: Expected {expected}, got {result.numpy()}")
                
    except Exception as e:
        print(f"✗ Basic RealDiv Test Failed: {e}")

def test_realdiv_scalar_broadcast():
    """测试标量广播除法"""
    print("\n--- 测试标量广播 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 标量除以向量
            print("1. 标量除以向量:")
            scalar = tf.constant(100.0, dtype=tf.float32)
            vector = tf.constant([2.0, 4.0, 5.0, 10.0], dtype=tf.float32)
            
            result1 = tf.divide(scalar, vector)
            print(f"   {scalar.numpy()} / {vector.numpy()} = {result1.numpy()}")
            
            # 向量除以标量
            print("2. 向量除以标量:")
            result2 = tf.divide(vector, scalar)
            print(f"   {vector.numpy()} / {scalar.numpy()} = {result2.numpy()}")
            
            # 验证结果
            expected1 = np.array([50.0, 25.0, 20.0, 10.0], dtype=np.float32)
            expected2 = np.array([0.02, 0.04, 0.05, 0.1], dtype=np.float32)
            
            if (np.allclose(result1.numpy(), expected1, rtol=1e-5) and
                np.allclose(result2.numpy(), expected2, rtol=1e-5)):
                print("✓ Scalar Broadcast Test Passed")
            else:
                print("✗ Scalar Broadcast Test Failed")
                
    except Exception as e:
        print(f"✗ Scalar Broadcast Test Failed: {e}")

def test_realdiv_matrix_broadcast():
    """测试矩阵广播除法"""
    print("\n--- 测试矩阵广播 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 矩阵除以向量（广播）
            print("1. 矩阵除以向量（广播）:")
            matrix = tf.constant([[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]], dtype=tf.float32)
            vector = tf.constant([2.0, 2.0, 2.0], dtype=tf.float32)
            
            result1 = tf.divide(matrix, vector)
            print(f"   Matrix shape: {matrix.shape}")
            print(f"   Vector shape: {vector.shape}")
            print(f"   Result shape: {result1.shape}")
            print(f"   Result:\n{result1.numpy()}")
            
            # 向量除以矩阵（广播）
            print("\n2. 向量除以矩阵（广播）:")
            vector2 = tf.constant([10.0, 20.0, 30.0], dtype=tf.float32)
            result2 = tf.divide(vector2, matrix)
            print(f"   Vector: {vector2.numpy()}")
            print(f"   Result:\n{result2.numpy()}")
            
            # 验证结果
            expected1 = np.array([[0.5, 1.0, 1.5],
                                   [2.0, 2.5, 3.0],
                                   [3.5, 4.0, 4.5]], dtype=np.float32)
            
            if np.allclose(result1.numpy(), expected1, rtol=1e-5):
                print("✓ Matrix Broadcast Test Passed")
            else:
                print("✗ Matrix Broadcast Test Failed")
                
    except Exception as e:
        print(f"✗ Matrix Broadcast Test Failed: {e}")

def test_realdiv_different_dtypes():
    """测试不同数据类型的除法"""
    print("\n--- 测试不同数据类型 ---")
    
    test_cases = [
        (tf.float32, "float32"),
        (tf.float16, "float16"),
    ]
    
    for dtype, dtype_name in test_cases:
        print(f"\n测试数据类型: {dtype_name}")
        try:
            tf.config.set_soft_device_placement(True)

            with tf.device('/device:MUSA:0'):
                # 创建测试数据
                if dtype in [tf.float32, tf.float64]:
                    dividend = tf.constant([10.5, 20.5, 30.5], dtype=dtype)
                    divisor = tf.constant([2.0, 5.0, 10.0], dtype=dtype)
                else:
                    dividend = tf.constant([10, 20, 30], dtype=dtype)
                    divisor = tf.constant([2, 5, 10], dtype=dtype)
                
                print(f"   Dividend: {dividend.numpy()}")
                print(f"   Divisor: {divisor.numpy()}")
                
                # 执行除法
                result = tf.divide(dividend, divisor)
                
                print(f"   Result: {result.numpy()}")
                print(f"   Result dtype: {result.dtype}")
                
                # 验证结果
                if dtype in [tf.float32, tf.float64]:
                    expected = dividend.numpy() / divisor.numpy()
                else:
                    expected = dividend.numpy() // divisor.numpy()  # 整数除法
                
                if np.allclose(result.numpy(), expected, rtol=1e-5):
                    print(f"✓ {dtype_name} RealDiv Test Passed")
                else:
                    print(f"✗ {dtype_name} RealDiv Test Failed")
                    
        except Exception as e:
            print(f"✗ {dtype_name} RealDiv Test Failed: {e}")

def main():
    """主测试函数"""
    # 运行所有测试
    test_realdiv_basic()
    test_realdiv_scalar_broadcast()
    test_realdiv_matrix_broadcast()
    test_realdiv_different_dtypes()

    print("\n" + "="*60)
    print("All RealDiv Operator Tests Completed!")
    print("="*60)

if __name__ == "__main__":
    main()
