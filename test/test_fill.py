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
print("Fill Operator Test")
print("="*60)

def test_fill_1d():
    """测试1D张量的填充"""
    print("\n--- 测试1D张量填充 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 创建形状 [5]，填充值为 3.14
            shape = [5]
            value = 3.14
            
            print(f"Shape: {shape}")
            print(f"Value: {value}")
            
            # 使用tf.fill创建张量
            result = tf.fill(shape, value)
            
            print(f"Fill result: {result.numpy()}")
            print(f"Result shape: {result.shape}")
            print(f"Result dtype: {result.dtype}")
            print(f"Result device: {result.device}")
            
            # 验证结果
            expected = np.full(shape, value, dtype=np.float32)
            if np.allclose(result.numpy(), expected, rtol=1e-5):
                print("✓ 1D Fill Test Passed")
            else:
                print(f"✗ 1D Fill Test Failed: Expected {expected}, got {result.numpy()}")
                
    except Exception as e:
        print(f"✗ 1D Fill Test Failed: {e}")

def test_fill_2d():
    """测试2D张量的填充"""
    print("\n--- 测试2D张量填充 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 创建形状 [3, 4]，填充值为 2.5
            shape = [3, 4]
            value = 2.5
            
            print(f"Shape: {shape}")
            print(f"Value: {value}")
            
            # 使用tf.fill创建张量
            result = tf.fill(shape, value)
            
            print(f"Fill result shape: {result.shape}")
            print(f"Fill result:\n{result.numpy()}")
            print(f"Result dtype: {result.dtype}")
            print(f"Result device: {result.device}")
            
            # 验证结果
            expected = np.full(shape, value, dtype=np.float32)
            if np.allclose(result.numpy(), expected, rtol=1e-5):
                print("✓ 2D Fill Test Passed")
            else:
                print(f"✗ 2D Fill Test Failed")
                
    except Exception as e:
        print(f"✗ 2D Fill Test Failed: {e}")

def test_fill_3d():
    """测试3D张量的填充"""
    print("\n--- 测试3D张量填充 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 创建形状 [2, 3, 4]，填充值为 -1.0
            shape = [2, 3, 4]
            value = -1.0
            
            print(f"Shape: {shape}")
            print(f"Value: {value}")
            
            # 使用tf.fill创建张量
            result = tf.fill(shape, value)
            
            print(f"Fill result shape: {result.shape}")
            print(f"Result dtype: {result.dtype}")
            print(f"Result device: {result.device}")
            
            # 打印部分内容验证
            print(f"First element: {result.numpy()[0, 0, 0]}")
            print(f"Last element: {result.numpy()[-1, -1, -1]}")
            
            # 验证结果
            expected = np.full(shape, value, dtype=np.float32)
            if np.allclose(result.numpy(), expected, rtol=1e-5):
                print("✓ 3D Fill Test Passed")
            else:
                print(f"✗ 3D Fill Test Failed")
                
    except Exception as e:
        print(f"✗ 3D Fill Test Failed: {e}")

def test_fill_scalar():
    """测试标量（0维张量）的填充"""
    print("\n--- 测试标量填充 ---")
    try:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:MUSA:0'):
            # 创建标量，形状 []，填充值为 42.0
            shape = []
            value = 42.0
            
            print(f"Shape: {shape} (scalar)")
            print(f"Value: {value}")
            
            # 使用tf.fill创建张量
            result = tf.fill(shape, value)
            
            print(f"Fill result: {result.numpy()}")
            print(f"Result shape: {result.shape}")
            print(f"Result dtype: {result.dtype}")
            
            # 验证结果
            expected = np.array(value, dtype=np.float32)
            if np.allclose(result.numpy(), expected, rtol=1e-5):
                print("✓ Scalar Fill Test Passed")
            else:
                print(f"✗ Scalar Fill Test Failed: Expected {expected}, got {result.numpy()}")
                
    except Exception as e:
        print(f"✗ Scalar Fill Test Failed: {e}")

def test_fill_different_dtypes():
    print("\n--- 测试不同数据类型 ---")
    
    test_cases = [
        (tf.float32, 3.14, "float32"),
        (tf.float64, 2.718281828459045, "float64"),
        (tf.int32, 42, "int32"),
        (tf.int64, 100, "int64"),
    ]
    
    for dtype, value, dtype_name in test_cases:
        print(f"\n测试数据类型: {dtype_name}")
        try:
            with tf.device('/device:MUSA:0'):
                shape = [2, 3]
                
                # 关键点：通过控制输入 value 的类型来控制输出结果的类型
                value_tensor = tf.cast(value, dtype=dtype)
                result = tf.fill(shape, value_tensor)
                
                print(f"Result shape: {result.shape}")
                print(f"Result dtype: {result.dtype}")
                print(f"First few values: {result.numpy().flatten()[:3]}")
                
                expected = np.full(shape, value, dtype=dtype.as_numpy_dtype)
                if np.allclose(result.numpy(), expected):
                    print(f"✓ {dtype_name} Fill Test Passed")
                else:
                    print(f"✗ {dtype_name} Fill Test Failed")
        except Exception as e:
            print(f"✗ {dtype_name} Fill Test Failed: {e}")

def main():
    """主测试函数"""
    # 运行所有测试
    test_fill_1d()
    test_fill_2d()
    test_fill_3d()
    test_fill_scalar()
    test_fill_different_dtypes()
    
    print("\n" + "="*60)
    print("All Fill Operator Tests Completed!")
    print("="*60)

if __name__ == "__main__":
    main()

