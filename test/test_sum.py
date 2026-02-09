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

print("\n--- 3. Testing Basic Sum Operations ---")

def test_sum_1d():
    """测试1D张量的求和"""
    print("\n--- 测试1D张量 ---")
    try:
        with tf.device('/device:MUSA:0'):
            # 创建测试数据
            x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
            print(f"Input: {x.numpy()}")
            
            # 全局求和
            result_all = tf.reduce_sum(x)
            print(f"Global sum: {result_all.numpy()}")
            
            # 指定轴求和（这里1D张量只有0轴）
            result_axis0 = tf.reduce_sum(x, axis=0)
            print(f"Sum along axis 0: {result_axis0.numpy()}")
            
            # 测试keepdims=True
            result_keepdims = tf.reduce_sum(x, keepdims=True)
            print(f"Sum with keepdims=True: {result_keepdims.numpy()}")
            print(f"Result shape with keepdims: {result_keepdims.shape}")
            
            # 验证结果
            expected_all = 10.0
            assert abs(result_all.numpy() - expected_all) < 0.0001
            print("✓ 1D Sum Test Passed")
            
    except Exception as e:
        print(f"✗ 1D Sum Test Failed: {e}")

def test_sum_2d():
    """测试2D张量的求和"""
    print("\n--- 测试2D张量 ---")
    try:
        with tf.device('/device:MUSA:0'):
            # 创建测试数据
            x = tf.constant([[1.0, 2.0, 3.0], 
                             [4.0, 5.0, 6.0]], dtype=tf.float32)
            print(f"Input shape: {x.shape}")
            print(f"Input:\n{x.numpy()}")
            
            # 全局求和
            result_all = tf.reduce_sum(x)
            print(f"Global sum: {result_all.numpy()}")
            
            # 按行求和（axis=1）
            result_row = tf.reduce_sum(x, axis=1)
            print(f"Sum along rows (axis=1): {result_row.numpy()}")
            
            # 按列求和（axis=0）
            result_col = tf.reduce_sum(x, axis=0)
            print(f"Sum along columns (axis=0): {result_col.numpy()}")
            
            # 测试多个轴求和
            result_axes = tf.reduce_sum(x, axis=[0, 1])
            print(f"Sum along axes [0,1]: {result_axes.numpy()}")
            
            # 测试keepdims
            result_keepdims = tf.reduce_sum(x, axis=0, keepdims=True)
            print(f"Sum with keepdims (axis=0): {result_keepdims.numpy()}")
            print(f"Shape with keepdims: {result_keepdims.shape}")
            
            # 验证结果
            assert abs(result_all.numpy() - 21.0) < 0.0001
            assert np.allclose(result_row.numpy(), [6.0, 15.0])
            assert np.allclose(result_col.numpy(), [5.0, 7.0, 9.0])
            print("✓ 2D Sum Test Passed")
            
    except Exception as e:
        print(f"✗ 2D Sum Test Failed: {e}")

def test_sum_3d():
    """测试3D张量的求和"""
    print("\n--- 测试3D张量 ---")
    try:
        with tf.device('/device:MUSA:0'):
            # 创建3D测试数据
            x = tf.constant([[[1.0, 2.0], [3.0, 4.0]],
                             [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float32)
            print(f"Input shape: {x.shape}")
            print(f"Input[0]:\n{x.numpy()[0]}")
            print(f"Input[1]:\n{x.numpy()[1]}")
            
            # 测试不同轴的求和
            result_axis0 = tf.reduce_sum(x, axis=0)
            print(f"Sum along axis 0:\n{result_axis0.numpy()}")
            
            result_axis1 = tf.reduce_sum(x, axis=1)
            print(f"Sum along axis 1:\n{result_axis1.numpy()}")
            
            result_axis2 = tf.reduce_sum(x, axis=2)
            print(f"Sum along axis 2:\n{result_axis2.numpy()}")
            
            result_axes_01 = tf.reduce_sum(x, axis=[0, 1])
            print(f"Sum along axes [0,1]: {result_axes_01.numpy()}")
            
            result_axes_012 = tf.reduce_sum(x, axis=[0, 1, 2])
            print(f"Sum along axes [0,1,2]: {result_axes_012.numpy()}")
            
            print("✓ 3D Sum Test Passed")
            
    except Exception as e:
        print(f"✗ 3D Sum Test Failed: {e}")

def test_sum_different_dtypes():
    """测试不同数据类型的求和"""
    print("\n--- 测试不同数据类型 ---")
    
    dtypes_to_test = [
        (tf.float32, "float32"),
        (tf.float64, "float64"),
        (tf.int32, "int32"),
        (tf.int64, "int64"),
    ]
    
    for dtype, dtype_name in dtypes_to_test:
        print(f"\n测试数据类型: {dtype_name}")
        try:
            with tf.device('/device:MUSA:0'):
                # 创建测试数据
                if dtype in [tf.float32, tf.float64]:
                    x = tf.constant([1.5, 2.5, 3.5, 4.5], dtype=dtype)
                else:
                    x = tf.constant([1, 2, 3, 4], dtype=dtype)
                
                result = tf.reduce_sum(x)
                print(f"Input: {x.numpy()}")
                print(f"Sum result: {result.numpy()} (dtype: {result.dtype})")
                
                # 验证结果
                if dtype in [tf.float32, tf.float64]:
                    expected = 12.0
                    assert abs(result.numpy() - expected) < 0.0001
                else:
                    expected = 10
                    assert result.numpy() == expected
                    
                print(f"✓ {dtype_name} Sum Test Passed")
                
        except Exception as e:
            print(f"✗ {dtype_name} Sum Test Failed: {e}")

def test_sum_with_negative_axes():
    """测试负轴索引的求和"""
    print("\n--- 测试负轴索引 ---")
    try:
        with tf.device('/device:MUSA:0'):
            x = tf.constant([[1.0, 2.0, 3.0], 
                             [4.0, 5.0, 6.0]], dtype=tf.float32)
            print(f"Input shape: {x.shape}")
            
            # 使用负轴索引（-1表示最后一个轴）
            result_neg1 = tf.reduce_sum(x, axis=-1)
            print(f"Sum along axis -1 (last axis): {result_neg1.numpy()}")
            
            # 验证与正轴索引结果一致
            result_pos1 = tf.reduce_sum(x, axis=1)
            assert np.allclose(result_neg1.numpy(), result_pos1.numpy())
            
            print("✓ Negative Axes Test Passed")
            
    except Exception as e:
        print(f"✗ Negative Axes Test Failed: {e}")

def main():
    """主测试函数"""
    print("=" * 60)
    print("MUSA Sum Operator Test Suite")
    print("=" * 60)
    
    # 运行所有测试
    test_sum_1d()
    test_sum_2d()
    test_sum_3d()
    test_sum_different_dtypes()
    test_sum_with_negative_axes()
    
    print("\n" + "=" * 60)
    print("All Sum Operator Tests Completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
