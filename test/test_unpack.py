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
print("Pack and Unpack Operator Test Suite")
print("="*60)

def test_pack_basic():
    """测试基本 Pack 操作"""
    print("\n--- 测试基本 Pack 操作 ---")
    try:
        tf.config.set_soft_device_placement(True)
        with tf.device('/device:MUSA:0'):
            # 测试不同轴的 pack
            test_cases = [
                # (输入列表, axis, 期望输出 shape, 描述)
                ([tf.constant([1, 2]), tf.constant([3, 4])], 0, [2, 2], "1D vectors, axis=0"),
                ([tf.constant([1, 2]), tf.constant([3, 4])], 1, [2, 2], "1D vectors, axis=1"),
                ([tf.constant([[1, 2]]), tf.constant([[3, 4]])], 0, [2, 1, 2], "2D matrices, axis=0"),
                ([tf.constant([[1], [2]]), tf.constant([[3], [4]])], 2, [2, 2, 1], "2D matrices, axis=2"),
                ([tf.constant(5.0), tf.constant(6.0)], 0, [2], "Scalars, axis=0"),
            ]
            
            for inputs, axis, expected_shape, description in test_cases:
                print(f"\n测试: {description}")
                print(f"  输入数量: {len(inputs)}")
                print(f"  Axis: {axis}")
                
                # 执行 pack
                packed = tf.stack(inputs, axis=axis)
                actual_shape = packed.shape.as_list()
                
                print(f"  输出 Shape: {actual_shape}")
                print(f"  期望 Shape: {expected_shape}")
                print(f"  输出值:\n{packed.numpy()}")
                
                if actual_shape == expected_shape:
                    print(f"  ✓ {description} Pack Test Passed")
                else:
                    print(f"  ✗ {description} Pack Test Failed")
                    
    except Exception as e:
        print(f"✗ Basic Pack Test Failed: {e}")

def test_unpack_basic():
    """测试基本 Unpack 操作"""
    print("\n--- 测试基本 Unpack 操作 ---")
    try:
        tf.config.set_soft_device_placement(True)
        with tf.device('/device:MUSA:0'):
            # 测试不同轴的 unpack
            test_cases = [
                # (输入张量, axis, 期望输出数量, 期望单个输出 shape, 描述)
                (tf.constant([[1, 2], [3, 4]]), 0, 2, [2], "2x2 matrix, axis=0"),
                (tf.constant([[1, 2], [3, 4]]), 1, 2, [2], "2x2 matrix, axis=1"),
                #(tf.constant([, ), 0, 2, [1, 1], "3D tensor, axis=0"),
                #(tf.constant([5, 6, 7]), 0, 3, [], "1D vector, axis=0 (scalars)"),
            ]
            
            for input_tensor, axis, expected_num_outputs, expected_output_shape, description in test_cases:
                print(f"\n测试: {description}")
                print(f"  输入 Shape: {input_tensor.shape}")
                print(f"  Axis: {axis}")
                
                # 执行 unpack
                unpacked = tf.unstack(input_tensor, axis=axis)
                actual_num_outputs = len(unpacked)
                actual_output_shape = unpacked[0].shape.as_list() if unpacked else []
                
                print(f"  输出数量: {actual_num_outputs}")
                print(f"  期望数量: {expected_num_outputs}")
                print(f"  单个输出 Shape: {actual_output_shape}")
                print(f"  期望单个 Shape: {expected_output_shape}")
                
                # 验证数量和形状
                shape_ok = (actual_num_outputs == expected_num_outputs and 
                           actual_output_shape == expected_output_shape)
                
                if shape_ok:
                    print(f"  ✓ {description} Unpack Test Passed")
                    # 可选：打印前几个输出值
                    if unpacked:
                        print(f"    示例输出: {unpacked[0].numpy()}")
                else:
                    print(f"  ✗ {description} Unpack Test Failed")
                    
    except Exception as e:
        print(f"✗ Basic Unpack Test Failed: {e}")

def test_pack_unpack_inverse():
    """测试 Pack 和 Unpack 的互逆性"""
    print("\n--- 测试 Pack/Unpack 互逆性 ---")
    try:
        tf.config.set_soft_device_placement(True)
        with tf.device('/device:MUSA:0'):
            # 创建测试张量列表
            original_tensors = [
                tf.constant([[1.0, 2.0], [3.0, 4.0]]),
                tf.constant([[5.0, 6.0], [7.0, 8.0]]),
                tf.constant([[9.0, 10.0], [11.0, 12.0]])
            ]
            
            test_axes = [0, 1, 2]  # 测试不同 axis
            
            for axis in test_axes:
                print(f"\n测试互逆性 (axis={axis}):")
                
                # Pack
                packed = tf.stack(original_tensors, axis=axis)
                print(f"  Pack 后 Shape: {packed.shape}")
                
                # Unpack
                unpacked = tf.unstack(packed, axis=axis)
                print(f"  Unpack 后数量: {len(unpacked)}")
                
                # 验证每个张量是否相等
                inverse_ok = True
                for i, (orig, unpacked_tensor) in enumerate(zip(original_tensors, unpacked)):
                    if not tf.reduce_all(tf.equal(orig, unpacked_tensor)):
                        inverse_ok = False
                        print(f"    ✗ 张量 {i} 不匹配!")
                        break
                
                if inverse_ok:
                    print(f"  ✓ Axis={axis} 互逆性测试通过")
                else:
                    print(f"  ✗ Axis={axis} 互逆性测试失败")
                    
    except Exception as e:
        print(f"✗ Pack/Unpack Inverse Test Failed: {e}")

def test_pack_different_dtypes():
    """测试不同数据类型的 Pack 操作"""
    print("\n--- 测试不同数据类型 (Pack) ---")
    dtypes_to_test = [
        (tf.float32, "float32"),
        (tf.int32, "int32"),
        (tf.int64, "int64"),
    ]
    
    for dtype, dtype_name in dtypes_to_test:
        print(f"\n测试数据类型: {dtype_name}")
        try:
            tf.config.set_soft_device_placement(True)
            with tf.device('/device:MUSA:0'):
                if dtype == tf.float32:
                    inputs = [tf.constant([1.5, 2.5], dtype=dtype), 
                             tf.constant([3.5, 4.5], dtype=dtype)]
                else:
                    inputs = [tf.constant([1, 2], dtype=dtype), 
                             tf.constant([3, 4], dtype=dtype)]
                
                packed = tf.stack(inputs, axis=0)
                print(f"  输入类型: {inputs[0].dtype}")
                print(f"  输出类型: {packed.dtype}")
                print(f"  输出 Shape: {packed.shape}")
                print(f"  输出值: {packed.numpy()}")
                
                if packed.dtype == dtype and packed.shape == [2, 2]:
                    print(f"  ✓ {dtype_name} Pack Test Passed")
                else:
                    print(f"  ✗ {dtype_name} Pack Test Failed")
                    
        except Exception as e:
            print(f"  ✗ {dtype_name} Pack Test Failed: {e}")

def test_edge_cases():
    """测试边界情况"""
    print("\n--- 测试边界情况 ---")
    try:
        tf.config.set_soft_device_placement(True)
        with tf.device('/device:MUSA:0'):
            # 单个张量 pack/unpack
            single_tensor = tf.constant([10, 20])
            packed_single = tf.stack([single_tensor], axis=0)
            unpacked_single = tf.unstack(packed_single, axis=0)
            
            if tf.reduce_all(tf.equal(single_tensor, unpacked_single[0])):
                print("✓ 单张量 Pack/Unpack 测试通过")
            else:
                print("✗ 单张量 Pack/Unpack 测试失败")
            
            # 标量测试
            scalars = [tf.constant(1.0), tf.constant(2.0), tf.constant(3.0)]
            packed_scalars = tf.stack(scalars, axis=0)
            unpacked_scalars = tf.unstack(packed_scalars, axis=0)
            
            if (packed_scalars.shape == [3] and 
                len(unpacked_scalars) == 3 and
                all(tf.equal(s, u).numpy() for s, u in zip(scalars, unpacked_scalars))):
                print("✓ 标量 Pack/Unpack 测试通过")
            else:
                print("✗ 标量 Pack/Unpack 测试失败")
                
    except Exception as e:
        print(f"✗ Edge Cases Test Failed: {e}")

def main():
    """主测试函数"""
    # 运行所有测试
    test_pack_basic()
    test_unpack_basic()
    test_pack_unpack_inverse()
    test_pack_different_dtypes()
    test_edge_cases()

    print("\n" + "="*60)
    print("All Pack and Unpack Operator Tests Completed!")
    print("="*60)

if __name__ == "__main__":
    main()
