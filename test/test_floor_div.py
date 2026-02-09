import tensorflow as tf
import numpy as np
import time
import os
import sys

# --- 1. 加载插件 ---
def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        try:
            tf.load_library(plugin_path)
            print("SUCCESS: MUSA plugin loaded!")
            return True
        except Exception as e:
            print(f"FAILED: Error loading plugin: {e}")
            return False
    else:
        print(f"ERROR: Plugin not found at {plugin_path}")
        return False

# --- 2. 核心验证函数 ---
def verify_floordiv(test_name, shape_x, shape_y, dtype):
    print(f"\n--- Testing {test_name} [{dtype.name}] ---")
    print(f"Shapes: {shape_x} // {shape_y}")
    
    np_dtype = dtype.as_numpy_dtype
    
    # 根据类型生成不同的测试数据
    if dtype in [tf.int32, tf.int64]:
        # 整数生成逻辑：范围 [-100, 100]
        x_np = np.random.randint(-100, 100, size=shape_x).astype(np_dtype)
        # 避免除以 0，生成 [-10, -1] 和 [1, 10]
        y_raw = np.random.randint(1, 10, size=shape_y).astype(np_dtype)
        sign = np.random.choice([-1, 1], size=shape_y).astype(np_dtype)
        y_np = y_raw * sign
    else:
        # 浮点数生成逻辑
        x_np = np.random.uniform(-100, 100, size=shape_x).astype(np_dtype)
        y_np = np.random.uniform(0.1, 10, size=shape_y).astype(np_dtype)
        # 随机反转符号以测试负数除法
        sign = np.random.choice([-1.0, 1.0], size=shape_y).astype(np_dtype)
        y_np = y_np * sign
        # 避免绝对值太小导致数值不稳定
        y_np = np.where(np.abs(y_np) < 0.1, 0.1, y_np)

    # 1. CPU 基准结果
    with tf.device('/CPU:0'):
        x_cpu = tf.constant(x_np, dtype=dtype)
        y_cpu = tf.constant(y_np, dtype=dtype)
        start = time.time()
        res_cpu = tf.math.floordiv(x_cpu, y_cpu)
        cpu_time = (time.time() - start) * 1000
    
    # 2. MUSA 测试结果
    try:
        # 使用 soft_device_placement=True，如果 MUSA 未注册 int，应该自动回退到 CPU 而不报错
        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x_np, dtype=dtype)
            y_musa = tf.constant(y_np, dtype=dtype)
            
            # 预热一次
            _ = tf.math.floordiv(x_musa, y_musa)
            
            start = time.time()
            res_musa = tf.math.floordiv(x_musa, y_musa)
            musa_time = (time.time() - start) * 1000
        
        val_cpu = res_cpu.numpy()
        val_musa = res_musa.numpy()
        
        # 比较结果
        if dtype in [tf.int32, tf.int64]:
            # 整数必须完全相等
            diff = np.abs(val_cpu - val_musa)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            passed = np.all(val_cpu == val_musa)
        else:
            # 浮点数允许微小误差
            diff = np.abs(val_cpu - val_musa)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            passed = max_diff < 1e-4

        print(f"Max diff: {max_diff}")
        print(f"Time: CPU {cpu_time:.2f}ms | MUSA {musa_time:.2f}ms")
        
        if passed:
            print("Result: PASS")
            return True
        else:
            print(f"Result: FAIL (Values do not match!)")
            # 打印前几个错误值方便调试
            mismatch_idx = np.where(val_cpu != val_musa)
            if len(mismatch_idx[0]) > 0:
                print(f"First mismatch at index 0: CPU={val_cpu.flatten()[0]}, MUSA={val_musa.flatten()[0]}")
            return False
            
    except Exception as e:
        print(f"Result: ERROR/CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- 3. 测试用例集 ---

def test_float_cases():
    print("\n" + "="*30 + "\nRunning FLOAT Tests\n" + "="*30)
    test_cases = [
        ([1024], [1024], tf.float32, "Vector 1K"),
        ([2, 2], [2, 2], tf.float32, "Matrix 2x2"),
        ([10, 10], [10, 10], tf.float32, "Matrix 10x10"),
    ]
    passed = 0
    for shape_x, shape_y, dtype, name in test_cases:
        if verify_floordiv(name, shape_x, shape_y, dtype):
            passed += 1
    return passed, len(test_cases)

def test_int_cases():
    print("\n" + "="*30 + "\nRunning INT Tests (Check for Crash/Fallback)\n" + "="*30)
    # 测试 Int32 和 Int64
    dtypes_to_test = [tf.int32, tf.int64]
    
    test_cases = []
    for dt in dtypes_to_test:
        test_cases.append(([4], [4], dt, f"Small Vector {dt.name}"))
        test_cases.append(([1024], [1024], dt, f"Large Vector {dt.name}"))
        # 广播测试
        test_cases.append(([5, 5], [1, 5], dt, f"Broadcast {dt.name}"))

    passed = 0
    for shape_x, shape_y, dtype, name in test_cases:
        # 注意：如果 C++ 实现有 bug，这里可能会直接导致进程崩溃，try-except 无法捕获 SegFault
        if verify_floordiv(name, shape_x, shape_y, dtype):
            passed += 1
    return passed, len(test_cases)

def test_edge_cases_int():
    print("\n" + "="*30 + "\nRunning INT Edge Cases (Negative Division)\n" + "="*30)
    # 重点测试 Python 语义的向下取整： -5 // 2 应该等于 -3，而不是 -2
    
    x_vals = [-5, -5, 5, 5, -10, 0]
    y_vals = [ 2, -2, 2, -2, 3, 5]
    expected_results = [-3, 2, 2, -3, -4, 0] # Python 语义结果
    
    dtypes = [tf.int32, tf.int64]
    all_passed = True
    
    for dt in dtypes:
        print(f"\n--- Testing Specific Values [{dt.name}] ---")
        x_tf = tf.constant(x_vals, dtype=dt)
        y_tf = tf.constant(y_vals, dtype=dt)
        
        try:
            with tf.device('/device:MUSA:0'):
                res = tf.math.floordiv(x_tf, y_tf)
            
            res_np = res.numpy()
            expected_np = np.array(expected_results, dtype=dt.as_numpy_dtype)
            
            if np.array_equal(res_np, expected_np):
                print(f"Values: {x_vals} // {y_vals}")
                print(f"Result: {res_np}")
                print("Result: PASS (Correct Python semantics)")
            else:
                print(f"Values: {x_vals} // {y_vals}")
                print(f"Got:      {res_np}")
                print(f"Expected: {expected_np}")
                print("Result: FAIL (Incorrect semantics - likely C++ truncation)")
                all_passed = False
                
        except Exception as e:
            print(f"Result: CRASHED: {e}")
            all_passed = False
            
    return all_passed

# --- 主程序 ---
if __name__ == "__main__":
    if not load_musa_plugin():
        sys.exit(1)
    
    # 必须开启软放置，这样如果 MUSA 不支持 Int，会自动切回 CPU 而不是报错
    tf.config.set_soft_device_placement(True)
    
    # 开启日志，观察 Int 算子到底是在 MUSA 上跑还是 CPU 上跑
    # 如果你在日志里看到 "MUSA_TRACE_AUTO" 出现在 INT 测试中，说明跑在 MUSA 上。
    # 如果没看到，说明回退到了 CPU。
    # tf.debugging.set_log_device_placement(True) 

    total_passed = 0
    total_tests = 0
    
    # 1. Float 测试
    p, t = test_float_cases()
    total_passed += p
    total_tests += t
    
    # 2. Int 测试
    p, t = test_int_cases()
    total_passed += p
    total_tests += t
    
    # 3. Int 边缘/语义测试
    if test_edge_cases_int():
        total_passed += 1
    total_tests += 1
    
    print("\n" + "="*50)
    print(f"FINAL SUMMARY: {total_passed}/{total_tests} Test Suites Passed")
    print("="*50)

    if total_passed == total_tests:
        print("SUCCESS: All types handled correctly (either via MUSA or Fallback).")
    else:
        print("FAILURE: Some tests failed or crashed.")