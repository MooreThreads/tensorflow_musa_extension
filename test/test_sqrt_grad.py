import tensorflow as tf
import numpy as np
import os
import traceback

PLUGIN_PATH = "/workspace/tensorflow_musa/build/libmusa_plugin.so"

def load_musa_plugin():
    if os.path.exists(PLUGIN_PATH):
        try:
            tf.load_library(PLUGIN_PATH)
            print(">> [INFO] MUSA Plugin Loaded")
            return True
        except Exception as e:
            print(f">> [ERROR] Failed to load plugin: {e}")
            return False
    return False

def check_result(test_name, musa_val, expected_val, rtol=1e-3, atol=1e-3):
    if musa_val is None: return False
    
    musa_np = musa_val.numpy().astype(np.float32) if hasattr(musa_val, 'numpy') else musa_val
    exp_np = expected_val.numpy().astype(np.float32) if hasattr(expected_val, 'numpy') else expected_val
    
    if hasattr(musa_val, 'dtype') and musa_val.dtype == tf.bfloat16:
        rtol = 3e-2 
        atol = 3e-2
    
    if musa_np.shape != exp_np.shape:
        print(f"[FAIL] {test_name}: Shape mismatch {musa_np.shape} vs {exp_np.shape}")
        return False
        
    if np.allclose(musa_np, exp_np, rtol=rtol, atol=atol):
        print(f"[PASS] {test_name}")
        return True
    else:
        diff = np.max(np.abs(musa_np - exp_np))
        print(f"[FAIL] {test_name}: Max diff {diff} (Tol: {atol})")
        return False

def test_sqrt_grad_logic(shape, dtype):
    """直接测试 SqrtGrad 算子逻辑 (不依赖前向 Sqrt)"""
    test_name = f"Direct_{dtype.name}_{shape}"
    try:
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
        y_np = np.abs(np.random.randn(*shape).astype(np_dtype)) + 0.5
        dy_np = np.random.randn(*shape).astype(np_dtype)
        
        expected = 0.5 * dy_np / y_np
        
        with tf.device('/device:MUSA:0'):
            y = tf.cast(tf.constant(y_np), dtype)
            dy = tf.cast(tf.constant(dy_np), dtype)
            res = tf.raw_ops.SqrtGrad(y=y, dy=dy)
            
        check_result(test_name, res, expected)
    except Exception as e:
        print(f"[FAIL] {test_name} Error: {e}")

def test_sqrt_integration(shape, dtype):
    """测试完整反向传播 (依赖前向 Sqrt)"""
    test_name = f"Tape_{dtype.name}_{shape}"
    try:
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
        x_np = np.abs(np.random.randn(*shape).astype(np_dtype)) + 0.1
        
        try:
            with tf.device('/device:MUSA:0'):
                x_test = tf.cast(tf.constant(x_np), dtype)
                _ = tf.math.sqrt(x_test)
        except Exception as e:
            print(f"[SKIP] {test_name}: Forward 'Sqrt' failed, skipping grad test.")
            return

        with tf.device('/device:MUSA:0'):
            x = tf.cast(tf.constant(x_np), dtype)
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = tf.math.sqrt(x)
            grad_musa = tape.gradient(y, x)
            
        with tf.device('/CPU:0'):
            x_cpu = tf.cast(tf.constant(x_np), dtype)
            with tf.GradientTape() as tape:
                tape.watch(x_cpu)
                y_cpu = tf.math.sqrt(x_cpu)
            grad_cpu = tape.gradient(y_cpu, x_cpu)
            
        check_result(test_name, grad_musa, grad_cpu)

    except Exception as e:
        print(f"[FAIL] {test_name} Error: {e}")

def main():
    if not load_musa_plugin(): return
    
    shapes = [(10,), (5, 5)]
    dtypes = [tf.float32, tf.float16]
    try: dtypes.append(tf.bfloat16) 
    except: pass
    
    print("\n=== 1. Direct SqrtGrad Logic (Safe) ===")
    for s in shapes:
        for d in dtypes:
            test_sqrt_grad_logic(s, d)
            
    test_sqrt_grad_logic((10,), tf.float64)

    print("\n=== 2. Integration Test ===")
    for s in shapes:
        for d in dtypes:
            test_sqrt_integration(s, d)

if __name__ == "__main__":
    main()
