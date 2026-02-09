import tensorflow as tf
import numpy as np
import os
import traceback

PLUGIN_PATH = "/workspace/tensorflow_musa/build/libmusa_plugin.so"

def load_musa_plugin():
    if os.path.exists(PLUGIN_PATH):
        try:
            tf.load_library(PLUGIN_PATH)
            print(">> [INFO] MUSA Plugin Loaded Successfully")
            return True
        except Exception as e:
            print(f">> [ERROR] Failed to load plugin: {e}")
            return False
    else:
        print(f">> [ERROR] Plugin file not found at {PLUGIN_PATH}")
        return False

def check_result(test_name, musa_result, expected_val, rtol=1e-4, atol=1e-4):

    musa_val = musa_result.numpy()
    
    if musa_val.shape != expected_val.shape:
        print(f"[FAIL] {test_name}: Shape mismatch")
        print(f"  MUSA: {musa_val.shape}")
        print(f"  Exp : {expected_val.shape}")
        return False
        
    if np.allclose(musa_val, expected_val, rtol=rtol, atol=atol):
        print(f"[PASS] {test_name}")
        return True
    else:
        print(f"[FAIL] {test_name}: Value mismatch")
        diff = np.abs(musa_val - expected_val)
        print(f"  Max Diff: {np.max(diff)}")
        if musa_val.size < 20:
            print(f"  MUSA:\n{musa_val}")
            print(f"  Exp :\n{expected_val}")
        return False

def run_forward_test(x_shape, y_shape, dtype=tf.float32, test_name=""):
    print(f"\n--- Test: {test_name} [Forward] {dtype.name} {x_shape} / {y_shape} ---")
    try:
        np_dtype = dtype.as_numpy_dtype
        x_np = np.random.randn(*x_shape).astype(np_dtype) * 10
        y_np = np.random.randn(*y_shape).astype(np_dtype) * 10
        
        if y_np.size > 0:
            indices = np.random.choice(y_np.size, size=min(3, y_np.size), replace=False)
            y_np.ravel()[indices] = 0.0

        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x_np, dtype=dtype)
            y_musa = tf.constant(y_np, dtype=dtype)
            z_musa = tf.raw_ops.DivNoNan(x=x_musa, y=y_musa)
            
        z_expected = np.divide(x_np, y_np, out=np.zeros_like(x_np, shape=z_musa.shape), where=(y_np!=0))
            
        check_result(test_name, z_musa, z_expected)
        
    except Exception as e:
        print(f"[FAIL] {test_name} Exception: {e}")
        traceback.print_exc()

def run_backward_test(x_shape, y_shape, dtype=tf.float32, test_name=""):
    print(f"\n--- Test: {test_name} [Backward] {dtype.name} {x_shape} / {y_shape} ---")
    try:
        np_dtype = dtype.as_numpy_dtype
        x_np = np.random.randn(*x_shape).astype(np_dtype)
        y_np = np.random.randn(*y_shape).astype(np_dtype)
        
        if y_np.size > 0:
            y_np.ravel()[0] = 0.0
        
        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x_np, dtype=dtype)
            y_musa = tf.constant(y_np, dtype=dtype)
            with tf.GradientTape() as tape:
                tape.watch([x_musa, y_musa])
                z_musa = tf.raw_ops.DivNoNan(x=x_musa, y=y_musa)
                loss_musa = tf.reduce_sum(z_musa)
            grad_musa = tape.gradient(loss_musa, [x_musa, y_musa])

        broadcast_shape = z_musa.shape
        
        # 将 x, y 广播到输出形状以便计算
        x_broad = np.broadcast_to(x_np, broadcast_shape)
        y_broad = np.broadcast_to(y_np, broadcast_shape)
        
        # 计算 dL/dz (全是 1，因为 loss = sum(z))
        dl_dz = np.ones(broadcast_shape, dtype=np_dtype)
        
        # 计算 dz/dx 和 dz/dy
        dz_dx = np.divide(1.0, y_broad, out=np.zeros_like(y_broad), where=(y_broad!=0))
        dz_dy = np.divide(-x_broad, np.square(y_broad), out=np.zeros_like(y_broad), where=(y_broad!=0))
        
        # 链式法则: dL/dx_broad = dL/dz * dz/dx
        grad_x_broad = dl_dz * dz_dx
        grad_y_broad = dl_dz * dz_dy
        
        def reduce_grad(grad_broad, input_shape):
            # 处理维度数量差异
            extra_dims = grad_broad.ndim - len(input_shape)
            if extra_dims > 0:
                # 对前几维求和
                grad_broad = np.sum(grad_broad, axis=tuple(range(extra_dims)))
            
            # 处理维度大小差异
            keep_dims = []
            for i, dim in enumerate(input_shape):
                if dim == 1 and grad_broad.shape[i] > 1:
                    keep_dims.append(i)
            
            if keep_dims:
                grad_broad = np.sum(grad_broad, axis=tuple(keep_dims), keepdims=True)
                
            return grad_broad.reshape(input_shape).astype(np_dtype)

        dx_expected = reduce_grad(grad_x_broad, x_shape)
        dy_expected = reduce_grad(grad_y_broad, y_shape)
            
        check_result(f"{test_name}_dx", grad_musa[0], dx_expected)
        check_result(f"{test_name}_dy", grad_musa[1], dy_expected)

    except Exception as e:
        print(f"[FAIL] {test_name} Exception: {e}")

def main():
    if not load_musa_plugin():
        return
    
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))

    print("\n=== 1. 基础前向传播测试 ===")
    run_forward_test((100,), (100,), tf.float32, "Basic_1D")
    run_forward_test((10, 10), (10, 10), tf.float32, "Basic_2D")
    
    print("\n=== 2. 广播机制测试 ===")
    run_forward_test((10, 10), (1,), tf.float32, "Broadcast_Scalar")
    run_forward_test((10, 10), (10, 1), tf.float32, "Broadcast_Col_Vec")
    run_forward_test((1, 4, 8, 8), (1, 4, 1, 1), tf.float32, "Broadcast_NCHW")
    
    print("\n=== 3. 双精度测试 ===")
    run_forward_test((10,), (10,), tf.float64, "Float64_1D")

    print("\n=== 4. 梯度测试 ===")
    run_backward_test((10,), (10,), tf.float32, "Grad_Basic")
    run_backward_test((5, 5), (5, 1), tf.float32, "Grad_Broadcast")

if __name__ == "__main__":
    main()