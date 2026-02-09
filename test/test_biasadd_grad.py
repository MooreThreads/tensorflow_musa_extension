import tensorflow as tf
import numpy as np
import os

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)

def verify_biasadd_grad(input_shape, data_format, dtype):
    print(f"\n--- Testing BiasAddGrad [{dtype.name}] format={data_format} ---")
    
    channel_axis = 1 if data_format == 'NCHW' else -1
    feature_dim = input_shape[channel_axis]
    
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    
    grad_np = np.random.randn(*input_shape).astype(np_dtype)

    with tf.device('/CPU:0'):
        g_cpu = tf.constant(grad_np, dtype=dtype)
        res_cpu = tf.raw_ops.BiasAddGrad(out_backprop=g_cpu, data_format=data_format)

    try:
        with tf.device('/device:MUSA:0'):
            g_musa = tf.constant(grad_np, dtype=dtype)
            res_musa = tf.raw_ops.BiasAddGrad(out_backprop=g_musa, data_format=data_format)
            
            print(f"  > Output Device: {res_musa.device}")

        val_cpu = tf.cast(res_cpu, tf.float32).numpy()
        val_musa = tf.cast(res_musa, tf.float32).numpy()
        
        if val_musa.shape != (feature_dim,):
            print(f"Result:  FAIL (Shape mismatch. Expected {(feature_dim,)}, got {val_musa.shape})")
            return

        mae = np.mean(np.abs(val_cpu - val_musa))
        
        if dtype == tf.bfloat16:
            atol = 5.0  
            rtol = 1e-1 
        elif dtype == tf.float16:
            atol = 1.0
            rtol = 5e-2
        else:
            # FP32
            atol = 1e-4
            rtol = 1e-4

        is_close = np.allclose(val_cpu, val_musa, rtol=rtol, atol=atol)
        
        if is_close:
            print(f"Result:  PASS (MAE: {mae:.6e})")
        else:
            print(f"Result:  FAIL (MAE: {mae:.6e})")
            print(f"  CPU (first 5): {val_cpu[:5]}")
            print(f"  MUSA(first 5): {val_musa[:5]}")
            diff = np.abs(val_cpu - val_musa)
            max_diff_idx = np.argmax(diff)
            print(f"  Max Diff: {diff[max_diff_idx]:.4f} at index {max_diff_idx} (CPU={val_cpu[max_diff_idx]:.4f}, MUSA={val_musa[max_diff_idx]:.4f})")
            
    except Exception as e:
        print(f"Result:  CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))
    
    if tf.config.list_physical_devices('MUSA'):
        shapes_nhwc = [4, 32, 32, 64]
        shapes_nchw = [4, 64, 32, 32]
        
        for dt in [tf.float32, tf.float16, tf.bfloat16]:
            verify_biasadd_grad(shapes_nhwc, 'NHWC', dt)
            verify_biasadd_grad(shapes_nchw, 'NCHW', dt)

