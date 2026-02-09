import os
import sys
import time
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_musa_plugin():
    base_dir = "/workspace/tensorflow_musa/build"
    candidates = ["libmusa_plugin.so", "libmusa_ops.so"]
    found = False
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            tf.load_library(path)
            found = True
            break
    if not found:
        sys.exit(1)

def show_pack_visual_logic():
    print("=" * 20)
    print("PACK OPERATOR VISUAL DEMO")
    print("=" * 20)
    t1 = tf.constant([[1, 1], [1, 1]], dtype=tf.int32)
    t2 = tf.constant([[2, 2], [2, 2]], dtype=tf.int32)
    t3 = tf.constant([[3, 3], [3, 3]], dtype=tf.int32)
    
    print("Input T1:\n", t1.numpy())
    print("Input T2:\n", t2.numpy())
    print("Input T3:\n", t3.numpy())
    
    packed = tf.stack([t1, t2, t3], axis=0)
    print("\nPacked Result (tf.stack on axis=0):")
    print(packed.numpy())
    print("New Shape:", packed.shape)
    print("=" * 50)

def is_int_dtype(dt):
    return dt in (tf.int32, tf.int64)

def gen_inputs(shape, num_inputs, dt):
    np_dt = dt.as_numpy_dtype
    xs_np = []
    for i in range(num_inputs):
        if is_int_dtype(dt):
            x = np.random.randint(i*10, (i+1)*10, size=shape).astype(np_dt)
        else:
            x = np.random.uniform(-1, 1, size=shape).astype(np_dt)
        xs_np.append(x)
    return xs_np

def verify_pack_performance(shape, axis, num_inputs, dt, warmup=10, iters=100):
    xs_np = gen_inputs(shape, num_inputs, dt)
    
    with tf.device("/CPU:0"):
        xs_cpu = [tf.constant(x, dtype=dt) for x in xs_np]
        _ = tf.stack(xs_cpu, axis=axis)
        t0 = time.perf_counter()
        for _ in range(10):
            y_cpu = tf.stack(xs_cpu, axis=axis)
        cpu_ms = (time.perf_counter() - t0) / 10 * 1000.0
        y_cpu_np = y_cpu.numpy()

    try:
        with tf.device("/device:MUSA:0"):
            xs_musa = [tf.identity(tf.constant(x, dtype=dt)) for x in xs_np]
            for _ in range(warmup):
                _ = tf.stack(xs_musa, axis=axis)
            
            t0 = time.perf_counter()
            for _ in range(iters):
                y_musa = tf.stack(xs_musa, axis=axis)
            _ = y_musa.numpy() 
            gpu_ms = (time.perf_counter() - t0) / iters * 1000.0
            y_musa_np = y_musa.numpy()
            dev = y_musa.device

        if is_int_dtype(dt):
            mismatches = np.sum(y_cpu_np != y_musa_np)
            ok = (mismatches == 0)
            metric_str = f"Mismatches: {mismatches}"
        else:
            mae = np.mean(np.abs(y_cpu_np.astype(np.float32) - y_musa_np.astype(np.float32)))
            ok = mae < (1e-4 if dt != tf.float64 else 1e-9)
            metric_str = f"MAE: {mae:.6e}"

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0.0

        print(f"[Test Case] Shape: {shape}, Axis: {axis}, NumInputs: {num_inputs}, Dtype: {dt.name}")
        print(f"Device    : {dev}")
        print(f"{metric_str}")
        print(f"CPU Time  : {cpu_ms:.3f} ms")
        print(f"MUSA Time : {gpu_ms:.3f} ms")
        print(f"Speedup   : {speedup:.2f}x")
        print(f"Result    : {'PASS' if ok else 'FAIL'}")
        print("-" * 50)

    except Exception as e:
        print(f"\n===== Pack Case Failed =====")
        print(f"Config: Shape={shape}, Axis={axis}, Dtype={dt.name}")
        print(f"CRASH: {e}")
        print("-" * 50)

def run_pack_suite():
    dtypes = [tf.float32, tf.float16, tf.int32, tf.int64]
    cases = [
        ((1024, 512), 0, 2),
        ((256, 4096), 1, 4),
        ((1024, 1024), -1, 8),
        ((64, 128, 256), 2, 2),
    ]

    print("\n" + "="*20 + " MUSA Pack Op Performance Test " + "="*20)
    for shape, axis, num_inputs in cases:
        for dt in dtypes:
            verify_pack_performance(shape, axis, num_inputs, dt)

if __name__ == "__main__":
    load_musa_plugin()
    
    musa_devices = tf.config.list_physical_devices('MUSA')
    if not musa_devices:
        sys.exit(1)
    
    show_pack_visual_logic()
    run_pack_suite()
