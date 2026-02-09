import tensorflow as tf
import numpy as np
import os
import traceback

def load_musa_plugin():
    plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
    if os.path.exists(plugin_path):
        tf.load_library(plugin_path)
        print(">> MUSA Plugin Loaded")
    else:
        print(">> MUSA Plugin Not Found")

def assert_all_equal(v1, v2, msg=""):
    v1 = v1.numpy() if isinstance(v1, tf.Tensor) else v1
    v2 = v2.numpy() if isinstance(v2, tf.Tensor) else v2
    
    dtype_v1 = getattr(v1.dtype, 'name', str(v1.dtype))
    dtype_v2 = getattr(v2.dtype, 'name', str(v2.dtype))

    if dtype_v1 == 'bfloat16': 
        v1 = v1.astype(np.float32)
    if dtype_v2 == 'bfloat16': 
        v2 = v2.astype(np.float32)
    
    if np.array_equal(v1, v2):
        print(f"[PASS] {msg}")
    else:
        print(f"[FAIL] {msg}")
        print(f"  Expected shape: {v2.shape}, Got: {v1.shape}")

def assert_less(v, threshold, msg=""):
    if v < threshold:
        print(f"[PASS] {msg} (Error: {v:.6e} < {threshold})")
    else:
        print(f"[FAIL] {msg} (Error: {v:.6e} >= {threshold})")

def testBroadcastToBasic():
    print("\n--- testBroadcastToBasic ---")
    for dtype in [np.float32, np.int32, np.int64]:
        try:
            x = np.array([1, 2, 3], dtype=dtype)
            with tf.device('/device:MUSA:0'):
                x_tf = tf.constant(x)
                v_tf = tf.broadcast_to(x_tf, [3, 3])
                if dtype == np.float32: 
                    print(f"  > Device: {v_tf.device}")
            v_np = np.broadcast_to(x, [3, 3])
            assert_all_equal(v_tf, v_np, f"dtype={np.dtype(dtype).name}")
        except Exception as e:
            print(f"[FAIL] Basic dtype={np.dtype(dtype).name} Error: {e}")
            traceback.print_exc()

def testBroadcastToBool():
    print("\n--- testBroadcastToBool ---")
    try:
        x = np.array([True, False, True], dtype=bool)
        with tf.device('/device:MUSA:0'):
            v_tf = tf.broadcast_to(tf.constant(x), [3, 3])
        v_np = np.broadcast_to(x, [3, 3])
        assert_all_equal(v_tf, v_np, "Bool check")
    except Exception as e:
        print(f"[FAIL] Bool Error: {e}")
        traceback.print_exc()

def testBroadcastToShape():
    print("\n--- testBroadcastToShape ---")
    for input_dim in range(1, 6):
        for output_dim in range(input_dim, 6):
            try:
                input_shape = [2] * input_dim
                output_shape = [2] * output_dim
                x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
                with tf.device('/device:MUSA:0'):
                    v_tf = tf.broadcast_to(tf.constant(x), output_shape)
                v_np = np.broadcast_to(x, output_shape)
                assert_all_equal(v_tf, v_np, f"InDim={input_dim} OutDim={output_dim}")
            except Exception as e:
                 print(f"[FAIL] Shape In={input_dim} Out={output_dim} Error: {e}")
                 traceback.print_exc()

def testBroadcastToShapeInnerDim():
    print("\n--- testBroadcastToShapeInnerDim ---")
    try:
        input_shape = [2, 1, 3]
        output_shape = [2, 5, 3]
        x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
        with tf.device('/device:MUSA:0'):
            v_tf = tf.broadcast_to(tf.constant(x), output_shape)
        v_np = np.broadcast_to(x, output_shape)
        assert_all_equal(v_tf, v_np, "InnerDim")
    except Exception as e:
        print(f"[FAIL] InnerDim Error: {e}")
        traceback.print_exc()

def testBroadcastToShapeLargerDim():
    print("\n--- testBroadcastToShapeLargerDim ---")
    try:
        input_shape = [2, 1, 3, 2, 2, 2]
        output_shape = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 15, 3, 2, 2, 2]
        x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
        with tf.device('/device:MUSA:0'):
            v_tf = tf.broadcast_to(tf.constant(x), output_shape)
        v_np = np.broadcast_to(x, output_shape)
        assert_all_equal(v_tf, v_np, "LargerDim")
    except Exception as e:
        print(f"[FAIL] LargerDim Error: {e}")
        traceback.print_exc()

def testBroadcastToScalar():
    print("\n--- testBroadcastToScalar ---")
    try:
        x = np.array(1, dtype=np.int32)
        with tf.device('/device:MUSA:0'):
            v_tf = tf.broadcast_to(tf.constant(x), [3, 3])
        v_np = np.broadcast_to(x, [3, 3])
        assert_all_equal(v_tf, v_np, "Scalar")
    except Exception as e:
        print(f"[FAIL] Scalar Error: {e}")
        traceback.print_exc()

def testGradientForScalar():
    print("\n--- testGradientForScalar ---")
    x_val = np.array(1.0, dtype=np.float32)
    try:
        with tf.device('/device:MUSA:0'):
            x = tf.constant(x_val)
            with tf.GradientTape() as tape:
                tape.watch(x)
                v = tf.broadcast_to(x, [2, 4, 3])
                y = 2 * v
            grad = tape.gradient(y, x)
        expected_grad = 2.0 * (2 * 4 * 3)
        if grad is None:
             print("[FAIL] Gradient is None (Check Sum op registration)")
        else:
             err = np.abs(grad.numpy() - expected_grad)
             assert_less(err, 1e-4, f"Gradient check (Expected {expected_grad}, Got {grad.numpy()})")
    except Exception as e:
        print(f"[FAIL] Gradient Error: {e}")
        traceback.print_exc()

def testGradientWithIncreasingRank():
    print("\n--- testGradientWithIncreasingRank ---")
    x_val = np.array([[1.0], [2.0]], dtype=np.float32)
    try:
        with tf.device('/device:MUSA:0'):
            x = tf.constant(x_val)
            with tf.GradientTape() as tape:
                tape.watch(x)
                v = tf.broadcast_to(x, [5, 2, 3])
                y = 2 * v
            grad = tape.gradient(y, x)
        expected_grad = np.ones_like(x_val) * 30.0
        if grad is None:
             print("[FAIL] Gradient is None")
        else:
             err = np.max(np.abs(grad.numpy() - expected_grad))
             assert_less(err, 1e-4, "Gradient check increasing rank")
    except Exception as e:
         print(f"[FAIL] Gradient Rank Error: {e}")
         traceback.print_exc()

if __name__ == "__main__":
    load_musa_plugin()
    print("MUSA Devices:", tf.config.list_physical_devices('MUSA'))
    if tf.config.list_physical_devices('MUSA'):
        testBroadcastToBasic()
        testBroadcastToBool()
        testBroadcastToShape()
        testBroadcastToShapeInnerDim()
        testBroadcastToShapeLargerDim()
        testBroadcastToScalar()
        testGradientForScalar()
        testGradientWithIncreasingRank()
