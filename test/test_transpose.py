import os
import numpy as np
import tensorflow as tf

PLUGIN_PATH = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
if os.path.exists(PLUGIN_PATH):
    print(f"--- Loading MUSA Plugin from {PLUGIN_PATH} ---")
    tf.load_op_library(PLUGIN_PATH)
else:
    print(f"ERROR: Plugin not found at {PLUGIN_PATH}")

def test_transpose_nhwc_to_nchw():
    shape = (1, 4, 4, 3)
    x_np = np.random.rand(*shape).astype(np.float32)

    perm = [0, 3, 1, 2]

    print("\n--- Running Transpose Test ---")

    try:
        with tf.device('/device:MUSA:0'):
            x_musa = tf.constant(x_np)
            y_musa = tf.transpose(x_musa, perm=perm)

            y_musa_np = y_musa.numpy()

        print("MUSA Transpose executed successfully!")
    except Exception as e:
        print(f"MUSA Execution Failed: {e}")
        return

    with tf.device('/cpu:0'):
        y_cpu_np = tf.transpose(tf.constant(x_np), perm=perm).numpy()

    np.testing.assert_allclose(y_musa_np, y_cpu_np, rtol=1e-5, atol=1e-5)

    print("--- Accuracy Check Passed! ---")
    print(f"Input Shape:  {x_np.shape} (NHWC)")
    print(f"Output Shape: {y_musa_np.shape} (NCHW)")

    print("\nSample Data (Channel 0):")
    print("Original (NHWC, slice channel 0):\n", x_np[0, :, :, 0])
    print("Transposed (NCHW, slice channel 0):\n", y_musa_np[0, 0, :, :])

if __name__ == "__main__":
    devices = tf.config.list_physical_devices('MUSA')
    print(f"Available MUSA devices: {devices}")

    if devices:
        test_transpose_nhwc_to_nchw()
    else:
        print("No MUSA device detected. Check your device_register.cc logic.")

