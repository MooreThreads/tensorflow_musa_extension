import tensorflow as tf
import ctypes
import os

plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
ctypes.CDLL(plugin_path)

print("Quick test of Shape operator on MUSA")
print("=" * 40)

os.environ['LD_LIBRARY_PATH'] = '/usr/local/musa/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

with tf.device('/device:MUSA:0'):
    scalar = tf.constant(42.0)
    scalar_shape = tf.shape(scalar)
    print(f"Scalar shape: {scalar_shape.numpy()}")
    
    vector = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    vector_shape = tf.shape(vector)
    print(f"Vector shape: {vector_shape.numpy()}")
    
    matrix = tf.constant([[1, 2], [3, 4]])
    matrix_shape = tf.shape(matrix)
    print(f"Matrix shape: {matrix_shape.numpy()}")
    
    matrix_shape_int64 = tf.shape(matrix, out_type=tf.int64)
    print(f"Matrix shape (int64): {matrix_shape_int64.numpy()}")
    
    if (scalar_shape.numpy() == []).all() and \
       (vector_shape.numpy() == [5]).all() and \
       (matrix_shape.numpy() == [2, 2]).all():
        print("\n All tests passed!")
    else:
        print("\n Tests failed")
