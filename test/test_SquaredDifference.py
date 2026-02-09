import tensorflow as tf
import numpy as np

# 确保加载了你的 musa 插件
try:
    tf.load_op_library('/workspace/tensorflow_musa/build/libmusa_plugin.so')
except:
    pass

def test_squared_difference():
    # 测试不同形状，包括广播情况
    shapes = [
        ((2, 3), (2, 3)),          # 同形状
        ((5,), (1,)),              # 标量广播
        ((1, 3, 224, 224), (1, 3, 1, 1))  # 典型 CNN 广播
    ]
    
    dtypes = [np.float32, np.float16]

    for shape_x, shape_y in shapes:
        for dtype in dtypes:
            # 准备数据
            x_np = np.random.randn(*shape_x).astype(dtype)
            y_np = np.random.randn(*shape_y).astype(dtype)

            with tf.device('/CPU:0'):
                x_cpu = tf.constant(x_np)
                y_cpu = tf.constant(y_np)
                expected = tf.math.squared_difference(x_cpu, y_cpu)

            with tf.device('/device:MUSA:0'):
                x_musa = tf.constant(x_np)
                y_musa = tf.constant(y_np)
                actual = tf.math.squared_difference(x_musa, y_musa)

            # 验证结果
            # float16 的 atol 设置稍大一点
            tol = 1e-3 if dtype == np.float16 else 1e-5
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=tol)
            print(f"✅ Pass: shape={shape_x}-{shape_y}, dtype={dtype}")

if __name__ == "__main__":
    test_squared_difference()
