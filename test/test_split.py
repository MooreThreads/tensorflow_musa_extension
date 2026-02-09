import tensorflow as tf
import numpy as np
import os

# 加载 MUSA 插件
so_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
if os.path.exists(so_path):
    tf.load_op_library(so_path)
    print("MUSA Plugin loaded.")

class MusaSplitTest(tf.test.TestCase):

    def testSplitBasic(self):
        """测试基础的 float32 切片"""
        with self.session() as sess:
            with tf.device('/device:MUSA:0'):
                x_np = np.random.rand(4, 6).astype(np.float32)
                x = tf.constant(x_np)
                
                # 修正参数名为 num_or_size_splits
                num_parts = 3
                y = tf.split(x, num_or_size_splits=num_parts, axis=1)
                
                y_out = sess.run(y)
                
                for i in range(num_parts):
                    expected = x_np[:, i*2:(i+1)*2]
                    self.assertAllClose(y_out[i], expected)
                    print(f"Split float32 index {i} shape: {y_out[i].shape} - OK")

    def testSplitFP16(self):
        """测试 AMP 关键的 float16 支持"""
        with self.session() as sess:
            with tf.device('/device:MUSA:0'):
                x_np = np.random.rand(2, 8).astype(np.float16)
                x = tf.constant(x_np)
                # 修正参数名
                y = tf.split(x, num_or_size_splits=2, axis=1)
                y_out = sess.run(y)
                
                self.assertAllClose(y_out[0], x_np[:, 0:4])
                self.assertAllClose(y_out[1], x_np[:, 4:8])
                print(f"Split float16 (AMP compatibility) - OK")

    def testSplitNegativeAxis(self):
        """测试负数轴索引 (split_dim < 0)"""
        with self.session() as sess:
            with tf.device('/device:MUSA:0'):
                x_np = np.random.rand(10, 4).astype(np.float32)
                x = tf.constant(x_np)
                # 修正参数名
                y = tf.split(x, num_or_size_splits=2, axis=-1)
                y_out = sess.run(y)
                
                self.assertEqual(y_out[0].shape, (10, 2))
                self.assertAllClose(y_out[0], x_np[:, 0:2])
                print(f"Split negative axis (-1) - OK")

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
