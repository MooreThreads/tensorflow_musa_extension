# random_standard_normal_test.py
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class RandomStandardNormalTest(MUSATestCase):
    """RandomStandardNormal 算子专用测试"""

    def test_basic_functionality(self):
        """基础功能：生成标准正态分布随机数"""
        shape = [1000, 100]  # 足够大的样本用于统计验证
        with tf.device('/device:MUSA:0'):
            result = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
        
        val = result.numpy()
        self.assertEqual(val.shape, tuple(shape))
        self.assertEqual(result.dtype, tf.float32)
        
        # 统计验证：均值接近0，标准差接近1（允许小误差）
        self.assertAlmostEqual(np.mean(val), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(val), 1.0, delta=0.1)

    def test_different_shapes(self):
        """测试不同形状的张量"""
        shapes = [
            [],           # 标量
            [1],          # 向量
            [5, 5],       # 矩阵
            [2, 3, 4],    # 3D张量
            [2, 3, 4, 5]  # 4D张量
        ]
        for shape in shapes:
            with tf.device('/device:MUSA:0'):
                result = tf.random.normal(shape, dtype=tf.float32)
            
            self.assertEqual(result.shape, tuple(shape))
            self.assertEqual(result.dtype, tf.float32)
            # 验证非空张量有合理分布
            if np.prod(shape) > 0:
                val = result.numpy()
                self.assertGreater(np.abs(np.mean(val)), 0)  # 非全零

    def test_float_types(self):
        """测试不同浮点类型支持"""
        for dtype, tol in [(tf.float16, 0.3), (tf.float32, 0.1), (tf.float64, 0.1)]:
            shape = [10000]
            with tf.device('/device:MUSA:0'):
                result = tf.random.normal(shape, dtype=dtype)
            
            self.assertEqual(result.dtype, dtype)
            val = result.numpy().astype(np.float32)  # 转为float32便于统计
            
            # 验证分布特性（float16精度较低，放宽容差）
            self.assertAlmostEqual(np.mean(val), 0.0, delta=tol)
            self.assertAlmostEqual(np.std(val), 1.0, delta=tol)

    def test_randomness(self):
        """验证两次调用结果不同（随机性）"""
        shape = [10, 10]
        with tf.device('/device:MUSA:0'):
            r1 = tf.random.normal(shape, dtype=tf.float32).numpy()
            r2 = tf.random.normal(shape, dtype=tf.float32).numpy()
        
        # 两次结果应不同（极小概率相等，但100元素同时相等概率≈1e-300）
        self.assertFalse(np.allclose(r1, r2), 
                        msg="两次随机调用结果相同，可能未正确实现随机性")

    def test_empty_tensor(self):
        """空张量处理：不崩溃且返回正确形状"""
        shape = [0, 5]
        with tf.device('/device:MUSA:0'):
            result = tf.random.normal(shape, dtype=tf.float32)
        
        self.assertEqual(result.shape, (0, 5))
        self.assertEqual(result.numpy().size, 0)

    def test_large_tensor(self):
        """大张量生成：验证内存和性能稳定性"""
        shape = [1000, 1000]  # 1M 元素
        with tf.device('/device:MUSA:0'):
            result = tf.random.normal(shape, dtype=tf.float32)
        
        val = result.numpy()
        self.assertEqual(val.shape, (1000, 1000))
        # 基础统计验证
        self.assertGreater(np.min(val), -10.0)  # 3σ原则：99.7%数据在[-3,3]内
        self.assertLess(np.max(val), 10.0)

    def test_distribution_properties(self):
        """深入验证正态分布特性（68-95-99.7法则）"""
        shape = [100000]  # 大样本
        with tf.device('/device:MUSA:0'):
            result = tf.random.normal(shape, dtype=tf.float32)
        
        val = result.numpy()
        # 68% 数据应在 [-1, 1] 内
        within_1sigma = np.mean(np.abs(val) <= 1.0)
        self.assertAlmostEqual(within_1sigma, 0.68, delta=0.03)
        
        # 95% 数据应在 [-2, 2] 内
        within_2sigma = np.mean(np.abs(val) <= 2.0)
        self.assertAlmostEqual(within_2sigma, 0.95, delta=0.03)

if __name__ == "__main__":
    tf.test.main()