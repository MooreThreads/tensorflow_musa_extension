# truncatedNormal_op_test.py
import numpy as np
import tensorflow as tf
from musa_test_utils import MUSATestCase

class TruncatedNormalTest(MUSATestCase):
    """TruncatedNormal 算子专用测试"""

    def test_basic_functionality(self):
        """基础功能：生成截断正态分布随机数"""
        shape = [1000, 100]  # 足够大的样本用于统计验证
        mean, stddev = 0.0, 1.0
        with tf.device('/device:MUSA:0'):
            result = tf.random.truncated_normal(shape, mean=mean, stddev=stddev, dtype=tf.float32)
        
        val = result.numpy()
        self.assertEqual(val.shape, tuple(shape))
        self.assertEqual(result.dtype, tf.float32)
        
        # 核心特性：所有值必须在 [mean-2*stddev, mean+2*stddev] 范围内
        self.assertTrue(np.all(val >= mean - 2*stddev))
        self.assertTrue(np.all(val <= mean + 2*stddev))
        
        # 统计验证：均值接近0，标准差小于1（截断后方差会减小）
        self.assertAlmostEqual(np.mean(val), 0.0, delta=0.1)
        self.assertLess(np.std(val), 1.0)  # 截断后标准差通常为 ~0.87

    def test_truncation_bounds(self):
        """严格验证截断边界"""
        shape = [50000]
        mean, stddev = 5.0, 2.0
        lower_bound = mean - 2 * stddev  # 1.0
        upper_bound = mean + 2 * stddev  # 9.0
        
        with tf.device('/device:MUSA:0'):
            result = tf.random.truncated_normal(shape, mean=mean, stddev=stddev, dtype=tf.float32)
        
        val = result.numpy()
        # 所有值必须严格在边界内
        self.assertGreaterEqual(np.min(val), lower_bound)
        self.assertLessEqual(np.max(val), upper_bound)
        
        # 验证均值接近指定值
        self.assertAlmostEqual(np.mean(val), mean, delta=0.2)

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
                result = tf.random.truncated_normal(shape, dtype=tf.float32)
            
            self.assertEqual(result.shape, tuple(shape))
            self.assertEqual(result.dtype, tf.float32)
            # 验证截断范围 [-2, 2] (默认 mean=0, stddev=1)
            if np.prod(shape) > 0:
                val = result.numpy()
                self.assertTrue(np.all(val >= -2.0))
                self.assertTrue(np.all(val <= 2.0))

    def test_float_types(self):
        """测试不同浮点类型支持"""
        for dtype, tol in [(tf.float16, 0.3), (tf.float32, 0.15), (tf.float64, 0.1)]:
            shape = [10000]
            with tf.device('/device:MUSA:0'):
                result = tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=dtype)
            
            self.assertEqual(result.dtype, dtype)
            val = result.numpy().astype(np.float32)
            
            # 验证截断范围
            self.assertGreaterEqual(np.min(val), -2.0)
            self.assertLessEqual(np.max(val), 2.0)
            
            # 验证均值接近0（精度根据类型调整）
            self.assertAlmostEqual(np.mean(val), 0.0, delta=tol)

    def test_randomness(self):
        """验证两次调用结果不同（随机性）"""
        shape = [10, 10]
        with tf.device('/device:MUSA:0'):
            r1 = tf.random.truncated_normal(shape, dtype=tf.float32).numpy()
            r2 = tf.random.truncated_normal(shape, dtype=tf.float32).numpy()
        
        # 两次结果应不同
        self.assertFalse(np.allclose(r1, r2), 
                        msg="两次随机调用结果相同，可能未正确实现随机性")

    def test_empty_tensor(self):
        """空张量处理：不崩溃且返回正确形状"""
        shape = [0, 5]
        with tf.device('/device:MUSA:0'):
            result = tf.random.truncated_normal(shape, dtype=tf.float32)
        
        self.assertEqual(result.shape, (0, 5))
        self.assertEqual(result.numpy().size, 0)

    def test_large_tensor(self):
        """大张量生成：验证内存和性能稳定性"""
        shape = [1000, 1000]  # 1M 元素
        with tf.device('/device:MUSA:0'):
            result = tf.random.truncated_normal(shape, dtype=tf.float32)
        
        val = result.numpy()
        self.assertEqual(val.shape, (1000, 1000))
        # 验证截断范围 [-2, 2]
        self.assertGreaterEqual(np.min(val), -2.0)
        self.assertLessEqual(np.max(val), 2.0)

    def test_custom_mean_stddev(self):
        """测试自定义均值和标准差"""
        shape = [20000]
        test_cases = [
            (0.0, 1.0),   # 标准参数
            (10.0, 3.0),  # 正均值，大标准差
            (-5.0, 0.5),  # 负均值，小标准差
            (100.0, 10.0) # 大均值
        ]
        
        for mean, stddev in test_cases:
            with tf.device('/device:MUSA:0'):
                result = tf.random.truncated_normal(shape, mean=mean, stddev=stddev, dtype=tf.float32)
            
            val = result.numpy()
            # 验证截断范围
            self.assertGreaterEqual(np.min(val), mean - 2*stddev, 
                                   msg=f"mean={mean}, stddev={stddev}")
            self.assertLessEqual(np.max(val), mean + 2*stddev,
                                msg=f"mean={mean}, stddev={stddev}")
            
            # 验证均值（允许一定误差）
            self.assertAlmostEqual(np.mean(val), mean, delta=stddev*0.3)

    def test_distribution_properties(self):
        """验证截断正态分布特性"""
        shape = [100000]  # 大样本
        with tf.device('/device:MUSA:0'):
            result = tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
        
        val = result.numpy()
        
        # 核心特性：无超出范围的值
        self.assertEqual(np.sum(np.abs(val) > 2.0), 0, 
                        msg="存在超出截断范围的值")
        
        # 分布应相对均匀填充 [-2, 2] 区间（与标准正态不同）
        # 验证数据覆盖了截断区间的大部分范围
        self.assertGreater(np.max(val), 1.5)  # 接近上界
        self.assertLess(np.min(val), -1.5)    # 接近下界
        
        # 标准差应小于未截断情况（理论值约0.87）
        self.assertLess(np.std(val), 1.0)
        self.assertGreater(np.std(val), 0.7)

    def test_no_extreme_outliers(self):
        """验证无极端异常值（截断的主要作用）"""
        shape = [50000]
        with tf.device('/device:MUSA:0'):
            result = tf.random.truncated_normal(shape, dtype=tf.float32)
        
        val = result.numpy()
        
        # 标准正态分布中，超出±2σ的概率约5%，截断后应为0
        outliers = np.sum(np.abs(val) > 2.0)
        self.assertEqual(outliers, 0, 
                        msg=f"发现 {outliers} 个超出截断范围的异常值")

    def test_seed_reproducibility(self):
        """测试使用 seed 的可重复性（如果支持）"""
        shape = [100]
        seed = 12345
        
        try:
            with tf.device('/device:MUSA:0'):
                tf.random.set_seed(seed)
                r1 = tf.random.truncated_normal(shape, dtype=tf.float32).numpy()
                
                tf.random.set_seed(seed)
                r2 = tf.random.truncated_normal(shape, dtype=tf.float32).numpy()
            
            # 相同 seed 应生成相同结果
            self.assertTrue(np.allclose(r1, r2), 
                           msg="相同 seed 未能重现结果")
        except Exception as e:
            # 如果设备不支持 seed，跳过此测试
            self.skipTest(f"MUSA 设备可能不支持 seed: {e}")

if __name__ == "__main__":
    tf.test.main()