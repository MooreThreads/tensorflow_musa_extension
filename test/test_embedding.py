import tensorflow as tf
import numpy as np
import os
import time
import sys

# ==========================================
# 0. 自动加载动态库
# ==========================================
def load_plugin():
    base_dir = "/workspace/tensorflow_musa/build"
    candidates = [
        "libmusa_embedding_final.so", 
        "libmusa_embedding_v888.so", 
        "libmusa_plugin1.so",
        "libmusa_plugin.so"
    ]
    
    for name in candidates:
        if os.path.exists(name):
            return tf.load_op_library(f"./{name}")
            
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return tf.load_op_library(path)
            
    print("❌ 错误：找不到任何 .so 库文件！")
    sys.exit(1)

try:
    musa_ops = load_plugin()
    print("📚 算子库加载成功！开始全类型性能测试...\n")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    sys.exit(1)

# ==========================================
# 1. 通用基准测试函数
# ==========================================
def run_benchmark_case(dtype, idx_dtype, vocab_size=100000, emb_dim=1024, num_lookups=200000):
    # 打印表头
    print(f"{'-'*70}")
    print(f"🧪 测试组合: Data=[{dtype.name}]  Index=[{idx_dtype.name}]")
    
    # --- 1. 数据生成 (High Load) ---
    # 确保数据量足够大，足以让 CPU L3 缓存爆掉，体现 GPU 显存带宽优势
    if dtype in [tf.float32, tf.float64]:
        params_np = np.random.randn(vocab_size, emb_dim).astype(dtype.as_numpy_dtype)
    else:
        params_np = np.random.randint(0, 100, (vocab_size, emb_dim)).astype(dtype.as_numpy_dtype)
    
    ids_np = np.random.randint(0, vocab_size, num_lookups).astype(idx_dtype.as_numpy_dtype)
    
    t_params = tf.constant(params_np, dtype=dtype)
    t_ids = tf.constant(ids_np, dtype=idx_dtype)

    # --- 2. CPU 基准 (TensorFlow Native) ---
    cpu_avg_ms = 0.0
    try:
        with tf.device("/CPU:0"):
            # 预热
            for _ in range(2): _ = tf.nn.embedding_lookup(t_params, t_ids)
            
            # 测速: CPU 是同步的，直接计时
            t0 = time.time()
            loops = 5
            for _ in range(loops):
                _ = tf.nn.embedding_lookup(t_params, t_ids)
            cpu_avg_ms = (time.time() - t0) / loops * 1000
    except Exception as e:
        print(f"  ❌ CPU Error: {e}")
        return

    # --- 3. MUSA GPU 极限测速 (Your Op) ---
    musa_avg_ms = 0.0
    device_name = "Unknown"
    try:
        with tf.device("/device:MUSA:0"):
            # 关键步骤：先把数据搬到 GPU，模拟真实训练场景
            # 避免把 PCIe 传输时间算进算子计算时间里
            t_params_gpu = tf.identity(t_params)
            t_ids_gpu = tf.identity(t_ids)
            
            # 预热
            for _ in range(5):
                _ = musa_ops.musa_embedding(t=t_params_gpu, tindices=t_ids_gpu)
            
            # 测速: 采用 "Throughput Mode" (高吞吐模式)
            # 连续提交 20 个任务，让 GPU 跑满，最后同步一次
            loops = 20
            t0 = time.time()
            
            for _ in range(loops):
                res_musa = musa_ops.musa_embedding(t=t_params_gpu, tindices=t_ids_gpu)
            
            # 强制同步：等待所有 GPU 任务完成
            _ = res_musa.numpy()
            
            musa_avg_ms = (time.time() - t0) / loops * 1000
            device_name = res_musa.device

    except Exception as e:
        print(f"  ❌ MUSA Error: {e}")
        return

    # --- 4. 结果输出与对比 ---
    # 验证是否在 MUSA 上
    is_on_gpu = "MUSA" in device_name
    gpu_status = "✅ YES" if is_on_gpu else "❌ NO"
    
    # 计算加速比
    speedup = cpu_avg_ms / musa_avg_ms if musa_avg_ms > 0 else 0
    
    print(f"  ► 运行设备: {device_name} (GPU确认: {gpu_status})")
    print(f"  ► CPU耗时 : {cpu_avg_ms:.2f} ms")
    print(f"  ► GPU耗时 : {musa_avg_ms:.2f} ms")
    
    if speedup > 1.0:
        print(f"  🚀 加速比 : {speedup:.2f} x (WIN!)")
    else:
        print(f"  🐢 加速比 : {speedup:.2f} x (LOSE)")

# ==========================================
# 主程序：8 种组合全覆盖
# ==========================================
if __name__ == "__main__":
    if not tf.config.list_physical_devices('MUSA'):
        print("❌ 致命错误: 未检测到 MUSA 设备！无法进行 GPU 测试。")
        sys.exit(1)

    # 定义全类型组合
    data_types = [tf.float32, tf.float64, tf.int32, tf.int64]
    index_types = [tf.int32, tf.int64]

    print(f"🚀 启动全覆盖性能轰炸 (Total {len(data_types) * len(index_types)} cases)...")
    print("   注意：为了体现 GPU 优势，使用了大显存负载 (1024 dim)。\n")

    for dt in data_types:
        for idx_dt in index_types:
            # 针对 64 位数据，适当减小一点查表量，防止显存小的卡 OOM
            # 32位: 20万 lookups; 64位: 15万 lookups
            current_lookups = 200000 if dt in [tf.float32, tf.int32] else 150000
            
            run_benchmark_case(
                dtype=dt, 
                idx_dtype=idx_dt,
                vocab_size=100000,   # 10万词表
                emb_dim=1024,        # 1024 维度 (高带宽需求)
                num_lookups=current_lookups
            )
            
    print(f"\n{'-'*70}")
    print("🎉 所有测试结束！如果全绿且加速比 > 1，说明算子优化完美！")


