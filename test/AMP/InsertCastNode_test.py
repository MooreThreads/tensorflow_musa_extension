import os
import time
import json
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# =========================
# 1. 环境变量
# =========================
# 日志等级：0=INFO, 1=WARNING, 2=ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "2"
# 只打开 musa_graph_optimizer 的 VLOG 1
# 这个变量名来自你给的模板
os.environ["TF_CPP_VMODULE"] = "musa_graph_optimizer_FixAMP=2"


# =========================
# 2. 可调测试参数
# =========================
BATCH = 4
SEQ = 128
HIDDEN = 768
NUM_BLOCKS = 1         # 可以调大到 32 / 48，让冗余 cast 更明显
WARMUP_STEPS = 30
BENCH_STEPS = 10
SEED = 1234

# 你也可以切换成 BF16
PRECISION_MODE = "FP16"   # "FP16" or "BF16"

# 是否使用 aggressive_mode
AGGRESSIVE_MODE = False

# 为了让测试更聚焦 AMP，这里默认关闭 layout optimizer
DISABLE_LAYOUT_OPTIMIZER = True






musa_plugin_path = "/workspace/tensorflow_musa_extension/build/libmusa_plugin.so"

# ==========================================
# 3. 加载 MUSA 插件
# ==========================================
def load_musa_plugin():
    if os.path.exists(musa_plugin_path):
        try:
            tf.load_op_library(musa_plugin_path)
            print(f">>>> [MUSA] Plugin loaded successfully from: {musa_plugin_path}")
        except Exception as e:
            print(f"!!!! [MUSA] Failed to load plugin: {e}")
    else:
        print(f"!!!! [MUSA] Plugin not found at {musa_plugin_path}, assuming built-in.")

# =========================
# 4. 图构建
# =========================
def dense_block(x, in_dim, out_dim, block_id):
    """
    MatMul -> BiasAdd -> Relu
    用 numpy 预生成常量，避免在 MUSA 上创建随机初始化 op。
    """
    rng = np.random.RandomState(SEED + block_id)

    w_np = rng.randn(in_dim, out_dim).astype(np.float32) * 0.02
    b_np = rng.randn(out_dim).astype(np.float32)

    with tf.name_scope(f"block_{block_id}"):
        w = tf.constant(w_np, dtype=tf.float32, name="w")
        b = tf.constant(b_np, dtype=tf.float32, name="b")

        y = tf.matmul(x, w, name="matmul")
        y = tf.nn.bias_add(y, b, name="bias_add")
        y = tf.nn.relu(y, name="relu")
        return y


def build_test_graph(num_blocks=NUM_BLOCKS):
    """
    输入 [BATCH, SEQ, HIDDEN]
    先 reshape 成二维，再串很多 block，最后做一个 reduce_mean，
    保证图里既有 AMP 候选，又有输出 fetch。
    """
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/device:MUSA:0"):
            x = tf.compat.v1.placeholder(
                tf.float32, shape=[BATCH, SEQ, HIDDEN], name="input"
            )

            y = tf.reshape(x, [BATCH * SEQ, HIDDEN], name="flatten")

            for i in range(num_blocks):
                y = dense_block(y, HIDDEN, HIDDEN, i)

            # 避免整个输出过大，fetch 一个较小结果
            out = tf.reduce_mean(y, axis=1, name="reduce_mean")
            out = tf.identity(out, name="final_output")

    return graph


# =========================
# 5. Session 配置
# =========================
def make_session_config(enable_amp):
    config = tf.compat.v1.ConfigProto()

    rewriter = config.graph_options.rewrite_options
    opt = rewriter.custom_optimizers.add()
    opt.name = "musa_graph_optimizer"

    # 只使用你当前文件里已经存在的参数
    opt.parameter_map["aggressive_mode"].b = AGGRESSIVE_MODE
    opt.parameter_map["precision_mode"].s = PRECISION_MODE.encode("utf-8")
    opt.parameter_map["disable_layout_optimizer"].b = DISABLE_LAYOUT_OPTIMIZER
    opt.parameter_map["disable_amp"].b = (not enable_amp)

    return config


# =========================
# 6. 计时函数
# =========================
def benchmark_one_case(graph, enable_amp, input_data):
    """
    返回:
      {
        "enable_amp": bool,
        "warmup_avg_ms": ...,
        "bench_avg_ms": ...,
        "bench_p50_ms": ...,
        "bench_p90_ms": ...,
        "bench_p95_ms": ...,
        "bench_min_ms": ...,
        "bench_max_ms": ...,
      }
    """
    config = make_session_config(enable_amp=enable_amp)

    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        # 初始化变量
        sess.run(tf.compat.v1.global_variables_initializer())

        x = graph.get_tensor_by_name("input:0")
        out = graph.get_tensor_by_name("final_output:0")

        # Warmup
    warmup_times = []
    for _ in range(WARMUP_STEPS):
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            t0 = time.perf_counter()
            _ = sess.run(out, feed_dict={x: input_data})
            t1 = time.perf_counter()
            warmup_times.append((t1 - t0) * 1000.0)

        # Benchmark
    bench_times = []
    bench_results = []
    for _ in range(BENCH_STEPS):
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            t0 = time.perf_counter()
            result = sess.run(out, feed_dict={x: input_data})
            t1 = time.perf_counter()
            bench_times.append((t1 - t0) * 1000.0)
            bench_results.append(result)
    if(enable_amp):
        with open('result_AMP.txt', 'w') as f:
            for line in bench_results:
                f.write(f"{line}\n")
    bench_arr = np.array(bench_times, dtype=np.float64)
    warmup_arr = np.array(warmup_times, dtype=np.float64)

    return {
        "enable_amp": enable_amp,
        "warmup_avg_ms": float(np.mean(warmup_arr)),
        "bench_avg_ms": float(np.mean(bench_arr)),
        "bench_avg_result": float(np.mean(bench_results)),
        "bench_p50_ms": float(np.percentile(bench_arr, 50)),
        "bench_p90_ms": float(np.percentile(bench_arr, 90)),
        "bench_p95_ms": float(np.percentile(bench_arr, 95)),
        "bench_min_ms": float(np.min(bench_arr)),
        "bench_max_ms": float(np.max(bench_arr)),
    }


# =========================
# 7. 主流程
# =========================
def main():
    load_musa_plugin()
    np.random.seed(SEED)

    graph = build_test_graph(num_blocks=NUM_BLOCKS)

    input_data = np.random.randn(BATCH, SEQ, HIDDEN).astype(np.float32)

    print("=" * 80)
    print("Benchmark config")
    print(f"BATCH={BATCH}, SEQ={SEQ}, HIDDEN={HIDDEN}")
    print(f"NUM_BLOCKS={NUM_BLOCKS}")
    print(f"WARMUP_STEPS={WARMUP_STEPS}, BENCH_STEPS={BENCH_STEPS}")
    print(f"PRECISION_MODE={PRECISION_MODE}")
    print(f"AGGRESSIVE_MODE={AGGRESSIVE_MODE}")
    print(f"DISABLE_LAYOUT_OPTIMIZER={DISABLE_LAYOUT_OPTIMIZER}")
    print("=" * 80)

    # Case 1: AMP 关闭
    print("AMP OFF =============")
    result_no_amp = benchmark_one_case(
        graph=graph,
        enable_amp=False,
        input_data=input_data,
    )
    print("AMP ON =============")
    # Case 2: AMP 开启（当前版本可能含冗余 cast）
    result_amp = benchmark_one_case(
        graph=graph,
        enable_amp=True,
        input_data=input_data,
    )

    print("\n[Result] AMP OFF")
    print(json.dumps(result_no_amp, indent=2))

    print("\n[Result] AMP ON")
    print(json.dumps(result_amp, indent=2))

    speedup = result_no_amp["bench_avg_ms"] / result_amp["bench_avg_ms"]
    print("\n[Summary]")
    print(f"Speedup (AMP OFF / AMP ON) = {speedup:.4f}x")

    print("\n[How to use this script]")
    print("1) 先用当前 optimizer 跑一遍，记录 AMP ON 的 bench_avg_ms")
    print("2) 修改 OptimizeAMP，去掉冗余 cast")
    print("3) 重新编译 plugin")
    print("4) 用完全相同的脚本和参数再跑一遍")
    print("5) 对比两次 AMP ON 的 bench_avg_ms / p50 / p95")


if __name__ == "__main__":
    main()
