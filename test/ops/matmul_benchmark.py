
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PLUGIN_PATH = ROOT_DIR / 'build' / 'libmusa_plugin.so'


def resolve_plugin_path(cli_path):
    if cli_path:
        p = Path(cli_path).resolve()
        if not p.exists():
            raise FileNotFoundError('plugin not found: {}'.format(p))
        return p
    if DEFAULT_PLUGIN_PATH.exists():
        return DEFAULT_PLUGIN_PATH
    raise FileNotFoundError('plugin not found: {}'.format(DEFAULT_PLUGIN_PATH))


def run(args):
    plugin_path = resolve_plugin_path(args.musa_plugin)
    tf.load_op_library(str(plugin_path))

    rng = np.random.RandomState(args.seed)
    a_np = rng.standard_normal((args.m, args.k)).astype(np.float32)
    b_np = rng.standard_normal((args.k, args.n)).astype(np.float32)

    expected = None
    if args.check:
        expected = np.matmul(a_np, b_np)

    graph = tf.Graph()
    with graph.as_default():
        with tf.device('/device:MUSA:0'):
            a_ph = tf.placeholder(tf.float32, shape=[args.m, args.k], name='a')
            b_ph = tf.placeholder(tf.float32, shape=[args.k, args.n], name='b')
            out = tf.matmul(a_ph, b_ph)
            out = tf.identity(out, name='matmul_out')
            with tf.control_dependencies([out]):
                step = tf.no_op(name='step')

    cfg = tf.ConfigProto()
    cfg.allow_soft_placement = True

    with tf.Session(graph=graph, config=cfg) as sess:
        feed = {a_ph: a_np, b_ph: b_np}
        got = sess.run(out, feed_dict=feed)
        if expected is not None:
            np.testing.assert_allclose(got, expected, rtol=args.rtol, atol=args.atol)

        for _ in range(args.warmup):
            sess.run(step, feed_dict=feed)

        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            sess.run(step, feed_dict=feed)
            ts.append((time.perf_counter() - t0) * 1000.0)

    result = {
        'm': args.m,
        'k': args.k,
        'n': args.n,
        'warmup': args.warmup,
        'iters': args.iters,
        'mean_ms': float(np.mean(ts)),
        'p50_ms': float(np.percentile(ts, 50)),
        'p90_ms': float(np.percentile(ts, 90)),
        'min_ms': float(np.min(ts)),
        'max_ms': float(np.max(ts)),
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description='MUSA MatMul benchmark')
    parser.add_argument('--m', type=int, default=3600)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--rtol', type=float, default=1e-2)
    parser.add_argument('--atol', type=float, default=1e-2)
    parser.add_argument('--musa-plugin', type=str, default='')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
