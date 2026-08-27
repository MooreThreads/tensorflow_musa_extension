#!/usr/bin/env python3
"""Standalone TensorFlow 2.15 GraphDef CPU/MUSA accuracy checker.

默认用于 JD platform dnn GraphDef：

  python3 test/multi_domain_dnn/tf215_graph_accuracy.py

脚本会：
1. 使用当前 Python 环境中的 TensorFlow 2.15 和 MUSA runtime；仅在未安装 TensorFlow 时回退到开发期 site-packages；
2. 生成确定性 mock inputs；
3. 在独立 worker 进程中分别运行 CPU no-XLA 与 MUSA XLA；
4. 对比输出 shape/dtype/finite/allclose，并写出 JSON report。

默认可直接在 tensorflow_musa_extension 仓库根目录运行。普通 wheel 路径不需要 PJRT
插件路径；开发调试时可用命令行参数覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT_TENSOR = "WeightedSum/add:0"
DEFAULT_MODEL_REL = (
    "model/JD/real_models/platform/"
    "dnn__multi-domain-task-simplifyv3__2026022622335783__tf_models__frozen_grpah.pb"
)
ENV_BOOTSTRAP_MARKER = "TF215_GRAPH_ACCURACY_STANDALONE_BOOTSTRAPPED"
ENV_BUNDLED_TF_MARKER = "TF215_GRAPH_ACCURACY_USE_BUNDLED_TF"


def find_repo_root(path: Path) -> Path:
    """Find tensorflow_musa_extension or OpenXLA-style repo root."""
    for parent in [path.parent, *path.parents]:
        if (parent / "setup.py").exists() and (parent / "python").is_dir() and (parent / "musa_ext").is_dir():
            return parent
        if (parent / "WORKSPACE").exists() or (parent / "WORKSPACE.bazel").exists():
            return parent
    return path.parent


REPO_ROOT = Path(
    os.environ.get(
        "TF_MUSA_EXTENSION_ROOT",
        os.environ.get("OPENXLA_REPO_ROOT", find_repo_root(SCRIPT_PATH)),
    )
).resolve()
DEFAULT_GRAPH_PATH = Path(os.environ.get("MUSA_MODEL_PATH", REPO_ROOT / DEFAULT_MODEL_REL))
OPENXLA_MUSA_ROOT = Path(
    os.environ.get("OPENXLA_MUSA_ROOT", REPO_ROOT.parent / "openxla_musa")
).resolve()
DEFAULT_TF_SITE_PACKAGES = Path(
    os.environ.get(
        "TF215_SITE_PACKAGES",
        os.environ.get(
            "TENSORFLOW_SITE_PACKAGES",
            OPENXLA_MUSA_ROOT / "pjrt_plugin/tf2.15/lib/python3.10/site-packages",
        ),
    )
)
MUSA_ENV_NAMES = {
    "MUSA_PJRT_ALLOCATION_MODEL",
    "MUSA_PJRT_ALLOW_EVENT_REUSE",
    "MUSA_PJRT_FORCE_HOST_BUFFER_COPY",
    "MUSA_PJRT_WAIT_EXECUTE_DONE",
    "MUSA_BLAS_RESET_STREAM_AFTER_CALL",
    "XLA_PJRT_SE_COALESCE_DEFINITION_EVENT_WAITS",
}


def prepend_path_env(env: dict[str, str], name: str, entries: list[Path | str]) -> None:
    values = [str(item) for item in entries if str(item)]
    current = env.get(name, "")
    if current:
        values.extend(item for item in current.split(":") if item)
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    env[name] = ":".join(deduped)


def ensure_bootstrap_env() -> None:
    """Re-exec once so LD_LIBRARY_PATH/PYTHONPATH are effective at process start."""
    if os.environ.get(ENV_BOOTSTRAP_MARKER) == "1":
        if os.environ.get(ENV_BUNDLED_TF_MARKER) == "1" and str(DEFAULT_TF_SITE_PACKAGES) not in sys.path:
            sys.path.insert(0, str(DEFAULT_TF_SITE_PACKAGES))
        return

    env = os.environ.copy()
    env[ENV_BOOTSTRAP_MARKER] = "1"
    # A normal wheel installation has tensorflow==2.15.1 in this interpreter.
    # Do not shadow it with the OpenXLA checkout merely because the target is MUSA.
    use_bundled_tf = importlib.util.find_spec("tensorflow") is None
    env[ENV_BUNDLED_TF_MARKER] = "1" if use_bundled_tf else "0"
    env.setdefault("MUSA_HOME", "/usr/local/musa")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("GLOG_minloglevel", "2")
    if use_bundled_tf:
        prepend_path_env(env, "PYTHONPATH", [DEFAULT_TF_SITE_PACKAGES])
    prepend_path_env(
        env,
        "LD_LIBRARY_PATH",
        [
            Path(env["MUSA_HOME"]) / "lib",
            *([DEFAULT_TF_SITE_PACKAGES / "tensorflow"] if use_bundled_tf else []),
            "/usr/local/openmpi/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
        ],
    )
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def normalize_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("需要逗号分隔的正整数")
    return items


def default_device_index() -> int:
    visible_devices = os.environ.get("MUSA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return 0
    first = visible_devices.split(",", 1)[0].strip()
    return int(first) if first else 0


def selected_runtime(args: argparse.Namespace) -> str:
    return "pjrt" if args.xla else "native"


def import_tensorflow_musa() -> object:
    try:
        import tensorflow_musa
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "未找到 tensorflow_musa 包。请先在 tensorflow_musa_extension 根目录构建并安装 wheel，"
            "或在包含 tensorflow_musa 的 Python 环境中运行。"
        ) from exc
    return tensorflow_musa


def import_tensorflow_v1():
    try:
        import tensorflow.compat.v1 as tf
        from tensorflow.core.framework import graph_pb2
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None)
        if missing_name == "tensorflow":
            raise RuntimeError(
                f"未找到 TensorFlow。已尝试加入 {DEFAULT_TF_SITE_PACKAGES}，"
                "请确认 TF2.15 wheel/site-packages 已存在，或设置 TF215_SITE_PACKAGES。"
            ) from exc
        raise RuntimeError(
            f"导入 TensorFlow 时缺少依赖模块: {missing_name!r}。"
            f"请把该依赖安装到 {DEFAULT_TF_SITE_PACKAGES} 或当前 Python 环境。"
        ) from exc
    tf.disable_v2_behavior()
    tf.disable_eager_execution()
    return tf, graph_pb2


def canonical_device_type(device: str) -> str:
    mapping = {"cpu": "CPU", "cuda": "GPU", "musa": "MUSA"}
    key = device.lower()
    if key not in mapping:
        raise ValueError(f"不支持的设备类型: {device}")
    return mapping[key]


def clean_graph_input_name(input_name: str) -> str:
    return input_name.split(":")[0].lstrip("^")


def const_tensor_first_dim(node) -> Optional[int]:
    if node is None or "value" not in node.attr:
        return None
    shape = node.attr["value"].tensor.tensor_shape
    if not shape.dim:
        return None
    first_dim = shape.dim[0].size
    return int(first_dim) if first_dim > 0 else None


def infer_integer_placeholder_upper_bound(graph_def, nodes_by_name, placeholder_name: str):
    bounds = []
    for node in graph_def.node:
        if node.op not in {"Gather", "GatherV2", "ResourceGather"} or len(node.input) < 2:
            continue
        if clean_graph_input_name(node.input[1]) != placeholder_name:
            continue
        first_dim = const_tensor_first_dim(nodes_by_name.get(clean_graph_input_name(node.input[0])))
        if first_dim is not None:
            bounds.append(first_dim)
    return min(bounds) if bounds else None


def load_graph_and_placeholders(tf, graph_pb2, graph_path: Path):
    if not graph_path.is_file():
        raise FileNotFoundError(f"GraphDef 不存在: {graph_path}")
    graph_def = graph_pb2.GraphDef()
    with tf.io.gfile.GFile(str(graph_path), "rb") as file_handle:
        graph_def.ParseFromString(file_handle.read())

    dtype_map = {
        tf.float32.as_datatype_enum: np.float32,
        tf.int32.as_datatype_enum: np.int32,
        tf.int64.as_datatype_enum: np.int64,
        tf.bool.as_datatype_enum: np.bool_,
        tf.string.as_datatype_enum: np.str_,
    }
    nodes_by_name = {node.name: node for node in graph_def.node}
    placeholders = {}
    for node in graph_def.node:
        if node.op != "Placeholder":
            continue
        shape = []
        if "shape" in node.attr and not node.attr["shape"].shape.unknown_rank:
            shape = [
                None if dim.size < 0 else int(dim.size)
                for dim in node.attr["shape"].shape.dim
            ]
        dtype = dtype_map.get(node.attr["dtype"].type, np.float32)
        placeholders[node.name] = {
            "dtype": dtype,
            "shape": shape,
            "int_upper_bound": infer_integer_placeholder_upper_bound(
                graph_def, nodes_by_name, node.name
            )
            if dtype in {np.int32, np.int64}
            else None,
        }
    return graph_def, placeholders


def create_mock_data(
    placeholders: dict,
    batch_size: int,
    seed: int,
    int_upper_bound: int | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feed_dict = {}
    for name, info in placeholders.items():
        shape = [batch_size if dim is None or dim == -1 else dim for dim in info["shape"]]
        dtype = info["dtype"]
        if dtype == np.float32:
            value = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        elif dtype in {np.int32, np.int64}:
            inferred_upper_bound = int(info.get("int_upper_bound") or 100)
            high = (
                min(inferred_upper_bound, int_upper_bound)
                if int_upper_bound is not None
                else inferred_upper_bound
            )
            high = max(high, 1)
            value = rng.integers(0, high, size=shape, dtype=dtype)
        elif dtype == np.bool_:
            value = rng.choice(np.array([False, True], dtype=np.bool_), size=shape)
        elif dtype == np.str_:
            value = np.full(shape if shape else (), "mock", dtype=np.str_)
        else:
            value = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        feed_dict[f"{name}:0"] = value
    return feed_dict


def import_graph(
    tf,
    graph_def,
    device_name: str,
    xla: bool,
    device_type: str,
    xla_jit_scope: bool,
) -> None:
    with ExitStack() as stack:
        stack.enter_context(tf.device(device_name))
        if xla and xla_jit_scope and canonical_device_type(device_type) in {"GPU", "MUSA"}:
            stack.enter_context(tf.xla.experimental.jit_scope(compile_ops=True))
        tf.import_graph_def(graph_def, name="")


def session_config(
    tf,
    device_type: str,
    xla: bool,
    allow_soft_placement: bool,
    inter_threads: int,
    intra_threads: int,
):
    from tensorflow.core.protobuf import rewriter_config_pb2

    config = tf.ConfigProto()
    config.allow_soft_placement = allow_soft_placement
    config.log_device_placement = False
    config.inter_op_parallelism_threads = inter_threads
    config.intra_op_parallelism_threads = intra_threads
    device_upper = canonical_device_type(device_type)
    if device_upper == "GPU":
        config.gpu_options.allow_growth = True
    if device_upper == "MUSA":
        rewrite = config.graph_options.rewrite_options
        rewrite.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
        rewrite.remapping = rewriter_config_pb2.RewriterConfig.OFF
        rewrite.custom_optimizers.add().name = "musa_graph_optimizer"
    if xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
    return config


def _configure_pjrt_development_overrides(args: argparse.Namespace) -> None:
    """Apply explicit development-only artifact overrides to the provider."""
    if args.musa_plugin_path:
        os.environ["TENSORFLOW_MUSA_PJRT_LIBRARY_PATH"] = str(
            normalize_path(args.musa_plugin_path)
        )
    if args.musa_bridge_path:
        os.environ["TENSORFLOW_MUSA_PJRT_BRIDGE_PATH"] = str(
            normalize_path(args.musa_bridge_path)
        )


def prepare_musa_env(args: argparse.Namespace) -> bool:
    os.environ["MUSA_VISIBLE_DEVICES"] = str(args.device_index)
    if not args.xla:
        for name in [
            "TENSORFLOW_MUSA_PJRT_LIBRARY_PATH",
            "TENSORFLOW_MUSA_PJRT_BRIDGE_PATH",
            "TENSORFLOW_MUSA_PJRT_PROVIDER_MODULE",
            "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
            "PJRT_NAMES_AND_LIBRARY_PATHS",
            "NEXT_PLUGGABLE_DEVICE_USE_C_API",
            "MUSA_TFNPD_IS_PLUGGABLE_DEVICE",
            "MUSA_PJRT_WAIT_EXECUTE_DONE",
            "MUSA_PJRT_MAX_INFLIGHT_COMPILES",
            "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS",
            "MUSA_PJRT_MAX_INFLIGHT_EXECUTES",
            "MTCC20X_TOOLS_DIR",
            "TF_XLA_FLAGS",
        ]:
            os.environ.pop(name, None)
        import_tensorflow_musa()
        return False

    _configure_pjrt_development_overrides(args)
    os.environ["MUSA_PJRT_WAIT_EXECUTE_DONE"] = "0"
    os.environ["MUSA_PJRT_MAX_INFLIGHT_COMPILES"] = str(args.musa_pjrt_max_inflight_compiles)
    os.environ["MUSA_PJRT_MAX_INFLIGHT_TRANSFERS"] = str(args.musa_pjrt_max_inflight_transfers)
    os.environ["MUSA_PJRT_MAX_INFLIGHT_EXECUTES"] = str(args.musa_pjrt_max_inflight_executes)
    tensorflow_musa = import_tensorflow_musa()
    status = tensorflow_musa.get_pjrt_status()
    if status.get("mode") != "pjrt" or status.get("phase") != "ready":
        raise RuntimeError(f"OpenXLA-MUSA PJRT bootstrap did not become ready: {status}")
    print(f"[musa] PJRT_STATUS = {status}")
    return True


def prepare_worker_runtime(args: argparse.Namespace):
    if args.worker == "cpu":
        for name in [
            "TF_XLA_FLAGS",
            "XLA_FLAGS",
            "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
            "PJRT_NAMES_AND_LIBRARY_PATHS",
            "NEXT_PLUGGABLE_DEVICE_USE_C_API",
            "MUSA_VISIBLE_DEVICES",
            "MUSA_TFNPD_IS_PLUGGABLE_DEVICE",
        ]:
            os.environ.pop(name, None)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf, graph_pb2 = import_tensorflow_v1()
        return tf, graph_pb2, "/device:CPU:0", False, True

    if args.worker == "cuda":
        for name in [
            "TF_XLA_FLAGS",
            "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
            "PJRT_NAMES_AND_LIBRARY_PATHS",
            "NEXT_PLUGGABLE_DEVICE_USE_C_API",
            "MUSA_VISIBLE_DEVICES",
            "MUSA_TFNPD_IS_PLUGGABLE_DEVICE",
            "MUSA_PJRT_WAIT_EXECUTE_DONE",
            "MUSA_PJRT_MAX_INFLIGHT_COMPILES",
            "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS",
            "MUSA_PJRT_MAX_INFLIGHT_EXECUTES",
        ]:
            os.environ.pop(name, None)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
        if args.cuda_disable_tf32:
            tf = None
            graph_pb2 = None
            tf, graph_pb2 = import_tensorflow_v1()
            tf.config.experimental.enable_tensor_float_32_execution(False)
            return tf, graph_pb2, "/device:GPU:0", True, not args.strict_placement
        tf, graph_pb2 = import_tensorflow_v1()
        return tf, graph_pb2, "/device:GPU:0", True, not args.strict_placement

    for name in MUSA_ENV_NAMES:
        os.environ.pop(name, None)
    use_xla = prepare_musa_env(args)
    tf, graph_pb2 = import_tensorflow_v1()
    logical_musa = [
        item for item in tf.config.list_logical_devices() if item.device_type.upper() == "MUSA"
    ]
    if not logical_musa:
        raise RuntimeError("TensorFlow 未发现 MUSA 逻辑设备")
    return tf, graph_pb2, "/device:MUSA:0", use_xla, not args.strict_placement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone TF2.15 GraphDef CPU/MUSA accuracy validation"
    )
    parser.add_argument("--device", choices=["musa", "cuda"], default="musa")
    parser.add_argument(
        "--xla",
        action="store_true",
        help="使用 OpenXLA-MUSA PJRT；不加该参数时使用 native",
    )
    parser.add_argument("--graph_path", default=str(DEFAULT_GRAPH_PATH))
    parser.add_argument("--output_tensor", default=DEFAULT_OUTPUT_TENSOR)
    parser.add_argument("--batches", type=parse_int_list, default=parse_int_list("1,2,8,32,100"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20260421"))
    parser.add_argument(
        "--int_upper_bound",
        type=int,
        default=None,
        help="整型 mock 输入的全局上界；设置后对每个 placeholder 取更小的安全值",
    )
    parser.add_argument("--device_index", type=int, default=default_device_index())
    parser.add_argument("--warmup_runs_per_case", type=int, default=1)
    parser.add_argument("--runs_per_case", type=int, default=2)
    parser.add_argument("--strict_placement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xla_jit_scope", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cuda_disable_tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅 CUDA 生效；默认关闭 TF32 以便和 CPU 更容易对齐",
    )
    parser.add_argument(
        "--trace_backend_execution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="抓取一次 RunMetadata，检查 backend 是否真的执行在目标设备上",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--output_dir",
        help=(
            "输出目录；默认按运行模式分离为 "
            "tmp/e2e_accuracy_standalone_<native|pjrt>，避免 native/PJRT report 互相覆盖"
        ),
    )
    parser.add_argument(
        "--musa_plugin_path",
        help="开发调试时覆盖 provider wheel 内的 PJRT plugin；正常 wheel 路径无需设置",
    )
    parser.add_argument(
        "--musa_bridge_path",
        help="开发调试时覆盖 provider wheel 内的 TF registry bridge；正常 wheel 路径无需设置",
    )
    parser.add_argument("--musa_pjrt_max_inflight_compiles", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_transfers", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_executes", type=int, default=1)
    parser.add_argument("--worker", choices=["cpu", "musa", "cuda"], help=argparse.SUPPRESS)
    parser.add_argument("--manifest", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / f"tmp/e2e_accuracy_standalone_{selected_runtime(args)}")
    return args


def generate_manifest(args: argparse.Namespace, output_dir: Path) -> Path:
    tf, graph_pb2 = import_tensorflow_v1()
    graph_path = normalize_path(args.graph_path)
    _, placeholders = load_graph_and_placeholders(tf, graph_pb2, graph_path)
    if not placeholders:
        raise RuntimeError("GraphDef 中没有 Placeholder")

    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    placeholder_names = [f"{name}:0" for name in placeholders]
    cases = []
    for batch in args.batches:
        for seed in args.seeds:
            case_name = f"batch{batch}_seed{seed}"
            feed_dict = create_mock_data(
                placeholders,
                batch,
                seed,
                args.int_upper_bound,
            )
            input_path = input_dir / f"{case_name}.npz"
            np.savez(
                input_path,
                **{
                    f"input_{index:03d}": feed_dict[name]
                    for index, name in enumerate(placeholder_names)
                },
            )
            cases.append(
                {
                    "name": case_name,
                    "batch": batch,
                    "seed": seed,
                    "input_file": str(input_path.relative_to(output_dir)),
                    "input_sha256": sha256_file(input_path),
                }
            )

    manifest = {
        "graph_path": str(graph_path),
        "graph_sha256": sha256_file(graph_path),
        "output_tensor": args.output_tensor,
        "placeholder_names": placeholder_names,
        "cases": cases,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {manifest_path}")
    print(f"[cases] {len(cases)}, placeholders={len(placeholder_names)}")
    return manifest_path


def run_worker(args: argparse.Namespace) -> None:
    if not args.worker or not args.manifest:
        raise RuntimeError("worker mode requires --worker and --manifest")
    output_root = normalize_path(args.output_dir)
    manifest_path = normalize_path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    label = args.worker
    output_dir = output_root / "outputs" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    tf, graph_pb2, device_name, xla, allow_soft = prepare_worker_runtime(args)
    graph_path = normalize_path(args.graph_path)
    if sha256_file(graph_path) != manifest["graph_sha256"]:
        raise RuntimeError("GraphDef SHA256 mismatch")
    graph_def, _ = load_graph_and_placeholders(tf, graph_pb2, graph_path)

    with tf.Graph().as_default() as graph:
        import_graph(tf, graph_def, device_name, xla, args.worker, args.xla_jit_scope)
        output_tensor = graph.get_tensor_by_name(manifest["output_tensor"])
        input_tensors = [graph.get_tensor_by_name(name) for name in manifest["placeholder_names"]]
        config = session_config(
            tf,
            args.worker,
            xla,
            allow_soft_placement=allow_soft,
            inter_threads=1,
            intra_threads=1,
        )
        with tf.Session(graph=graph, config=config) as session:
            print(f"[{label}] visible devices:")
            for device in session.list_devices():
                print(f"  - {device.name} [{device.device_type}]")
            for case in manifest["cases"]:
                input_path = output_root / case["input_file"]
                if sha256_file(input_path) != case["input_sha256"]:
                    raise RuntimeError(f"input sha256 mismatch: {input_path}")
                with np.load(input_path, allow_pickle=False) as arrays:
                    feed = {
                        tensor: arrays[f"input_{index:03d}"]
                        for index, tensor in enumerate(input_tensors)
                    }
                for _ in range(args.warmup_runs_per_case):
                    session.run(output_tensor, feed_dict=feed)
                run_options = None
                run_metadata = None
                if args.trace_backend_execution:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                result = session.run(
                    output_tensor,
                    feed_dict=feed,
                    options=run_options,
                    run_metadata=run_metadata,
                )
                for run_index in range(1, args.runs_per_case):
                    repeat = session.run(output_tensor, feed_dict=feed)
                    if not np.allclose(result, repeat, atol=args.atol, rtol=args.rtol):
                        diff = np.abs(result.astype(np.float64) - repeat.astype(np.float64))
                        raise RuntimeError(
                            f"{label} {case['name']} unstable at run {run_index + 1}, "
                            f"max_abs={float(diff.max(initial=0.0)):.8e}"
                        )
                if args.trace_backend_execution and args.worker != "cpu":
                    execution_devices = {
                        item.device
                        for item in (run_metadata.step_stats.dev_stats if run_metadata else [])
                        if item.device
                    }
                    expected_device = "MUSA" if args.worker == "musa" else "GPU"
                    if not any(expected_device in device for device in execution_devices):
                        raise RuntimeError(
                            f"{label} {case['name']} 未观察到 {expected_device} 执行痕迹，"
                            f"devices={sorted(execution_devices)}"
                        )
                np.save(output_dir / f"{case['name']}.npy", result)
                print(
                    f"[{label}] {case['name']} shape={result.shape} "
                    f"dtype={result.dtype} finite={bool(np.isfinite(result).all())}"
                )


def launch_worker(args: argparse.Namespace, manifest_path: Path, worker: str) -> None:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        worker,
        "--manifest",
        str(manifest_path),
        "--graph_path",
        str(args.graph_path),
        "--output_tensor",
        args.output_tensor,
        "--output_dir",
        str(args.output_dir),
        "--device_index",
        str(args.device_index),
        "--warmup_runs_per_case",
        str(args.warmup_runs_per_case),
        "--runs_per_case",
        str(args.runs_per_case),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--musa_pjrt_max_inflight_compiles",
        str(args.musa_pjrt_max_inflight_compiles),
        "--musa_pjrt_max_inflight_transfers",
        str(args.musa_pjrt_max_inflight_transfers),
        "--musa_pjrt_max_inflight_executes",
        str(args.musa_pjrt_max_inflight_executes),
    ]
    if args.xla:
        command.append("--xla")
    if args.musa_plugin_path:
        command.extend(["--musa_plugin_path", str(args.musa_plugin_path)])
    if args.musa_bridge_path:
        command.extend(["--musa_bridge_path", str(args.musa_bridge_path)])
    command.append("--strict_placement" if args.strict_placement else "--no-strict_placement")
    command.append("--xla_jit_scope" if args.xla_jit_scope else "--no-xla_jit_scope")
    command.append("--cuda_disable_tf32" if args.cuda_disable_tf32 else "--no-cuda_disable_tf32")
    command.append(
        "--trace_backend_execution"
        if args.trace_backend_execution
        else "--no-trace_backend_execution"
    )
    print(f"\n[launch] {' '.join(command)}")
    subprocess.run(command, check=True)


def compare_outputs(args: argparse.Namespace, manifest_path: Path) -> bool:
    output_root = normalize_path(args.output_dir)
    manifest = json.loads(manifest_path.read_text())
    comparisons = []
    device_label = args.device
    for case in manifest["cases"]:
        cpu = np.load(output_root / "outputs/cpu" / f"{case['name']}.npy")
        device = np.load(output_root / f"outputs/{device_label}" / f"{case['name']}.npy")
        same_shape = cpu.shape == device.shape
        same_dtype = cpu.dtype == device.dtype
        finite = bool(np.isfinite(cpu).all() and np.isfinite(device).all())
        if same_shape and np.issubdtype(cpu.dtype, np.number):
            abs_diff = np.abs(cpu.astype(np.float64) - device.astype(np.float64))
            rel_diff = abs_diff / np.maximum(np.abs(cpu.astype(np.float64)), 1e-12)
            mismatch = int(np.count_nonzero(~np.isclose(cpu, device, atol=args.atol, rtol=args.rtol)))
            max_abs = float(abs_diff.max(initial=0.0))
            mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
            max_rel = float(rel_diff.max(initial=0.0))
        else:
            mismatch = 0 if np.array_equal(cpu, device) else -1
            max_abs = mean_abs = max_rel = float("inf") if mismatch else 0.0
        passed = same_shape and same_dtype and finite and mismatch == 0
        comparisons.append(
            {
                "device": device_label,
                "case": case["name"],
                "batch": case["batch"],
                "seed": case["seed"],
                "passed": passed,
                "shape": list(device.shape),
                "dtype": str(device.dtype),
                "finite": finite,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "max_rel": max_rel,
                "mismatch_count": mismatch,
            }
        )
        print(
            f"[{'PASS' if passed else 'FAIL'}] {case['name']} "
            f"max_abs={max_abs:.8e} max_rel={max_rel:.8e} mismatch={mismatch}"
        )

    report = {
        "passed": all(item["passed"] for item in comparisons),
        "method": "same deterministic mock inputs, CPU no-XLA golden vs selected backend",
        "runtime_mode": selected_runtime(args),
        "graph_path": manifest["graph_path"],
        "graph_sha256": manifest["graph_sha256"],
        "output_tensor": manifest["output_tensor"],
        "strict_placement": args.strict_placement,
        "xla_jit_scope": args.xla_jit_scope,
        "trace_backend_execution": args.trace_backend_execution,
        "tf_interop_threads": 1,
        "tf_intraop_threads": 1,
        "cuda_disable_tf32": args.cuda_disable_tf32 if device_label == "cuda" else None,
        "device": device_label,
        "atol": args.atol,
        "rtol": args.rtol,
        "device_plugin_path": (
            str(normalize_path(args.musa_plugin_path))
            if args.musa_plugin_path
            else None
        ),
        "device_plugin_sha256": (
            sha256_file(normalize_path(args.musa_plugin_path))
            if device_label == "musa" and args.musa_plugin_path
            else None
        ),
        "device_plugin_source": (
            "development override" if args.musa_plugin_path else "provider wheel auto-discovery"
        ),
        "comparisons": comparisons,
    }
    report_path = output_root / "accuracy_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[report] {report_path}")
    print(f"[result] {'PASS' if report['passed'] else 'FAIL'}")
    return bool(report["passed"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    ensure_bootstrap_env()
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    if args.warmup_runs_per_case < 0 or args.runs_per_case <= 0:
        raise ValueError("warmup must be >=0 and runs must be >0")
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
    output_dir = normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = generate_manifest(args, output_dir)
    launch_worker(args, manifest_path, "cpu")
    launch_worker(args, manifest_path, args.device)
    if not compare_outputs(args, manifest_path):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
