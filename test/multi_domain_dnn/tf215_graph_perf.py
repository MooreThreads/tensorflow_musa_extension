#!/usr/bin/env python3
"""Standalone TensorFlow 2.15 GraphDef performance runner.

默认命令：

  python3 test/multi_domain_dnn/tf215_graph_perf.py

默认运行 JD platform dnn GraphDef 的 MUSA XLA 路径，使用 `time.perf_counter()`
统计 `Session.run` 端到端耗时；可通过 `--device cpu|cuda|musa` 切换设备。

默认可直接在 tensorflow_musa_extension 仓库根目录运行；如 PJRT 插件或 TF2.15
site-packages 不在默认位置，可通过环境变量覆盖。
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import queue
import shlex
import sys
import threading
import time
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
ENV_BOOTSTRAP_MARKER = "TF215_GRAPH_PERF_STANDALONE_BOOTSTRAPPED"
ENV_BUNDLED_TF_MARKER = "TF215_GRAPH_PERF_USE_BUNDLED_TF"


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
DEFAULT_MUSA_PLUGIN_PATH = Path(
    os.environ.get(
        "TENSORFLOW_MUSA_PJRT_LIBRARY_PATH",
        os.environ.get(
            "MUSA_PJRT_PLUGIN_PATH",
            OPENXLA_MUSA_ROOT / "bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so",
        ),
    )
)
DEFAULT_MUSA_BRIDGE_PATH = Path(
    os.environ.get(
        "MUSA_TF215_BRIDGE_PATH",
        OPENXLA_MUSA_ROOT / "bazel-bin/pjrt_plugin/libmusa_tf215_registry_bridge.so",
    )
)
_MUSA_RUNTIME = None
_MUSA_DEVICE_SYNCHRONIZE = None
_CUDA_RUNTIME = None
_CUDA_DEVICE_SYNCHRONIZE = None


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
    if os.environ.get(ENV_BOOTSTRAP_MARKER) == "1":
        if os.environ.get(ENV_BUNDLED_TF_MARKER) == "1" and str(DEFAULT_TF_SITE_PACKAGES) not in sys.path:
            sys.path.insert(0, str(DEFAULT_TF_SITE_PACKAGES))
        return
    env = os.environ.copy()
    env[ENV_BOOTSTRAP_MARKER] = "1"
    selected_device = "musa"
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--device":
            selected_device = sys.argv[index + 1]
    use_bundled_tf = selected_device == "musa" or importlib.util.find_spec("tensorflow") is None
    env[ENV_BUNDLED_TF_MARKER] = "1" if use_bundled_tf else "0"
    env.setdefault("MUSA_HOME", "/usr/local/musa")
    env.setdefault("MTCC20X_TOOLS_DIR", str(OPENXLA_MUSA_ROOT / "mtcc20x_tools"))
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
            DEFAULT_MUSA_PLUGIN_PATH.parent,
            DEFAULT_MUSA_BRIDGE_PATH.parent,
            OPENXLA_MUSA_ROOT / "bazel-bin/pjrt_plugin",
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


def default_device_index() -> int:
    visible_devices = os.environ.get("MUSA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return 0
    first = visible_devices.split(",", 1)[0].strip()
    return int(first) if first else 0


def runtime_mode(args: argparse.Namespace) -> str:
    return getattr(args, "runtime_mode", "pjrt")


def import_tensorflow_musa() -> object:
    try:
        import tensorflow_musa
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "未找到 tensorflow_musa 包。请先在 tensorflow_musa_extension 根目录构建并安装 wheel，"
            "或在包含 tensorflow_musa 的 Python 环境中运行。"
        ) from exc
    return tensorflow_musa


def append_env_flag(env_name: str, flag: str) -> None:
    current = os.environ.get(env_name, "").strip()
    try:
        tokens = shlex.split(current) if current else []
    except ValueError:
        tokens = current.split()
    if flag not in tokens:
        os.environ[env_name] = f"{current} {flag}".strip() if current else flag


def append_env_entry(env_name: str, entry: str) -> None:
    current = os.environ.get(env_name, "").strip()
    entries = [item.strip() for item in current.split(",") if item.strip()]
    if entry not in entries:
        entries.append(entry)
    os.environ[env_name] = ",".join(entries)


def load_dynamic_library(path: Path) -> ctypes.CDLL:
    mode = getattr(ctypes, "RTLD_GLOBAL", None)
    return ctypes.CDLL(str(path), mode=mode) if mode is not None else ctypes.CDLL(str(path))


def maybe_call_symbol(
    library: ctypes.CDLL,
    symbol_name: str,
    restype=None,
    argtypes: Optional[list[object]] = None,
):
    symbol = getattr(library, symbol_name, None)
    if symbol is None:
        return None
    if argtypes is not None:
        symbol.argtypes = list(argtypes)
    if restype is not None:
        symbol.restype = restype
    return symbol()


def call_musa_profiler_api(symbol_name: str) -> None:
    runtime_library = load_dynamic_library(Path("libmusart.so"))
    status = maybe_call_symbol(
        runtime_library, symbol_name, restype=ctypes.c_int, argtypes=[]
    )
    if status is None:
        raise RuntimeError(f"libmusart.so does not export {symbol_name}")
    if status == 801:
        print(
            f"[profiler] {symbol_name} returned musaErrorNotSupported (801); "
            "msys capture trigger remains active"
        )
        return
    if status != 0:
        raise RuntimeError(f"{symbol_name} failed with status {status}")
    print(f"[profiler] {symbol_name} succeeded")


def call_cuda_profiler_api(symbol_name: str) -> None:
    runtime_library = None
    load_error = None
    for library_name in ("libcudart.so.12", "libcudart.so"):
        try:
            runtime_library = load_dynamic_library(Path(library_name))
            break
        except OSError as error:
            load_error = error
    if runtime_library is None:
        raise RuntimeError(f"failed to load CUDA runtime: {load_error}")

    status = maybe_call_symbol(
        runtime_library, symbol_name, restype=ctypes.c_int, argtypes=[]
    )
    if status is None:
        raise RuntimeError(f"CUDA runtime does not export {symbol_name}")
    if status != 0:
        raise RuntimeError(f"{symbol_name} failed with status {status}")
    print(f"[profiler] {symbol_name} succeeded")


def synchronize_musa_device() -> None:
    global _MUSA_RUNTIME, _MUSA_DEVICE_SYNCHRONIZE
    if _MUSA_DEVICE_SYNCHRONIZE is None:
        _MUSA_RUNTIME = load_dynamic_library(Path("libmusart.so"))
        _MUSA_DEVICE_SYNCHRONIZE = getattr(_MUSA_RUNTIME, "musaDeviceSynchronize")
        _MUSA_DEVICE_SYNCHRONIZE.argtypes = []
        _MUSA_DEVICE_SYNCHRONIZE.restype = ctypes.c_int
    status = _MUSA_DEVICE_SYNCHRONIZE()
    if status != 0:
        raise RuntimeError(f"musaDeviceSynchronize failed with status {status}")


def synchronize_cuda_device() -> None:
    global _CUDA_RUNTIME, _CUDA_DEVICE_SYNCHRONIZE
    if _CUDA_DEVICE_SYNCHRONIZE is None:
        load_error = None
        for library_name in ("libcudart.so.12", "libcudart.so"):
            try:
                _CUDA_RUNTIME = load_dynamic_library(Path(library_name))
                break
            except OSError as error:
                load_error = error
        if _CUDA_RUNTIME is None:
            raise RuntimeError(f"failed to load CUDA runtime: {load_error}")
        _CUDA_DEVICE_SYNCHRONIZE = getattr(_CUDA_RUNTIME, "cudaDeviceSynchronize")
        _CUDA_DEVICE_SYNCHRONIZE.argtypes = []
        _CUDA_DEVICE_SYNCHRONIZE.restype = ctypes.c_int
    status = _CUDA_DEVICE_SYNCHRONIZE()
    if status != 0:
        raise RuntimeError(f"cudaDeviceSynchronize failed with status {status}")


def synchronize_device(device: str) -> None:
    if device == "musa":
        synchronize_musa_device()
    elif device == "cuda":
        synchronize_cuda_device()
    else:
        raise RuntimeError(f"device synchronization is unsupported for {device}")


def import_tensorflow_v1():
    try:
        import tensorflow.compat.v1 as tf
        from tensorflow.core.framework import graph_pb2
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None)
        if missing_name == "tensorflow":
            raise RuntimeError(
                f"未找到 TensorFlow，已尝试加入 {DEFAULT_TF_SITE_PACKAGES}；"
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


def runtime_device_name(device: str, device_index: int) -> str:
    if device.lower() == "musa":
        return "/device:MUSA:0"
    if device.lower() == "cuda":
        return "/device:GPU:0"
    return f"/device:{canonical_device_type(device)}:{device_index}"


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


def create_mock_data(placeholders: dict, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feed_dict = {}
    for name, info in placeholders.items():
        shape = [batch_size if dim is None or dim == -1 else dim for dim in info["shape"]]
        dtype = info["dtype"]
        if dtype == np.float32:
            value = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        elif dtype == np.int32:
            high = max(int(info.get("int_upper_bound") or 100), 1)
            value = rng.integers(0, high, size=shape, dtype=np.int32)
        elif dtype == np.int64:
            high = max(int(info.get("int_upper_bound") or 100), 1)
            value = rng.integers(0, high, size=shape, dtype=np.int64)
        elif dtype == np.bool_:
            value = rng.choice(np.array([False, True], dtype=np.bool_), size=shape)
        elif dtype == np.str_:
            value = np.full(shape if shape else (), "mock", dtype=np.str_)
        else:
            value = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        feed_dict[f"{name}:0"] = value
    return feed_dict


def load_npz_inputs(placeholders: dict, path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        feed = {}
        for index, name in enumerate(placeholders):
            tensor_name = f"{name}:0"
            key = tensor_name if tensor_name in arrays else name if name in arrays else f"input_{index:03d}"
            if key not in arrays:
                raise KeyError(f"{path} 缺少输入 {tensor_name} / {name} / {key}")
            feed[tensor_name] = arrays[key]
        return feed


def load_npy_input(placeholders: dict, path: Path) -> dict[str, np.ndarray]:
    if len(placeholders) != 1:
        raise RuntimeError("--input_npy 只支持单 Placeholder 图，多输入图请使用 --input_npz")
    name = next(iter(placeholders))
    return {f"{name}:0": np.load(path, allow_pickle=False)}


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


def import_graph_with_device_variables(
    tf,
    graph_def,
    feed_dict: dict[str, np.ndarray],
    device_name: str,
    xla: bool,
    xla_jit_scope: bool,
):
    input_map = {}
    variables = []
    with tf.device(device_name):
        for tensor_name, value in feed_dict.items():
            input_name = f"{tensor_name[:-2].replace('/', '_')}_device_variable"
            variable = tf.Variable(
                value,
                trainable=False,
                use_resource=True,
                dtype=tf.as_dtype(value.dtype),
                name=input_name,
            )
            variables.append(variable)
            device_tensor = variable.read_value()
            device_tensor.set_shape(value.shape)
            input_map[tensor_name] = device_tensor

    with ExitStack() as stack:
        stack.enter_context(tf.device(device_name))
        if xla and xla_jit_scope:
            stack.enter_context(tf.xla.experimental.jit_scope(compile_ops=True))
        tf.import_graph_def(graph_def, input_map=input_map, name="")
    return variables


def terminal_tensor_names(graph) -> list[str]:
    skip_ops = {"NoOp", "Assert", "Print", "Const", "Placeholder"}
    candidates = []
    for op in graph.get_operations():
        if op.type in skip_ops or not op.outputs:
            continue
        if all(len(output.consumers()) == 0 for output in op.outputs):
            candidates.extend(output.name for output in op.outputs)
    return candidates


def resolve_output_tensor(graph, output_tensor: str):
    name = output_tensor if ":" in output_tensor else f"{output_tensor}:0"
    try:
        return graph.get_tensor_by_name(name)
    except KeyError as exc:
        candidates = terminal_tensor_names(graph)
        if len(candidates) == 1:
            print(f"[output] {name} not found, use only terminal tensor {candidates[0]}")
            return graph.get_tensor_by_name(candidates[0])
        raise RuntimeError(f"找不到输出张量 {name}; candidates={candidates[:20]}") from exc


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
    config.allow_soft_placement = allow_soft_placement or device_type.lower() == "cpu"
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


def load_tensorflow_musa_plugin(tf, plugin_path: Path) -> None:
    env_plugin_paths = os.environ.get("TF_PLUGGABLE_DEVICE_LIBRARY_PATH", "")
    if str(plugin_path) in env_plugin_paths:
        return
    try:
        from tensorflow.python.framework import load_library as tf_load_library
        loader = getattr(tf_load_library, "load_pluggable_device_library", None)
        if loader is not None:
            loader(str(plugin_path))
            return
    except Exception:
        pass
    tf.load_op_library(str(plugin_path))


def prepare_runtime(args: argparse.Namespace):
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    xla_enabled = args.xla and not args.host_calibration
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        if not xla_enabled:
            os.environ.pop("TF_XLA_FLAGS", None)
        return
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
        if xla_enabled:
            append_env_flag("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
            append_env_flag("TF_XLA_FLAGS", "--tf_xla_min_cluster_size=1")
        else:
            os.environ.pop("TF_XLA_FLAGS", None)
        return

    os.environ["MUSA_VISIBLE_DEVICES"] = str(args.device_index)
    if runtime_mode(args) == "native":
        args.xla = False
        for name in [
            "TENSORFLOW_MUSA_ENABLE_PJRT",
            "TENSORFLOW_MUSA_PJRT_LIBRARY_PATH",
            "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
            "PJRT_NAMES_AND_LIBRARY_PATHS",
            "NEXT_PLUGGABLE_DEVICE_USE_C_API",
            "MUSA_TFNPD_IS_PLUGGABLE_DEVICE",
            "MUSA_PJRT_WAIT_EXECUTE_DONE",
            "MUSA_PJRT_MAX_INFLIGHT_COMPILES",
            "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS",
            "MUSA_PJRT_MAX_INFLIGHT_EXECUTES",
            "TF_XLA_FLAGS",
        ]:
            os.environ.pop(name, None)
        import_tensorflow_musa()
        return

    plugin_path = normalize_path(args.musa_plugin_path)
    bridge_path = normalize_path(args.musa_bridge_path)
    if not plugin_path.is_file():
        raise FileNotFoundError(f"MUSA PJRT plugin 不存在: {plugin_path}")
    os.environ["TENSORFLOW_MUSA_ENABLE_PJRT"] = "1"
    os.environ["TENSORFLOW_MUSA_PJRT_LIBRARY_PATH"] = str(plugin_path)
    os.environ.setdefault("NEXT_PLUGGABLE_DEVICE_USE_C_API", "true")
    os.environ["MUSA_TFNPD_IS_PLUGGABLE_DEVICE"] = "1"
    os.environ["MUSA_PJRT_WAIT_EXECUTE_DONE"] = "1" if args.wait_execute_done else "0"
    os.environ["MUSA_PJRT_MAX_INFLIGHT_COMPILES"] = str(args.musa_pjrt_max_inflight_compiles)
    os.environ["MUSA_PJRT_MAX_INFLIGHT_TRANSFERS"] = str(args.musa_pjrt_max_inflight_transfers)
    os.environ["MUSA_PJRT_MAX_INFLIGHT_EXECUTES"] = str(args.musa_pjrt_max_inflight_executes)
    append_env_entry("TF_PLUGGABLE_DEVICE_LIBRARY_PATH", str(plugin_path))
    append_env_entry("PJRT_NAMES_AND_LIBRARY_PATHS", f"MUSA:{plugin_path}")
    if xla_enabled:
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_use_device_api=true")
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_use_device_api_for_auto_jit=true")
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_min_cluster_size=1")
    else:
        os.environ.pop("TF_XLA_FLAGS", None)

    tf, _ = import_tensorflow_v1()
    plugin_lib = load_dynamic_library(plugin_path)
    load_tensorflow_musa_plugin(tf, plugin_path)
    maybe_call_symbol(plugin_lib, "ForceRegisterMusa")
    if bridge_path.is_file():
        bridge_lib = load_dynamic_library(bridge_path)
        maybe_call_symbol(
            bridge_lib,
            "MusaTf215_RegisterRuntimeFactory",
            restype=ctypes.c_int,
            argtypes=[],
        )


def summarize_result(result: Optional[np.ndarray]) -> dict:
    if result is None:
        return {"fetched": False}
    summary = {"fetched": True, "shape": list(result.shape), "dtype": str(result.dtype)}
    if result.size and np.issubdtype(result.dtype, np.number):
        summary.update(
            {
                "min": float(np.min(result)),
                "max": float(np.max(result)),
                "mean": float(np.mean(result)),
                "finite": bool(np.isfinite(result).all()),
            }
        )
    return summary


def compare_output(result: Optional[np.ndarray], reference_path: Optional[str], atol: float, rtol: float) -> None:
    if reference_path is None:
        return
    if result is None:
        raise RuntimeError("--compare_output_path requires --fetch_output")
    reference = np.load(normalize_path(reference_path), allow_pickle=False)
    if reference.shape != result.shape:
        raise RuntimeError(f"output shape mismatch: current={result.shape}, reference={reference.shape}")
    abs_diff = np.abs(result.astype(np.float64) - reference.astype(np.float64))
    mismatch = int(np.count_nonzero(~np.isclose(result, reference, atol=atol, rtol=rtol)))
    max_abs = float(abs_diff.max(initial=0.0))
    print(f"[compare] max_abs={max_abs:.8e} mismatch={mismatch} atol={atol} rtol={rtol}")
    if mismatch:
        raise RuntimeError(f"output mismatch: mismatch_count={mismatch}, max_abs={max_abs:.8e}")


def start_profiler(args: argparse.Namespace):
    if not args.profiler_range:
        return None
    if args.device == "musa":
        call_musa_profiler_api("musaProfilerStart")
        return lambda: call_musa_profiler_api("musaProfilerStop")
    if args.device == "cuda":
        call_cuda_profiler_api("cudaProfilerStart")
        return lambda: call_cuda_profiler_api("cudaProfilerStop")
    raise RuntimeError("--profiler_range requires musa or cuda")


def measure_session(
    session,
    fetch,
    session_feed,
    args: argparse.Namespace,
    final_device_sync: bool = False,
    delete_result_handle: bool = False,
    keep_result: bool = False,
):
    for _ in range(args.warmup_runs):
        current = session.run(fetch, feed_dict=session_feed)
        if final_device_sync:
            synchronize_device(args.device)
        if delete_result_handle:
            current.delete()

    latencies = []
    result = None
    profiler_stop = start_profiler(args)
    try:
        for index in range(args.num_runs):
            start = time.perf_counter()
            current = session.run(fetch, feed_dict=session_feed)
            if final_device_sync:
                synchronize_device(args.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            if delete_result_handle:
                current.delete()
            elif keep_result:
                result = current
            if args.print_run_times:
                print(f"[run {index + 1:04d}] {elapsed_ms:.4f} ms")
    finally:
        if profiler_stop is not None:
            profiler_stop()
    return latencies, result


def measure_multistream_session(
    session,
    fetch,
    session_feed,
    args: argparse.Namespace,
    querys: int,
    batch_size: int,
    keep_result: bool = False,
):
    warmup_handles = []
    for _ in range(args.warmup_runs):
        for _ in range(args.streams):
            current = session.run(fetch, feed_dict=session_feed)
            if not args.fetch_output:
                warmup_handles.append(current)
    if warmup_handles:
        synchronize_device(args.device)
        for handle in warmup_handles:
            handle.delete()

    work_queue: queue.Queue[int] = queue.Queue()
    for _ in range(querys):
        work_queue.put_nowait(1)
    ready_barrier = threading.Barrier(args.streams + 1)
    start_event = threading.Event()
    errors = []
    error_lock = threading.Lock()
    result_lock = threading.Lock()
    latencies = []
    results = []
    handle_results = []
    profiler_stop = start_profiler(args)

    def worker() -> None:
        try:
            ready_barrier.wait()
            start_event.wait()
            while True:
                try:
                    work_queue.get_nowait()
                except queue.Empty:
                    return
                for _ in range(batch_size - 1):
                    try:
                        work_queue.get_nowait()
                    except queue.Empty:
                        break
                start = time.perf_counter()
                current = session.run(fetch, feed_dict=session_feed)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                with result_lock:
                    latencies.append(elapsed_ms)
                    if keep_result and len(results) == 0:
                        results.append(current)
                    elif not keep_result:
                        handle_results.append(current)
        except BaseException as error:
            with error_lock:
                errors.append(error)

    try:
        threads = [threading.Thread(target=worker) for _ in range(args.streams)]
        for thread in threads:
            thread.start()
        ready_barrier.wait()
        start = time.perf_counter()
        start_event.set()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("one or more TF streams failed") from errors[0]
        if not args.fetch_output:
            synchronize_device(args.device)
        wall_time_ms = (time.perf_counter() - start) * 1000.0
    finally:
        if profiler_stop is not None:
            profiler_stop()
        for handle in handle_results:
            handle.delete()
    return latencies, (results[0] if results else None), wall_time_ms


def build_stats(
    args: argparse.Namespace,
    execution_device: str,
    input_mode: str,
    import_ms: float,
    latencies,
    result,
    final_device_sync: bool,
    wall_time_ms: Optional[float] = None,
    querys: Optional[int] = None,
) -> dict:
    values = np.asarray(latencies, dtype=np.float64)
    multi_stream = wall_time_ms is not None and querys is not None
    runs_per_second = (
        len(values) * 1000.0 / wall_time_ms
        if multi_stream
        else 1000.0 / values.mean()
    )
    samples_per_second = (
        querys * 1000.0 / wall_time_ms
        if multi_stream
        else args.batchsize * 1000.0 / values.mean()
    )
    return {
        "device": args.device,
        "execution_device": execution_device,
        "runtime_mode": runtime_mode(args),
        "input_mode": input_mode,
        "host_calibration": args.host_calibration,
        "device_index": args.device_index,
        "xla": args.xla,
        "xla_jit_scope": args.xla_jit_scope,
        "cuda_disable_tf32": args.cuda_disable_tf32 if args.device == "cuda" else None,
        "fetch_output": args.fetch_output,
        "wait_execute_done": args.wait_execute_done if args.device == "musa" else None,
        "final_device_sync": final_device_sync,
        "profiler_range": args.profiler_range,
        "batchsize": args.batchsize,
        "warmup_runs": args.warmup_runs,
        "num_runs": len(values),
        "requested_num_runs": args.num_runs,
        "import_ms": import_ms,
        "total_ms": float(values.sum()),
        "avg_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "runs_per_second": float(runs_per_second),
        "samples_per_second": float(samples_per_second),
        "us_per_sample": float(1_000_000.0 / samples_per_second),
        "latencies_ms": latencies if args.save_latencies else None,
        "output": summarize_result(result),
        "_result": result,
        "streams": args.streams,
        "querys": querys,
        "wall_time_ms": wall_time_ms,
        "qps": querys * 1000.0 / wall_time_ms if querys and wall_time_ms else None,
        "tf_interop_threads": args.tf_interop_threads,
        "tf_intraop_threads": args.tf_intraop_threads,
        "musa_pjrt_max_inflight_compiles": (
            args.musa_pjrt_max_inflight_compiles if args.device == "musa" else None
        ),
        "musa_pjrt_max_inflight_transfers": (
            args.musa_pjrt_max_inflight_transfers if args.device == "musa" else None
        ),
        "musa_pjrt_max_inflight_executes": (
            args.musa_pjrt_max_inflight_executes if args.device == "musa" else None
        ),
    }


def run_host_benchmark(
    tf, graph_def, feed_dict: dict[str, np.ndarray], args: argparse.Namespace
) -> dict:
    execution_device = "cpu" if args.host_calibration else args.device
    device_name = runtime_device_name(execution_device, args.device_index)
    with tf.Graph().as_default() as graph:
        import_start = time.perf_counter()
        import_graph(
            tf,
            graph_def,
            device_name,
            args.xla and not args.host_calibration,
            execution_device,
            args.xla_jit_scope,
        )
        import_ms = (time.perf_counter() - import_start) * 1000.0
        session_feed = {graph.get_tensor_by_name(name): value for name, value in feed_dict.items()}
        if args.host_calibration:
            with tf.device("/device:CPU:0"):
                feed_identities = [
                    tf.identity(tensor, name=f"host_calibration_feed_{index:03d}")
                    for index, tensor in enumerate(session_feed)
                ]
                with tf.control_dependencies(feed_identities):
                    fetch = tf.constant(0.0, name="host_calibration_result")
            output_tensor = None
        else:
            output_tensor = resolve_output_tensor(graph, args.output_tensor)
            fetch = output_tensor if args.fetch_output else output_tensor.op
        config = session_config(
            tf,
            execution_device,
            args.xla and not args.host_calibration,
            allow_soft_placement=args.allow_soft_placement,
            inter_threads=args.tf_interop_threads,
            intra_threads=args.tf_intraop_threads,
        )
        with tf.Session(graph=graph, config=config) as session:
            print("[devices]")
            for device in session.list_devices():
                print(f"  - {device.name} [{device.device_type}]")
            latencies, result = measure_session(
                session,
                fetch,
                session_feed,
                args,
                keep_result=args.fetch_output and not args.host_calibration,
            )
    return build_stats(
        args, execution_device, "host", import_ms, latencies, result, False
    )


def run_device_benchmark(
    tf, graph_def, feed_dict: dict[str, np.ndarray], args: argparse.Namespace
) -> dict:
    if args.device not in {"musa", "cuda"}:
        raise ValueError("--input_mode device supports --device musa or cuda")
    if args.host_calibration:
        raise ValueError("--host_calibration requires --input_mode host")
    if (args.streams > 1 or args.querys is not None) and args.wait_execute_done:
        raise ValueError("multi-stream mode requires --no-wait_execute_done")

    device_name = runtime_device_name(args.device, args.device_index)
    with tf.Graph().as_default() as graph:
        import_start = time.perf_counter()
        variables = import_graph_with_device_variables(
            tf,
            graph_def,
            feed_dict,
            device_name,
            args.xla,
            args.xla_jit_scope,
        )
        import_ms = (time.perf_counter() - import_start) * 1000.0
        init_op = tf.variables_initializer(variables)
        session_feed = {}
        output_tensor = resolve_output_tensor(graph, args.output_tensor)
        fetch = (
            output_tensor
            if args.fetch_output
            else tf.get_session_handle(output_tensor)
        )
        config = session_config(
            tf,
            args.device,
            args.xla,
            allow_soft_placement=args.allow_soft_placement,
            inter_threads=args.tf_interop_threads,
            intra_threads=args.tf_intraop_threads,
        )
        with tf.Session(graph=graph, config=config) as session:
            print("[devices]")
            for device in session.list_devices():
                print(f"  - {device.name} [{device.device_type}]")

            session.run(init_op)
            synchronize_device(args.device)
            print(
                f"[input] device-resident resource variables: {len(feed_dict)} inputs "
                "(initial transfer excluded from timed runs)"
            )
            multi_stream = args.streams > 1 or args.querys is not None
            final_device_sync = not args.fetch_output and not args.wait_execute_done
            if multi_stream:
                latencies, result, wall_time_ms = measure_multistream_session(
                    session,
                    fetch,
                    session_feed,
                    args,
                    args.querys or args.num_runs * args.batchsize,
                    args.batchsize,
                    keep_result=args.fetch_output,
                )
            else:
                latencies, result = measure_session(
                    session,
                    fetch,
                    session_feed,
                    args,
                    final_device_sync=final_device_sync,
                    delete_result_handle=not args.fetch_output,
                    keep_result=args.fetch_output,
                )
                wall_time_ms = None

    return build_stats(
        args,
        args.device,
        "device",
        import_ms,
        latencies,
        result,
        final_device_sync,
        wall_time_ms=wall_time_ms,
        querys=(args.querys or args.num_runs * args.batchsize)
        if args.streams > 1 or args.querys is not None
        else None,
    )


def run_benchmark(
    tf, graph_def, feed_dict: dict[str, np.ndarray], args: argparse.Namespace
) -> dict:
    if args.input_mode == "device" and not args.host_calibration:
        stats = run_device_benchmark(tf, graph_def, feed_dict, args)
    else:
        stats = run_host_benchmark(tf, graph_def, feed_dict, args)

    result = stats.pop("_result")
    if args.save_output_path:
        if result is None:
            raise RuntimeError("--save_output_path requires --fetch_output")
        save_path = normalize_path(args.save_output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, result)
        stats["save_output_path"] = str(save_path)
    compare_output(result, args.compare_output_path, args.compare_atol, args.compare_rtol)
    return stats


def parse_args() -> argparse.Namespace:
    default_device = os.environ.get("TF215_GRAPH_PERF_DEVICE", "musa")
    parser = argparse.ArgumentParser(description="Standalone TF2.15 GraphDef performance runner")
    parser.add_argument("--device", choices=["cpu", "cuda", "musa"], default=default_device)
    parser.add_argument(
        "--runtime_mode",
        choices=["native", "pjrt"],
        default=os.environ.get("TENSORFLOW_MUSA_DNN_RUNTIME", "pjrt"),
        help="musa 设备执行模式：native 使用 tensorflow_musa 原生 kernel；pjrt 使用 OpenXLA-MUSA PJRT",
    )
    parser.add_argument("--device_index", type=int, default=default_device_index())
    parser.add_argument("--graph_path", default=str(DEFAULT_GRAPH_PATH))
    parser.add_argument("--output_tensor", default=DEFAULT_OUTPUT_TENSOR)
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--querys", type=int)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--input_npz", help="多输入 .npz；key 可为 tensor name/op name/input_000")
    parser.add_argument("--input_npy", help="单输入 .npy")
    parser.add_argument(
        "--input_mode",
        choices=["host", "device"],
        help="默认 MUSA/CUDA 使用 device，CPU 使用 host",
    )
    parser.add_argument("--xla", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xla_jit_scope", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cuda_disable_tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅 CUDA 生效；默认保留 TensorFlow 的 TF32 设置",
    )
    parser.add_argument("--fetch_output", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--host_calibration",
        action="store_true",
        help="复用全部输入 feed，只执行 CPU Identity 调度链，用于跨机器对齐 host 基线",
    )
    parser.add_argument("--wait_execute_done", action=argparse.BooleanOptionalAction)
    parser.add_argument("--warmup_runs", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--print_run_times", action="store_true")
    parser.add_argument(
        "--profiler_range",
        action="store_true",
        help="用 musaProfilerApi/cudaProfilerApi 只包围 warmup 后的正式 runs",
    )
    parser.add_argument("--save_latencies", action="store_true")
    parser.add_argument("--allow_soft_placement", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--musa_plugin_path", default=str(DEFAULT_MUSA_PLUGIN_PATH))
    parser.add_argument("--musa_bridge_path", default=str(DEFAULT_MUSA_BRIDGE_PATH))
    parser.add_argument("--save_output_path")
    parser.add_argument("--compare_output_path")
    parser.add_argument("--compare_atol", type=float, default=1e-5)
    parser.add_argument("--compare_rtol", type=float, default=1e-5)
    parser.add_argument(
        "--json_output",
        help=(
            "输出文件；默认按 runtime_mode 分离为 "
            "tmp/e2e_perf_standalone_<native|pjrt>/perf_report.json，避免 native/PJRT report 互相覆盖"
        ),
    )
    parser.add_argument("--tf_interop_threads", type=int, default=1)
    parser.add_argument("--tf_intraop_threads", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_compiles", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_transfers", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_executes", type=int)
    args = parser.parse_args()
    if args.json_output is None:
        args.json_output = str(REPO_ROOT / f"tmp/e2e_perf_standalone_{runtime_mode(args)}/perf_report.json")
    return args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    ensure_bootstrap_env()
    args = parse_args()
    if args.input_mode is None:
        args.input_mode = "device" if args.device in {"musa", "cuda"} else "host"
    if args.host_calibration:
        args.input_mode = "host"
    if args.fetch_output is None:
        args.fetch_output = args.input_mode == "host"
    if args.wait_execute_done is None:
        args.wait_execute_done = args.input_mode == "host"
    if args.musa_pjrt_max_inflight_executes is None:
        args.musa_pjrt_max_inflight_executes = args.streams if args.streams > 1 else 1
    if (
        args.warmup_runs < 0
        or args.num_runs <= 0
        or args.batchsize <= 0
        or args.streams <= 0
        or args.musa_pjrt_max_inflight_compiles <= 0
        or args.musa_pjrt_max_inflight_transfers <= 0
        or args.musa_pjrt_max_inflight_executes <= 0
        or (args.querys is not None and args.querys <= 0)
    ):
        raise ValueError("warmup/runs/batchsize/streams/querys must be positive")
    if (args.streams > 1 or args.querys is not None) and args.input_mode != "device":
        raise ValueError("multi-stream mode requires --input_mode device")
    if args.device == "cpu" or (args.device == "musa" and runtime_mode(args) == "native"):
        args.xla = False if args.xla is True else args.xla

    total_start = time.perf_counter()
    prepare_runtime(args)
    tf, graph_pb2 = import_tensorflow_v1()
    if args.device == "cuda" and args.cuda_disable_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    if args.device != "cpu":
        visible = [
            item for item in tf.config.list_logical_devices() if item.device_type.upper() == canonical_device_type(args.device)
        ]
        if not visible:
            raise RuntimeError(f"TensorFlow 未发现 {canonical_device_type(args.device)} 设备")

    graph_path = normalize_path(args.graph_path)
    graph_def, placeholders = load_graph_and_placeholders(tf, graph_pb2, graph_path)
    if args.input_npz:
        feed_dict = load_npz_inputs(placeholders, normalize_path(args.input_npz))
        input_source = str(normalize_path(args.input_npz))
    elif args.input_npy:
        feed_dict = load_npy_input(placeholders, normalize_path(args.input_npy))
        input_source = str(normalize_path(args.input_npy))
    else:
        feed_dict = create_mock_data(placeholders, args.batchsize, args.seed)
        input_source = f"mock(seed={args.seed}, batch={args.batchsize})"

    print("[config]")
    print(f"  graph={graph_path}")
    print(f"  output_tensor={args.output_tensor}")
    print(f"  input_source={input_source}")
    print(
        f"  device={args.device}:{args.device_index}, xla={args.xla}, "
        f"xla_jit_scope={args.xla_jit_scope}"
    )
    print(f"  input_mode={args.input_mode}")
    print(f"  host_calibration={args.host_calibration}")
    print(f"  fetch_output={args.fetch_output}, wait_execute_done={args.wait_execute_done}")
    if args.streams > 1 or args.querys is not None:
        querys = args.querys or args.num_runs * args.batchsize
        print(
            f"  warmup={args.warmup_runs}, streams={args.streams}, "
            f"querys={querys}, batch={args.batchsize}"
        )
    else:
        print(f"  warmup={args.warmup_runs}, runs={args.num_runs}, batch={args.batchsize}")

    stats = run_benchmark(tf, graph_def, feed_dict, args)
    stats["graph_path"] = str(graph_path)
    stats["output_tensor"] = args.output_tensor
    stats["input_source"] = input_source
    stats["total_process_ms"] = (time.perf_counter() - total_start) * 1000.0

    print("\n[performance]")
    print(f"  runs:        {stats['num_runs']}")
    print(f"  total_ms:    {stats['total_ms']:.4f}")
    print(f"  avg_ms:      {stats['avg_ms']:.4f}")
    print(f"  min_ms:      {stats['min_ms']:.4f}")
    print(f"  max_ms:      {stats['max_ms']:.4f}")
    print(f"  p50_ms:      {stats['p50_ms']:.4f}")
    print(f"  p95_ms:      {stats['p95_ms']:.4f}")
    print(f"  p99_ms:      {stats['p99_ms']:.4f}")
    print(f"  qps:         {stats['runs_per_second']:.4f}")
    print(f"  samples/s:   {stats['samples_per_second']:.4f}")
    print(f"  us/sample:   {stats['us_per_sample']:.4f}")
    if stats["qps"] is not None:
        print(f"  query_qps:   {stats['qps']:.4f}")
        print(f"  wall_ms:     {stats['wall_time_ms']:.4f}")

    json_path = normalize_path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, indent=2))
    print(f"\n[json] {json_path}")


if __name__ == "__main__":
    main()
