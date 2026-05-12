# CI 固定命令（Pluggable / 默认路径）

避免在跑 PluggableDevice 相关测试前通过 `import musa_test_utils` 等方式**提前** `dlopen` 插件；默认路径会在第一次加载 `libmusa_plugin.so` 时走 `SE_InitPlugin`。如需验证迁移期 legacy C++ 路径，必须在第一次加载前设置 `TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1`。

拉取新版本或新增了 `musa_ext/**/*.cc` 后，请先 **`cmake -S . -B <builddir>`** 再 **`cmake --build`**：CMake 当前用 `GLOB_RECURSE` 收集源码，未重新生成构建文件时可能不会把新 `.cc` 编入 `libmusa_plugin.so`。

## 构建

本仓库工程由 **CMake** 生成，`libmusa_plugin.so` 输出在 `build/`（或你指定的 `-B` 目录）：

```bash
cd /path/to/tensorflow_musa_extension
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target musa_plugin
```

（之后在 `build/` 内也可用 `cmake --build . ...`。**务必**在该目录已用 `cmake -S/-B` 生成过 Ninja/Makefile。）

## 默认 PluggableDevice 路径（需 MUSA 设备时由测试 skip）

使用与编译 TensorFlow 一致的 Python（例如 conda env）；并设置 **`PYTHONPATH=test`**：

```bash
cd /path/to/tensorflow_musa_extension
PYTHONPATH=test python -m pytest test/ops/pluggable_default_import_test.py \
  test/ops/pluggable_device_compliance_test.py -q
```

## SE API / 纯子进程（不依赖 `musa_test_utils` 预加载）

```bash
PYTHONPATH=test python -m pytest test/ops/pluggable_se_api_test.py -q
```

## Pluggable 端到端 eager（子进程：`load_pluggable_device_library`、`load_op_library`）

**需要可用的 MUSA 设备**并完成上述构建；驱动报 0 台设备或未导出枚举时跳过；若驱动枚举到设备但 TF 未发现 MUSA 物理设备则**失败**（用于发现 SE 注册问题）。

```bash
PYTHONPATH=test python -m pytest test/ops/pluggable_se_eager_add_test.py -q
```

## 一次性跑以上 Pluggable 相关测试

```bash
PYTHONPATH=test python -m pytest test/ops/pluggable_se_api_test.py \
  test/ops/pluggable_default_import_test.py \
  test/ops/pluggable_device_compliance_test.py \
  test/ops/pluggable_se_eager_add_test.py -q
```
