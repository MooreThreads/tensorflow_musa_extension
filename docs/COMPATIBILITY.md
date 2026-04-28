# TensorFlow MUSA 插件：版本与构建矩阵

本文说明 `tensorflow_musa` 扩展与 **TensorFlow**、**Python** 及 **ABI** 的对应关系。插件通过 CMake 与当前环境中的 TensorFlow 头文件/链接标志绑定，**必须针对要运行的 TensorFlow 版本重新构建**同一份 `libmusa_plugin.so`（或随 wheel 打包的库）。

## 当前声明的支持矩阵

| 维度 | 说明 |
|------|------|
| TensorFlow | **主要验证**：`2.6.1`。中期目标在独立分支/独立 wheel 中支持 `2.8.x`、`2.10.x` 等，**每个 TF 大版本/次版本一一对应一版预编译或源码构建产物**。 |
| Python | 与 TensorFlow 官方 manylinux wheel 一致，典型为 **3.7 / 3.8 / 3.9**（以目标 TF 实际发布为准）。 |
| C++ ABI | 与 **pip 安装的 TensorFlow** 使用相同的 `_GLIBCXX_USE_CXX11_ABI`：当前 CMake 强制 **`0`** 以与常见 manylinux 轮子对齐。若官方轮子改为 `1`，须同步调整并重建。 |
| MUSA SDK | 由本仓库 `MUSA_PATH`（默认 `/usr/local/musa`）及对应 `musa` / `mublas` / muDNN 版本约束，与 TensorFlow 版本独立但需在 CI 中固定。 |

## Wheel 与依赖策略

- **包名**：`tensorflow_musa`；动态库为 `libmusa_plugin.so`（随 `python/` 安装）。
- **不在 `install_requires` 中声明 `tensorflow`**：避免在构建/安装 wheel 时由 pip 拉取错误版本。用户须先 `pip install tensorflow==<与构建时相同>`。
- **目标 TensorFlow 版本**：
  - 通过环境变量 **`TENSORFLOW_MUSA_TARGET_TF`** 指定，逗号分隔，例如 `2.6.1` 或 `2.6.1,2.8.0`；**`setup.py` 校验 `tf.__version__` 须在该集合内**（未设置时默认 `2.6.1`）。
  - **逗号列表不是“一个 wheel 同时兼容多个已安装 TensorFlow 小版本”**：它只是在 **构建/安装** 时校验 `tf.__version__` 是否被允许。每个预编译/源码产物仍应 **在对应 TensorFlow 版本下编译并测试**；跨 minor 复用同一份 `libmusa_plugin.so` 会触发 **头文件 `SP_*` 结构、ABI、行为** 不一致风险。
- 多版本共仓库时，建议 **为每个 TF 版本打独立 tag 的 wheel 文件名或 build tag**，例如 `+tf261`，并在发布说明中写明。

## 构建时 TensorFlow 来源

- CMake 通过 `python3 -c "import tensorflow; tf.sysconfig.get_include()..."` 获取路径，因此 **构建用的 Python/TensorFlow 即为运行时目标**；混用会触发 ABI/头文件不一致和难以排查的 crash。

## PluggableDevice C API 与 C++ 设备路径（互斥）

同一份 `libmusa_plugin.so` 同时包含 **StreamExecutor C API**（`SE_InitPlugin`）与 **历史 C++ 路径**（`DeviceFactory::Register("MUSA", ...)` + 供 C++ StreamExecutor 使用的 `MusaPlatform`）。两者都向 TensorFlow 注册名为 `MUSA` 的 device/platform，**不能在同一次进程加载里各跑一遍**。

**生产/默认路径**仍是 C++ 注册 + `MusaDevice`：这是当前算子、allocator、muDNN handle 的完整实现所在。

**`MUSA_ENABLE_SE_PLUGIN=1`（SE-only，实验性）** 只注册 StreamExecutor C API 侧设备，**不**再注册 C++ `DeviceFactory` 与 C++ `MusaPlatform`。该路径用于验证 TensorFlow PluggableDevice 发现与 `SE_InitPlugin` 契约；**算子层仍要求 `MusaDevice`（C++ 路径）**，在纯 Pluggable 设备上执行本仓库的 MUSA kernel 尚未全量打通（见下节 “kernel 与设备实现”）。

**重要：`MUSA_ENABLE_SE_PLUGIN` 在 `.so` 的静态初始化阶段读取。** 必须在**第一次** `dlopen` / `import` 该插件**之前**在环境中设为 `1`。进程内先 `tf.load_op_library` 或 `import tensorflow_musa` 再改环境变量**无效**。

| 环境变量 | 含义 |
|----------|------|
| **未设置或 `MUSA_ENABLE_SE_PLUGIN` ≠ `1`**（默认） | 走 **C++ 注册**：`tf.load_op_library` / `import tensorflow_musa` 与当前行为一致。此时若 TensorFlow 仍调用 `SE_InitPlugin`，实现会返回 **`TF_UNIMPLEMENTED`**（仅在使用 `tensorflow-plugins` 等会强调 SE 的场景下出现；正常只走 `load_op_library` 时不会调用）。 |
| **`MUSA_ENABLE_SE_PLUGIN=1`** | 走 **仅 Pluggable SE 路径**：**不**执行 C++ `DeviceFactory::Register` 与 C++ `RegisterPlatform("MUSA")`，由 `SE_InitPlugin` 完成注册。适用于将 `.so` 放入官方 Pluggable 插件目录、由 TensorFlow 按 C API 加载的场景。 |

### kernel 与设备实现

本仓库的 MUSA 算子通过 `GetHandleByCtx` / `GetMusaStreamByCtx`（内部经 `QueryMusaKernelRuntimeView`）从 **`MusaDevice`** 取 **muDNN**；从 **`MusaKernelRuntimeView`** 取 **stream** 与 **device_id**。C++ 默认路径上 `MusaDevice` 非空；**SE/Pluggable 路径**上 `musa_device` 可能为空：**stream 仅从当前 `DeviceContext` 对应的 `StreamExecutor::GpuStreamHack()` 解析**，不做按 ordinal 猜测（多 stream 无法用单句柄安全表示）。依赖 **muDNN** 的算子仍需要 `musa_device` 或后续接线的 `mudnn_handle`（未迁移路径应返回 `Unimplemented`）。runtime registry（`musa_runtime_registry`）仅跟踪 **device 生命周期** hook，不包含 per-stream 调度表。

| 环境变量 | 含义 |
|----------|------|
| **`MUSA_STRICT_DEVICE_ENUM=1`** | 默认 C++ 路径的 `ListPhysicalDevices` / `CreateDevices` 与 **`SE_InitPlugin` 下 `get_device_count`** 在 `musaGetDeviceCount` 失败时均返回 **错误**（便于排障）。 |
| **未设置** | 上述路径在 `musaGetDeviceCount` 失败时均返回 **OK**（空列表、0 台设备、不创建 MUSA `Device`），便于无 MUSA/无驱动的 CI。 |

**Pluggable 路径上的内存**：`SP_Platform::use_bfc_allocator` 置为 **true**，使 TensorFlow Pluggable 侧可用 BFC 包装插件的 `allocate/deallocate`；与现有 `MusaDevice` 内 BFC **仍属两条可能路径**，同一进程不要混用两种注册方式。

**算子主路径**：默认仍以 C++ `REGISTER_KERNEL_BUILDER` 静态注册为准；与上表设备注册方式独立，但需与构建时 TensorFlow 版本一致。

**Kernel 按类别扩大覆盖（SE-only 迁移顺序）**：

1. Elementwise 自定义快路径（`AddV2` 等，见 `MUSA_ADDV2_ENABLE_CUSTOM_KERNEL`）— 已具备从 `MusaKernelRuntimeView` 取 stream 的通路。
2. 形状/拷贝类（`Identity`、`Reshape`、`ZerosLike` 等）— 同样优先绕开 muDNN。
3. **muDNN** 类算子 — 需 `MusaKernelRuntimeView::mudnn_handle` 在 Pluggable 侧完整提供后再迁移。
4. **Stateful / ResourceVariable** 算子 — 最后迁移（同步与生命周期风险最高）。

## 发布与验证矩阵（按 TensorFlow 次版本门控）

| TensorFlow 版本 | 构建/测试 | `SE_InitPlugin` / `MUSA_ENABLE_SE_PLUGIN` | 枚举 / `MUSA_STRICT_DEVICE_ENUM` | 驱动/memcpy：`MUSA_TestSeRuntimeSmoke`（宽松）/`MUSA_TestSeRuntimeSmokeStrict`、`MUSA_TestRegistryDeviceLifecycle` | SE-only eager `AddV2`（`load_pluggable`） | 全量算子（C++ `MusaDevice`） |
|-----------------|-----------|---------------------------------------------|------------------------------------|----------------------------------------|------------------------------|----------------------------------|
| **2.6.1**（主验证） | 仓库内 `cmake` **重新 configure**（见 CI）构建 `libmusa_plugin.so` + `pytest` | 覆盖：`test/ops/pluggable_se_api_test.py` | `MUSA_TestPluginGetDeviceCount` | 宽松：`MUSA_TestSeRuntimeSmoke`；严格：`MUSA_TestSeRuntimeSmokeStrict`（无 GPU 时为非 `TF_OK`）；`MUSA_TestRegistryDeviceLifecycle` | `pluggable_se_eager_add_test`（**需 GPU**，无设备 skip） | 现有 kernel 测试矩阵 |
| **2.8.x / 2.10.x 等** | **须单独分支/单独预编译**；勿跨 minor 复用同一份 `libmusa_plugin.so` | 计划项 | 计划项 | 计划项 | 计划项 | 同目标版本完整回归 |

详细固定 CI 命令见 **[`docs/CI.md`](CI.md)**。
