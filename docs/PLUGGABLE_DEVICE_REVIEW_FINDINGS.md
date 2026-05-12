# PluggableDevice 未提交变更 Review 记录

文档索引：`PLUGGABLE_DEVICE_DOCS.md`

日期：2026-05-12
分支：`pluggable_device`
范围：当前工作区未提交修改，重点覆盖默认 PluggableDevice 加载、SE C API runtime、legacy gating、测试改造和 Python loader。

## 结论摘要

当前修改方向整体正确：默认切到 PluggableDevice / StreamExecutor C API，legacy C++ 设备注册改为显式 fallback，并且 stream 解析从不安全的 `GpuStreamHack()` 探测改成优先解包 TensorFlow `CStream::Handle()`。

但在合入前建议优先处理以下问题：

1. `python/_loader.py` 默认路径对同一个 `libmusa_plugin.so` 先调用 `load_pluggable_device_library()`，再调用 `tf.load_op_library()`，需要确认不会造成重复注册或加载状态不一致。
2. `plugin_se_allocate()` 在 `musaSetDevice()` 失败时返回 `opaque == nullptr` 但保留 `size != 0`，不符合“失败分配应表现为空分配”的约定，可能污染 BFC/后续 memcpy 路径。
3. SE 路径按 stream 缓存 muDNN handle，但 stream destroy 时没有清理对应 cache entry，长进程 eager 场景可能泄漏 handle 或复用已销毁 stream 的旧 slot。
4. 新增/修改的 subprocess 测试依赖 `os.getcwd()` 是仓库根目录；按当前文档从 `test/` 目录运行时会找错 `python/` 和 `test/` 路径。
5. `_loader.py` 的 `is_plugin_loaded()` / `get_musa_devices()` 仍用字符串方式检查 `PhysicalDevice` 对象，会吞异常并错误返回未加载/空设备。

## 急需修改项

### 1. 默认 loader 对同一 `.so` 双加载风险

位置：

- `python/_loader.py:119-123`
- `docs/PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md:40-45` 也记录了该设计。

当前默认导入流程是：

1. `load_pluggable_device_library(plugin_path)`
2. `tf.load_op_library(plugin_path)`

TensorFlow 2.6.1 的 `load_pluggable_device_library()` 文档说明：加载 PluggableDevice library 后，library 中通过 StreamExecutor C API 以及 Kernel/Op Registration C API 注册的 devices/kernels 会在 TensorFlow 进程中可用。因此再对同一个 so 调用 `tf.load_op_library()` 需要谨慎：它可能再次走 TensorFlow library loading / op registration 路径，带来重复 kernel/op registration、重复静态初始化、或 loader 状态不一致的风险。

建议：

- 首选方案：验证 `load_pluggable_device_library()` 后是否已经足够注册 kernels；如果 Python custom op wrappers 仍必须通过 `tf.load_op_library()` 获取，建议补一个无硬件 subprocess 测试覆盖“默认 import 不崩溃、不重复注册、可重复 import”。
- 更稳妥方案：把 PluggableDevice/device-kernel plugin 和 Python custom op wrapper library 拆成两个加载单元，避免同一 so 被两种 loader API 加载。
- 短期至少要明确记录该约束，并增加测试防止 `AlreadyExists`/重复注册类回归。

优先级：高。

### 2. `plugin_se_allocate()` 失败状态不符合分配语义

位置：

- `musa_ext/mu/musa_se_plugin.cc:77-103`

问题代码路径：

- 进入 `plugin_se_allocate()` 时先把 `mem->opaque = nullptr`、`mem->size = 0`。
- 如果 `musaSetDevice(ordinal)` 失败，当前代码会设置 `mem->size = size` 后返回。
- 这会产生一个“非零大小但空指针”的 device memory record。

风险：

- StreamExecutor allocation callback 没有 `TF_Status`，调用方通常只能通过 allocation representation 判断成功/失败。
- 非零 `size` 可能让上层 allocator/BFC 或调试路径误判该分配是有意义的。
- 当前 memcpy 系列函数只检查 wrapper 指针，不统一拒绝 `opaque == nullptr` 的非零拷贝，后续可能把空 device pointer 传入 MUSART。

建议：

- `musaSetDevice()` 或 `musaMalloc()` 失败时保持 `opaque == nullptr` 且 `size == 0`。
- 在 `memcpy_dtoh`、`memcpy_htod`、`memcpy_dtod`、同步 memcpy 入口统一检查非零 size 下的 device pointer 是否为空。
- 可保留日志来区分失败原因，不要用 `mem->size` 表达失败原因。

优先级：高。

### 3. SE per-stream muDNN handle 缓存缺少 stream 生命周期清理

位置：

- `musa_ext/mu/musa_runtime_registry.cc:62-100`
- `musa_ext/mu/musa_se_plugin.cc:200-208`

当前 `MusaSeRegistryEnsureMudnnForDevice()` 以 `reinterpret_cast<uintptr_t>(musaStream_t)` 为 key，为每个 stream 创建/cache 一个 muDNN handle。`plugin_se_destroy_stream()` 销毁 MUSART stream 后，没有通知 registry 删除该 stream 对应的 handle slot。

风险：

- 长进程 eager workload 如果频繁创建/销毁 stream，会持续积累 muDNN handle，直到 device destroy 才清理。
- 如果底层 runtime 复用相同 stream handle 值，旧 slot 可能被复用，语义上依赖 `SetStream()` 重新绑定，维护和排查都比较困难。

建议：

- 在 registry 增加 `MusaSeRegistryOnStreamDestroyed(int32_t ordinal, musaStream_t stream)`。
- 在 `plugin_se_destroy_stream()` 中销毁 stream 前或后调用该 hook，删除 `mudnn_by_stream[stream_key]`。
- 增加 lifecycle 单测覆盖：create device → ensure handle → destroy stream → registry slot 被删除。

优先级：高。

### 4. subprocess 测试依赖 cwd，按文档入口会失败

位置：

- `test/ops/pluggable_device_compliance_test.py:47-55`
- `test/ops/pluggable_default_import_test.py:53-62`

当前测试用：

```python
shutil.copytree(os.path.join(os.getcwd(), "python"), package_dir, symlinks=True)
env["PYTHONPATH"] = os.pathsep.join([tmpdir, os.path.join(os.getcwd(), "test"), ...])
```

这只在 cwd 是仓库根目录时成立。项目文档里的常用测试入口是：

```bash
cd tensorflow_musa_extension/test
python test_runner.py --single xxx.py
```

此时 `os.getcwd()` 是 `.../tensorflow_musa_extension/test`，会错误寻找 `test/python`。

建议：

- 从 `__file__` 推导 repo root，而不是用 cwd。
- `_plugin_path()` 已经采用类似策略，可以复用同样思路。

示例：

```python
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(repo_root, "python")
test_dir = os.path.join(repo_root, "test")
```

优先级：高。

### 5. loader health check 对 `PhysicalDevice` 判断错误

位置：

- `python/_loader.py:145-162`
- `python/_loader.py:165-182`

当前逻辑：

```python
for device in tf.config.list_physical_devices():
    if "MUSA" in device:
        return True
```

`device` 是 TensorFlow `PhysicalDevice` 对象，不是字符串；`"MUSA" in device` 会抛异常并被吞掉，导致 `is_plugin_loaded()` 即使在 MUSA device 存在时也可能返回 `False`，`get_musa_devices()` 也会返回空列表。

建议：

- 直接使用 `tf.config.list_physical_devices("MUSA")`。
- 或检查 `device.name` / `device.device_type`。

优先级：中高。该问题不一定影响核心加载，但会削弱默认 PluggableDevice 路径的健康检查和用户可诊断性。

## 从代码可读性角度

### 当前较好的地方

- legacy 与 PluggableDevice 的 env gate 已集中到 `musa_plugin_env.h`，比散落检查更清晰。
- stream 解析被隔离在 `tf_compat::GpuStreamFromTfStream()`，kernel 侧不直接接触 TensorFlow 内部 hack，方向正确。
- `MusaKernelRuntimeView` 统一承载 legacy/pluggable 两条路径的 runtime 视图，可读性比在每个 op 中分支更好。

### 建议改进

- 测试里 staging 临时 package 的逻辑已经在多个文件重复，建议提一个小 helper，但不要过度抽象；至少避免多个地方各自写 cwd 推导。
- `musa_se_plugin.cc` 已经超过千行，后续如果继续增长，可以考虑按 allocator/stream/event/memcpy/test-export/platform-fns 分段拆文件；当前阶段不是必须，但新增逻辑应尽量保持分区清楚。
- 对“同一个 so 双加载”的意图需要在代码附近写清楚，因为这是非显然行为；如果保留，应说明为什么安全以及依赖哪些 TensorFlow 版本行为。

## 从稳定性角度

### 进程加载顺序

`TENSORFLOW_MUSA_USE_LEGACY_DEVICE` 是 load-time gate：library 首次 dlopen 时如果没有设置该 env，C++ `DeviceFactory` 和 C++ `MusaPlatform` 就会被跳过；之后再设置 env 也无法恢复同进程的 legacy 注册。

建议增加测试覆盖：

- 默认 import 后再设置 legacy env 不应声称可切换到 legacy。
- legacy env 在首次 import 前设置时，`SE_InitPlugin()` 返回 `TF_UNIMPLEMENTED` 且 legacy 注册路径可用。
- 重复 import `tensorflow_musa` 不应重复加载或重复注册。

### CPU-only CI 覆盖不足

当前多项测试在没有 MUSA physical device 时 skip。这样能保持 CPU CI 绿色，但也会漏掉默认 import、loader double-load、重复注册等不依赖硬件的回归。

建议至少增加一个 no-hardware 测试：

- 在 fresh subprocess 中 import `tensorflow` + `tensorflow_musa`。
- 如果 driver enum 返回 0 或 relaxed mode 返回 0 devices，测试仍应断言进程退出码为 0，且没有重复注册崩溃。

## 从健壮性角度

### 指针/size 校验

memcpy 系列入口应对以下情况做一致处理：

- wrapper 指针为空。
- host 指针为空且 size 非零。
- device memory wrapper 非空但 `opaque == nullptr` 且 size 非零。
- stream wrapper magic 不匹配。

当前 stream 的 magic 校验已有改善；device memory 的 `opaque` 校验建议补齐。

### ABI / struct_size 兼容

位置：

- `musa_ext/mu/musa_se_plugin.cc:966-975`

当前 `SE_InitPlugin()` 对 `SE_PlatformRegistrationParams::struct_size` 小于当前编译期 `SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE` 直接报 `TF_INVALID_ARGUMENT`。这对“只支持当前目标 TF 版本构建”的策略是可以接受的，但它不是最宽松的 ABI negotiation 行为。

建议：

- 如果项目目标是每个 TF 版本单独构建和测试，则保留也可以，但文档要明确“不承诺一个二进制跨 TF minor 版本”。
- 如果希望更符合 PluggableDevice ABI 习惯，应按实际访问字段的 offset 做兼容判断，而不是要求完整 struct size 等于/大于当前头文件。

## 从 PluggableDevice 规范角度

### 符合方向

- 默认 `SE_InitPlugin()` 返回 `TF_OK` 并填充 `SP_Platform` / `SP_PlatformFns`，legacy 模式下返回明确 `TF_UNIMPLEMENTED`，避免双注册。
- `SP_Stream` 使用本插件 wrapper 并带 magic，TensorFlow `CStream::Handle()` 解包后再取底层 `musaStream_t`，比旧的 `GpuStreamHack()` 猜测式解引用更符合安全边界。
- `get_device_count` 的 strict / relaxed 行为已通过 env 统一，便于 CPU-only CI。

### 需要补齐/确认

- allocation failure representation 应符合 C API 注释：“failure returns nullptr”。不要返回非零 size 的 null allocation。
- `load_pluggable_device_library()` 本身已经声明会使 devices/kernels available；同一个 library 再用 `tf.load_op_library()` 需要测试或拆分设计支撑。
- platform/device/stream executor 的 struct size 初始化是必要的，但对 registration params 的 struct size 检查应与目标 TF 兼容策略一致。

## 建议修复顺序

1. 修复测试 cwd 依赖和 `_loader.py` device helper 判断，成本低、收益高。
2. 修复 `plugin_se_allocate()` 失败状态和 memcpy 空 opaque 校验。
3. 增加 stream destroy 清理 per-stream muDNN handle 的 registry hook。
4. 明确并测试默认 loader 的双加载行为；如果存在重复注册风险，优先改设计。
5. 根据兼容策略决定是否放宽 `SE_PlatformRegistrationParams::struct_size` 检查。

## 建议补充测试

- `test_import_tensorflow_musa_no_hardware_clean_exit`：无设备时也验证默认 import 不崩溃。
- `test_repeated_import_default_path`：fresh subprocess 中连续 import/reload，确认没有重复注册问题。
- `test_legacy_env_must_be_set_before_load`：覆盖 load-time env gate 的行为。
- `test_allocator_failure_returns_empty_allocation`：可通过测试导出函数或 mock/invalid ordinal 覆盖失败 allocation representation。
- `test_stream_destroy_removes_mudnn_slot`：覆盖 per-stream registry 清理。
