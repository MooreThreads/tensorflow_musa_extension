# PluggableDevice 全量迁移计划

文档索引：`PLUGGABLE_DEVICE_DOCS.md`

本文档描述 `tensorflow_musa` 从当前 legacy C++ `MusaDevice` 路径完全迁移到 TensorFlow PluggableDevice / StreamExecutor C API 路径的分阶段计划。

## 背景

当前 `libmusa_plugin.so` 同时存在两条设备注册路径：

1. 默认 legacy C++ 路径：通过 `DeviceFactory::Register("MUSA", ...)` 注册自定义 `MusaDevice`。
2. 第一阶段已切换为默认 SE / PluggableDevice 路径：未设置 legacy fallback 时，由 `SE_InitPlugin()` 注册 StreamExecutor C API platform/device。
3. 迁移期 legacy fallback：设置 `TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1` 后才启用 C++ device factory。

后续目标是完全转向 PluggableDevice，但仍保留用户侧入口：

```python
import tensorflow_musa
```

最终 `import tensorflow_musa` 应负责加载 PluggableDevice library 和 custom op library，而不是依赖 C++ static constructor 注册 `MusaDevice`。

## 最终目标状态

- `SE_InitPlugin()` 是唯一设备注册入口。
- `import tensorflow_musa` 默认加载 PluggableDevice，无需用户设置额外环境变量。
- `DeviceFactory::Register("MUSA", ...)` 和 legacy `MusaDevice` 注册路径被删除或仅在短期 fallback 中保留。
- kernel runtime 不再依赖 `MusaDevice*`、`MusaDeviceContext*` 或 `musa_device->mudnn_handle()`。
- stream、allocator、muDNN handle、collective runtime、ResourceVariable 同步语义都来自 PluggableDevice/SE runtime。
- `MUSA_OP_REQUIRES_CPP_MUSA_DEVICE`、`TryGetMusaDeviceFromContext()` 等兼容接口最终删除。

## Phase 0：冻结 legacy 扩张

目标：避免继续扩大双路径兼容成本。

工作项：

- 明确 legacy C++ path 仅为过渡兼容路径。
- 新功能和新 kernel 默认适配 PluggableDevice runtime。
- 标记以下接口为 migration-only / deprecated：
  - `musa_ext/kernels/utils_op.h` 中的 `MUSA_OP_REQUIRES_CPP_MUSA_DEVICE`。
  - `musa_ext/kernels/musa_kernel_runtime.h` 中的 `TryGetMusaDeviceFromContext()`。
- CI 中把 PluggableDevice 相关测试作为迁移门禁。

关键文件：

- `docs/COMPATIBILITY.md`
- `docs/CI.md`
- `musa_ext/kernels/utils_op.h`
- `musa_ext/kernels/musa_kernel_runtime.h`

验收标准：

- 文档明确 PluggableDevice 是后续主线。
- 新增 kernel 不再直接依赖 `MusaDevice`。

## Phase 1：Python 入口改为 PluggableDevice 优先

目标：用户仍使用 `import tensorflow_musa`，但内部优先走 PluggableDevice 加载。

当前 `python/_loader.py` 主要通过：

```python
tf.load_op_library(plugin_path)
```

加载插件。这会注册 custom ops，同时依赖 `.so` static constructor 触发 legacy device 注册。

改造方向：

1. 找到 `libmusa_plugin.so`。
2. 调用 TensorFlow PluggableDevice loader 加载设备插件。
3. 再调用 `tf.load_op_library()` 注册 custom ops。
4. 记录加载状态，避免同一进程内重复注册。

示例结构：

```python
from tensorflow.python.framework.load_library import load_pluggable_device_library

load_pluggable_device_library(plugin_path)
_op_module = tf.load_op_library(plugin_path)
```

短期可以保留 legacy fallback 环境变量，例如：

```text
TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1
```

但该 fallback 不应长期存在。

关键文件：

- `python/_loader.py`
- `python/__init__.py`
- `test/ops/pluggable_se_eager_add_test.py`
- `test/ops/pluggable_device_compliance_test.py`

验收标准：

```python
import tensorflow as tf
import tensorflow_musa

devices = tf.config.list_physical_devices("MUSA")
assert len(devices) > 0
```

且用户不需要手动设置额外环境变量。

## Phase 2：PluggableDevice 成为默认注册路径

目标：`SE_InitPlugin()` 默认成功，legacy C++ device factory 不再是默认路径。

当前行为：

- `musa_ext/mu/device_register.cc` 默认跳过 C++ `DeviceFactory::Register`；仅 `TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1` 时注册。
- `musa_ext/mu/device/musa_platform.cc` 默认跳过 C++ `MusaPlatform` 注册；仅 legacy fallback 时注册。
- `musa_ext/mu/musa_se_plugin.cc` 默认让 `SE_InitPlugin()` 返回 `TF_OK`；legacy fallback 模式下返回 `TF_UNIMPLEMENTED`。

改造步骤：

1. **已实施第一阶段默认切换**：PluggableDevice / SE 现在是默认路径。
2. 如需短期 fallback，改用反向开关，例如：

   ```text
   TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1
   ```

3. `SE_InitPlugin()` 默认完成注册，不再因 env 未设置返回 `TF_UNIMPLEMENTED`。
4. static constructor 默认不执行 `DeviceFactory::Register("MUSA", ...)`。
5. 后续阶段删除 legacy fallback。

关键文件：

- `musa_ext/mu/musa_plugin_env.h`
- `musa_ext/mu/device_register.cc`
- `musa_ext/mu/device/musa_platform.cc`
- `musa_ext/mu/musa_se_plugin.cc`
- `docs/COMPATIBILITY.md`
- `test/ops/pluggable_se_api_test.py`

验收标准：

- 默认 import / load plugin 走 PluggableDevice。
- legacy fallback 只能通过显式迁移开关启用。
- `SE_InitPlugin()` 在默认配置下返回 `TF_OK`。

## Phase 3：修正 PluggableDevice stream 获取机制

目标：stream 获取必须安全、确定，不依赖对未知指针的探测式解引用。

当前风险点：

- `musa_ext/mu/tf_compat/pluggable_tf_compat.cc` 中 `ResolveStreamFromHack()` 会将 `GpuStreamHack()` 返回值解释为 `SP_Stream_st*` 并读取 `magic`。
- 但注释同时说明 `GpuStreamHack()` 可能返回裸 `musaStream_t`，此时解引用可能导致 crash。

改造原则：

1. 优先使用 `GpuStreamMemberHack()`。
2. 如果只能使用 `GpuStreamHack()`，必须明确其返回语义。
3. 不对未知 `void*` 做 `wrapper->magic` 这类非法地址探测。
4. 如果 TensorFlow 某版本保证 SE stream 返回 `SP_Stream_st*`，该保证必须由明确版本/类型路径判断支撑。
5. 无法安全解析 stream 时返回 `nullptr`，由上层报 `Unimplemented` 或 `Internal`，不能崩溃。

关键文件：

- `musa_ext/mu/tf_compat/pluggable_tf_compat.cc`
- `musa_ext/mu/tf_compat/pluggable_tf_compat.h`
- `musa_ext/mu/musa_plugin_sp_stream.h`
- `musa_ext/kernels/musa_kernel_runtime.cc`

验收标准：

- 裸 `musaStream_t` fallback 不会被当作 host pointer 解引用。
- PluggableDevice eager / graph op 能稳定取得 stream。
- stream 获取失败时错误可诊断，不出现 segfault。

## Phase 4：kernel runtime 完全迁移到 PluggableDevice

目标：所有 kernel 通过 PluggableDevice runtime 获取 device id、stream、allocator 和 muDNN handle。

当前 `MusaKernelRuntimeView` 仍有双路径：

- legacy：从 `MusaDevice` 获取 stream / muDNN handle。
- PluggableDevice：从 `DeviceContext` 和 `musa_runtime_registry` 获取 stream / muDNN handle。

最终结构应收敛为：

```cpp
struct MusaKernelRuntimeView {
  int device_id = -1;
  musaStream_t stream = nullptr;
  Allocator* allocator = nullptr;
  ::musa::dnn::Handle* mudnn_handle = nullptr;
};
```

迁移步骤：

1. 所有 kernel 统一使用：
   - `QueryMusaKernelRuntimeView(ctx)`
   - `GetMusaStreamForKernelContext(ctx)`
   - `GetMusaDeviceIdForKernelContext(ctx)`
   - `GetHandleByCtx(ctx)` 或后续替代接口。
2. 移除 kernel 内直接依赖：
   - `MusaDevice*`
   - `MusaDeviceContext*`
   - `musa_device->GetStream()`
   - `musa_device->mudnn_handle()`
   - `musa_device->GetMemMaintainer()`
3. 所有 muDNN-backed kernel 在获取 handle 前必须有统一 guard，例如：
   - `MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx)`。
4. 当前依赖 `MUSA_OP_REQUIRES_CPP_MUSA_DEVICE(ctx)` 的 op 需要逐个迁移或明确 unsupported。

建议迁移顺序：

1. Elementwise / shape / memcpy 类：Add、Cast、Identity、Reshape、ZerosLike。
2. muDNN-backed math / nn：MatMul、Conv2D、Softmax、Reduce、BiasAdd。
3. fusion op：LayerNorm、GELU、MatMul fusion 等。
4. stateful / ResourceVariable op。
5. collective / distribute。

关键文件：

- `musa_ext/kernels/musa_kernel_runtime.h`
- `musa_ext/kernels/musa_kernel_runtime.cc`
- `musa_ext/kernels/utils_op.h`
- `musa_ext/kernels/state/musa_resource_variable_op.cc`

验收标准：

- kernel 中不再需要 `TryGetMusaDeviceFromContext()`。
- muDNN kernel 在 PluggableDevice 路径下失败时返回清晰错误，不使用 sink handle 继续执行。
- 常见 eager 和 graph op 在 PluggableDevice 下通过。

## Phase 5：完善 allocator / memory / host callback 语义

目标：PluggableDevice 内存路径具备生产可用的错误传播和测试覆盖。

当前基础：

- `musa_ext/mu/musa_se_plugin.cc` 中 `SP_Platform::use_bfc_allocator = 1`，由 TensorFlow PluggableDevice 侧 BFC 包装插件 allocator。

需要补强：

1. `plugin_se_deallocate()` 在 `musaSetDevice()` 失败时不应静默清空 `mem->opaque`。
2. `plugin_se_get_allocator_stats()` 目前未提供有效统计。
3. `plugin_se_device_memory_usage()` 需要稳定覆盖成功和失败路径。
4. host memory / device memory allocation 需要压力测试。
5. OOM 需要返回可诊断错误。

关键文件：

- `musa_ext/mu/musa_se_plugin.cc`
- `test/ops/pluggable_se_api_test.py`

建议新增测试：

- 多次 allocate/free。
- 大 tensor 分配。
- BFC reuse。
- OOM 错误传播。
- host callback 执行顺序。

验收标准：

- allocator 压力测试通过。
- deallocate 失败不会隐藏潜在 leak。
- TensorFlow OOM / allocation failure 能得到明确错误。

## Phase 6：完善 muDNN handle registry

目标：handle 生命周期、并发和 stream 绑定语义明确。

当前 registry 按 device ordinal 和 stream 维护 muDNN handle：

- `musa_ext/mu/musa_runtime_registry.cc`
- `musa_ext/mu/musa_runtime_registry.h`

风险点：

- `MusaSeRegistryEnsureMudnnForDevice()` 返回裸 handle 指针。
- device destroy 时会 clear registry。
- 生命周期依赖 TensorFlow device teardown 一定晚于 kernel 完成。
- 全局锁内执行 handle 创建和 `SetStream()`，可能成为热路径竞争点。

改造建议：

1. 在头文件中明确 lifetime contract：
   - handle 仅在 device live 且当前 kernel 调用期间有效。
   - kernel 不允许缓存 handle。
   - device destroy 前必须无并发 kernel 使用。
2. 降低锁粒度，避免在全局锁内做慢速初始化。
3. 清理或实现 `init_failed` 语义。
4. 增加多 stream / 多 device 测试。

关键文件：

- `musa_ext/mu/musa_runtime_registry.h`
- `musa_ext/mu/musa_runtime_registry.cc`
- `musa_ext/kernels/musa_kernel_runtime.cc`

验收标准：

- 多 stream 下不会共享错误的 muDNN stream 状态。
- device destroy 后 registry 清理正确。
- kernel 不缓存 registry 返回的 handle。

## Phase 7：补齐 device 属性和 placement 信息

目标：PluggableDevice 向 TensorFlow 提供足够准确的设备属性。

当前部分属性为 stub：

- `plugin_get_numa_node()` 返回 `-1`。
- `plugin_get_memory_bandwidth()` 返回 `-1`。
- `plugin_get_gflops()` 返回 `-1.0`。
- `pci_bus_id` 为空或 stub。

需要补齐：

- PCI bus id。
- memory bandwidth。
- GFLOPS / compute capability 等性能信息。
- NUMA node。
- physical device description。

关键文件：

- `musa_ext/mu/musa_se_plugin.cc`
- `musa_ext/mu/musa_runtime_registry.cc`

验收标准：

- `tf.config.list_physical_devices("MUSA")` 日志和设备描述可诊断。
- TensorFlow placement / cost model 不再依赖大量未知属性。

## Phase 8：测试体系升级

目标：从 smoke 级验证升级到迁移门禁级验证。

当前已有测试：

- `test/ops/pluggable_se_api_test.py`
- `test/ops/pluggable_se_eager_add_test.py`
- `test/ops/pluggable_device_compliance_test.py`
- `test/ops/pluggable_tf_distribute_smoke_test.py`

建议新增或扩展：

### 8.1 import 入口测试

验证：

```python
import tensorflow_musa
tf.config.list_physical_devices("MUSA")
```

无需设置额外环境变量。

### 8.2 PluggableDevice eager 基础 op

覆盖：

- Add
- Cast
- Identity
- Reshape
- ZerosLike

### 8.3 PluggableDevice muDNN op

覆盖：

- MatMul
- Conv2D
- Softmax
- ReduceSum

### 8.4 graph mode

覆盖：

```python
@tf.function
def f(x, y):
    return ...
```

### 8.5 ResourceVariable

覆盖：

```python
v = tf.Variable(...)
v.assign(...)
v.read_value()
```

### 8.6 allocator 压力

覆盖：

- 大 tensor 分配。
- 反复分配释放。
- OOM。
- 多 stream 并发。

### 8.7 distribute / collective

当前 `pluggable_tf_distribute_smoke_test.py` 会把 collective / NCCL gap 转成 skip。最终应收敛为：

- 明确 supported 并 pass；或
- 明确 unsupported，文档和测试都体现该限制。

验收标准：

- PluggableDevice 默认路径有独立 CI 门禁。
- 至少覆盖 eager、graph、muDNN、stateful、allocator 基础路径。

## Phase 9：删除 legacy `MusaDevice` 注册路径

目标：完成单路径 PluggableDevice 架构收敛。

删除或收敛对象：

- `musa_ext/mu/device_register.cc` 中的 `DeviceFactory::Register("MUSA", ...)`。
- 自定义 `MusaDevice` 作为 TensorFlow device 的注册路径。
- kernel 中的 legacy 依赖：
  - `TryGetMusaDeviceFromContext()`
  - `MUSA_OP_REQUIRES_CPP_MUSA_DEVICE`
  - `MusaDeviceContext`
  - `musa_device->GetStream()`
  - `musa_device->mudnn_handle()`
- 文档中 legacy 默认路径说明。

删除前置条件：

- `import tensorflow_musa` 默认 PluggableDevice 通过。
- eager 基础 op 通过。
- graph mode 通过。
- muDNN op 通过。
- ResourceVariable 通过。
- allocator 压力测试通过。
- 至少一个单卡模型 inference 通过。
- distribute / collective 有明确支持状态。

验收标准：

- 删除 legacy path 后主测试矩阵仍通过。
- 用户入口仍保持 `import tensorflow_musa`。
- 仓库中不再存在双设备注册路径。

## 推荐里程碑

### Milestone A：PluggableDevice 默认加载

验收：

```python
import tensorflow_musa
tf.config.list_physical_devices("MUSA")
```

无需设置额外环境变量。

### Milestone B：基础 op 不依赖 legacy

验收：

- Add / Cast / Reshape / ZerosLike / Identity 在 eager 和 graph 下通过。

### Milestone C：muDNN op 可用

验收：

- MatMul / Conv2D / Softmax / ReduceSum 在 PluggableDevice 下通过。
- 所有 muDNN kernel 都有统一 handle guard。

### Milestone D：stateful op 可用

验收：

- `tf.Variable`。
- `assign`。
- `read_value`。
- optimizer step 的最小训练 case。

### Milestone E：legacy path 删除

验收：

- 删除 `DeviceFactory::Register("MUSA")` 后主测试矩阵仍通过。
- `import tensorflow_musa` 用户入口不变。

## 近期优先级

建议优先执行以下 5 项：

1. 修复 `GpuStreamHack()` fallback 的不安全解引用。
2. 让 `import tensorflow_musa` 支持 PluggableDevice 优先加载。
3. 默认路径已经是 PluggableDevice；后续移除历史实验开关语义与文档残留。
4. 系统性给 muDNN kernel 加 `MUSA_OP_REQUIRES_MUDNN_HANDLE` 或等价强制错误处理。
5. 新增 PluggableDevice MatMul / Softmax / graph mode 测试。

这 5 项完成后，PluggableDevice 路径才适合作为默认路径继续扩展。