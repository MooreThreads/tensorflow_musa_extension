# PluggableDevice 迁移开发日志

文档索引：`PLUGGABLE_DEVICE_DOCS.md`

日期：2026-05-12
分支：`pluggable_device`

## 背景

本轮工作围绕 TensorFlow MUSA 插件的 PluggableDevice 规范改造展开。目标不是让同一个 `libmusa_plugin.so` 或 wheel 二进制兼容所有 TF2.x 版本，而是让源码和构建流程支持针对不同 TF2.x 目标版本分别构建、测试和发布。

当前第一阶段聚焦于：默认 PluggableDevice 加载、安全 stream 获取、最小 eager Add 样板、构建脚本和文档测试同步。

## 已完成内容

### 1. 默认设备注册路径切换

- 默认路径切换为 PluggableDevice / StreamExecutor C API。
- legacy C++ `MusaDevice` 路径改为显式 fallback：

  ```bash
  TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1
  ```

- 未设置 legacy fallback 时：
  - 跳过 C++ `DeviceFactory::Register("MUSA", ...)`。
  - 跳过 C++ `MusaPlatform` 注册。
  - `SE_InitPlugin()` 默认返回 `TF_OK` 并完成 SE platform/device 注册。
- 设置 `TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1` 时：
  - 启用 legacy C++ `MusaDevice` 注册路径。
  - `SE_InitPlugin()` 返回明确的 `TF_UNIMPLEMENTED`，避免同进程双注册。

涉及文件：

- `musa_ext/mu/musa_plugin_env.h`
- `musa_ext/mu/device_register.cc`
- `musa_ext/mu/device/musa_platform.cc`
- `musa_ext/mu/musa_se_plugin.cc`

### 2. Python loader 改为 PluggableDevice 优先

`python/_loader.py` 已调整为默认：

1. 查找 `libmusa_plugin.so`。
2. 调用 TensorFlow 的 `load_pluggable_device_library(plugin_path)` 加载 PluggableDevice。
3. 再调用 `tf.load_op_library(plugin_path)` 注册 custom op wrappers。
4. 记录加载状态，避免重复加载。

legacy fallback 模式下仍保留旧的 `tf.load_op_library()` 行为。

涉及文件：

- `python/_loader.py`

### 3. stream 获取安全边界修复

修复了之前对未知 `GpuStreamHack()` 返回值做 `SP_Stream_st*` 探测式解引用的风险。

当前策略：

- 对 TensorFlow PluggableDevice 路径，优先判断 `stream_executor::CStream`，通过 `CStream::Handle()` 获取 `SP_Stream`。
- 仅在确认是本插件创建的 `SP_Stream` 且 magic 匹配时取出底层 `musaStream_t`。
- legacy 路径 fallback 到 `GpuStreamMemberHack()`。
- 不再对未知 `GpuStreamHack()` 返回的 `void*` 做解引用。
- 无法安全解析 stream 时返回 `nullptr`，由上层返回可诊断错误，而不是 segfault。

涉及文件：

- `musa_ext/mu/tf_compat/pluggable_tf_compat.cc`
- `musa_ext/mu/tf_compat/pluggable_tf_compat.h`

### 4. eager Add 最小样板修复

Add fast path 现在在获取 stream 后做空指针检查；stream 为空时不再 launch 到默认 stream。

涉及文件：

- `musa_ext/kernels/math/musa_add_op.cc`

### 5. runtime / error message 同步

- legacy-only kernel 的错误文案改为提示 `TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1`。
- ResourceVariable 路径仍未迁移到 PluggableDevice，但错误文案已明确说明该 op 目前仍需要 `MusaDeviceContext`。

涉及文件：

- `musa_ext/kernels/utils_op.h`
- `musa_ext/kernels/state/musa_resource_variable_op.cc`
- `musa_ext/mu/musa_runtime_registry.h`

### 6. 构建策略同步

- `build.sh` 不再硬编码只允许 TensorFlow `2.6.1`，改为使用 `TENSORFLOW_MUSA_TARGET_TF` allowlist。
- 默认 allowlist 仍是 `2.6.1`。
- `CMakeLists.txt` 改为使用传入的 Python executable 查询 TensorFlow sysconfig，而不是固定 `python3`。

涉及文件：

- `build.sh`
- `CMakeLists.txt`
- `setup.py` 仍保留目标 TF 版本校验思路。

### 7. 测试更新

新增或更新了以下测试：

- `test/ops/pluggable_default_import_test.py`
- `test/ops/pluggable_device_compliance_test.py`
- `test/ops/pluggable_se_api_test.py`
- `test/ops/pluggable_se_eager_add_test.py`
- `test/ops/pluggable_tf_distribute_smoke_test.py`

重点调整：

- 默认测试不再依赖 `MUSA_ENABLE_SE_PLUGIN=1`。
- 默认 import/compliance 测试使用临时 `tensorflow_musa` package，并复制当前 `build/libmusa_plugin.so`，避免误加载 conda/site-packages 中已安装的旧 wheel。
- `pluggable_device_compliance_test.py` 不再在模块导入阶段通过 `musa_test_utils` 抢先加载插件，避免旧包和当前构建产物混用。

### 8. 文档同步

更新了 PluggableDevice 默认路径、legacy fallback、TF 版本兼容策略和测试建议。

涉及文件：

- `docs/PLUGGABLE_DEVICE_MIGRATION_PLAN.md`
- `docs/COMPATIBILITY.md`
- `docs/CI.md`
- `docs/KERNEL_EXPANSION.md`

## 已验证结果

验证环境：

- conda env：`tf261`
- TensorFlow：`2.6.1`
- GPU 测试前缀：`MUSA_VISIBLE_DEVICES=5`

已通过命令：

```bash
conda run -n tf261 ./build.sh release
```

```bash
python3 -m py_compile \
  test/ops/pluggable_device_compliance_test.py \
  test/ops/pluggable_default_import_test.py \
  test/ops/pluggable_se_api_test.py \
  test/ops/pluggable_se_eager_add_test.py \
  test/ops/pluggable_tf_distribute_smoke_test.py
```

```bash
git diff --check
```

```bash
conda run -n tf261 bash -lc 'MUSA_VISIBLE_DEVICES=5 python test/ops/pluggable_default_import_test.py'
```

```bash
conda run -n tf261 bash -lc 'PYTHONPATH=test MUSA_VISIBLE_DEVICES=5 python test/ops/pluggable_se_api_test.py'
```

```bash
conda run -n tf261 bash -lc 'PYTHONPATH=test MUSA_VISIBLE_DEVICES=5 python test/ops/pluggable_se_eager_add_test.py'
```

```bash
conda run -n tf261 bash -lc 'MUSA_VISIBLE_DEVICES=5 python test/ops/pluggable_device_compliance_test.py'
```

结果：以上构建、格式检查、Python 编译检查和四个核心 PluggableDevice 测试均通过。

## 当前已知限制

1. 目前只完成第一阶段最小闭环，不代表全量 kernel 已迁移。
2. 大量 kernel 仍依赖 legacy `MusaDevice` / `MusaDeviceContext`。
3. ResourceVariable 路径仍不支持 PluggableDevice，只是错误路径已更清晰。
4. muDNN-backed kernel 仍需要逐个迁移和验证，例如：
   - MatMul
   - Conv2D
   - Softmax
   - ReduceSum
5. graph mode、allocator 压力、OOM、host callback、collective/distribute 还没有生产级覆盖。
6. legacy C++ 路径尚未删除，只是变成显式 fallback。
7. 多 TF2.x 版本尚未实际构建验证；目前主验证仍是 TF `2.6.1`。
8. 尚未运行全量 operator 测试矩阵。
9. 本轮没有创建 commit。

## 建议 review 重点

请重点 review 以下方向：

1. 默认路径切换是否会影响现有用户导入行为。
2. `TENSORFLOW_MUSA_USE_LEGACY_DEVICE=1` 的 fallback 边界是否清晰。
3. `SE_InitPlugin()` 默认成功、legacy 模式返回 `TF_UNIMPLEMENTED` 的语义是否合理。
4. `python/_loader.py` 中先 `load_pluggable_device_library()` 后 `tf.load_op_library()` 是否符合目标 TensorFlow 版本行为。
5. `pluggable_tf_compat.cc` 中对 `stream_executor::CStream` 的 `dynamic_cast` 和 `SP_Stream` magic 校验是否足够稳妥。
6. Add fast path 在 stream 获取失败时 fallback/报错路径是否符合预期。
7. 临时 package 测试策略是否适合长期 CI。
8. 构建脚本的 `TENSORFLOW_MUSA_TARGET_TF` allowlist 是否满足后续多 TF 版本构建流程。

## 后续建议任务

### P0：review 后修正

- 根据代码 review 意见修正第一阶段实现。
- 再次运行当前四个核心测试和 `git diff --check`。

### P1：扩大 PluggableDevice kernel 覆盖

建议按以下顺序迁移：

1. Elementwise / shape / memcpy 类：
   - Cast
   - Identity
   - Reshape
   - ZerosLike
   - Sub / Mul / unary elementwise
2. muDNN-backed math / nn：
   - MatMul
   - Conv2D
   - Softmax
   - ReduceSum
   - BiasAdd
3. fusion op：
   - LayerNorm
   - GELU
   - MatMul fusion
4. stateful / ResourceVariable op。
5. collective / distribute。

### P2：ResourceVariable 迁移

当前 ResourceVariable 仍要求 legacy `MusaDeviceContext`。后续需要设计 PluggableDevice 下的：

- assign
- read_value
- copy-on-read
- stream ordering
- resource lifetime
- allocator 语义

### P3：测试矩阵补强

需要新增或补强：

- graph mode 测试。
- muDNN-backed op 的 PluggableDevice 测试。
- allocator 多次 allocate/free。
- OOM 可诊断错误。
- 多 stream 并发。
- ResourceVariable 最小训练 case。
- distribute / collective 明确 supported 或 unsupported。

### P4：多 TensorFlow 版本验证

按目标版本分别构建和测试，例如：

- TF 2.6.1
- TF 2.8.x
- TF 2.10.x

每个版本都应独立构建 `libmusa_plugin.so` / wheel，不跨 TF minor 复用同一二进制产物。

### P5：legacy 路径删除

只有在 PluggableDevice 路径覆盖 eager、graph、muDNN、ResourceVariable、allocator、模型 smoke 之后，再删除：

- C++ `DeviceFactory::Register("MUSA", ...)`
- C++ `MusaPlatform` 注册路径
- kernel 中直接依赖 `MusaDevice` / `MusaDeviceContext` 的接口
- `MUSA_OP_REQUIRES_CPP_MUSA_DEVICE`
- `TryGetMusaDeviceFromContext()`

## 工作区注意事项

当前工作区包含本轮修改文件，也包含一些原本就存在或本轮新增的 untracked 文件。提交前需要人工确认哪些文件应纳入 commit，尤其是：

- `.gitignore`
- `docs/PLUGGABLE_DEVICE_MIGRATION_PLAN.md`
- `docs/TEST_FRAMEWORK_REFACTOR_PLAN.md`
- `docs/eager_graph_fusion_development_plan.md`
- `docs/fusion_kernel_remediation_plan.md`
- `test/ops/pluggable_default_import_test.py`
- 本开发日志文档
