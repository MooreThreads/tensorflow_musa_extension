# 内存 & H2D/D2H 优化改造

> **想直接了解当前架构与环境变量？** 请移步 [`architecture-and-memory.md`](./architecture-and-memory.md)——那是"当前状态快照"。本文是"**改造历程**"，按 commit 记录每一步引入了什么。

本文档介绍 `tensorflow_musa_extension` 在 **内存管理 / H2D / D2H / Python 诊断接口 / 基准** 方面的一轮系统改造（commits `cb8f7b2` → `4451f6e`，共 9 个 commit）。改造灵感来自 `torch_musa` 的 `MUSACachingAllocator` / `CachingHostAllocator` / `driver_api` / `memory_snapshot` 等模块，但所有改动**只发生在 TensorFlow PluggableDevice C ABI 之下**，不和具体 TF 版本耦合，兼容矩阵保持 `tensorflow>=2.6, <2.17`（详见 [`tf-compat-matrix.md`](./tf-compat-matrix.md)）。

## 1. 总览

| 阶段 | Commit | 范围 | 说明 |
|------|--------|------|------|
| C0 | `a00bcd5` | TF 版本解耦 | 移除 TF 2.6.1 硬绑定，引入 `tf_compat.h` 与版本范围校验 |
| C1 | `cb8f7b2` | Host 侧缓存分配器 | 新增 `HostCachingAllocator`，统一 pinned host 内存与 H2D staging |
| C2 | `913d802` | 构建架构拆分 | `libmusa_core.so` / `libmusa_plugin.so` / `tensorflow_musa._C.so` 三库共享 singleton |
| C3 | `41afa96` | Device 侧缓存分配器 MVP | 自研 caching allocator 替代 TF BFC (`use_bfc_allocator=0`) |
| C4 | `2fbfe2c` | Allocator stats / memory fraction / OOM | 扩展统计指标、加入 per-process 上限、输出结构化 OOM 诊断 |
| C5 | `4a1d2fb` | VMM Expandable Segments | dlopen MUSA driver VMM 符号，通过 env 可选开启 |
| C6 | `acc18e2` + `bb3cad2` | Python 上层 API | `tensorflow_musa.memory` / `tensorflow_musa.device` + snapshot 工具链 |
| C7 | `816de4e` | 基准与回归 | `benchmark/bench_h2d.py` / `bench_resnet.py` / `bench_alloc_churn.py` + CI 兜底测试 |
| Opt | `4451f6e` | 可选项：Peer-to-Peer Access | `musaDeviceCanAccessPeer` 缓存 + 幂等 enable，多卡用户可手动启用 |

## 2. 核心架构

```
┌──────────────────────────────────┐
│      tensorflow_musa.*           │   Python 上层 (memory / device / snapshot)
└───────────────┬──────────────────┘
                │ CPython API
┌───────────────▼──────────────────┐
│     tensorflow_musa._C.so        │   pybind 扩展（诊断 + 控制接口）
└───────────────┬──────────────────┘
                │ 直接链接
┌───────────────▼──────────────────┐
│        libmusa_core.so           │   HostCachingAllocator / DeviceCachingAllocator
│                                  │   AllocatorConfig / DriverApi / EventPool
│                                  │   PeerAccess / ExpandableSegment
└───────────────┬──────────────────┘
                │ 同一组 singleton
┌───────────────▼──────────────────┐
│       libmusa_plugin.so          │   PluggableDevice callbacks (allocate / memcpy / …)
└───────────────┬──────────────────┘
                │ PluggableDevice C ABI (SE_MAJOR=0)
┌───────────────▼──────────────────┐
│           TensorFlow             │
└──────────────────────────────────┘
```

**关键点**：allocator 等 singleton 一律放在 `libmusa_core.so`，`libmusa_plugin.so` 和 `_C.so` 各自链接这个核心库 —— 保证 TF 侧的 `allocate/deallocate` 回调与 Python 侧的 `memory.*` 看到的是**同一套状态**；所有 singleton 采用 leaked pointer 模式，避免进程退出时静态析构顺序带来的 segfault。

## 3. 功能分解

### 3.1 Host 侧：`HostCachingAllocator`

替换 `host_memory_allocate/deallocate` 中每次直调 `musaHostAlloc + musaFreeHost` 的做法：

- 按 2 的幂次大小分桶缓存已释放的 pinned 块
- 通过 `EventPool` 跟踪"尚在流上的在用块"，流结束后自动回收
- 同时接管 `PinnedStagingPool`，H2D staging buffer 与普通 pinned 分配共用同一池
- 通过 env 可控：
  - `TF_MUSA_HOST_ALLOC_MAX_POOL_MB`（默认 2048）
  - `TF_MUSA_DISABLE_HOST_CACHING=1` 回退到原始直调路径（用于 A/B 对比）

### 3.2 Device 侧：`DeviceCachingAllocator`

自研 device 分配器，**替代 TF BFC**（`platform->use_bfc_allocator = 0`）：

- 单池、按地址排序的 free list + block 链表
- 支持 **块切分 / 合并**，接入 allocator config 的 `max_split_size_bytes`
- 不走 `musaMallocAsync`，底层仅用 `musaMalloc/musaFree`，回避 mudnn 某些 kernel 对 async pool 的已知兼容问题
- 通过 env 可控：
  - `TF_MUSA_DEVICE_ALLOCATOR=caching|passthrough`（缺省 caching，`passthrough` 用于 bisect）
  - `TF_MUSA_ALLOC_VERBOSE_OOM=1` 在 stderr 输出结构化 OOM 诊断
  - `TF_MUSA_ALLOC_CONF="key:value,key:value"`（见下）

### 3.3 `TF_MUSA_ALLOC_CONF` 可选项

| key | 类型 | 说明 |
|-----|------|------|
| `expandable_segments` | bool | 开启 VMM 扩展段（见 3.4） |
| `max_split_size_bytes` | bytes | 超过此尺寸的块禁止切分；支持 `MB/GB` 单位 |
| `roundup_power2_divisions` | int | 请求尺寸向上取整时 2 的幂次分度 |
| `garbage_collection_threshold` | float (0..1] | GC 触发阈值（预留位，后续扩展） |

### 3.4 Expandable Segments（VMM）

参考 torch 4.0+ 与 torch_musa 的做法，通过 MUSA driver 的虚拟内存管理 API 避免碎片：

- 启动时 `dlopen("libmusa.so")` + `dlsym` 解析 `muMemAddressReserve / muMemCreate / muMemMap / muMemSetAccess / muMemGetAllocationGranularity / muDeviceGetAttribute / muGetErrorString` 等符号
- 缺失任一符号或设备能力不支持时，**静默回退**到 `musaMalloc` 路径
- 开关由 `TF_MUSA_ALLOC_CONF=expandable_segments:true` 打开，默认关闭
- 相关代码：
  - `musa_ext/mu/device/driver_api.{h,cc}`：dlsym 表 + 设备能力探测
  - `musa_ext/mu/device/expandable_segment.{h,cc}`：单段 RAII 包装
  - `caching_allocator.cc` 在 cache miss 的分段创建分支集成

### 3.5 诊断指标

`SP_AllocatorStats` + 内部 `DeviceCachingAllocatorStats` 均已扩展，Python 侧一次性抓到：

```python
tensorflow_musa.memory.memory_stats()
# {
#   "in_use_bytes": ..., "reserved_bytes": ..., "cached_bytes": ...,
#   "peak_in_use_bytes": ...,
#   "alloc_requests": ..., "cache_hits": ..., "cache_misses": ...,
#   "oom_events": ..., "splits": ..., "merges": ...,
#   "segments": ..., "limit_bytes": ..., "total_device_bytes": ...,
# }
```

另外提供 `_device_segment_snapshot(ordinal)` 返回每个活动段的 `(address, size, in_use, num_blocks, num_free_blocks, largest_free_block, is_expandable)` 列表，用于碎片分析。

### 3.6 Python API 精简版

> 设计原则：**只暴露必要的诊断与调优接口**，不对齐 `torch.musa.*` 的 Stream/Event/Random/Graph/AMP。

| 子模块 | 函数 | 功能 |
|--------|------|------|
| `tensorflow_musa.memory` | `empty_cache()` | 释放空闲段回驱动 |
| | `memory_allocated(device=None)` | 当前分配字节 |
| | `memory_reserved(device=None)` | 当前保留字节（含 free cache） |
| | `max_memory_allocated(device=None)` | 自上次 reset 的峰值 |
| | `reset_peak_memory_stats(device=None)` | 重置峰值计数 |
| | `memory_stats(device=None)` | 完整计数器字典 |
| | `set_per_process_memory_fraction(f, device=None)` | 硬上限（0 < f ≤ 1） |
| | `mem_get_info(device=None)` | 驱动视角的 `(free, total)` |
| | `get_allocator_backend()` | `"caching"` 或 `"passthrough"` |
| | `memory_snapshot(device=None)` | 合成 dict：stats + segments + driver + config + VMM + history |
| | `_dump_snapshot(path)` | 原子写 JSON（先 `.tmp` 再 `os.replace`） |
| | `_record_memory_history(enabled, …)` | 后台轮询采样 `memory_stats`，环形缓冲 |
| `tensorflow_musa.device` | `device_count()` / `is_available()` | 走 TF，尊重 `MUSA_VISIBLE_DEVICES` |
| | `current_device()` / `get_device_name(device=None)` | 基本设备查询 |
| | `can_access_peer(src, dst)` | P2P 能力（带 cache） |
| | `enable_peer_access(src, dst)` | 幂等启用 P2P |
| | `peer_access_snapshot()` | 已观察到的 P2P 表 |

**明确不做**（见 plan §6.3 原文）：`Stream` / `Event` 类、`set_device` / `synchronize`、完整的 `get_device_properties`、`MemPool` / `use_mem_pool`、`register_musa_device_module`。

### 3.7 Peer-to-Peer Access（可选）

多卡用户可手动启用：

```python
import tensorflow_musa as tf_musa

if tf_musa.device.can_access_peer(0, 1):
    tf_musa.device.enable_peer_access(0, 1)  # 幂等
```

当前 `memcpy_dtod` 回调尚未自动切换到 `musaMemcpyPeerAsync`（这是后续独立 commit 的任务，且需要从 allocator bookkeeping 中反查 `SP_DeviceMemoryBase` 的 ordinal）。启用之后用户可以在自定义 kernel 或 `musaMemcpyPeerAsync` 调用里直接使用跨卡 DMA。

## 4. 基准与回归

`benchmark/` 下新增三个脚本：

| 脚本 | 作用 |
|------|------|
| `bench_h2d.py` | 纯 H2D / D2H 吞吐扫频 (4K–16M)，输出 mean / p50 / p95 / GB/s |
| `bench_resnet.py` | 100-step 训练循环；keras ResNet50 可用则用，否则回退到手写 conv stack + 手写 SGD（绕开 TF 2.6 keras 版本兼容问题） |
| `bench_alloc_churn.py` | 混合尺寸压力测试；收尾断言：无泄漏、无误 OOM、`empty_cache` ≥95% 释放、`cache_hits>0` |

`test/test_bench_alloc_churn.py` 用小 iters 把 `bench_alloc_churn.py` 跑到 pytest 里 —— **作为 allocator 的 CI 回归兜底**，未来任何破坏分配器不变量的改动都会在这里红灯。

## 5. 环境变量一览

> 完整分组 / 解读 / 调试剧本见 [`architecture-and-memory.md §5`](./architecture-and-memory.md#5-环境变量全集) 与 [`DEBUG_GUIDE.md`](./DEBUG_GUIDE.md)。本节只列"这一轮改造引入或受影响"的变量，保留改造视角。

### 5.1 Device allocator

| 变量 | 缺省 | 作用 |
|------|------|------|
| `TF_MUSA_DEVICE_ALLOCATOR` | `caching` | 切换 device allocator 后端（`caching` / `passthrough`） |
| `TF_MUSA_ALLOC_CONF` | `""` | VMM 等高级选项，`key:value[,key:value]` |
| `TF_MUSA_ALLOC_VERBOSE_OOM` | `0` | OOM 时打印详细诊断到 stderr |
| `TF_MUSA_DEVICE_ALLOC_MAX_POOL_MB` | `32768` | 分配器 reserved 软上限（MiB），`0` = 无上限 |
| `TF_MUSA_ENABLE_ASYNC_ALLOC` | `0` | 启用 runtime 的 mempool async-alloc 路径（与本分配器互斥，默认关） |

### 5.2 Host pinned allocator

| 变量 | 缺省 | 作用 |
|------|------|------|
| `TF_MUSA_HOST_ALLOC_MAX_POOL_MB` | `2048` | Host caching allocator 池上限（MiB） |
| `TF_MUSA_DISABLE_HOST_CACHING` | `0` | 关闭 host caching，回退到直调 `musaHostAlloc/Free` |

### 5.3 H2D staging & auto-pin

| 变量 | 缺省 | 作用 |
|------|------|------|
| `TF_MUSA_H2D_STAGING_THRESHOLD_BYTES` | `0`（关） | `>0` 开启 staging 并在进程退出打印统计；推荐 `524288` 或 `1048576` |
| `TF_MUSA_H2D_STAGING_MEMCPY_THREADS` | `4` | staging 路径里 host→pinned 的并行 memcpy 线程数 [1,16] |
| `TF_MUSA_H2D_STAGING_SKIP_MEMCPY` | `0` | **调试用**：跳过 CPU memcpy（会发送未定义内容到 device） |
| `TF_MUSA_H2D_STAGING_MAX_POOL_MB` | ⚠️ | **已废弃**，由 `TF_MUSA_HOST_ALLOC_MAX_POOL_MB` 统一管理 |
| `TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES` | `0`（关） | `>0` 开启：自动 `musaHostRegister` 大于此值的 H2D 源（仅适合"同一块地址反复用"场景） |
| `TF_MUSA_DIAG_H2D_PINNED` | `0` | 调试用：打印前 50 次 H2D 源指针的 pinned 状态 |

### 5.4 EventPool

| 变量 | 缺省 | 作用 |
|------|------|------|
| `TF_MUSA_DISABLE_EVENT_POOL` | `0` | 关闭 `musaEvent_t` 重用池（bisect 用；默认开启） |

## 6. PluggableDevice 合规性

- `use_bfc_allocator = 0` 是 TF 2.5+ PluggableDevice 合约内的字段，CUDA 插件也用；本改造不引入未公开字段
- `SP_Stream / SP_Event` 依旧对 TF 透明为 `musaStream_t / musaEvent_t`，未动结构体布局
- `_C` 模块只读/写 allocator 自身状态，**不触发 TF-visible 分配**，不会扰动 TF 的引用计数
- 所有 TF 头文件引用收敛到 `musa_ext/mu/tf_compat.h`，并用 `static_assert(SE_MAJOR == 0)` 做兼容断言

## 7. 与 torch_musa 的对应关系

| torch_musa | 本扩展 | 差异 |
|------------|--------|------|
| `MUSACachingAllocator.{h,cpp}` | `caching_allocator.{h,cc}` | MVP：单池、未做 per-stream reuse |
| `CachingHostAllocator.{h,cpp}` | `host_caching_allocator.{h,cc}` | 一致 |
| `MUSAAllocatorConfig.{h,cpp}` | `allocator_config.{h,cc}` | `TF_MUSA_ALLOC_CONF` 字段对齐 |
| `driver_api.{h,cpp}` | `driver_api.{h,cc}` | 一致，使用 dlopen 而非 weak link |
| `memory_snapshot.cpp` | `python/snapshot.py` | **采样式实现**，不含符号化 stack |
| `MUSAStream.{h,cpp}` | — | 不移植：TF 自己管理 stream |
| `Module.cpp`（pybind） | `musa_ext/python/_C.cpp` | 使用裸 CPython API，暂不引入 pybind11 |
| `torch_musa/core/memory.py` | `python/memory.py` | 仅保留必要子集 |
| `torch_musa/core/device.py` | `python/device.py` | 通过 TF 查设备 |

## 8. 后续待办（可选 commit）

以下项目在 plan §5.6 明确为"可选，按业务优先级推进"：

- `memcpy_dtod` 的 peer-aware dispatch（依赖 allocator 侧 ordinal 反查）
- Expandable segments + 多卡 `muMemSetAccess` peer 映射
- `MUSAGraph / graph_pool_handle` 支持
- `MusaIPC` 跨进程共享
- `supports_unified_memory=1` / Unified Allocator
- `_record_memory_history` 升级为 allocator 端 hook（可含 stream/event 信息，替换当前的轮询采样）

## 9. 快速上手

```python
import tensorflow as tf
import tensorflow_musa as tf_musa

with tf.device("/device:MUSA:0"):
    a = tf.random.uniform([1024, 1024])
    b = a + 1.0
    _ = b.numpy()

print("in_use:", tf_musa.memory.memory_allocated(), "B")
print("reserved:", tf_musa.memory.memory_reserved(), "B")

snap = tf_musa.memory.memory_snapshot()
tf_musa.memory._dump_snapshot("/tmp/musa_mem.json")

released = tf_musa.memory.empty_cache()
print("released:", released, "B")
```

开启 VMM 扩展段：

```bash
TF_MUSA_ALLOC_CONF="expandable_segments:true" python train.py
```

限制进程内存用量：

```python
tf_musa.memory.set_per_process_memory_fraction(0.5)  # 50% of total
```
