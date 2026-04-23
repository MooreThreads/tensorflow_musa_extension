# tensorflow_musa_extension 架构与内存调度总览

> 面向读者：**想在 10 分钟内了解本项目全貌，或在 30 分钟内定位一个内存/H2D 问题**。
>
> 既有文档分工：
> - 本文：当前架构 + 内存调度 + 环境变量一览
> - [`memory-optimization.md`](./memory-optimization.md)：改造历程（commit-by-commit）
> - [`tf-compat-matrix.md`](./tf-compat-matrix.md)：TF 版本兼容矩阵
> - [`DEBUG_GUIDE.md`](./DEBUG_GUIDE.md)：线上 / 现场调试命令手册

## 1. 一张图看懂整个栈

```
                 ┌────────────────────────────────────────────────┐
 用户 Python ──▶ │  tensorflow_musa.{memory, device, snapshot}    │  纯 Python，只做参数校验和字典重整
                 └───────────────────────┬────────────────────────┘
                                         │ CPython API 调用
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │  tensorflow_musa._C.so   (musa_ext/python/)    │  裸 CPython 扩展，暴露诊断/控制接口
                 └───────────────────────┬────────────────────────┘
                                         │ 直接链接
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │  libmusa_core.so    （musa_ext/mu/device/）    │  无 TF 依赖，承载所有 singleton
                 │  ┌──────────────────────────────────────────┐  │   DeviceCachingAllocator
                 │  │  AllocatorConfig  (TF_MUSA_ALLOC_CONF)   │  │   HostCachingAllocator
                 │  │  DriverApi  (libmusa.so dlsym)           │  │   PinnedStagingPool
                 │  │  EventPool                               │  │   ExpandableSegment (VMM)
                 │  │  PeerAccess                              │  │   HostPinCache (可选)
                 │  └──────────────────────────────────────────┘  │
                 └───────────────────────┬────────────────────────┘
                                         │ singleton 引用
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │  libmusa_plugin.so  （musa_ext/mu/device/      │  PluggableDevice C ABI 回调
                 │                       musa_se_callbacks.cc）   │   allocate / deallocate
                 │                                                │   memcpy_htod / dtoh / dtod
                 │                                                │   host_memory_allocate / free
                 │                                                │   create_event / wait / synchronize
                 └───────────────────────┬────────────────────────┘
                                         │ SE_MAJOR=0 PluggableDevice C ABI
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │              TensorFlow (2.6 ~ 2.16)            │
                 └────────────────────────────────────────────────┘
```

**三库拆分的关键约束**：所有 singleton（分配器、池、driver 句柄、事件池、P2P 表、配置）都只在 `libmusa_core.so` 里定义一次。`_C.so`（Python 诊断）和 `libmusa_plugin.so`（TF 回调）各自链接 core，**永远看到同一份状态**。所有 singleton 都用 "leaked pointer" 模式——进程退出时不析构，避免静态对象析构顺序踩到已关闭的 driver。

## 2. 目录速查

```
musa_ext/
├── mu/
│   ├── device/                     ← 内存相关全部在这里
│   │   ├── caching_allocator.{h,cc}      Device 侧自研分配器
│   │   ├── host_caching_allocator.{h,cc} Host 侧 pinned 缓存分配器
│   │   ├── allocator_config.{h,cc}       TF_MUSA_ALLOC_CONF 解析
│   │   ├── driver_api.{h,cc}             libmusa.so dlsym 表（VMM 用）
│   │   ├── expandable_segment.{h,cc}     VMM 段 RAII
│   │   ├── event_pool.{h,cc}             musaEvent_t 重用池
│   │   ├── peer_access.{h,cc}            多卡 P2P 表
│   │   ├── musa_se_callbacks.{h,cc}      PluggableDevice 全部 C 回调 ←
│   │   │                                 staging pool / auto-pin / diag
│   │   │                                 都在这个文件里
│   │   ├── musa_memcpy.{h,cc}            memcpy 小工具
│   │   ├── musa_memset.{h,cc}
│   │   ├── musa_resource_mgr.{h,cc}      ordinal → driver handle 映射
│   │   └── musa_telemetry.{h,cc}         轻量事件流（可选外接）
│   ├── optimizer/                  ← 图优化（与内存部分独立）
│   │   └── musa_graph_optimizer.cc       PinHostComputeToCpu 等
│   ├── kernel_register.{h,cc}
│   ├── device_register.{h,cc}
│   └── tf_compat.h                 ← 所有 TF 头文件引用的唯一入口
├── python/
│   └── _C.cpp                      裸 CPython 扩展入口
└── kernels/                        MUSA kernel 实现（和内存调度无关）

python/                             Python 侧上层 API
├── __init__.py         import 时加载 _C，初始化 hook
├── _loader.py          libmusa_plugin.so 定位与动态加载
├── _ext.py             从 _C 取出原始接口，做类型统一
├── memory.py           用户可见：memory_stats / empty_cache / ...
├── device.py           用户可见：device_count / can_access_peer / ...
└── snapshot.py         memory_snapshot / _record_memory_history

docs/
├── architecture-and-memory.md   ← 本文
├── memory-optimization.md       改造历程
├── tf-compat-matrix.md          TF 版本矩阵
└── DEBUG_GUIDE.md               调试手册

benchmark/
├── bench_h2d.py                 H2D/D2H 吞吐扫频
├── bench_resnet.py              100 步训练
├── bench_alloc_churn.py         分配器压力 + 不变量断言
└── bench_host_alloc.py          pinned host 池压力
```

## 3. 内存调度四条路径

### 3.1 Device 内存（`allocate` / `deallocate`）

TF 请求显存时走 `libmusa_plugin.so::plugin_memory_allocate` → core 里的 `DeviceCachingAllocator`。

```
TF::BFC-like request
      │
      ▼
DeviceCachingAllocator::Allocate(size)
      │
      ├─ RoundUp(size, 512 B) = rounded        ← 对齐保证
      ├─ 查 free_blocks (size-ordered set)     ← cache-hit 走这里
      │     命中 → MaybeSplitLocked → 标记 allocated → 返回
      │
      └─ cache-miss
          ├─ SegmentSize = ceil(rounded, 2 MiB)
          ├─ 先看 TF_MUSA_ALLOC_CONF expandable_segments 是否开
          │     是 → muMemCreate + muMemMap（VMM）
          │     否 → musaMalloc
          ├─ 创建 Block（is_segment_head=true）
          └─ 进入 active_blocks
```

**要点**：
- 分配路径的 `allocated=true`、`active_blocks.emplace`、`cache_hits++` 这三件事都在**同一个 critical section** 内（修复过一次 cache-hit vs. merge race）。
- `musaFree`（或 VMM 的 `muMemUnmap`）只在 `EmptyCache` 或 OOM 回退时才发生，稳态下完全无 driver-side 分配调用。
- 后端可以通过 `TF_MUSA_DEVICE_ALLOCATOR=passthrough` 切回"直调 `musaMalloc/Free`"模式，用于 bisect。

### 3.2 Host pinned 内存（`host_memory_allocate` / `deallocate`）

TF 为 `feed_dict` 和 output tensor 准备 host buffer 时走这条路。

```
TF host_memory_allocate(size)
      │
      ▼
HostCachingAllocator::Allocate(size)
      │
      ├─ 向上取整到 2^k bucket
      ├─ bucket 里找到可用块 → 命中返回
      │     注意："可用" = allocated=false 且流未再引用
      │          → 通过 EventPool 里的 musaEvent_t 判断
      └─ cache-miss → musaHostAlloc(Portable) → 入 bucket

HostCachingAllocator::Free(ptr)
      │
      ├─ 若曾被 RecordStream 注册过流
      │     → 把块挂到 "等待流完成" 队列
      │     → 等流上的 musaEvent_t 就绪后回 bucket
      └─ 否则直接回 bucket
```

**要点**：
- 池上限 = `TF_MUSA_HOST_ALLOC_MAX_POOL_MB`（默认 2048 MiB）。超过 cap 时新分配仍然成功，但**不入池**（直接交给 OS）。
- `TF_MUSA_DISABLE_HOST_CACHING=1` → 整条路径短路到 `musaHostAlloc/Free`。
- 共用一套 `EventPool`：staging pool 的"等流完成才能回收"逻辑和这里是同一个机制。

### 3.3 H2D staging（`memcpy_htod` 大包加速）

> 关键：这条路径是 **H2D** 专用，D2H 不走 staging。

```
memcpy_htod(device_dst, host_src, size, stream)
      │
      ├─ PinnedStagingPool::TryStagingCopy ?
      │     ├─ threshold_ == 0          → 不开，直接 async H2D
      │     ├─ size < threshold_        → 太小，直接 async H2D
      │     ├─ musaHostGetFlags(src) OK → 源已经 pinned，staging 反而多一次 memcpy，直接 async H2D
      │     │
      │     └─ 走 staging：
      │         ├─ stage = HostCachingAllocator.Allocate(size)     ← 借一块 pinned
      │         ├─ ParallelMemcpy(stage, host_src, size)           ← 多线程 memcpy
      │         ├─ musaMemcpyAsync(device_dst, stage, size, H2D, stream)
      │         ├─ 在 stream 上挂一个 musaEvent_t
      │         └─ HostCachingAllocator.RecordStream(stage, stream) + Free(stage)
      │                                                            ← 等流完成后才真正还池
      └─ 否则 → musaMemcpyAsync(device_dst, host_src, size, H2D, stream)
```

**要点**：
- staging **只在 pageable 源 + 超过阈值**时触发。
- 进程退出时在 stderr 打印统计：`staged=N (X MiB) already_pinned=P pool_allocs=A`。
- `TF_MUSA_H2D_STAGING_MEMCPY_THREADS` 可调 host memcpy 并行度（默认 4，范围 1–16）。
- staging buffer 的 pinned 池与 `host_memory_allocate` **共用同一个 `HostCachingAllocator`**，所以 `TF_MUSA_HOST_ALLOC_MAX_POOL_MB` 同时限制两者。

### 3.4 D2H（`memcpy_dtoh`）

```
memcpy_dtoh(host_dst, device_src, size, stream)
      │
      └─ musaMemcpyAsync(host_dst, device_src.opaque, size, D2H, stream)
```

D2H **没有 staging**。如果对 D2H 瓶颈敏感，解决方向是：用户侧把 `host_dst` 分配到 pinned 内存（例如通过 `HostPinCache` 自动 pin——见 §4.4）。

## 4. 进阶机制

### 4.1 VMM Expandable Segments

长跑服务 / 多阶段切形的工作负载会在 BFC-style 分配器下产生不可回收碎片。解法是用 MUSA Driver 的虚拟内存 API 把"地址空间"和"物理页"解耦：

```
muMemAddressReserve(8 GiB)      一次性保留一大段连续 VA
muMemCreate(2 MiB)              按需创建物理页
muMemMap(va + offset, phys)     拼接到 VA 上
muMemSetAccess(dev, rw)         开读写权限
```

- 所有符号通过 `dlopen("libmusa.so") + dlsym` 解析，**任何符号缺失都会静默回退**到 `musaMalloc`，不影响老驱动兼容。
- 开关：`TF_MUSA_ALLOC_CONF=expandable_segments:true`（默认关）。
- 观测：`tf_musa.memory.memory_snapshot()` 的 `segments[].is_expandable` 字段。

### 4.2 EventPool

`musaEventCreateWithFlags + Destroy` 在热路径里是 OS 级调用，`HostCachingAllocator` 的流序回收会大量用到事件。`EventPool` 给每 ordinal 维护一个 free list（cap 256），`Acquire/Return` 零系统调用。

- 关闭：`TF_MUSA_DISABLE_EVENT_POOL=1`（bisect 用；默认开启）。
- 用途：staging、pinned 回收、未来的 per-stream block reuse。

### 4.3 PeerAccess（多卡 P2P）

`musaDeviceCanAccessPeer` 在旧驱动上是 ~100 μs 级调用，重复访问会拖慢一切。`PeerAccess` 用一张 `(src, dst) → bool` 的 lazy 表 + 幂等 `EnablePeerAccess`：

```python
import tensorflow_musa as tf_musa
if tf_musa.device.can_access_peer(0, 1):
    tf_musa.device.enable_peer_access(0, 1)   # 幂等
```

注意：目前 `memcpy_dtod` 仍然走 `musaMemcpyAsync`（而非 `musaMemcpyPeerAsync`），多卡直连 DMA 要靠用户自己在 kernel 里用。

### 4.4 HostPinCache（auto-pin，可选）

**默认不开**。当已经 *知道* 客户在反复喂同一批 pageable 大 buffer（比如一个长期存在的 NumPy 数组）时，可以让运行时"自动 `musaHostRegister` 这块区域"，避免每次都走 staging CPU memcpy：

- `TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES=N`（默认 0 = 关闭）。
- 设为 `N` 后，H2D 源地址大小 ≥ N 的区间会被 `musaHostRegister` 注册（对齐到 page），后续命中就走原生 async H2D，staging 的 CPU memcpy 省掉。
- 成本：注册操作本身是 OS 级，**只适合"同一块地址反复用"**。普通 framework feed（每次新 numpy）不适合开。

## 5. 环境变量全集

> 下表按"作用域 / 子系统"分组。**粗体** = 运维 / 调优常用；其余为调试 / A-B-bisect 用途。

### 5.1 Device Allocator

| 变量 | 类型 | 缺省 | 作用 |
|---|---|---|---|
| **`TF_MUSA_DEVICE_ALLOCATOR`** | enum | `caching` | `caching` = 走 `DeviceCachingAllocator`；`passthrough` = 直调 `musaMalloc/Free`。bisect 用。 |
| **`TF_MUSA_ALLOC_CONF`** | `k:v,k:v` | `""` | 见 §5.6。 |
| `TF_MUSA_DEVICE_ALLOC_MAX_POOL_MB` | int (MiB) | 32768 | 分配器 "reserved" 软上限。超了仍然会服务 live 请求；`0` 或非法值 = 无上限。 |
| `TF_MUSA_ALLOC_VERBOSE_OOM` | bool | `0` | OOM 时把结构化诊断（各 size bucket、free 块数、最大空闲块）打到 stderr。Python 侧 `_device_last_oom_message` 无论此变量是否打开都能拿到。 |
| `TF_MUSA_ENABLE_ASYNC_ALLOC` | bool | `0` | 开启 MUSA runtime 的 mempool async-alloc 路径（`musaMallocAsync`）。**默认关**：mudnn 某些 kernel 与 async pool 混用会崩，仅在明确知道不会踩到时才开。 |

### 5.2 Host Pinned Allocator

| 变量 | 类型 | 缺省 | 作用 |
|---|---|---|---|
| **`TF_MUSA_HOST_ALLOC_MAX_POOL_MB`** | int (MiB) | 2048 | Pinned host 缓存池上限。超限后新分配不入池（每次都 `musaHostAlloc`，会影响命中率）。 |
| `TF_MUSA_DISABLE_HOST_CACHING` | bool | `0` | 关闭 host caching，回退到直调 `musaHostAlloc/musaFreeHost`。bisect 用。 |

### 5.3 H2D Staging Pool

| 变量 | 类型 | 缺省 | 作用 |
|---|---|---|---|
| **`TF_MUSA_H2D_STAGING_THRESHOLD_BYTES`** | int (bytes) | `0`（关） | `>0` 开启 staging：源是 pageable 且 size ≥ 此值才走 staging。同时在进程退出时打印统计。推荐值 `524288`（512 KiB）或 `1048576`（1 MiB）。 |
| `TF_MUSA_H2D_STAGING_MEMCPY_THREADS` | int | `4` | staging 里 host→pinned CPU memcpy 的并行线程数。范围 [1, 16]。 |
| `TF_MUSA_H2D_STAGING_SKIP_MEMCPY` | bool | `0` | **调试用**：跳过 CPU memcpy，只测 "pinned buffer 分配 + async H2D" 的开销。正常场景不要开，它会发送未定义内容到 device。 |
| `TF_MUSA_H2D_STAGING_MAX_POOL_MB` | ⚠️ | — | **已废弃**。staging 的 pinned 池现在由 `TF_MUSA_HOST_ALLOC_MAX_POOL_MB` 统一管理，设置此变量只会打印一条 deprecation 警告。 |

### 5.4 H2D 自动 Pin（可选加速器）

| 变量 | 类型 | 缺省 | 作用 |
|---|---|---|---|
| `TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES` | int | `0`（关） | `>0` 开启：H2D 源地址大小 ≥ 此值时 `musaHostRegister` 自动固定内存。**仅适合"同一块地址反复使用"的工作负载**（比如长期存在的 tf.data 缓冲）。 |
| `TF_MUSA_DIAG_H2D_PINNED` | bool | `0` | 调试用：打印前 50 次 H2D 源指针的 `pointerType / isPinned` 诊断，用来确认 feed_dict 是不是 pageable。 |

### 5.5 EventPool

| 变量 | 类型 | 缺省 | 作用 |
|---|---|---|---|
| `TF_MUSA_DISABLE_EVENT_POOL` | bool | `0` | 关闭 `musaEvent_t` 池，每次回到 `musaEventCreateWithFlags + Destroy`。bisect 用。 |

### 5.6 `TF_MUSA_ALLOC_CONF` 子键

格式：`key:value` 或 `key=value`，多项逗号分隔。未知键会打 warning 但不报错。

| key | 类型 | 说明 |
|---|---|---|
| `expandable_segments` | bool (`1/true/on/...`) | 开启 VMM 段（§4.1）。 |
| `max_split_size_mb` | int (MiB) | 超过此尺寸的块禁止切分，避免大块被切碎。与 `max_split_size_bytes` 同义，优先前者（torch 对齐）。 |
| `max_split_size_bytes` | 支持 `123`、`123kb`、`4mb`、`1gb` | 同上。 |
| `roundup_power2_divisions` | int | 请求尺寸向上取整时的 2 的幂次分度。预留位。 |
| `garbage_collection_threshold` | float (0, 1] | 预留位。 |

### 5.7 图优化器（与内存调度独立但常一起调）

| 变量 | 类型 | 缺省 | 作用 |
|---|---|---|---|
| `TF_MUSA_DISABLE_HOST_COMPUTE_PIN` | bool | `0` | 关闭 `PinHostComputeToCpu`。这个 pass 会把 shape/index 等小计算固定在 CPU 上，避免把 int32 shape 到处搬。关掉一般只会变慢。 |

## 6. Python API 速查

完整清单见 `memory-optimization.md §3.6`，这里给 **top-7 日常用**：

```python
import tensorflow_musa as tf_musa

tf_musa.memory.memory_stats()                 # 全部计数器
tf_musa.memory.memory_allocated()             # 当前 in-use 字节
tf_musa.memory.memory_reserved()              # 当前 reserved (含 cache)
tf_musa.memory.max_memory_allocated()         # 峰值
tf_musa.memory.reset_peak_memory_stats()

tf_musa.memory.empty_cache()                                  # 释放全部 free 段回驱动
tf_musa.memory.set_per_process_memory_fraction(0.5, device=0) # 硬上限

snap = tf_musa.memory.memory_snapshot()
tf_musa.memory._dump_snapshot("/tmp/musa.json")

tf_musa.device.device_count()
tf_musa.device.can_access_peer(0, 1)
tf_musa.device.enable_peer_access(0, 1)
```

## 7. 常见调试剧本

### 7.1 "显存疑似泄漏"

```bash
# 1. 打开 verbose OOM，顺便抓堆栈
TF_MUSA_ALLOC_VERBOSE_OOM=1 python your_script.py 2>err.log
grep "OOM" err.log | head

# 2. 导 snapshot 看碎片
python -c "
import tensorflow_musa as tf_musa
tf_musa.memory._dump_snapshot('/tmp/snap.json')
"
jq '.segments | map({addr, size, num_blocks, num_free_blocks, largest_free_block})' /tmp/snap.json

# 3. 看是不是碎片问题（largest_free_block 远小于 free 总和）
#    → 开 VMM
TF_MUSA_ALLOC_CONF=expandable_segments:true python your_script.py
```

### 7.2 "H2D 很慢"

```bash
# 1. 先确认 feed 是 pageable 还是 pinned
TF_MUSA_DIAG_H2D_PINNED=1 python your_script.py 2>&1 | grep "\[MUSA\] h2d"

# 2. 如果 pageable，开 staging 看统计
TF_MUSA_H2D_STAGING_THRESHOLD_BYTES=524288 python your_script.py 2>&1 | tail -3

# 3. 对照：关掉 staging
TF_MUSA_H2D_STAGING_THRESHOLD_BYTES=0 python your_script.py

# 4. 如果 feed 是反复用的同一块地址，试试 auto-pin
TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES=1048576 python your_script.py
```

### 7.3 "怀疑分配器有 bug / 要 bisect"

```bash
# 逐层关掉本项目的缓存，看哪个引入问题

# 1) 全关（最保守，等价原生 TF 路径）
TF_MUSA_DEVICE_ALLOCATOR=passthrough \
TF_MUSA_DISABLE_HOST_CACHING=1 \
TF_MUSA_H2D_STAGING_THRESHOLD_BYTES=0 \
TF_MUSA_DISABLE_EVENT_POOL=1 \
python your_script.py

# 2) 只关 device，其余开
TF_MUSA_DEVICE_ALLOCATOR=passthrough python your_script.py

# 3) 只关 host
TF_MUSA_DISABLE_HOST_CACHING=1 python your_script.py
```

### 7.4 "长期跑碎片严重 / reserved 只增不减"

```bash
# 定时 emit snapshot
python -c "
import threading, time, tensorflow_musa as tf_musa
def dump():
    while True:
        tf_musa.memory._dump_snapshot('/tmp/musa.json')
        time.sleep(60)
threading.Thread(target=dump, daemon=True).start()
# ... 你的主循环 ...
"

# 开 VMM
TF_MUSA_ALLOC_CONF=expandable_segments:true python your_script.py

# 周期性手动收缩（在业务低峰期）
#   tf_musa.memory.empty_cache()
```

## 8. 设计原则与边界

**做**：
- 替换或增强 TF 交给 PluggableDevice 的内存接口。
- 暴露**诊断、限额、清理**能力，让客户能自己排查问题。
- 可选路径（staging、VMM、auto-pin）默认关闭或低干扰默认值，开关驱动。

**不做**：
- 不对齐 `torch.musa` 的 Stream/Event/Random/Graph/AMP Python 类——TF 不需要。
- 不引入 TF 私有字段。`use_bfc_allocator=0` 是合约内字段（CUDA 插件也用）。
- 不改变 `SP_Stream / SP_Event` 的 ABI 布局。
- D2H 目前不做 staging（需要的时候可以补；但在现有 workload 上没观察到瓶颈）。

**边界条件**：
- 多进程共享 GPU：务必用 `set_per_process_memory_fraction`，否则第一个进程会把 `reserved` 抬得很高，后来者触发 OOM 回退路径（慢几十倍）。
- 长期服务：`empty_cache()` 不是免费的（要等所有流排空），放在业务低峰期调用，或用 VMM 段来避免"碎片却无法还给 OS"的境地。
- `TF_MUSA_ENABLE_ASYNC_ALLOC` 与 `DeviceCachingAllocator` 是互斥概念——前者让 runtime 管池，后者我们自己管。**不要同时依赖两者**。

## 9. 一张表小结"我该调什么"

| 现象 / 目标 | 第一档调 | 第二档调 | 验证指标 |
|---|---|---|---|
| 大 pageable H2D 慢 | `TF_MUSA_H2D_STAGING_THRESHOLD_BYTES=524288` | `TF_MUSA_H2D_STAGING_MEMCPY_THREADS=8` | 退出时 `staged_bytes` 比例、ms/step |
| 同一块 host buffer 反复喂 | `TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES=1048576` | — | `already_pinned` 计数上升 |
| OOM / 长期跑碎片 | `TF_MUSA_ALLOC_CONF=expandable_segments:true` | `empty_cache()` 定时 | `segments[].largest_free_block` |
| 多进程共享 GPU | `set_per_process_memory_fraction(1/N)` | `TF_MUSA_DEVICE_ALLOC_MAX_POOL_MB` | `reserved_bytes` 稳定 |
| 怀疑内存 bug | 逐步切 `passthrough` / `DISABLE_HOST_CACHING` / staging 关 | — | 问题是否复现 |
| 在线诊断 | `memory_snapshot()` + `_dump_snapshot` | `TF_MUSA_ALLOC_VERBOSE_OOM=1` | JSON 文件、stderr 日志 |
| 运行时 stats | `memory_stats()` / `host_memory_stats()` | — | `cache_hits/alloc_requests` 命中率 |

---

**一句话内化**：底层两个 caching allocator + 一个 staging pool + 一个 EventPool，在 TF PluggableDevice C ABI **之下**把 MUSA 的内存/传输路径包了一层；每条路径都有一个 kill-switch 环境变量用来 A/B；Python 侧只开放诊断和控制（不重建 torch.musa 生态）。所有不变量在 `benchmark/bench_alloc_churn.py` 里有回归兜底。
