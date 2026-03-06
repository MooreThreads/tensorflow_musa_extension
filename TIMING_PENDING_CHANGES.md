# TIMING Pending Changes

> 状态：设计已落地，以下内容用于记录当前实现约束与后续可选增强

## 1. 目标

支持在 kernel compute 中使用通用阶段计时：

- `MUSA_KERNEL_TRACE_START(stage_id)`
- `MUSA_KERNEL_TRACE_END(stage_id)`

并允许输出布局按 kernel 级别自定义，不再硬编码 `Mem Alloc/Mem Cpy/Kernel/Sync`。

## 2. 设计结论（已确认）

1. `START/END` 只传“阶段标识”（字符串或枚举 ID），保持轻量高频调用。
2. 自定义的 `vector/struct` 放在 `GUARD` 里定义“输出布局”，例如：
   - 阶段顺序
   - 显示名
   - 是否显示 0 值
3. 统计按“阶段标识”累计，不硬编码固定阶段。
4. 打印按布局输出，未定义阶段归入 `Other`。
5. `MUSA_KERNEL_DEBUG=OFF` 时宏保持空实现，release 性能不变。

## 3. 为什么布局不放到每个 START/END 参数

1. `START/END` 是高频路径，传复杂对象会增加侵入和开销。
2. 每次都传布局会重复，维护成本高。
3. 布局本质是 compute 级全局配置，应在 `GUARD` 一次性声明。

## 4. 期望输出示例

```text
[MUSA_KERNEL_TIMING] MatMul [[1,10],[10,5]], host_total_ms=0.201, device_total_ms=0.186, ABC=0.120, BCA=0.045, Other=0.021
```

## 5. 最小改动实现方向（框架优先）

1. 在 timing scope 内引入“动态阶段桶”（map），按 `stage_id` 聚合。
2. 增加 `GUARD` 的可选布局参数（默认兼容旧行为）。
3. 保留旧宏接口兼容：
   - 旧 `MUSA_KERNEL_TRACE(x)` 可保留为 one-shot/兼容路径。
4. 输出层按布局渲染；无布局时回退默认顺序。

## 6. 验收点

1. LEVEL=1 输出总时间不变。
2. LEVEL=2 可按自定义阶段名输出。
3. 未覆盖时间进入 `Other`。
4. 关闭 `MUSA_KERNEL_DEBUG` 后无额外开销（宏空实现）。
