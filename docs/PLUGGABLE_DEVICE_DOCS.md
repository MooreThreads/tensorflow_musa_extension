# PluggableDevice 文档索引

本文档用于串联 PluggableDevice 迁移过程中的计划、开发记录、review 记录和验证文档。后续每次分阶段修改时，优先从这里找到对应的 review 文档和验收入口。

## 使用方式

1. 先根据本次修改所属阶段查看“阶段索引”。
2. 修改代码前，对照迁移计划确认阶段目标和不做事项。
3. 修改完成后，在对应开发记录中补充实现、验证结果和已知限制。
4. review 时，在对应 review 文档中记录问题、结论、修复顺序和补测建议。
5. 合并前，对照 CI、兼容性和 kernel 扩展文档确认验收命令与覆盖范围。

## 文档角色

| 文档 | 角色 | 维护时机 |
| --- | --- | --- |
| `PLUGGABLE_DEVICE_MIGRATION_PLAN.md` | 全量迁移计划，定义 Phase 0-9 和 Milestone A-E | 阶段边界、目标或优先级变化时更新 |
| `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md` | 当前阶段开发日志，记录已完成内容、验证结果、限制和后续任务 | 每轮实现完成后更新 |
| `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md` | 当前阶段 review 记录，记录问题、风险、修复顺序和补测建议 | 每轮 review 后更新 |
| `CI.md` | Pluggable 默认路径的固定构建/测试命令 | 测试入口、CI 门禁或命令变化时更新 |
| `COMPATIBILITY.md` | TensorFlow 版本、构建矩阵、legacy/Pluggable 路径兼容说明 | 支持矩阵或默认路径语义变化时更新 |
| `KERNEL_EXPANSION.md` | kernel 扩展 checklist | 新增或迁移 kernel 类别时更新 |

## 阶段索引

| 阶段 | 计划入口 | 开发记录入口 | Review 入口 | 验收/辅助文档 |
| --- | --- | --- | --- | --- |
| Phase 0：冻结 legacy 扩张 | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-0冻结-legacy-扩张` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#后续建议任务` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#从-pluggabledevice-规范角度` | `COMPATIBILITY.md#pluggabledevice-c-api-与-c-设备路径互斥` |
| Phase 1：Python 入口 PluggableDevice 优先 | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-1python-入口改为-pluggabledevice-优先` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#2-python-loader-改为-pluggabledevice-优先` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#1-默认-loader-对同一-so-双加载风险` | `CI.md#默认-pluggabledevice-路径需-musa-设备时由测试-skip` |
| Phase 2：默认注册路径切换 | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-2pluggabledevice-成为默认注册路径` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#1-默认设备注册路径切换` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#进程加载顺序` | `COMPATIBILITY.md#pluggabledevice-c-api-与-c-设备路径互斥` |
| Phase 3：stream 获取机制修正 | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-3修正-pluggabledevice-stream-获取机制` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#3-stream-获取安全边界修复` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#指针size-校验` | `CI.md#pluggable-端到端-eager子进程load_pluggable_device_libraryload_op_library` |
| Phase 4：kernel runtime 迁移 | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-4kernel-runtime-完全迁移到-pluggabledevice` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#p1扩大-pluggabledevice-kernel-覆盖` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#建议补充测试` | `KERNEL_EXPANSION.md` |
| Phase 5：allocator / memory / callback | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-5完善-allocator--memory--host-callback-语义` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#5-runtime--error-message-同步` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#2-plugin_se_allocate-失败状态不符合分配语义` | `COMPATIBILITY.md#kernel-与设备实现` |
| Phase 6：muDNN handle registry | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-6完善-mudnn-handle-registry` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#p3测试矩阵补强` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#3-se-per-stream-mudnn-handle-缓存缺少-stream-生命周期清理` | `COMPATIBILITY.md#kernel-与设备实现` |
| Phase 7：device 属性与 placement | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-7补齐-device-属性和-placement-信息` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#p3测试矩阵补强` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#5-loader-health-check-对-physicaldevice-判断错误` | `CI.md#se-api--纯子进程不依赖-musa_test_utils-预加载` |
| Phase 8：测试体系升级 | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-8测试体系升级` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#7-测试更新` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#cpu-only-ci-覆盖不足` | `CI.md#一次性跑以上-pluggable-相关测试` |
| Phase 9：删除 legacy 路径 | `PLUGGABLE_DEVICE_MIGRATION_PLAN.md#phase-9删除-legacy-musadevice-注册路径` | `PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#p5legacy-路径删除` | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md#建议修复顺序` | `COMPATIBILITY.md#发布与验证矩阵按-tensorflow-次版本门控` |

## 当前阶段映射

当前已有开发记录和 review 记录主要覆盖 Milestone A / Phase 1-3，并包含 Phase 4-8 的后续任务入口：

- 默认 PluggableDevice 加载：开发记录 `已完成内容` 1、2；review 记录问题 1、4、5。
- SE C API runtime 与 stream：开发记录 `已完成内容` 3、5；review 记录问题 2、3 和健壮性章节。
- eager Add 最小闭环：开发记录 `已完成内容` 4、7；review 记录建议补充测试。
- 后续 kernel 扩展：迁移计划 Phase 4、Phase 8；开发记录 P1/P3；`KERNEL_EXPANSION.md`。

## 后续 review 文档命名规则

当 review 开始按阶段拆分时，使用以下命名，避免一个 review 文件无限增长：

| 阶段范围 | 建议 review 文件 |
| --- | --- |
| Phase 1-3 / Milestone A | `PLUGGABLE_DEVICE_REVIEW_FINDINGS.md` |
| Phase 4 / Milestone B | `PLUGGABLE_DEVICE_REVIEW_PHASE4_KERNEL_RUNTIME.md` |
| Phase 5-6 / allocator + muDNN registry | `PLUGGABLE_DEVICE_REVIEW_PHASE5_6_RUNTIME.md` |
| Phase 7-8 / device 属性 + 测试体系 | `PLUGGABLE_DEVICE_REVIEW_PHASE7_8_TESTING.md` |
| Phase 9 / legacy 删除 | `PLUGGABLE_DEVICE_REVIEW_PHASE9_LEGACY_REMOVAL.md` |

新增 review 文档后，需要在本文档的“阶段索引”和“当前阶段映射”中补充对应入口。

## 每轮修改的最小记录模板

开发记录建议追加以下内容：

```markdown
## YYYY-MM-DD：<阶段 / 主题>

### 范围
- 对应阶段：Phase N / Milestone X
- 主要文件：...

### 已完成
- ...

### 验证
- 命令：...
- 结果：...

### 已知限制
- ...

### 对应 review
- `PLUGGABLE_DEVICE_REVIEW_*.md#...`
```

Review 记录建议追加以下内容：

```markdown
## YYYY-MM-DD：<阶段 / 主题> review

### 范围
- 对应开发记录：`PLUGGABLE_DEVICE_DEVELOPMENT_LOG.md#...`
- 对应计划阶段：`PLUGGABLE_DEVICE_MIGRATION_PLAN.md#...`

### 结论
- ...

### 必修项
- [ ] ...

### 建议项
- [ ] ...

### 补测建议
- ...
```
