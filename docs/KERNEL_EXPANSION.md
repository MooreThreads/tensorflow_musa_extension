# SE-only kernel expansion checklist

Order matches the Pure PluggableDevice migration plan; each row should add an
**SE-only subprocess test** before broad kernel edits, then run default C++
`MusaDevice` regression.

| Phase | Category | Examples / notes |
|-------|----------|------------------|
| 1 | Elementwise fast path | `AddV2`, `Sub`, `Mul`, unary — custom launch + `QueryMusaKernelRuntimeView` stream |
| 2 | Shape / copy | `Identity`, `Reshape`, `ZerosLike`, memcpy-style copies |
| 3 | muDNN-backed | Use `GetHandleByCtx` / `MUSA_OP_REQUIRES_MUDNN_HANDLE`; SE path via registry `mudnn_handle` |
| 4 | Stateful / `ResourceVariable` | Last — ordering, allocator, and sync semantics |

Cross-links: [`COMPATIBILITY.md`](COMPATIBILITY.md), [`CI.md`](CI.md).
