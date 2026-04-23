# Internal Documentation

This directory holds **internal-only** documentation that is **not intended for the open-source release** of `tensorflow_musa_extension`.

## What lives here

| File | Purpose |
|------|---------|
| [`architecture-and-memory.md`](./architecture-and-memory.md) | Full architecture walkthrough + memory scheduling + debugging playbooks (the 10-/30-minute onboarding doc). |
| [`memory-optimization.md`](./memory-optimization.md) | Change history of the Host/Device caching allocator work (commit-by-commit). |
| [`tf-compat-matrix.md`](./tf-compat-matrix.md) | TensorFlow version compatibility matrix + rationale. |
| [`DEBUG_GUIDE.md`](./DEBUG_GUIDE.md) | Production / on-site debugging command book (telemetry, kernel timing, memory coloring, etc.). |

## Why they are kept separate

Public users of the wheel only need:

- The **runtime knobs** — documented in [`../docs/environment-variables.md`](../docs/environment-variables.md).
- The **Python API** — documented in the package docstrings and sampled in [`../README.md`](../README.md).

The documents in this directory contain internal design rationale, commit bisection notes, raw debugging recipes and forward-looking TODOs — valuable for the team, but noise (and a maintenance burden) for external consumers.

## Keeping the public release clean

Exclude this directory when producing the open-source artifact. Suggested patterns:

```gitignore
# .gitignore for an open-source mirror / export
/internal-docs/
```

Or, when exporting via `rsync` / `tar`:

```bash
rsync -a --exclude='/internal-docs/' ./ /path/to/oss-mirror/
tar --exclude='internal-docs' -czf tensorflow_musa_extension.tar.gz .
```

Whichever mechanism you adopt, verify `docs/` contains **only** `environment-variables.md` before publishing.
