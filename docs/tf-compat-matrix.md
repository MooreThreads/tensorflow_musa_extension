# TensorFlow Compatibility Matrix

This document tracks the TensorFlow versions that `tensorflow_musa_extension`
is designed to support and which of them have been end-to-end validated.

## Design goal

The plugin only talks to TensorFlow through the public **PluggableDevice C
ABI** (`SP_Platform`, `SP_StreamExecutor`, `SP_Device`, `SP_DeviceMemoryBase`,
etc.). This ABI has been append-only with `SE_MAJOR=0` since TF 2.5. As long
as TF keeps that contract, the plugin does not need to be rebuilt against
internal TF headers for every release.

All TF C-API header inclusion goes through a single shim:

- [`musa_ext/mu/tf_compat.h`](../musa_ext/mu/tf_compat.h)

That shim does three things:

1. Centralizes the `#include` of TF PluggableDevice headers.
2. Pins `SE_MAJOR == 0` with `static_assert`, so a breaking major bump fails
   at compile time rather than at runtime.
3. Reserves space for `TF_MUSA_HAS_*` feature macros to be used from `.cc`
   files instead of raw TF version arithmetic.

## Supported version range

The plugin's build tooling (both `setup.py` and `build.sh`) validates the
installed TensorFlow version against the range:

- **MIN**: `2.6` (inclusive)
- **MAX**: `2.17` (exclusive)
- **RECOMMENDED** (primary test target): `2.6.1`

Anything outside the range causes the build to abort with a clear error
message. Anything inside the range builds; the note `"within the supported
range; the primary test matrix uses 2.6.1"` is printed when you use a
non-recommended version.

## Validation status

The columns below should be kept up-to-date as we validate new versions.

| TF version | Build on Linux x86_64 | Plugin load | Allocator smoke | Status |
|------------|:---------------------:|:-----------:|:---------------:|--------|
| 2.6.1      | ok                    | ok          | ok              | Primary test target |
| 2.10.x     | pending               | pending     | pending         | Planned (C0 CI matrix) |
| 2.15.x     | pending               | pending     | pending         | Planned (C0 CI matrix) |

"ok" means we verified it locally or in CI.
"pending" means the plan calls for it but evidence has not been recorded.

## How to add a new version to the matrix

1. Build and load the wheel against that TF version in a clean venv.
2. Run the smoke tests under `test/` that touch the PluggableDevice path
   (allocator, H2D/D2H, a minimal op) end-to-end.
3. If anything fails, either:
   - file a compatibility bug and keep the version marked as `fail`, or
   - add a feature macro in `tf_compat.h` and a conditional branch in the
     call sites, then re-test.
4. Update the table above.

## What to do when a future TF bumps the ABI

- **`SE_MAJOR` bump**: `tf_compat.h`'s `static_assert` will fail. Audit the
  entire `PopulateStreamExecutor` table (see
  [`musa_se_callbacks.cc`](../musa_ext/mu/device/musa_se_callbacks.cc)) for
  signature changes before raising the assert.
- **New field added, same major**: TF's `struct_size` handshake makes this a
  non-event for us. If we want to use the new field, add a
  `TF_MUSA_HAS_<NAME>` macro in `tf_compat.h` guarded by the appropriate
  `#if defined(SP_*_STRUCT_SIZE) && SP_*_STRUCT_SIZE >= ...` check, then
  `#ifdef`-gate the call site.
- **Runtime incompatibility**: `SE_InitPlugin` copies the `struct_size` from
  `SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE`. If our plugin is loaded by
  a TF that sees an incompatible size, TF refuses to enable the device and
  reports a clear error; we do NOT silently misbehave.

## Known compatibility boundaries

- `platform->use_bfc_allocator = 1` (current default) ties us to TF's BFC
  sub-allocator path. The plan's phase 2 flips this to 0 so the plugin owns
  caching end-to-end; that flip is also ABI-safe because `use_bfc_allocator`
  has existed since the experimental API was introduced.
- `platform->supports_unified_memory = 0` is our current default; do NOT set
  it to 1 without also populating `unified_memory_allocate/deallocate`.
