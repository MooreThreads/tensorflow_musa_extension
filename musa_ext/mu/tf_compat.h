/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// TensorFlow PluggableDevice compatibility shim.
//
// All PluggableDevice C API headers from upstream TensorFlow must be included
// through this file. The goal is to make the plugin tolerant of TensorFlow
// version differences (2.6.x ... 2.16.x tested) without touching every .cc
// file in the plugin every time TF bumps its C ABI:
//
//   * Header locations are centralized here, so if TF relocates a header in a
//     future release we only patch this one file.
//   * `SE_MAJOR` / `SE_MINOR` from the PluggableDevice C API are pinned via
//     static_assert, giving a clear build-time error instead of UB when the
//     struct layout changes in an incompatible way.
//   * Feature macros (TF_MUSA_HAS_*) are declared here; plugin code uses
//     `#ifdef` instead of hard-coding TF version checks.
//
// Important: the PluggableDevice C API contract is struct-size-versioned.
// TF fills `params->struct_size` with `SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE`
// and checks ours when it reads us back. New fields are append-only, so an
// older plugin still runs against a newer TF (TF only looks at the fields it
// announced via struct_size). This shim documents which fields we rely on.

#ifndef TENSORFLOW_MUSA_MU_TF_COMPAT_H_
#define TENSORFLOW_MUSA_MU_TF_COMPAT_H_

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"

// ---------------------------------------------------------------------------
// Version assertions.
// ---------------------------------------------------------------------------
//
// TF introduced SE_MAJOR / SE_MINOR / SE_PATCH in the 2.5 cycle. The plugin
// only supports the SE_MAJOR=0 ABI (that is the "experimental" ABI used from
// TF 2.5 through at least 2.16). If TF ever bumps SE_MAJOR, we need to audit
// the whole SE callback table, so surface that as a hard error.
#ifndef SE_MAJOR
#error "tf_compat.h: SE_MAJOR not defined by TF's stream_executor.h; \
cannot determine PluggableDevice ABI version."
#endif

static_assert(SE_MAJOR == 0,
              "tensorflow_musa_extension only supports the SE_MAJOR=0 "
              "PluggableDevice C ABI (TF 2.5+). A newer major version "
              "requires re-auditing SP_StreamExecutor / SP_Platform.");

// SE_MINOR grew as TF added fields. We do NOT fail on higher minors (that is
// the whole point of append-only ABI); we just document the minimum we
// require and rely on TF's struct_size checks at runtime.
#if SE_MAJOR == 0 && SE_MINOR < 0
#error "tf_compat.h: unsupported SE_MINOR."
#endif

// ---------------------------------------------------------------------------
// Feature macros.
// ---------------------------------------------------------------------------
//
// Add TF_MUSA_HAS_* macros here as we discover version-gated fields. Use them
// with #ifdef in .cc files rather than raw TF version arithmetic.
//
// Example (placeholder; enable when we actually consume a gated field):
//   #if defined(SP_STREAMEXECUTOR_STRUCT_SIZE) && \
//       (SP_STREAMEXECUTOR_STRUCT_SIZE >= <threshold>)
//   #define TF_MUSA_HAS_SE_SOME_NEW_CALLBACK 1
//   #endif

// ---------------------------------------------------------------------------
// Sanity checks for struct_size macros we consume in PopulateStreamExecutor
// and SE_InitPlugin. If TF removed or renamed any of these, we want a
// compile-time failure at the shim layer.
// ---------------------------------------------------------------------------

#ifndef SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE
#error "tf_compat.h: SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE missing."
#endif

#ifndef SP_PLATFORM_STRUCT_SIZE
#error "tf_compat.h: SP_PLATFORM_STRUCT_SIZE missing."
#endif

#ifndef SP_STREAMEXECUTOR_STRUCT_SIZE
#error "tf_compat.h: SP_STREAMEXECUTOR_STRUCT_SIZE missing."
#endif

#ifndef SP_DEVICE_STRUCT_SIZE
#error "tf_compat.h: SP_DEVICE_STRUCT_SIZE missing."
#endif

#ifndef SP_DEVICE_MEMORY_BASE_STRUCT_SIZE
#error "tf_compat.h: SP_DEVICE_MEMORY_BASE_STRUCT_SIZE missing."
#endif

#endif  // TENSORFLOW_MUSA_MU_TF_COMPAT_H_
