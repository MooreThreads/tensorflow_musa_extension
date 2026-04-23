cmake_minimum_required(VERSION 3.10)
project(musa_plugin)

# Force GCC compiler to align with TF core library ABI
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set MUSA path
set(MUSA_PATH "/usr/local/musa" CACHE STRING "Path to MUSA installation")
set(MCC_EXECUTABLE "${MUSA_PATH}/bin/mcc")

# Validate MUSA path exists
if(NOT EXISTS "${MUSA_PATH}")
    message(FATAL_ERROR "MUSA_PATH '${MUSA_PATH}' does not exist. Please set correct MUSA installation path.")
endif()

find_library(MUDNNCXX_LIBRARY NAMES mudnncxx PATHS "${MUSA_PATH}/lib" "${MUSA_PATH}/lib64" NO_DEFAULT_PATH)
find_library(MUDNN_LIBRARY NAMES mudnn PATHS "${MUSA_PATH}/lib" "${MUSA_PATH}/lib64" NO_DEFAULT_PATH)

if(MUDNNCXX_LIBRARY)
    set(MUSA_DNN_LIBRARY "${MUDNNCXX_LIBRARY}")
elseif(MUDNN_LIBRARY)
    set(MUSA_DNN_LIBRARY "${MUDNN_LIBRARY}")
else()
    message(FATAL_ERROR "Failed to find mudnncxx or mudnn under '${MUSA_PATH}/lib' or '${MUSA_PATH}/lib64'.")
endif()

# Get TensorFlow parameters
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE TF_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))"
    OUTPUT_VARIABLE TF_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

# Validate TensorFlow is available
if(NOT TF_INC)
    message(FATAL_ERROR "Failed to get TensorFlow include path. Please ensure TensorFlow is installed.")
endif()

# Clean ABI conflicts in TF_COMPILE_FLAGS
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=1" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
# Re-add the safest definition
set(TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# Configure logging-related compile definitions.
#
# Important: this plugin links against the pip TensorFlow wheel, which is built
# in release mode. Some TensorFlow headers (for example refcount.h) contain
# inline DCHECK-dependent side effects. If the plugin is compiled without
# NDEBUG while libtensorflow_framework is compiled with NDEBUG, the plugin and
# framework can disagree on RefCounted teardown semantics and trigger false
# refcount aborts during session/resource destruction. Keep NDEBUG enabled in
# every build to match the TensorFlow wheel.
#
# TF_PLUGGABLE_DEVICE is defined so plugin code can cheaply gate any
# diagnostics that only make sense in the PluggableDevice C API world.
set(TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS} -DMUSA_DISABLE_DEBUG_LOGGING -DNDEBUG -DMUSA_DISABLE_TRACE_LOGGING -DMUSA_DISABLE_AUTO_TRACE -DTF_PLUGGABLE_DEVICE")

# Include paths. The plugin now only consumes TF's public C API
# (tensorflow/c/experimental/stream_executor) and a small set of public core
# headers (framework, platform, stream_executor). The internal
# common_runtime/device headers are no longer needed: device/stream/allocator
# lifetime is owned by TF core via PluggableDevice.
include_directories("${TF_INC}" "${MUSA_PATH}/include")
if(EXISTS "${MUSA_PATH}/include/mudnncxx")
    include_directories("${MUSA_PATH}/include/mudnncxx")
endif()

# TensorFlow pip headers may place farmhash in different external include
# layouts. Some kernels still include `fingerprint.h` (which transitively
# depends on `<farmhash.h>`), so keep whichever directory actually ships with
# the installed wheel.
set(TF_FARMHASH_INC_SRC "${TF_INC}/external/farmhash_archive/src")
set(TF_FARMHASH_INC_VIRTUAL "${TF_INC}/external/farmhash_archive/_virtual_includes/farmhash")
if(EXISTS "${TF_FARMHASH_INC_SRC}/farmhash.h")
    include_directories("${TF_FARMHASH_INC_SRC}")
endif()
if(EXISTS "${TF_FARMHASH_INC_VIRTUAL}/farmhash.h")
    include_directories("${TF_FARMHASH_INC_VIRTUAL}")
endif()

# Define source root directory
set(SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/musa_ext")
include_directories("${SRC_ROOT}" "${SRC_ROOT}/mu" "${SRC_ROOT}/kernels")

# Validate source directory exists
if(NOT EXISTS "${SRC_ROOT}")
    message(FATAL_ERROR "Source directory '${SRC_ROOT}' does not exist.")
endif()

# Link paths
link_directories("${MUSA_PATH}/lib" "${MUSA_PATH}/lib64" "${TF_LIB_DIR}")

# -----------------------------------------------------------------------------
# libmusa_core.so — shared state that must live in exactly one place
# -----------------------------------------------------------------------------
#
# Any TU that defines a process-singleton must be listed here. Both
# libmusa_plugin.so and tensorflow_musa._C.so link against libmusa_core.so
# at runtime so they observe the same HostCachingAllocator / EventPool
# instances. Adding a new singleton (device caching allocator, telemetry
# snapshot, VMM reservations) means adding its .cc here — never in plugin.
set(MUSA_CORE_SOURCES
    "${SRC_ROOT}/mu/device/event_pool.cc"
    "${SRC_ROOT}/mu/device/host_caching_allocator.cc"
    "${SRC_ROOT}/mu/device/caching_allocator.cc"
    "${SRC_ROOT}/mu/device/driver_api.cc"
    "${SRC_ROOT}/mu/device/allocator_config.cc"
    "${SRC_ROOT}/mu/device/expandable_segment.cc"
    "${SRC_ROOT}/mu/device/peer_access.cc"
)

# -----------------------------------------------------------------------------
# libmusa_plugin.so — everything else in musa_ext/ minus core / python
# -----------------------------------------------------------------------------
file(GLOB_RECURSE ALL_CPP_SOURCES
    "${SRC_ROOT}/*.cc"
    "${SRC_ROOT}/mu/*.cc"
    "${SRC_ROOT}/kernels/*.cc"
)
# Exclude core sources (compiled into libmusa_core.so) and the Python
# extension entry point (compiled into tensorflow_musa._C.so).
list(REMOVE_ITEM ALL_CPP_SOURCES ${MUSA_CORE_SOURCES})
set(PYEXT_SOURCES "${SRC_ROOT}/python/_C.cpp")
foreach(exclude_file ${PYEXT_SOURCES})
    list(REMOVE_ITEM ALL_CPP_SOURCES ${exclude_file})
endforeach()
set(PLUGIN_SOURCES ${ALL_CPP_SOURCES})

# Automatically collect MUSA Kernel files
file(GLOB_RECURSE MU_SOURCES "${SRC_ROOT}/kernels/*.mu")

# TensorFlow headers pulled into `.mu` compilation require C++17, and mcc's
# aarch64 host predefines can incorrectly trigger NEON-only host headers.
set(MCC_COMMON_FLAGS
    -O3
    -fPIC
    -std=c++17
    -DEIGEN_DONT_VECTORIZE
    -U__ARM_NEON
    -U__ARM_NEON__
    -I${MUSA_PATH}/include
    -I${TF_INC}
)

# MCC compile MUSA Kernel
set(MU_OBJECTS "")
foreach(mu_file ${MU_SOURCES})
    get_filename_component(filename ${mu_file} NAME)
    if(NOT filename MATCHES "^disabled_")
        file(RELATIVE_PATH mu_rel_path "${SRC_ROOT}/kernels" "${mu_file}")
        string(REPLACE "/" "_" mu_rel_obj "${mu_rel_path}")
        string(REPLACE ".mu" "" mu_rel_obj "${mu_rel_obj}")
        set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${mu_rel_obj}.mu.o")
        set(mu_full_path "${mu_file}")

        add_custom_command(
            OUTPUT ${obj_file}
            COMMAND ${MCC_EXECUTABLE} -c ${mu_full_path} -o ${obj_file} ${MCC_COMMON_FLAGS}
            DEPENDS ${mu_full_path}
            COMMENT "Compiling MUSA kernel: ${filename}"
        )
        list(APPEND MU_OBJECTS ${obj_file})
    endif()
endforeach()

# -----------------------------------------------------------------------------
# Target: musa_core (SHARED) — singleton state
# -----------------------------------------------------------------------------
add_library(musa_core SHARED ${MUSA_CORE_SOURCES})
set_target_properties(musa_core PROPERTIES
    COMPILE_FLAGS "${TF_COMPILE_FLAGS}"
    PREFIX "lib"
    SUFFIX ".so"
    OUTPUT_NAME "musa_core"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    # $ORIGIN lets downstream .so files find core regardless of install path.
    INSTALL_RPATH "$ORIGIN;${MUSA_PATH}/lib;${MUSA_PATH}/lib64"
    # Position-independent code so both plugin and pybind ext can link.
    POSITION_INDEPENDENT_CODE ON
)
# Note: the MUSA runtime library is `libmusart.so` (not `libmusa.so`). The
# existing musa_plugin target used to pick it up transitively through
# mudnn/mublas, but musa_core links no such higher-level libs, so we depend
# on musart directly here.
target_link_libraries(musa_core PUBLIC musart ${CMAKE_DL_LIBS})

# -----------------------------------------------------------------------------
# Target: musa_plugin (SHARED) — TF PluggableDevice entry point
# -----------------------------------------------------------------------------
add_library(musa_plugin SHARED ${PLUGIN_SOURCES} ${MU_OBJECTS})
set_target_properties(musa_plugin PROPERTIES
    COMPILE_FLAGS "${TF_COMPILE_FLAGS}"
    PREFIX "lib"
    SUFFIX ".so"
    OUTPUT_NAME "musa_plugin"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    # Locate libmusa_core.so next to the plugin at runtime.
    INSTALL_RPATH "$ORIGIN;${TF_LIB_DIR};${MUSA_PATH}/lib;${MUSA_PATH}/lib64"
)
target_link_libraries(musa_plugin PRIVATE
    musa_core
    ${TF_LINK_FLAGS}
    musa
    mublas
    ${MUSA_DNN_LIBRARY}
    rt
)

# -----------------------------------------------------------------------------
# Target: tensorflow_musa__C — Python extension module
# -----------------------------------------------------------------------------
#
# Named `tensorflow_musa__C` inside CMake (target namespace uses no dots) but
# the produced shared object is `_C.<ext-suffix>.so` so that Python resolves
# it as `tensorflow_musa._C` after wheel install.
execute_process(COMMAND python3 -c "from sysconfig import get_paths; print(get_paths()['include'])"
    OUTPUT_VARIABLE PY_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_VARIABLE PY_EXT_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT PY_INC)
    message(FATAL_ERROR "Failed to locate Python include dir via sysconfig.")
endif()
if(NOT PY_EXT_SUFFIX OR PY_EXT_SUFFIX STREQUAL "None")
    # Fallback for old sysconfig layouts that return None.
    set(PY_EXT_SUFFIX ".so")
endif()

add_library(tensorflow_musa__C MODULE ${PYEXT_SOURCES})
target_include_directories(tensorflow_musa__C PRIVATE "${PY_INC}")
set_target_properties(tensorflow_musa__C PROPERTIES
    COMPILE_FLAGS "${TF_COMPILE_FLAGS}"
    PREFIX ""
    OUTPUT_NAME "_C"
    SUFFIX "${PY_EXT_SUFFIX}"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${MUSA_PATH}/lib;${MUSA_PATH}/lib64"
)
target_link_libraries(tensorflow_musa__C PRIVATE musa_core)

# -----------------------------------------------------------------------------
# Build summary and post-build message
# -----------------------------------------------------------------------------
add_custom_command(TARGET musa_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build completed successfully!"
    COMMENT "Plugin is located at: $<TARGET_FILE:musa_plugin>"
)

list(LENGTH PLUGIN_SOURCES PLUGIN_SOURCES_COUNT)
list(LENGTH MUSA_CORE_SOURCES CORE_SOURCES_COUNT)
list(LENGTH MU_SOURCES MU_SOURCES_COUNT)

message(STATUS "TensorFlow MUSA Plugin Configuration:")
message(STATUS "  MUSA Path: ${MUSA_PATH}")
message(STATUS "  MuDNN Library: ${MUSA_DNN_LIBRARY}")
message(STATUS "  TensorFlow Include: ${TF_INC}")
message(STATUS "  Python Include:     ${PY_INC}")
message(STATUS "  Python EXT_SUFFIX:  ${PY_EXT_SUFFIX}")
message(STATUS "  Core Sources:       ${CORE_SOURCES_COUNT}")
message(STATUS "  Plugin Sources:     ${PLUGIN_SOURCES_COUNT} C++ files, ${MU_SOURCES_COUNT} MUSA kernels")
message(STATUS "  Outputs:")
message(STATUS "    ${CMAKE_CURRENT_BINARY_DIR}/libmusa_core.so")
message(STATUS "    ${CMAKE_CURRENT_BINARY_DIR}/libmusa_plugin.so")
message(STATUS "    ${CMAKE_CURRENT_BINARY_DIR}/_C${PY_EXT_SUFFIX}")
