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

# Add compile definitions to disable debug logging
set(TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS} -DMUSA_DISABLE_DEBUG_LOGGING -DNDEBUG -DMUSA_DISABLE_TRACE_LOGGING -DMUSA_DISABLE_AUTO_TRACE")

# Include paths
include_directories("${TF_INC}" "${MUSA_PATH}/include")

# Define source root directory
set(SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/musa_ext")
include_directories("${SRC_ROOT}" "${SRC_ROOT}/mu")

# Validate source directory exists
if(NOT EXISTS "${SRC_ROOT}")
    message(FATAL_ERROR "Source directory '${SRC_ROOT}' does not exist.")
endif()

# Link paths
link_directories("${MUSA_PATH}/lib" "${MUSA_PATH}/lib64" "${TF_LIB_DIR}")

# Automatically collect source files
file(GLOB_RECURSE CPP_SOURCES
    "${SRC_ROOT}/*.cc"
    "${SRC_ROOT}/mu/*.cc"
    "${SRC_ROOT}/kernels/*.cc"
)

# filter out disabled kernels (if any)
# list(FILTER CPP_SOURCES EXCLUDE REGEX ".*test.*|.*disabled.*")
# Automatically collect MUSA Kernel files
file(GLOB MU_SOURCES "${SRC_ROOT}/kernels/*.mu")

# MCC compile MUSA Kernel
set(MU_OBJECTS "")
foreach(mu_file ${MU_SOURCES})
    get_filename_component(filename ${mu_file} NAME)
    if(NOT filename MATCHES "^disabled_")
        get_filename_component(basename ${mu_file} NAME_WE)
        set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${basename}.mu.o")
        set(mu_full_path "${mu_file}")

        add_custom_command(
            OUTPUT ${obj_file}
            COMMAND ${MCC_EXECUTABLE} -c ${mu_full_path} -o ${obj_file}
            -O3 -fPIC -I${MUSA_PATH}/include -I${TF_INC}
            DEPENDS ${mu_full_path}
            COMMENT "Compiling MUSA kernel: ${filename}"
        )
        list(APPEND MU_OBJECTS ${obj_file})
    endif()
endforeach()

# Generate plugin library
add_library(musa_plugin SHARED ${CPP_SOURCES} ${MU_OBJECTS})

# Apply TF compile flags (ensure strict ABI consistency)
set_target_properties(musa_plugin PROPERTIES COMPILE_FLAGS "${TF_COMPILE_FLAGS}")

# Set RPATH to ensure runtime library search
set_target_properties(musa_plugin PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${TF_LIB_DIR};${MUSA_PATH}/lib;${MUSA_PATH}/lib64"
)

# Link
target_link_libraries(musa_plugin PRIVATE ${TF_LINK_FLAGS} musa mublas mudnn rt)

# Output configuration
set_target_properties(musa_plugin PROPERTIES
    PREFIX "lib"
    SUFFIX ".so"
    OUTPUT_NAME "musa_plugin"
)

# Build completion message
add_custom_command(TARGET musa_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build completed successfully!"
    COMMENT "Plugin is located at: $<TARGET_FILE:musa_plugin>"
)

# Print build summary
list(LENGTH CPP_SOURCES CPP_SOURCES_COUNT)
list(LENGTH MU_SOURCES MU_SOURCES_COUNT)

message(STATUS "TensorFlow MUSA Plugin Configuration:")
message(STATUS "  MUSA Path: ${MUSA_PATH}")
message(STATUS "  TensorFlow Include: ${TF_INC}")
message(STATUS "  Source Files: ${CPP_SOURCES_COUNT} C++ files, ${MU_SOURCES_COUNT} MUSA kernels")
message(STATUS "  Output: ${CMAKE_CURRENT_BINARY_DIR}/libmusa_plugin.so")