cmake_minimum_required(VERSION 3.10)
project(musa_plugin)

# --- 强制指定编译器为 GCC ---
# 必须在 project() 之后立即设置，确保插件与 TF 核心库 ABI 对齐
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 设置 MUSA 路径
set(MUSA_PATH "/usr/local/musa" CACHE STRING "Path to MUSA installation")
set(MCC_EXECUTABLE "${MUSA_PATH}/bin/mcc")

# 验证 MUSA 路径是否存在
if(NOT EXISTS "${MUSA_PATH}")
    message(FATAL_ERROR "MUSA_PATH '${MUSA_PATH}' does not exist. Please set correct MUSA installation path.")
endif()

# 2. 获取 TensorFlow 参数
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE TF_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))"
    OUTPUT_VARIABLE TF_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

# 验证 TensorFlow 是否可用
if(NOT TF_INC)
    message(FATAL_ERROR "Failed to get TensorFlow include path. Please ensure TensorFlow is installed.")
endif()

# --- 核心修改 2: 清理 TF_COMPILE_FLAGS 中的 ABI 冲突 ---
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=1" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
# 重新加上最稳妥的定义
set(TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# 3. 包含路径
include_directories("${TF_INC}" "${MUSA_PATH}/include")

# 定义源码根目录
set(SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/musa_ext")
include_directories("${SRC_ROOT}" "${SRC_ROOT}/mu")

# 验证源码目录是否存在
if(NOT EXISTS "${SRC_ROOT}")
    message(FATAL_ERROR "Source directory '${SRC_ROOT}' does not exist.")
endif()

# 4. 链接路径
link_directories("${MUSA_PATH}/lib" "${MUSA_PATH}/lib64" "${TF_LIB_DIR}")

# 5. 自动收集源文件（简化配置）
# 收集所有 .cc 文件，包括 optimizer 目录中的特殊文件
file(GLOB_RECURSE CPP_SOURCES 
    "${SRC_ROOT}/*.cc"
    "${SRC_ROOT}/mu/*.cc"
    "${SRC_ROOT}/kernels/*.cc"
)

# 过滤掉可能不需要的文件（如果有的话）
# list(FILTER CPP_SOURCES EXCLUDE REGEX ".*test.*|.*disabled.*")

# 6. 自动收集 MUSA Kernel 文件
file(GLOB MU_SOURCES "${SRC_ROOT}/kernels/*.mu")

# 如果需要排除某些 .mu 文件，可以在这里注释掉对应的行
# 或者在文件名前加前缀如 "disabled_" 来排除

# 7. MCC 编译 MUSA Kernel
set(MU_OBJECTS "")
foreach(mu_file ${MU_SOURCES})
    # 跳过被注释或禁用的文件（如果文件名包含特定前缀）
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

# 8. 生成插件库
add_library(musa_plugin SHARED ${CPP_SOURCES} ${MU_OBJECTS})

# 9. 应用 TF 编译标志 (确保 ABI 严格一致)
# 使用与原文件相同的方式设置编译标志
set_target_properties(musa_plugin PROPERTIES COMPILE_FLAGS "${TF_COMPILE_FLAGS}")

# 10. 设置 RPATH 确保运行时库搜索
set_target_properties(musa_plugin PROPERTIES 
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${TF_LIB_DIR};${MUSA_PATH}/lib;${MUSA_PATH}/lib64"
)

# 11. 链接
target_link_libraries(musa_plugin PRIVATE ${TF_LINK_FLAGS} musa mublas mudnn rt)

# 12. 安装和输出配置
set_target_properties(musa_plugin PROPERTIES 
    PREFIX "lib" 
    SUFFIX ".so"
    OUTPUT_NAME "musa_plugin"
)

# 输出信息
add_custom_command(TARGET musa_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build completed successfully!"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:musa_plugin> /workspace/tensorflow-plugins/libmusa_plugin.so
    COMMENT "Copying plugin to /workspace/tensorflow-plugins/"
)

# 获取源文件数量用于显示
list(LENGTH CPP_SOURCES CPP_SOURCES_COUNT)
list(LENGTH MU_SOURCES MU_SOURCES_COUNT)

# 打印构建摘要
message(STATUS "TensorFlow MUSA Plugin Configuration:")
message(STATUS "  MUSA Path: ${MUSA_PATH}")
message(STATUS "  TensorFlow Include: ${TF_INC}")
message(STATUS "  Source Files: ${CPP_SOURCES_COUNT} C++ files, ${MU_SOURCES_COUNT} MUSA kernels")
message(STATUS "  Output: ${CMAKE_CURRENT_BINARY_DIR}/libmusa_plugin.so")