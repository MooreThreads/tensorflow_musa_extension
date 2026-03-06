cmake_minimum_required(VERSION 3.10) # [cite: 1]
project(musa_plugin) # [cite: 1]

# Force GCC compiler to align with TF core library ABI
set(CMAKE_C_COMPILER gcc) # [cite: 1]
set(CMAKE_CXX_COMPILER g++) # [cite: 1]

set(CMAKE_CXX_STANDARD 14) # [cite: 1]
set(CMAKE_CXX_STANDARD_REQUIRED ON) # [cite: 1]

# Set MUSA path
set(MUSA_PATH "/usr/local/musa" CACHE STRING "Path to MUSA installation") # [cite: 1]
set(MCC_EXECUTABLE "${MUSA_PATH}/bin/mcc") # [cite: 1]

# Validate MUSA path exists
if(NOT EXISTS "${MUSA_PATH}") # [cite: 1]
    message(FATAL_ERROR "MUSA_PATH '${MUSA_PATH}' does not exist. Please set correct MUSA installation path.") # [cite: 1]
endif() # [cite: 1]

# Get TensorFlow parameters
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())" # [cite: 1]
    OUTPUT_VARIABLE TF_INC OUTPUT_STRIP_TRAILING_WHITESPACE) # [cite: 1]
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))" # [cite: 1]
    OUTPUT_VARIABLE TF_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE) # [cite: 1]
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))" # [cite: 1]
    OUTPUT_VARIABLE TF_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE) # [cite: 1]
execute_process(COMMAND python3 -c "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))" # [cite: 1, 2]
    OUTPUT_VARIABLE TF_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE) # [cite: 2]

# Validate TensorFlow is available
if(NOT TF_INC) # [cite: 2]
    message(FATAL_ERROR "Failed to get TensorFlow include path. Please ensure TensorFlow is installed.") # [cite: 2]
endif() # [cite: 2]

# Clean ABI conflicts in TF_COMPILE_FLAGS
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}") # [cite: 2]
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=1" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}") # [cite: 2]
# Re-add the safest definition
set(TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0") # [cite: 2]

# Add compile definitions to disable debug logging
set(TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS} -DMUSA_DISABLE_DEBUG_LOGGING -DNDEBUG -DMUSA_DISABLE_TRACE_LOGGING -DMUSA_DISABLE_AUTO_TRACE") # [cite: 2]

# Include paths
include_directories("${TF_INC}" "${MUSA_PATH}/include") # [cite: 2]

# Define source root directory
set(SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/musa_ext") # [cite: 2]
include_directories("${SRC_ROOT}" "${SRC_ROOT}/mu" "${SRC_ROOT}/kernels") # [cite: 2]

# Validate source directory exists
if(NOT EXISTS "${SRC_ROOT}") # [cite: 2]
    message(FATAL_ERROR "Source directory '${SRC_ROOT}' does not exist.") # [cite: 2]
endif() # [cite: 2]

# Link paths
link_directories("${MUSA_PATH}/lib" "${MUSA_PATH}/lib64" "${TF_LIB_DIR}") # [cite: 2]

set(CPP_SOURCES
    "${SRC_ROOT}/mu/device_register.cc"
    "${SRC_ROOT}/mu/kernel_register.cc"
    "${SRC_ROOT}/mu/device/musa_device.cc"
    "${SRC_ROOT}/mu/device/musa_allocator.cc"
    "${SRC_ROOT}/mu/device/musa_memcpy.cc"
    "${SRC_ROOT}/mu/device/musa_memset.cc"
    "${SRC_ROOT}/mu/device/musa_platform.cc"
    "${SRC_ROOT}/mu/device/musa_utils.cc"
    "${SRC_ROOT}/mu/optimizer/musa_graph_optimizer.cc"
    "${SRC_ROOT}/kernels/utils_op.cc"

    "${SRC_ROOT}/kernels/array/musa_ZerosLike_op.cc"
    "${SRC_ROOT}/kernels/array/musa_broadcast_gradient_args_op.cc"
    "${SRC_ROOT}/kernels/array/musa_broadcast_to_op.cc"
    "${SRC_ROOT}/kernels/array/musa_concat_offset_op.cc"
    "${SRC_ROOT}/kernels/array/musa_concat_op.cc"
    "${SRC_ROOT}/kernels/array/musa_const_op.cc"
    "${SRC_ROOT}/kernels/array/musa_diag_part_op.cc"
    "${SRC_ROOT}/kernels/array/musa_expand_dims_op.cc"
    "${SRC_ROOT}/kernels/array/musa_fill_op.cc"
    "${SRC_ROOT}/kernels/array/musa_gather_nd_op.cc"
    "${SRC_ROOT}/kernels/array/musa_gather_op.cc"
    "${SRC_ROOT}/kernels/array/musa_identity_op.cc"
    "${SRC_ROOT}/kernels/array/musa_invertpermutation_op.cc"
    "${SRC_ROOT}/kernels/array/musa_pack_op.cc"
    "${SRC_ROOT}/kernels/array/musa_pad_op.cc"
    "${SRC_ROOT}/kernels/array/musa_placeholder_op.cc"
    "${SRC_ROOT}/kernels/array/musa_range_op.cc"
    "${SRC_ROOT}/kernels/array/musa_reshape_op.cc"
    "${SRC_ROOT}/kernels/array/musa_resourcegather_op.cc"
    "${SRC_ROOT}/kernels/array/musa_shape_op.cc"
    "${SRC_ROOT}/kernels/array/musa_size_op.cc"
    "${SRC_ROOT}/kernels/array/musa_slice_op.cc"
    "${SRC_ROOT}/kernels/array/musa_split_op.cc"
    "${SRC_ROOT}/kernels/array/musa_split_v_op.cc"
    "${SRC_ROOT}/kernels/array/musa_squeeze_op.cc"
    "${SRC_ROOT}/kernels/array/musa_stopgradient_op.cc"
    "${SRC_ROOT}/kernels/array/musa_strided_slice_op.cc"
    "${SRC_ROOT}/kernels/array/musa_tensorinteraction_op.cc"
    "${SRC_ROOT}/kernels/array/musa_tile_op.cc"
    "${SRC_ROOT}/kernels/array/musa_transpose_op.cc"
    "${SRC_ROOT}/kernels/array/musa_unique_op.cc"
    "${SRC_ROOT}/kernels/control_flow/musa_arg_op.cc"
    "${SRC_ROOT}/kernels/control_flow/musa_merge_op.cc"
    "${SRC_ROOT}/kernels/control_flow/musa_switch_op.cc"
    "${SRC_ROOT}/kernels/io/musa_merge_v2_checkpoints_op.cc"
    "${SRC_ROOT}/kernels/io/musa_restore_v2_op.cc"
    "${SRC_ROOT}/kernels/io/musa_save_v2_op.cc"
    "${SRC_ROOT}/kernels/io/musa_sharded_filename_op.cc"
    "${SRC_ROOT}/kernels/logging/musa_logging_ops.cc"
    "${SRC_ROOT}/kernels/math/musa_SquaredDifference_op.cc"
    "${SRC_ROOT}/kernels/math/musa_abs_op.cc"
    "${SRC_ROOT}/kernels/math/musa_add_op.cc"
    "${SRC_ROOT}/kernels/math/musa_addn_op.cc"
    "${SRC_ROOT}/kernels/math/musa_argmax_op.cc"
    "${SRC_ROOT}/kernels/math/musa_bitwise_and_op.cc"
    "${SRC_ROOT}/kernels/math/musa_cast_op.cc"
    "${SRC_ROOT}/kernels/math/musa_conv2d_backward_op.cc"
    "${SRC_ROOT}/kernels/math/musa_conv2d_op.cc"
    "${SRC_ROOT}/kernels/math/musa_div_no_nan_op.cc"
    "${SRC_ROOT}/kernels/math/musa_einsum_op.cc"
    "${SRC_ROOT}/kernels/math/musa_equal_op.cc"
    "${SRC_ROOT}/kernels/math/musa_erf_op.cc"
    "${SRC_ROOT}/kernels/math/musa_exp_op.cc"
    "${SRC_ROOT}/kernels/math/musa_expm1_op.cc"
    "${SRC_ROOT}/kernels/math/musa_floor_div_op.cc"
    "${SRC_ROOT}/kernels/math/musa_floor_mod_op.cc"
    "${SRC_ROOT}/kernels/math/musa_fused_matmul_op.cc"
    "${SRC_ROOT}/kernels/math/musa_greater_op.cc"
    "${SRC_ROOT}/kernels/math/musa_isnan_op.cc"
    "${SRC_ROOT}/kernels/math/musa_less_op.cc"
    "${SRC_ROOT}/kernels/math/musa_log1p_op.cc"
    "${SRC_ROOT}/kernels/math/musa_log_op.cc"
    "${SRC_ROOT}/kernels/math/musa_logical_not_op.cc"
    "${SRC_ROOT}/kernels/math/musa_logical_or_op.cc"
     "${SRC_ROOT}/kernels/math/musa_matmul_op.cc"
    "${SRC_ROOT}/kernels/math/musa_max_op.cc"
    "${SRC_ROOT}/kernels/math/musa_maximum_op.cc"
    "${SRC_ROOT}/kernels/math/musa_mean_op.cc"
    "${SRC_ROOT}/kernels/math/musa_minimum_op.cc"
    "${SRC_ROOT}/kernels/math/musa_multiply_op.cc"
    "${SRC_ROOT}/kernels/math/musa_neg_op.cc"
    "${SRC_ROOT}/kernels/math/musa_pow_op.cc"
    "${SRC_ROOT}/kernels/math/musa_prod_op.cc"
    "${SRC_ROOT}/kernels/math/musa_realdiv_op.cc"
    "${SRC_ROOT}/kernels/math/musa_reciprocal_op.cc"
    "${SRC_ROOT}/kernels/math/musa_round_op.cc"
    "${SRC_ROOT}/kernels/math/musa_select_op.cc"
    "${SRC_ROOT}/kernels/math/musa_sign_op.cc"
    "${SRC_ROOT}/kernels/math/musa_sqrt_grad_op.cc"
    "${SRC_ROOT}/kernels/math/musa_sqrt_op.cc"
    "${SRC_ROOT}/kernels/math/musa_square_op.cc"
    "${SRC_ROOT}/kernels/math/musa_sub_op.cc"
    "${SRC_ROOT}/kernels/math/musa_sum_op.cc"
    "${SRC_ROOT}/kernels/math/musa_unsorted_segment_sum_op.cc"
    "${SRC_ROOT}/kernels/math/rsqrt_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_biasadd_grad_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_biasadd_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_fused_batchnorm_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_layernorm_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_leakyrelu_grad_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_leakyrelu_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_relu_grad_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_relu_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_sigmoid_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_softmax_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_softplus_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_sparse_xent_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_tanh_grad_op.cc"
    "${SRC_ROOT}/kernels/nn/musa_tanh_op.cc"
    "${SRC_ROOT}/kernels/random/musa_random_op.cc"
    "${SRC_ROOT}/kernels/random/musa_random_ops.cc"
    "${SRC_ROOT}/kernels/state/musa_assign_op.cc"
    "${SRC_ROOT}/kernels/state/musa_resource_variable_op.cc"
    "${SRC_ROOT}/kernels/state/musa_variable_v2_op.cc"
    "${SRC_ROOT}/kernels/string/musa_static_regex_full_match_op.cc"
    "${SRC_ROOT}/kernels/string/musa_string_join_op.cc"
    "${SRC_ROOT}/kernels/string/musa_string_to_hash_bucket_fast_op.cc"
    "${SRC_ROOT}/kernels/training/musa_applyadam_op.cc"
    "${SRC_ROOT}/kernels/training/musa_train_op.cc"
    "${SRC_ROOT}/kernels/utils_op.cc"
)

set(MU_SOURCES 
    "${SRC_ROOT}/kernels/array/musa_diag_part_kernel.mu"
    "${SRC_ROOT}/kernels/array/musa_gather_kernel.mu"
    "${SRC_ROOT}/kernels/array/musa_invertpermutation_kernel.mu"
    "${SRC_ROOT}/kernels/array/musa_pack_kernel.mu"
    "${SRC_ROOT}/kernels/array/musa_range_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_addn_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_bitwise_and_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_cast_kernels.mu"
    "${SRC_ROOT}/kernels/math/musa_div_no_nan_op.mu"
    "${SRC_ROOT}/kernels/math/musa_erf_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_expm1_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_isnan_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_neg_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_prod_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_sign_kernel.mu"
    "${SRC_ROOT}/kernels/math/musa_unsorted_segment_sum_kernel.mu"
    "${SRC_ROOT}/kernels/nn/musa_softplus_kernel.mu"
    "${SRC_ROOT}/kernels/random/musa_normal_kernel.mu"
    "${SRC_ROOT}/kernels/random/musa_random_kernels.mu"
    "${SRC_ROOT}/kernels/state/musa_assign_kernel.mu"
)

# MCC compile MUSA Kernel
set(MU_OBJECTS "") # [cite: 3]
foreach(mu_file ${MU_SOURCES}) # [cite: 3]
    get_filename_component(filename ${mu_file} NAME) # [cite: 3]
    if(NOT filename MATCHES "^disabled_") # [cite: 3]
        file(RELATIVE_PATH mu_rel_path "${SRC_ROOT}/kernels" "${mu_file}") # [cite: 3]
        string(REPLACE "/" "_" mu_rel_obj "${mu_rel_path}") # [cite: 3]
        string(REPLACE ".mu" "" mu_rel_obj "${mu_rel_obj}") # [cite: 3]
        set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${mu_rel_obj}.mu.o") # [cite: 3]
        set(mu_full_path "${mu_file}") # [cite: 3]

        add_custom_command( # [cite: 4]
            OUTPUT ${obj_file} # [cite: 4]
            COMMAND ${MCC_EXECUTABLE} -c ${mu_full_path} -o ${obj_file} # [cite: 4]
            -O3 -fPIC -I${MUSA_PATH}/include -I${TF_INC} # [cite: 4]
            DEPENDS ${mu_full_path} # [cite: 4]
            COMMENT "Compiling MUSA kernel: ${filename}" # [cite: 4]
        ) # [cite: 4]
        list(APPEND MU_OBJECTS ${obj_file}) # [cite: 4]
    endif() # [cite: 5]
endforeach() # [cite: 5]

# Generate plugin library
add_library(musa_plugin SHARED ${CPP_SOURCES} ${MU_OBJECTS}) # [cite: 5]

# Apply TF compile flags (ensure strict ABI consistency)
set_target_properties(musa_plugin PROPERTIES COMPILE_FLAGS "${TF_COMPILE_FLAGS}") # [cite: 5]

# Set RPATH to ensure runtime library search
set_target_properties(musa_plugin PROPERTIES # [cite: 5]
    SKIP_BUILD_RPATH FALSE # [cite: 5]
    BUILD_WITH_INSTALL_RPATH TRUE # [cite: 5]
    INSTALL_RPATH "${TF_LIB_DIR};${MUSA_PATH}/lib;${MUSA_PATH}/lib64" # [cite: 5]
) # [cite: 5]

# Link
target_link_libraries(musa_plugin PRIVATE ${TF_LINK_FLAGS} musa mublas mudnn rt) # [cite: 5]

# Output configuration
set_target_properties(musa_plugin PROPERTIES # [cite: 5]
    PREFIX "lib" # [cite: 5]
    SUFFIX ".so" # [cite: 5]
    OUTPUT_NAME "musa_plugin" # [cite: 5]
) # [cite: 5]

# Build completion message
add_custom_command(TARGET musa_plugin POST_BUILD # [cite: 5]
    COMMAND ${CMAKE_COMMAND} -E echo "Build completed successfully!" # [cite: 5]
    COMMENT "Plugin is located at: $<TARGET_FILE:musa_plugin>" # [cite: 5]
) # [cite: 5]

# Print build summary
list(LENGTH CPP_SOURCES CPP_SOURCES_COUNT) # [cite: 5]
list(LENGTH MU_SOURCES MU_SOURCES_COUNT) # [cite: 5]

message(STATUS "TensorFlow MUSA Plugin Configuration:") # [cite: 5, 6]
message(STATUS "  MUSA Path: ${MUSA_PATH}") # [cite: 6]
message(STATUS "  TensorFlow Include: ${TF_INC}") # [cite: 6]
message(STATUS "  Source Files: ${CPP_SOURCES_COUNT} C++ files, ${MU_SOURCES_COUNT} MUSA kernels") # [cite: 6]
message(STATUS "  Output: ${CMAKE_CURRENT_BINARY_DIR}/libmusa_plugin.so") # [cite: 6]
