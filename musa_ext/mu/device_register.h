#ifndef TENSORFLOW_MUSA_MU1_DEVICE_REGISTER_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_REGISTER_H_

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"

extern const char NAME_MTGPU[];
#define DEVICE_MTGPU "MUSA"
#ifdef __cplusplus
extern "C" {
#endif

void plugin_get_device_count(const SP_Platform* platform, int* count,
                             TF_Status* const status);

void plugin_create_device(const SP_Platform* platform,
                          SE_CreateDeviceParams* params, TF_Status* status);
void plugin_destroy_device(const SP_Platform* platform, SP_Device* device);
void plugin_create_device_fns(const SP_Platform* platform,
                              SE_CreateDeviceFnsParams* params,
                              TF_Status* status);
void plugin_destroy_device_fns(const SP_Platform* platform,
                               SP_DeviceFns* device_fns);
void plugin_create_stream_executor(const SP_Platform* platform,
                                   SE_CreateStreamExecutorParams* params,
                                   TF_Status* status);
void plugin_destroy_stream_executor(const SP_Platform* platform,
                                    SP_StreamExecutor* stream_executor);
void plugin_create_timer_fns(const SP_Platform* platform, SP_TimerFns* timer,
                             TF_Status* status);
void plugin_destroy_timer_fns(const SP_Platform* platform,
                              SP_TimerFns* timer_fns);
void plugin_destroy_platform(SP_Platform* platform);
void plugin_destroy_platform_fns(SP_PlatformFns* platform_fns);

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status);

#ifdef __cplusplus
}
#endif

#endif
