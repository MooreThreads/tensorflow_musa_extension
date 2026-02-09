#include "musa_device.h"
#include "musa_allocator.h"
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "mu/device/musa_event.h"  
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/device_base.h"
#include <iostream>

#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {
namespace musa {

// ============================================================
// MusaDeviceContext 实现
// ============================================================

MusaDeviceContext::MusaDeviceContext(musaStream_t stream, ::stream_executor::StreamExecutor* executor) 
    : stream_handle_(stream) {
    
    implementation_ = new ::stream_executor::musa::MusaStream(stream);
    
    // 传入 executor 
    official_stream_ = new ::stream_executor::Stream(executor, implementation_);
    
    // 初始化 Stream
    official_stream_->Init(); 
}

MusaDeviceContext::~MusaDeviceContext() {
    if (official_stream_) {
        delete official_stream_;
    }
}

void MusaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                                              Tensor* device_tensor, StatusCallback done,
                                              bool sync_dst_compute) const {
    auto* musa_dev = static_cast<MusaDevice*>(device);
    musaSetDevice(musa_dev->get_device_id());

    const void* src = cpu_tensor->tensor_data().data();
    void* dst = const_cast<char*>(device_tensor->tensor_data().data());
    size_t bytes = cpu_tensor->TotalBytes();

    if (bytes > 0) {
        musaError_t err = musaMemcpy(dst, src, bytes, musaMemcpyHostToDevice);
        if (err != musaSuccess) {
            done(errors::Internal("MUSA H2D copy failed: ", musaGetErrorString(err)));
            return;
        }
    }
    done(Status::OK());
}

void MusaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                              StringPiece tensor_name, Device* device,
                                              Tensor* cpu_tensor, StatusCallback done) {
    auto* musa_dev = static_cast<MusaDevice*>(device);
    musaSetDevice(musa_dev->get_device_id());

    const void* src = device_tensor->tensor_data().data();
    void* dst = const_cast<char*>(cpu_tensor->tensor_data().data());
    size_t bytes = device_tensor->TotalBytes();

    if (bytes > cpu_tensor->TotalBytes()) {
        bytes = cpu_tensor->TotalBytes();
    }

    if (bytes > 0) {
        // 使用同步拷贝
        mStatus m_stat = MusaMemcpyD2H(dst, src, bytes);
        musaDeviceSynchronize(); // 确保拷贝完成

        if (m_stat != mStatus::SUCCESS) {
            done(errors::Internal("MUSA D2H copy failed."));
            return;
        }
    }
    done(Status::OK());
}

// ============================================================
// MusaDevice 实现
// ============================================================

// 参数跟头文件保持一致
MusaDevice::MusaDevice(Env* env, const DeviceAttributes& attributes, int device_id, 
                       ::stream_executor::StreamExecutor* executor)
    : Device(env, attributes), device_id_(device_id) {

    // 切卡
    musaSetDevice(device_id_);

    // 创建流
    musaStreamCreate(&stream_);

    // 初始化 muDNN
    mudnn_handle_.reset(new ::musa::dnn::Handle());
    ::musa::dnn::Status s = mudnn_handle_->SetStream(stream_);
    if (s != ::musa::dnn::Status::SUCCESS) {
        std::cerr << ">>> [MUSA] ERROR: Device " << device_id_ << " failed to bind muDNN handle!" << std::endl;
    }

    // 初始化 muBLAS
    mublasCreate(&mublas_handle_);
    mublasSetStream(mublas_handle_, stream_);

    // 初始化 Context
    // 直接使用参数里的 executor
    device_context_ = new MusaDeviceContext(stream_, executor);
    musa_allocator_ = new MusaRawAllocator(device_id_);

    gpu_device_info_.stream = device_context_->stream();
    gpu_device_info_.default_context = device_context_;
    gpu_device_info_.gpu_id = device_id_;

    set_tensorflow_gpu_device_info(&gpu_device_info_);

    std::cerr << ">>> [MUSA] Device " << device_id_ << " correctly initialized." << std::endl;
}

MusaDevice::~MusaDevice() {
    musaSetDevice(device_id_);
    if (device_context_) {
        device_context_->Unref();
    }
    if (mublas_handle_) {
        mublasDestroy(mublas_handle_);
    }
    if (musa_allocator_) {
        delete musa_allocator_;
    }
    musaStreamDestroy(stream_);
}

Allocator* MusaDevice::GetAllocator(AllocatorAttributes attr) {
    return attr.on_host() ? cpu_allocator() : musa_allocator_;
}

Status MusaDevice::Sync() {
    musaSetDevice(device_id_);
    musaError_t err = musaStreamSynchronize(stream_);
    return (err == musaSuccess) ? Status::OK() : errors::Internal("MUSA Device Sync Failed");
}

Status MusaDevice::TryGetDeviceContext(DeviceContext** out_context) {
    if (device_context_) {
        *out_context = device_context_;
        device_context_->Ref();
        return Status::OK();
    }
    return errors::Internal("MusaDeviceContext is null");
}

}  // namespace musa
}  // namespace tensorflow