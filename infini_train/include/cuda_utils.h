#pragma once
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <unordered_map>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train {

namespace detail {
struct CudaStreamPool {
    ~CudaStreamPool() {
        for (auto &[device_id, stream] : streams_) {
            cudaSetDevice(device_id);
            cudaStreamDestroy(stream);
        }
    }

    cudaStream_t Get(int device_id) {
        auto it = streams_.find(device_id);
        if (it != streams_.end()) return it->second;
        cudaSetDevice(device_id);
        cudaStream_t stream = nullptr;
        const cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        LOG_IF(FATAL, status != cudaSuccess)
            << "cudaStreamCreateWithFlags failed for device " << device_id << ": "
            << cudaGetErrorString(status);
        streams_[device_id] = stream;
        return stream;
    }

    std::unordered_map<int, cudaStream_t> streams_;
};

struct CublasHandlePool {
    ~CublasHandlePool() {
        for (auto &[device_id, h] : handles_) {
            cudaSetDevice(device_id);
            cublasDestroy(h);
        }
    }

    cublasHandle_t Get(int device_id) {
        auto it = handles_.find(device_id);
        if (it != handles_.end()) {
            return it->second;
        }
        cudaSetDevice(device_id);
        cublasHandle_t h;
        cublasStatus_t status = cublasCreate(&h);
        LOG_IF(FATAL, status != CUBLAS_STATUS_SUCCESS)
            << "cublasCreate failed for device " << device_id;
        handles_[device_id] = h;
        return h;
    }

    std::unordered_map<int, cublasHandle_t> handles_;
};
} // namespace detail

// Returns a non-blocking stream owned by the calling host thread and CUDA
// device. Compute kernels, cuBLAS calls, copies, and NCCL collectives can bind
// to this stream without serializing through CUDA's legacy default stream.
inline cudaStream_t ThreadLocalStream(int device_id = 0) {
    thread_local detail::CudaStreamPool pool;
    return pool.Get(device_id);
}

// Returns a persistent per-device cuBLAS handle for the calling thread.
// Replaces per-call cublasCreate/cublasDestroy to amortize handle overhead
// and enable future stream attachment for async execution.
inline cublasHandle_t GetCublasHandle(int device_id = 0) {
    thread_local detail::CublasHandlePool pool;
    return pool.Get(device_id);
}

// Maps TinyInfiniTrain DataType to the corresponding CUDA library type constant.
// Used by cublasGemmEx for mixed-precision matrix multiply.
//
// BF16 (kBFLOAT16): supported by cuBLAS on Ampere (sm_80+) and later.
// E4M3 currently uses portable software fake-quant plus FP32-accumulating GEMM,
// so no hardware FP8 cudaDataType_t is returned here.
inline cudaDataType_t ToCudaDtype(infini_train::DataType dt) {
    switch (dt) {
    case infini_train::DataType::kFLOAT32:  return CUDA_R_32F;
    case infini_train::DataType::kFLOAT64:  return CUDA_R_64F;
    case infini_train::DataType::kFLOAT16:  return CUDA_R_16F;
    case infini_train::DataType::kBFLOAT16: return CUDA_R_16BF;
    case infini_train::DataType::kINT32:    return CUDA_R_32I;
    case infini_train::DataType::kINT8:     return CUDA_R_8I;
    // Hardware FP8 constants exist in recent CUDA, but TinyInfiniTrain's FP8
    // tensors deliberately go through fake quantization before this boundary.
    default:
        LOG(FATAL) << "Unsupported DataType for CUDA: " << static_cast<int>(dt);
        return CUDA_R_32F; // unreachable
    }
}

} // namespace infini_train
#endif // USE_CUDA
