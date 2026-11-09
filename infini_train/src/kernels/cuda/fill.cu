#include "glog/logging.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "infini_train/include/cuda_utils.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {
struct Float8E4M3 {
    uint8_t bits;
};

__device__ uint8_t EncodeE4M3(float value) {
    if (isnan(value)) return 0x7f;
    const uint8_t sign = signbit(value) ? 0x80 : 0x00;
    float magnitude = fabsf(value);
    if (magnitude == 0.0f) return sign;
    if (!isfinite(magnitude) || magnitude >= 448.0f) return sign | 0x7e;
    if (magnitude < 0.015625f) {
        int mantissa = static_cast<int>(rintf(magnitude / 0.001953125f));
        mantissa = mantissa < 0 ? 0 : (mantissa > 7 ? 7 : mantissa);
        return sign | static_cast<uint8_t>(mantissa);
    }
    int exponent = static_cast<int>(floorf(log2f(magnitude)));
    int exponent_field = exponent + 7;
    int mantissa = static_cast<int>(rintf((magnitude / ldexpf(1.0f, exponent) - 1.0f) * 8.0f));
    if (mantissa == 8) {
        mantissa = 0;
        ++exponent_field;
    }
    if (exponent_field >= 15) {
        exponent_field = 15;
        mantissa = mantissa > 6 ? 6 : mantissa;
    }
    return sign | static_cast<uint8_t>((exponent_field << 3) | mantissa);
}

template <typename T> __device__ T FromFloat(float value) { return static_cast<T>(value); }
template <> __device__ __half FromFloat(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 FromFloat(float value) { return __float2bfloat16(value); }
template <> __device__ Float8E4M3 FromFloat(float value) { return Float8E4M3{EncodeE4M3(value)}; }

template <typename T> __global__ void FillKernel(T *data, size_t count, float value) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) { data[idx] = FromFloat<T>(value); }
}

template <typename T> void LaunchFill(const std::shared_ptr<Tensor> &tensor, float value) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((tensor->NumElements() + threads - 1) / threads);
    FillKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(tensor->GetDevice().Index())>>>(
        static_cast<T *>(tensor->DataPtr()), tensor->NumElements(), value);
}
} // namespace

void Fill(std::shared_ptr<Tensor> tensor, double value) {
    const float scalar = static_cast<float>(value);
    switch (tensor->Dtype()) {
    case DataType::kUINT8: LaunchFill<uint8_t>(tensor, scalar); break;
    case DataType::kINT8: LaunchFill<int8_t>(tensor, scalar); break;
    case DataType::kUINT16: LaunchFill<uint16_t>(tensor, scalar); break;
    case DataType::kINT16: LaunchFill<int16_t>(tensor, scalar); break;
    case DataType::kUINT32: LaunchFill<uint32_t>(tensor, scalar); break;
    case DataType::kINT32: LaunchFill<int32_t>(tensor, scalar); break;
    case DataType::kUINT64: LaunchFill<uint64_t>(tensor, scalar); break;
    case DataType::kINT64: LaunchFill<int64_t>(tensor, scalar); break;
    case DataType::kFLOAT8_E4M3: LaunchFill<Float8E4M3>(tensor, scalar); break;
    case DataType::kBFLOAT16: LaunchFill<__nv_bfloat16>(tensor, scalar); break;
    case DataType::kFLOAT16: LaunchFill<__half>(tensor, scalar); break;
    case DataType::kFLOAT32: LaunchFill<float>(tensor, scalar); break;
    case DataType::kFLOAT64: LaunchFill<double>(tensor, scalar); break;
    }
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FILL_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FILL_KERNEL(Fill)

#undef REGISTER_CUDA_FILL_KERNEL
