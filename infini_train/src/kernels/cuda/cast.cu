#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "glog/logging.h"

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

__device__ float DecodeE4M3(uint8_t bits) {
    const float sign = (bits & 0x80) ? -1.0f : 1.0f;
    const int exponent_field = (bits >> 3) & 0x0f;
    const int mantissa = bits & 0x07;
    if (exponent_field == 15 && mantissa == 7) return NAN;
    if (exponent_field == 0) return sign * static_cast<float>(mantissa) * 0.001953125f;
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent_field - 7);
}

template <typename T> __device__ float ToFloat(T value) { return static_cast<float>(value); }
template <> __device__ float ToFloat(__half value) { return __half2float(value); }
template <> __device__ float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }
template <> __device__ float ToFloat(Float8E4M3 value) { return DecodeE4M3(value.bits); }

template <typename T> __device__ T FromFloat(float value) { return static_cast<T>(value); }
template <> __device__ __half FromFloat(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 FromFloat(float value) { return __float2bfloat16(value); }
template <> __device__ Float8E4M3 FromFloat(float value) { return Float8E4M3{EncodeE4M3(value)}; }

template <typename Src, typename Dst>
__global__ void CastKernel(const Src *input, Dst *output, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) { output[idx] = FromFloat<Dst>(ToFloat(input[idx])); }
}

template <typename Src, typename Dst>
void LaunchCast(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &output) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((input->NumElements() + threads - 1) / threads);
    CastKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(input->GetDevice().Index())>>>(
        static_cast<const Src *>(input->DataPtr()), static_cast<Dst *>(output->DataPtr()), input->NumElements());
}

template <typename Src>
void DispatchDestination(const std::shared_ptr<Tensor> &input,
                         const std::shared_ptr<Tensor> &output) {
    switch (output->Dtype()) {
    case DataType::kFLOAT8_E4M3: LaunchCast<Src, Float8E4M3>(input, output); break;
    case DataType::kFLOAT16: LaunchCast<Src, __half>(input, output); break;
    case DataType::kBFLOAT16: LaunchCast<Src, __nv_bfloat16>(input, output); break;
    case DataType::kFLOAT32: LaunchCast<Src, float>(input, output); break;
    case DataType::kFLOAT64: LaunchCast<Src, double>(input, output); break;
    default: LOG(FATAL) << "Unsupported CUDA cast destination dtype";
    }
}

} // namespace

std::shared_ptr<Tensor> CastForward(const std::shared_ptr<Tensor> &input,
                                    DataType output_dtype) {
    auto output = std::make_shared<Tensor>(input->Dims(), output_dtype, input->GetDevice());
    switch (input->Dtype()) {
    case DataType::kFLOAT8_E4M3: DispatchDestination<Float8E4M3>(input, output); break;
    case DataType::kFLOAT16: DispatchDestination<__half>(input, output); break;
    case DataType::kBFLOAT16: DispatchDestination<__nv_bfloat16>(input, output); break;
    case DataType::kFLOAT32: DispatchDestination<float>(input, output); break;
    case DataType::kFLOAT64: DispatchDestination<double>(input, output); break;
    default: LOG(FATAL) << "Unsupported CUDA cast source dtype";
    }
    return output;
}

template <typename T>
__global__ void FP8FakeQuantizeKernel(const T *input, T *output, size_t count, float scale) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const Float8E4M3 quantized = FromFloat<Float8E4M3>(ToFloat(input[idx]) * scale);
    output[idx] = FromFloat<T>(ToFloat(quantized) / scale);
}

template <typename T>
void LaunchFP8FakeQuantize(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &output, float scale) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((input->NumElements() + threads - 1) / threads);
    FP8FakeQuantizeKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(input->GetDevice().Index())>>>(
        static_cast<const T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), input->NumElements(), scale);
}

std::shared_ptr<Tensor> FP8FakeQuantizeForward(const std::shared_ptr<Tensor> &input, float scale) {
    CHECK_GT(scale, 0.0f);
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    switch (input->Dtype()) {
    case DataType::kFLOAT16: LaunchFP8FakeQuantize<__half>(input, output, scale); break;
    case DataType::kBFLOAT16: LaunchFP8FakeQuantize<__nv_bfloat16>(input, output, scale); break;
    case DataType::kFLOAT32: LaunchFP8FakeQuantize<float>(input, output, scale); break;
    default: LOG(FATAL) << "Unsupported CUDA FP8 fake-quant input dtype";
    }
    return output;
}

} // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::DeviceType::kCUDA, CastForward,
                infini_train::kernels::cuda::CastForward)
REGISTER_KERNEL(infini_train::DeviceType::kCUDA, FP8FakeQuantizeForward,
                infini_train::kernels::cuda::FP8FakeQuantizeForward)
