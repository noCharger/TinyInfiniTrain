#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <memory>
#include <tuple>

#include "glog/logging.h"
#include "infini_train/include/cuda_utils.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {

template <typename T> __device__ float ToFloat(T value) { return static_cast<float>(value); }
template <> __device__ float ToFloat(__half value) { return __half2float(value); }
template <> __device__ float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> __device__ T FromFloat(float value) { return static_cast<T>(value); }
template <> __device__ __half FromFloat(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 FromFloat(float value) { return __float2bfloat16(value); }

__device__ float Sigmoid(float x) { return 0.5f * (tanhf(0.5f * x) + 1.0f); }

template <typename T>
__global__ void RMSNormForwardKernel(const T *input, const T *weight, T *output, float *rstd,
                                     int64_t rows, int64_t hidden, float eps) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float scratch[];
    float local_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float x = ToFloat(input[row * hidden + col]);
        local_sum += x * x;
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
    if (threadIdx.x == 0) rstd[row] = scale;
    for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        output[row * hidden + col]
            = FromFloat<T>(ToFloat(input[row * hidden + col]) * scale * ToFloat(weight[col]));
    }
}

template <typename T>
__global__ void RMSNormInputGradKernel(const T *input, const T *weight, const float *rstd,
                                       const T *grad_output, T *grad_input, int64_t rows,
                                       int64_t hidden) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float scratch[];
    float local_projection = 0.0f;
    for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        local_projection += ToFloat(grad_output[row * hidden + col]) * ToFloat(weight[col])
                            * ToFloat(input[row * hidden + col]);
    }
    scratch[threadIdx.x] = local_projection;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rstd[row];
    const float correction = scale * scale * scale * scratch[0] / static_cast<float>(hidden);
    for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float dx = ToFloat(grad_output[row * hidden + col]) * ToFloat(weight[col]) * scale
                         - ToFloat(input[row * hidden + col]) * correction;
        grad_input[row * hidden + col] = FromFloat<T>(dx);
    }
}

template <typename T>
__global__ void RMSNormWeightGradKernel(const T *input, const float *rstd, const T *grad_output,
                                        T *grad_weight, int64_t rows, int64_t hidden) {
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) return;
    float sum = 0.0f;
    for (int64_t row = 0; row < rows; ++row) {
        sum += ToFloat(grad_output[row * hidden + col]) * ToFloat(input[row * hidden + col]) * rstd[row];
    }
    grad_weight[col] = FromFloat<T>(sum);
}

template <typename T>
__global__ void RoPEForwardKernel(const T *input, const T *cos, const T *sin, T *output,
                                  int64_t rows, int64_t sequence, int64_t hidden, bool cache_1d) {
    const int64_t half = hidden / 2;
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * half) return;
    const int64_t row = index / half;
    const int64_t pair = index % half;
    const int64_t token = row % sequence;
    const int64_t cache_index = cache_1d ? pair : token * half + pair;
    const float c = ToFloat(cos[cache_index]);
    const float s = ToFloat(sin[cache_index]);
    const float first = ToFloat(input[row * hidden + pair]);
    const float second = ToFloat(input[row * hidden + half + pair]);
    output[row * hidden + pair] = FromFloat<T>(first * c - second * s);
    output[row * hidden + half + pair] = FromFloat<T>(first * s + second * c);
}

template <typename T>
__global__ void RoPEInputGradKernel(const T *grad_output, const T *cos, const T *sin, T *grad_input,
                                    int64_t rows, int64_t sequence, int64_t hidden, bool cache_1d) {
    const int64_t half = hidden / 2;
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * half) return;
    const int64_t row = index / half;
    const int64_t pair = index % half;
    const int64_t token = row % sequence;
    const int64_t cache_index = cache_1d ? pair : token * half + pair;
    const float c = ToFloat(cos[cache_index]);
    const float s = ToFloat(sin[cache_index]);
    const float first = ToFloat(grad_output[row * hidden + pair]);
    const float second = ToFloat(grad_output[row * hidden + half + pair]);
    grad_input[row * hidden + pair] = FromFloat<T>(first * c + second * s);
    grad_input[row * hidden + half + pair] = FromFloat<T>(-first * s + second * c);
}

template <typename T>
__global__ void RoPECacheGradKernel(const T *input, const T *grad_output, T *grad_cos, T *grad_sin,
                                    int64_t rows, int64_t sequence, int64_t hidden, bool cache_1d) {
    const int64_t half = hidden / 2;
    const int64_t cache_elements = (cache_1d ? 1 : sequence) * half;
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cache_elements) return;
    const int64_t token = cache_1d ? 0 : index / half;
    const int64_t pair = index % half;
    float cos_sum = 0.0f;
    float sin_sum = 0.0f;
    for (int64_t row = cache_1d ? 0 : token; row < rows; row += cache_1d ? 1 : sequence) {
        const float first = ToFloat(input[row * hidden + pair]);
        const float second = ToFloat(input[row * hidden + half + pair]);
        const float grad_first = ToFloat(grad_output[row * hidden + pair]);
        const float grad_second = ToFloat(grad_output[row * hidden + half + pair]);
        cos_sum += grad_first * first + grad_second * second;
        sin_sum += -grad_first * second + grad_second * first;
    }
    grad_cos[index] = FromFloat<T>(cos_sum);
    grad_sin[index] = FromFloat<T>(sin_sum);
}

template <typename T>
__global__ void SwiGLUForwardKernel(const T *input, T *output, int64_t rows, int64_t hidden) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * hidden) return;
    const int64_t row = index / hidden;
    const int64_t col = index % hidden;
    const float gate = ToFloat(input[row * hidden * 2 + col]);
    const float up = ToFloat(input[row * hidden * 2 + hidden + col]);
    output[index] = FromFloat<T>(gate * Sigmoid(gate) * up);
}

template <typename T>
__global__ void SwiGLUBackwardKernel(const T *input, const T *grad_output, T *grad_input,
                                     int64_t rows, int64_t hidden) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * hidden) return;
    const int64_t row = index / hidden;
    const int64_t col = index % hidden;
    const float gate = ToFloat(input[row * hidden * 2 + col]);
    const float up = ToFloat(input[row * hidden * 2 + hidden + col]);
    const float grad = ToFloat(grad_output[index]);
    const float sigmoid = Sigmoid(gate);
    grad_input[row * hidden * 2 + col]
        = FromFloat<T>(grad * up * (sigmoid + gate * sigmoid * (1.0f - sigmoid)));
    grad_input[row * hidden * 2 + hidden + col] = FromFloat<T>(grad * gate * sigmoid);
}

// FP32 vector path: four channels are loaded and stored in one transaction.
// CUDA allocations and row strides are at least 16-byte aligned when hidden % 4 == 0.
__global__ void SwiGLUForwardFloat4Kernel(const float4 *input, float4 *output,
                                          int64_t vectors_per_row, int64_t vector_count) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vector_count) return;
    const int64_t row = index / vectors_per_row;
    const int64_t vector = index % vectors_per_row;
    const float4 g = input[row * vectors_per_row * 2 + vector];
    const float4 u = input[row * vectors_per_row * 2 + vectors_per_row + vector];
    output[index] = make_float4(g.x * Sigmoid(g.x) * u.x, g.y * Sigmoid(g.y) * u.y,
                                g.z * Sigmoid(g.z) * u.z, g.w * Sigmoid(g.w) * u.w);
}

__global__ void SwiGLUBackwardFloat4Kernel(const float4 *input, const float4 *grad_output,
                                           float4 *grad_input, int64_t vectors_per_row,
                                           int64_t vector_count) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vector_count) return;
    const int64_t row = index / vectors_per_row;
    const int64_t vector = index % vectors_per_row;
    const float4 g = input[row * vectors_per_row * 2 + vector];
    const float4 u = input[row * vectors_per_row * 2 + vectors_per_row + vector];
    const float4 dy = grad_output[index];
    const float sx = Sigmoid(g.x), sy = Sigmoid(g.y), sz = Sigmoid(g.z), sw = Sigmoid(g.w);
    grad_input[row * vectors_per_row * 2 + vector] = make_float4(
        dy.x * u.x * (sx + g.x * sx * (1.0f - sx)),
        dy.y * u.y * (sy + g.y * sy * (1.0f - sy)),
        dy.z * u.z * (sz + g.z * sz * (1.0f - sz)),
        dy.w * u.w * (sw + g.w * sw * (1.0f - sw)));
    grad_input[row * vectors_per_row * 2 + vectors_per_row + vector]
        = make_float4(dy.x * g.x * sx, dy.y * g.y * sy, dy.z * g.z * sz, dy.w * g.w * sw);
}

void CheckSameTypeAndDevice(const std::shared_ptr<Tensor> &input,
                            const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(other->Dtype()));
    CHECK_EQ(input->GetDevice(), other->GetDevice());
}

void CheckRoPE(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &cos,
               const std::shared_ptr<Tensor> &sin) {
    CHECK_GE(input->Dims().size(), 2u);
    const int64_t hidden = input->Dims().back();
    const int64_t sequence = input->Dims()[input->Dims().size() - 2];
    CHECK_EQ(hidden % 2, 0);
    CheckSameTypeAndDevice(input, cos);
    CheckSameTypeAndDevice(input, sin);
    CHECK(cos->Dims() == sin->Dims());
    CHECK(cos->Dims().size() == 1u || cos->Dims().size() == 2u);
    if (cos->Dims().size() == 1u) {
        CHECK_EQ(cos->Dims()[0], hidden / 2);
    } else {
        CHECK_EQ(cos->Dims()[0], sequence);
        CHECK_EQ(cos->Dims()[1], hidden / 2);
    }
}

template <typename T>
void LaunchRMSNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                          const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &rstd,
                          int64_t rows, int64_t hidden, float eps) {
    constexpr int threads = 256;
    RMSNormForwardKernel<<<rows, threads, threads * sizeof(float),
                           infini_train::ThreadLocalStream(input->GetDevice().Index())>>>(
        static_cast<const T *>(input->DataPtr()), static_cast<const T *>(weight->DataPtr()),
        static_cast<T *>(output->DataPtr()), static_cast<float *>(rstd->DataPtr()), rows, hidden, eps);
}

template <typename T>
void LaunchRMSNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                           const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output,
                           const std::shared_ptr<Tensor> &grad_input,
                           const std::shared_ptr<Tensor> &grad_weight, int64_t rows, int64_t hidden) {
    constexpr int threads = 256;
    auto stream = infini_train::ThreadLocalStream(input->GetDevice().Index());
    RMSNormInputGradKernel<<<rows, threads, threads * sizeof(float), stream>>>(
        static_cast<const T *>(input->DataPtr()), static_cast<const T *>(weight->DataPtr()),
        static_cast<const float *>(rstd->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
        static_cast<T *>(grad_input->DataPtr()), rows, hidden);
    const int blocks = static_cast<int>((hidden + threads - 1) / threads);
    RMSNormWeightGradKernel<<<blocks, threads, 0, stream>>>(
        static_cast<const T *>(input->DataPtr()), static_cast<const float *>(rstd->DataPtr()),
        static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_weight->DataPtr()), rows, hidden);
}

template <typename T>
void LaunchRoPEForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &cos,
                       const std::shared_ptr<Tensor> &sin, const std::shared_ptr<Tensor> &output,
                       int64_t rows, int64_t sequence, int64_t hidden) {
    constexpr int threads = 256;
    const int64_t count = rows * hidden / 2;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    RoPEForwardKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(input->GetDevice().Index())>>>(
        static_cast<const T *>(input->DataPtr()), static_cast<const T *>(cos->DataPtr()),
        static_cast<const T *>(sin->DataPtr()), static_cast<T *>(output->DataPtr()), rows, sequence, hidden,
        cos->Dims().size() == 1u);
}

template <typename T>
void LaunchRoPEBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &cos,
                        const std::shared_ptr<Tensor> &sin, const std::shared_ptr<Tensor> &grad_output,
                        const std::shared_ptr<Tensor> &grad_input, const std::shared_ptr<Tensor> &grad_cos,
                        const std::shared_ptr<Tensor> &grad_sin, int64_t rows, int64_t sequence, int64_t hidden) {
    constexpr int threads = 256;
    const int64_t pair_count = rows * hidden / 2;
    const int pair_blocks = static_cast<int>((pair_count + threads - 1) / threads);
    const bool cache_1d = cos->Dims().size() == 1u;
    auto stream = infini_train::ThreadLocalStream(input->GetDevice().Index());
    RoPEInputGradKernel<<<pair_blocks, threads, 0, stream>>>(
        static_cast<const T *>(grad_output->DataPtr()), static_cast<const T *>(cos->DataPtr()),
        static_cast<const T *>(sin->DataPtr()), static_cast<T *>(grad_input->DataPtr()), rows, sequence, hidden,
        cache_1d);
    const int cache_blocks = static_cast<int>((cos->NumElements() + threads - 1) / threads);
    RoPECacheGradKernel<<<cache_blocks, threads, 0, stream>>>(
        static_cast<const T *>(input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
        static_cast<T *>(grad_cos->DataPtr()), static_cast<T *>(grad_sin->DataPtr()), rows, sequence, hidden,
        cache_1d);
}

template <typename T>
void LaunchSwiGLUForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &output,
                         int64_t rows, int64_t hidden) {
    constexpr int threads = 256;
    const int64_t count = rows * hidden;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    SwiGLUForwardKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(input->GetDevice().Index())>>>(
        static_cast<const T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), rows, hidden);
}

template <typename T>
void LaunchSwiGLUBackward(const std::shared_ptr<Tensor> &input,
                          const std::shared_ptr<Tensor> &grad_output,
                          const std::shared_ptr<Tensor> &grad_input, int64_t rows, int64_t hidden) {
    constexpr int threads = 256;
    const int64_t count = rows * hidden;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    SwiGLUBackwardKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(input->GetDevice().Index())>>>(
        static_cast<const T *>(input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
        static_cast<T *>(grad_input->DataPtr()), rows, hidden);
}

} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RMSNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, float eps) {
    CHECK_GE(input->Dims().size(), 1u);
    CHECK_EQ(weight->Dims().size(), 1u);
    CHECK_EQ(weight->Dims()[0], input->Dims().back());
    CheckSameTypeAndDevice(input, weight);
    const int64_t hidden = input->Dims().back();
    const int64_t rows = input->NumElements() / hidden;
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>{rows}, DataType::kFLOAT32, input->GetDevice());
    switch (input->Dtype()) {
    case DataType::kFLOAT16: LaunchRMSNormForward<__half>(input, weight, output, rstd, rows, hidden, eps); break;
    case DataType::kBFLOAT16:
        LaunchRMSNormForward<__nv_bfloat16>(input, weight, output, rstd, rows, hidden, eps); break;
    case DataType::kFLOAT32: LaunchRMSNormForward<float>(input, weight, output, rstd, rows, hidden, eps); break;
    default: LOG(FATAL) << "Unsupported CUDA RMSNorm dtype";
    }
    return {output, rstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RMSNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    CheckSameTypeAndDevice(input, weight);
    CheckSameTypeAndDevice(input, grad_output);
    const int64_t hidden = input->Dims().back();
    const int64_t rows = input->NumElements() / hidden;
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), weight->Dtype(), weight->GetDevice());
    switch (input->Dtype()) {
    case DataType::kFLOAT16:
        LaunchRMSNormBackward<__half>(input, weight, rstd, grad_output, grad_input, grad_weight, rows, hidden); break;
    case DataType::kBFLOAT16:
        LaunchRMSNormBackward<__nv_bfloat16>(input, weight, rstd, grad_output, grad_input, grad_weight, rows,
                                             hidden); break;
    case DataType::kFLOAT32:
        LaunchRMSNormBackward<float>(input, weight, rstd, grad_output, grad_input, grad_weight, rows, hidden); break;
    default: LOG(FATAL) << "Unsupported CUDA RMSNorm dtype";
    }
    return {grad_input, grad_weight};
}

std::shared_ptr<Tensor> RoPEForward(const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &cos,
                                    const std::shared_ptr<Tensor> &sin) {
    CheckRoPE(input, cos, sin);
    const int64_t hidden = input->Dims().back();
    const int64_t sequence = input->Dims()[input->Dims().size() - 2];
    const int64_t rows = input->NumElements() / hidden;
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    switch (input->Dtype()) {
    case DataType::kFLOAT16: LaunchRoPEForward<__half>(input, cos, sin, output, rows, sequence, hidden); break;
    case DataType::kBFLOAT16:
        LaunchRoPEForward<__nv_bfloat16>(input, cos, sin, output, rows, sequence, hidden); break;
    case DataType::kFLOAT32: LaunchRoPEForward<float>(input, cos, sin, output, rows, sequence, hidden); break;
    default: LOG(FATAL) << "Unsupported CUDA RoPE dtype";
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RoPEBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &cos,
             const std::shared_ptr<Tensor> &sin, const std::shared_ptr<Tensor> &grad_output) {
    CheckRoPE(input, cos, sin);
    CheckSameTypeAndDevice(input, grad_output);
    const int64_t hidden = input->Dims().back();
    const int64_t sequence = input->Dims()[input->Dims().size() - 2];
    const int64_t rows = input->NumElements() / hidden;
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto grad_cos = std::make_shared<Tensor>(cos->Dims(), cos->Dtype(), cos->GetDevice());
    auto grad_sin = std::make_shared<Tensor>(sin->Dims(), sin->Dtype(), sin->GetDevice());
    switch (input->Dtype()) {
    case DataType::kFLOAT16:
        LaunchRoPEBackward<__half>(input, cos, sin, grad_output, grad_input, grad_cos, grad_sin, rows, sequence,
                                   hidden); break;
    case DataType::kBFLOAT16:
        LaunchRoPEBackward<__nv_bfloat16>(input, cos, sin, grad_output, grad_input, grad_cos, grad_sin, rows,
                                          sequence, hidden); break;
    case DataType::kFLOAT32:
        LaunchRoPEBackward<float>(input, cos, sin, grad_output, grad_input, grad_cos, grad_sin, rows, sequence,
                                  hidden); break;
    default: LOG(FATAL) << "Unsupported CUDA RoPE dtype";
    }
    return {grad_input, grad_cos, grad_sin};
}

std::shared_ptr<Tensor> SwiGLUForward(const std::shared_ptr<Tensor> &input) {
    CHECK_GE(input->Dims().size(), 1u);
    CHECK_EQ(input->Dims().back() % 2, 0);
    const int64_t hidden = input->Dims().back() / 2;
    const int64_t rows = input->NumElements() / (hidden * 2);
    auto output_dims = input->Dims();
    output_dims.back() = hidden;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());
    if (input->Dtype() == DataType::kFLOAT32 && hidden % 4 == 0) {
        constexpr int threads = 256;
        const int64_t vectors_per_half = rows * hidden / 4;
        const int64_t vectors_per_row = hidden / 4;
        const int blocks = static_cast<int>((vectors_per_half + threads - 1) / threads);
        const auto *base = static_cast<const float *>(input->DataPtr());
        SwiGLUForwardFloat4Kernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(input->GetDevice().Index())>>>(
            reinterpret_cast<const float4 *>(base), reinterpret_cast<float4 *>(output->DataPtr()),
            vectors_per_row, vectors_per_half);
    } else {
        switch (input->Dtype()) {
        case DataType::kFLOAT16: LaunchSwiGLUForward<__half>(input, output, rows, hidden); break;
        case DataType::kBFLOAT16: LaunchSwiGLUForward<__nv_bfloat16>(input, output, rows, hidden); break;
        case DataType::kFLOAT32: LaunchSwiGLUForward<float>(input, output, rows, hidden); break;
        default: LOG(FATAL) << "Unsupported CUDA SwiGLU dtype";
        }
    }
    return output;
}

std::shared_ptr<Tensor> SwiGLUBackward(const std::shared_ptr<Tensor> &input,
                                      const std::shared_ptr<Tensor> &grad_output) {
    CheckSameTypeAndDevice(input, grad_output);
    const int64_t hidden = input->Dims().back() / 2;
    const int64_t rows = input->NumElements() / (hidden * 2);
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    if (input->Dtype() == DataType::kFLOAT32 && hidden % 4 == 0) {
        constexpr int threads = 256;
        const int64_t vectors_per_half = rows * hidden / 4;
        const int64_t vectors_per_row = hidden / 4;
        const int blocks = static_cast<int>((vectors_per_half + threads - 1) / threads);
        const auto *base = static_cast<const float *>(input->DataPtr());
        auto *grad_base = static_cast<float *>(grad_input->DataPtr());
        SwiGLUBackwardFloat4Kernel<<<blocks, threads, 0,
                                    infini_train::ThreadLocalStream(input->GetDevice().Index())>>>(
            reinterpret_cast<const float4 *>(base), reinterpret_cast<const float4 *>(grad_output->DataPtr()),
            reinterpret_cast<float4 *>(grad_base), vectors_per_row, vectors_per_half);
    } else {
        switch (input->Dtype()) {
        case DataType::kFLOAT16: LaunchSwiGLUBackward<__half>(input, grad_output, grad_input, rows, hidden); break;
        case DataType::kBFLOAT16:
            LaunchSwiGLUBackward<__nv_bfloat16>(input, grad_output, grad_input, rows, hidden); break;
        case DataType::kFLOAT32: LaunchSwiGLUBackward<float>(input, grad_output, grad_input, rows, hidden); break;
        default: LOG(FATAL) << "Unsupported CUDA SwiGLU dtype";
        }
    }
    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_TRANSFORMER_KERNEL(kernel_name)                                                                 \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_TRANSFORMER_KERNEL(RMSNormForward)
REGISTER_CUDA_TRANSFORMER_KERNEL(RMSNormBackward)
REGISTER_CUDA_TRANSFORMER_KERNEL(RoPEForward)
REGISTER_CUDA_TRANSFORMER_KERNEL(RoPEBackward)
REGISTER_CUDA_TRANSFORMER_KERNEL(SwiGLUForward)
REGISTER_CUDA_TRANSFORMER_KERNEL(SwiGLUBackward)

#undef REGISTER_CUDA_TRANSFORMER_KERNEL
