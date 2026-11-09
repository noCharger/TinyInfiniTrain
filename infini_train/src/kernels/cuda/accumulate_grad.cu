#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

template <typename T>
__global__ void AccumulateGradKernel(const T *grad_ptr, float rate, T *tensor_ptr, size_t num_elements) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] = FromFloat<T>(ToFloat(tensor_ptr[idx]) + rate * ToFloat(grad_ptr[idx]));
    }
}

template <typename T>
void LaunchAccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate,
                          const std::shared_ptr<Tensor> &tensor) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((gradient->NumElements() + threads - 1) / threads);
    AccumulateGradKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(tensor->GetDevice().Index())>>>(
        static_cast<const T *>(gradient->DataPtr()), rate, static_cast<T *>(tensor->DataPtr()),
        gradient->NumElements());
}

template <typename GradT, typename ParamT>
__global__ void AdamAccumulateGradKernel(const GradT *g, ParamT *p, float *m, float *v, float lr,
                                         float beta1, float beta2, float eps, float bc1, float bc2, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float grad = ToFloat(g[i]);
    m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    p[i] = FromFloat<ParamT>(ToFloat(p[i]) - lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps));
}

template <typename GradT, typename ParamT>
void LaunchAdam(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                float beta1, float beta2, float eps, float bc1, float bc2) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((grad->NumElements() + threads - 1) / threads);
    AdamAccumulateGradKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(param->GetDevice().Index())>>>(
        static_cast<const GradT *>(grad->DataPtr()), static_cast<ParamT *>(param->DataPtr()),
        static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
        bc1, bc2, grad->NumElements());
}

template <typename GradT>
void DispatchAdamParam(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                       const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                       float beta1, float beta2, float eps, float bc1, float bc2) {
    switch (param->Dtype()) {
    case DataType::kFLOAT16:
        LaunchAdam<GradT, __half>(grad, param, m, v, learning_rate, beta1, beta2, eps, bc1, bc2); break;
    case DataType::kBFLOAT16:
        LaunchAdam<GradT, __nv_bfloat16>(grad, param, m, v, learning_rate, beta1, beta2, eps, bc1, bc2); break;
    case DataType::kFLOAT32:
        LaunchAdam<GradT, float>(grad, param, m, v, learning_rate, beta1, beta2, eps, bc1, bc2); break;
    default: LOG(FATAL) << "Unsupported CUDA Adam parameter dtype";
    }
}

} // namespace

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    CHECK_EQ(gradient->NumElements(), tensor->NumElements());
    CHECK_EQ(static_cast<int>(gradient->Dtype()), static_cast<int>(tensor->Dtype()));
    switch (tensor->Dtype()) {
    case DataType::kFLOAT16: LaunchAccumulateGrad<__half>(gradient, rate, tensor); break;
    case DataType::kBFLOAT16: LaunchAccumulateGrad<__nv_bfloat16>(gradient, rate, tensor); break;
    case DataType::kFLOAT32: LaunchAccumulateGrad<float>(gradient, rate, tensor); break;
    default: LOG(FATAL) << "Unsupported CUDA gradient accumulation dtype";
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    CHECK_EQ(grad->NumElements(), param->NumElements());
    CHECK_EQ(grad->NumElements(), m->NumElements());
    CHECK_EQ(grad->NumElements(), v->NumElements());
    CHECK_EQ(static_cast<int>(m->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(v->Dtype()), static_cast<int>(DataType::kFLOAT32));
    const float bc1 = 1.0f - powf(beta1, static_cast<float>(t));
    const float bc2 = 1.0f - powf(beta2, static_cast<float>(t));
    switch (grad->Dtype()) {
    case DataType::kFLOAT16:
        DispatchAdamParam<__half>(grad, param, m, v, learning_rate, beta1, beta2, eps, bc1, bc2); break;
    case DataType::kBFLOAT16:
        DispatchAdamParam<__nv_bfloat16>(grad, param, m, v, learning_rate, beta1, beta2, eps, bc1, bc2); break;
    case DataType::kFLOAT32:
        DispatchAdamParam<float>(grad, param, m, v, learning_rate, beta1, beta2, eps, bc1, bc2); break;
    default: LOG(FATAL) << "Unsupported CUDA Adam gradient dtype";
    }
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
