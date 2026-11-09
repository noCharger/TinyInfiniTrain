#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"
#include "infini_train/include/cuda_utils.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {

#define CUBLAS_CHECK(call)                                                                                            \
    do {                                                                                                              \
        const cublasStatus_t status = (call);                                                                         \
        LOG_IF(FATAL, status != CUBLAS_STATUS_SUCCESS)                                                                \
            << "cuBLAS error " << cublasGetStatusString(status) << " at " << __FILE__ << ':' << __LINE__;          \
    } while (0)

template <typename T> __device__ float ToFloat(T value) { return static_cast<float>(value); }
template <> __device__ float ToFloat(__half value) { return __half2float(value); }
template <> __device__ float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> __device__ T FromFloat(float value) { return static_cast<T>(value); }
template <> __device__ __half FromFloat(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 FromFloat(float value) { return __float2bfloat16(value); }

void CheckCublasDtype(DataType dtype) {
    CHECK(dtype == DataType::kFLOAT32 || dtype == DataType::kFLOAT16 || dtype == DataType::kBFLOAT16)
        << "CUDA linear kernels support FP32, FP16, and BF16";
}

void CheckSameTypeAndDevice(const std::shared_ptr<Tensor> &left,
                            const std::shared_ptr<Tensor> &right) {
    CHECK_EQ(static_cast<int>(left->Dtype()), static_cast<int>(right->Dtype()));
    CHECK_EQ(left->GetDevice(), right->GetDevice());
    CheckCublasDtype(left->Dtype());
}

void Gemm(const std::shared_ptr<Tensor> &a, cublasOperation_t op_a, int64_t lda, int64_t stride_a,
          const std::shared_ptr<Tensor> &b, cublasOperation_t op_b, int64_t ldb, int64_t stride_b,
          const std::shared_ptr<Tensor> &c, int64_t ldc, int64_t stride_c,
          int64_t m, int64_t n, int64_t k, int64_t batch_count = 1, float beta = 0.0f) {
    const float alpha = 1.0f;
    const cudaDataType_t a_type = infini_train::ToCudaDtype(a->Dtype());
    const cudaDataType_t b_type = infini_train::ToCudaDtype(b->Dtype());
    const cudaDataType_t c_type = infini_train::ToCudaDtype(c->Dtype());
    cudaStream_t stream = infini_train::ThreadLocalStream(c->GetDevice().Index());
    cublasHandle_t handle = infini_train::GetCublasHandle(c->GetDevice().Index());
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    if (batch_count == 1) {
        CUBLAS_CHECK(cublasGemmEx(handle, op_a, op_b, static_cast<int>(m), static_cast<int>(n),
                                  static_cast<int>(k), &alpha, a->DataPtr(), a_type, static_cast<int>(lda),
                                  b->DataPtr(), b_type, static_cast<int>(ldb), &beta, c->DataPtr(), c_type,
                                  static_cast<int>(ldc), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, op_a, op_b, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), &alpha,
            a->DataPtr(), a_type, static_cast<int>(lda), stride_a, b->DataPtr(), b_type,
            static_cast<int>(ldb), stride_b, &beta, c->DataPtr(), c_type, static_cast<int>(ldc), stride_c,
            static_cast<int>(batch_count), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

template <typename T>
__global__ void BiasCopyKernel(T *output, const T *bias, int64_t count, int64_t out_features) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = bias[index % out_features];
}

template <typename T>
__global__ void BiasGradKernel(const T *grad_output, T *grad_bias, int64_t rows, int64_t out_features) {
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= out_features) return;
    float sum = 0.0f;
    for (int64_t row = 0; row < rows; ++row) sum += ToFloat(grad_output[row * out_features + col]);
    grad_bias[col] = FromFloat<T>(sum);
}

template <typename T>
void LaunchBiasCopy(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &bias,
                    int64_t rows, int64_t out_features) {
    constexpr int threads = 256;
    const int64_t count = rows * out_features;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    BiasCopyKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(output->GetDevice().Index())>>>(
        static_cast<T *>(output->DataPtr()), static_cast<const T *>(bias->DataPtr()), count, out_features);
}

template <typename T>
void LaunchBiasGrad(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &grad_bias,
                    int64_t rows, int64_t out_features) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((out_features + threads - 1) / threads);
    BiasGradKernel<<<blocks, threads, 0, infini_train::ThreadLocalStream(grad_output->GetDevice().Index())>>>(
        static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_bias->DataPtr()), rows, out_features);
}

void CopyBias(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &bias,
              int64_t rows, int64_t out_features) {
    switch (output->Dtype()) {
    case DataType::kFLOAT16: LaunchBiasCopy<__half>(output, bias, rows, out_features); break;
    case DataType::kBFLOAT16: LaunchBiasCopy<__nv_bfloat16>(output, bias, rows, out_features); break;
    case DataType::kFLOAT32: LaunchBiasCopy<float>(output, bias, rows, out_features); break;
    default: LOG(FATAL) << "Unsupported CUDA bias dtype";
    }
}

void ReduceBias(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &grad_bias,
                int64_t rows, int64_t out_features) {
    switch (grad_output->Dtype()) {
    case DataType::kFLOAT16: LaunchBiasGrad<__half>(grad_output, grad_bias, rows, out_features); break;
    case DataType::kBFLOAT16: LaunchBiasGrad<__nv_bfloat16>(grad_output, grad_bias, rows, out_features); break;
    case DataType::kFLOAT32: LaunchBiasGrad<float>(grad_output, grad_bias, rows, out_features); break;
    default: LOG(FATAL) << "Unsupported CUDA bias gradient dtype";
    }
}

} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input,
                                      const std::shared_ptr<Tensor> &other) {
    CheckSameTypeAndDevice(input, other);
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2u);
    CHECK_GE(other_dims.size(), 2u);
    const int64_t k = input_dims.back();
    const int64_t n = other_dims.back();
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);
    auto output_dims = input_dims;
    output_dims.back() = n;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    if (other_dims.size() == 2u) {
        const int64_t rows = input->NumElements() / k;
        Gemm(other, CUBLAS_OP_N, n, 0, input, CUBLAS_OP_N, k, 0, output, n, 0,
             n, rows, k);
        return output;
    }

    CHECK_EQ(input_dims.size(), other_dims.size());
    const int64_t m = input_dims[input_dims.size() - 2];
    int64_t batch = 1;
    for (size_t dim = 0; dim + 2 < input_dims.size(); ++dim) {
        CHECK_EQ(input_dims[dim], other_dims[dim]);
        batch *= input_dims[dim];
    }
    Gemm(other, CUBLAS_OP_N, n, k * n, input, CUBLAS_OP_N, k, m * k, output, n, m * n,
         n, m, k, batch);
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    CheckSameTypeAndDevice(input, other);
    CheckSameTypeAndDevice(input, grad_output);
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const int64_t k = input_dims.back();
    const int64_t n = other_dims.back();
    auto grad_input = std::make_shared<Tensor>(input_dims, input->Dtype(), input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, other->Dtype(), other->GetDevice());

    if (other_dims.size() == 2u) {
        const int64_t rows = input->NumElements() / k;
        Gemm(other, CUBLAS_OP_T, n, 0, grad_output, CUBLAS_OP_N, n, 0, grad_input, k, 0,
             k, rows, n);
        Gemm(grad_output, CUBLAS_OP_N, n, 0, input, CUBLAS_OP_T, k, 0, grad_other, n, 0,
             n, k, rows);
        return {grad_input, grad_other};
    }

    const int64_t m = input_dims[input_dims.size() - 2];
    int64_t batch = 1;
    for (size_t dim = 0; dim + 2 < input_dims.size(); ++dim) batch *= input_dims[dim];
    Gemm(other, CUBLAS_OP_T, n, k * n, grad_output, CUBLAS_OP_N, n, m * n,
         grad_input, k, m * k, k, m, n, batch);
    Gemm(grad_output, CUBLAS_OP_N, n, m * n, input, CUBLAS_OP_T, k, m * k,
         grad_other, n, k * n, n, k, m, batch);
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input,
                                      const std::shared_ptr<Tensor> &weight, bool transpose,
                                      const std::shared_ptr<Tensor> &bias) {
    CheckSameTypeAndDevice(input, weight);
    const auto &input_dims = input->Dims();
    const auto &weight_dims = weight->Dims();
    CHECK_GE(input_dims.size(), 2u);
    CHECK_EQ(weight_dims.size(), 2u);
    const int64_t in_features = input_dims.back();
    const int64_t out_features = weight_dims[transpose ? 0 : 1];
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int64_t rows = input->NumElements() / in_features;
    auto output_dims = input_dims;
    output_dims.back() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    float beta = 0.0f;
    if (bias) {
        CheckSameTypeAndDevice(input, bias);
        CHECK_EQ(bias->Dims().size(), 1u);
        CHECK_EQ(bias->Dims()[0], out_features);
        CopyBias(output, bias, rows, out_features);
        beta = 1.0f;
    }
    Gemm(weight, transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
         transpose ? in_features : out_features, 0,
         input, CUBLAS_OP_N, in_features, 0, output, out_features, 0,
         out_features, rows, in_features, 1, beta);
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
               bool transpose, int64_t out_features, const std::shared_ptr<Tensor> &grad_output,
               bool bias) {
    CheckSameTypeAndDevice(input, weight);
    CheckSameTypeAndDevice(input, grad_output);
    const auto &input_dims = input->Dims();
    const auto &weight_dims = weight->Dims();
    const int64_t in_features = input_dims.back();
    const int64_t rows = input->NumElements() / in_features;
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);
    auto grad_input = std::make_shared<Tensor>(input_dims, input->Dtype(), input->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, weight->Dtype(), weight->GetDevice());
    std::shared_ptr<Tensor> grad_bias;

    if (transpose) {
        Gemm(weight, CUBLAS_OP_N, in_features, 0, grad_output, CUBLAS_OP_N, out_features, 0,
             grad_input, in_features, 0, in_features, rows, out_features);
        Gemm(input, CUBLAS_OP_N, in_features, 0, grad_output, CUBLAS_OP_T, out_features, 0,
             grad_weight, in_features, 0, in_features, out_features, rows);
    } else {
        Gemm(weight, CUBLAS_OP_T, out_features, 0, grad_output, CUBLAS_OP_N, out_features, 0,
             grad_input, in_features, 0, in_features, rows, out_features);
        Gemm(grad_output, CUBLAS_OP_N, out_features, 0, input, CUBLAS_OP_T, in_features, 0,
             grad_weight, out_features, 0, out_features, in_features, rows);
    }
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, grad_output->Dtype(),
                                             grad_output->GetDevice());
        ReduceBias(grad_output, grad_bias, rows, out_features);
    }
    return {grad_input, grad_weight, grad_bias};
}

std::shared_ptr<Tensor> LinearForwardMixedPrecision(
    const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
    bool transpose, const std::shared_ptr<Tensor> &bias) {
    return LinearForward(input, weight, transpose, bias);
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForwardMixedPrecision)

#undef REGISTER_CUDA_LINEAR_KERNEL
