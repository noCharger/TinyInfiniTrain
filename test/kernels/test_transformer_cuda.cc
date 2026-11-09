#ifdef USE_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "infini_train/include/cuda_utils.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {
template <typename T, size_t N> void CopyFromHost(const std::shared_ptr<Tensor> &tensor, const T (&values)[N]) {
    const cudaStream_t stream = ThreadLocalStream(tensor->GetDevice().Index());
    ASSERT_EQ(cudaMemcpyAsync(tensor->DataPtr(), values, sizeof(values), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}
} // namespace

TEST(TransformerCudaKernelsTest, SwiGLUUsesFloat4Path) {
    const Device cuda(DeviceType::kCUDA, 0);
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 8}, DataType::kFLOAT32, cuda)->RequiresGrad();
    const float host_input[] = {
        0, 1, 2, 3, 2, 3, 4, 5,
        1, 0, -1, 2, 3, 2, 1, 4,
    };
    CopyFromHost(input, host_input);

    auto output = nn::function::SwiGLU(input);
    output->Backward();
    auto host_output = output->To(Device(DeviceType::kCPU, 0));
    auto host_grad = input->grad()->To(Device(DeviceType::kCPU, 0));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto *y = static_cast<const float *>(host_output.DataPtr());
    const auto *grad = static_cast<const float *>(host_grad.DataPtr());
    EXPECT_NEAR(y[0], 0.0f, 1e-6f);
    EXPECT_NEAR(y[1], 3.0f / (1.0f + std::exp(-1.0f)), 1e-5f);
    EXPECT_NEAR(y[4], 3.0f / (1.0f + std::exp(-1.0f)), 1e-5f);
    EXPECT_GT(grad[0], 0.9f);
    EXPECT_GT(grad[8], 2.0f);
}

TEST(TransformerCudaKernelsTest, RMSNormFP16PreservesDtype) {
    const Device cuda(DeviceType::kCUDA, 0);
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 2}, DataType::kFLOAT16, cuda)->RequiresGrad();
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT16, cuda)->RequiresGrad();
    const __half host_input[] = {__float2half(3.0f), __float2half(4.0f)};
    const __half host_weight[] = {__float2half(1.0f), __float2half(2.0f)};
    CopyFromHost(input, host_input);
    CopyFromHost(weight, host_weight);

    auto output = nn::function::RMSNorm(input, weight, 1e-6f);
    ASSERT_EQ(output->Dtype(), DataType::kFLOAT16);
    output->Backward();
    auto host_output = output->To(Device(DeviceType::kCPU, 0));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto *y = static_cast<const Eigen::half *>(host_output.DataPtr());
    EXPECT_NEAR(static_cast<float>(y[0]), 0.848528f, 2e-3f);
    EXPECT_NEAR(static_cast<float>(y[1]), 2.262742f, 2e-3f);
}

TEST(TransformerCudaKernelsTest, LayerNormBackwardMatchesAnalyticalGradient) {
    const Device cuda(DeviceType::kCUDA, 0);
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3}, DataType::kFLOAT32, cuda)->RequiresGrad();
    const float host_input[] = {1.0f, 2.0f, 4.0f};
    const float host_weight[] = {1.0f, 2.0f, 3.0f};
    CopyFromHost(input, host_input);
    nn::LayerNorm norm({3}, 1e-5f, cuda);
    CopyFromHost(*norm.mutable_parameter(nn::LayerNorm::kParamWeightName), host_weight);

    auto output = norm.Forward({input})[0];
    output->Backward();
    auto host_grad = input->grad()->To(Device());
    const auto *grad = static_cast<const float *>(host_grad.DataPtr());
    EXPECT_NEAR(grad[0], -0.1145446f, 1e-5f);
    EXPECT_NEAR(grad[1], 0.1718091f, 1e-5f);
    EXPECT_NEAR(grad[2], -0.0572646f, 1e-5f);
}

TEST(TransformerCudaKernelsTest, FP8FakeQuantizedBF16TrainingUsesMasterWeight) {
    const Device cuda(DeviceType::kCUDA, 0);
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{1, 1}, DataType::kBFLOAT16, cuda);
    auto weight = std::make_shared<Tensor>(
        std::vector<int64_t>{1, 1}, DataType::kBFLOAT16, cuda)->RequiresGrad();
    input->Fill<float>(1.0f);
    weight->Fill<float>(2.0f);
    optimizers::Adam optimizer({weight}, 0.1f);

    float initial_abs_output = 0.0f;
    float final_abs_output = 0.0f;
    for (int step = 0; step < 20; ++step) {
        optimizer.ZeroGrad();
        auto output = nn::function::FP8Matmul(input, weight);
        auto host_output = output->To(Device());
        const float abs_output = std::fabs(
            dtype::LoadAsFloat(host_output.DataPtr(), host_output.Dtype(), 0));
        if (step == 0) initial_abs_output = abs_output;
        final_abs_output = abs_output;
        output->Pow(2.0f)->Backward();
        optimizer.Step();
    }

    EXPECT_LT(final_abs_output, initial_abs_output * 0.25f);
}

#endif
