#ifdef USE_CUDA
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <memory>

#include "cuda_runtime_api.h"

#include "gtest/gtest.h"
#include "infini_train/include/cuda_utils.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/autograd/matmul.h"

using namespace infini_train;

namespace {
template <typename T, size_t N> void CopyFromHost(const std::shared_ptr<Tensor> &tensor, const T (&values)[N]) {
    const cudaStream_t stream = ThreadLocalStream(tensor->GetDevice().Index());
    ASSERT_EQ(cudaMemcpyAsync(tensor->DataPtr(), values, sizeof(values), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}
} // namespace

TEST(MatmulTest, BasicMatrixMultiplyCuda) {
    // 创建输入张量 (2x3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCUDA, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1, 2, 3, 4, 5, 6};
    CopyFromHost(input, input_values);

    // 创建权重张量 (3x2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCUDA, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {7, 8, 9, 10, 11, 12};
    CopyFromHost(other, other_values);

    // 执行正向传播
    autograd::Matmul matmul_op;
    auto output = matmul_op.Forward({input, other});
    auto cpu_output = output[0]->To(Device(DeviceType::kCPU, 0));

    // 验证输出结果 (2x2)
    float expected[] = {58, 64, 
                        139, 154};

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(cpu_output.DataPtr())[i], expected[i]);
    }
}

TEST(MatmulTest, BatchedMatrixMultiplyCuda) {
    // 创建输入张量 (2, 2, 3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCUDA, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1,2,3,4,5,6,    // batch 0
                          7,8,9,10,11,12}; // batch 1
    CopyFromHost(input, input_values);

    // 创建权重张量 (2, 3, 2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCUDA, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {1,2,3,4,5,6,    // batch 0
                          7,8,9,10,11,12}; // batch 1
    CopyFromHost(other, other_values);

    autograd::Matmul matmul_op;
    auto output = matmul_op.Forward({input, other});
    auto cpu_output = output[0]->To(Device(DeviceType::kCPU, 0));

    // 验证输出结果 (2, 2, 2)
    float expected[] = {22, 28, 49, 64,    // batch 0
                        220, 244, 301, 334}; // batch 1
    for (int i = 0; i < std::size(expected); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(cpu_output.DataPtr())[i], expected[i]);
    }
}

TEST(MatmulTest, BackwardPassCuda) {
    // 创建输入张量 (2x3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCUDA, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1, 2, 3, 
                          4, 5, 6};
    CopyFromHost(input, input_values);

    // 创建权重张量 (3x2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCUDA, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {7, 8, 
                          9, 10, 
                          11, 12};
    CopyFromHost(other, other_values);

    // 模拟梯度输出 (2x2)
    auto grad_output = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCUDA, 0)
    );
    float* grad_data = static_cast<float*>(grad_output->DataPtr());
    float grad_values[] = {0.1, 0.2, 
                         0.3, 0.4};
    CopyFromHost(grad_output, grad_values);

    auto output_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);
    autograd::Matmul matmul_op;
    matmul_op.SetupContext({input, other}, {output_tensor});
    auto output = matmul_op.Backward({grad_output});
    auto grad_input = output[0]->To(Device(DeviceType::kCPU, 0));
    auto grad_other = output[1]->To(Device(DeviceType::kCPU, 0));

    // 验证输入梯度 (2x3)make
    float expected_grad_input[] = {2.3, 2.9, 3.5, 
                                   5.3, 6.7, 8.1};
    for (int i = 0; i < std::size(expected_grad_input); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(grad_input.DataPtr())[i], expected_grad_input[i]);
    }

    // 验证权重梯度 (3x2)
    float expected_grad_other[] = {1.3, 1.8, 
                                   1.7, 2.4, 
                                   2.1, 3.0};
    for (int i = 0; i < std::size(expected_grad_other); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(grad_other.DataPtr())[i], expected_grad_other[i]);
    }
}

TEST(MatmulTest, FP16BatchedForwardAndBackwardCuda) {
    const Device cuda(DeviceType::kCUDA, 0);
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 1, 2}, DataType::kFLOAT16, cuda)->RequiresGrad();
    auto other = std::make_shared<Tensor>(std::vector<int64_t>{2, 2, 1}, DataType::kFLOAT16, cuda)->RequiresGrad();
    const __half input_values[] = {
        __float2half(1.0f), __float2half(2.0f), __float2half(3.0f), __float2half(4.0f)};
    const __half other_values[] = {
        __float2half(5.0f), __float2half(6.0f), __float2half(7.0f), __float2half(8.0f)};
    CopyFromHost(input, input_values);
    CopyFromHost(other, other_values);

    auto output = input->Matmul(other);
    ASSERT_EQ(output->Dtype(), DataType::kFLOAT16);
    output->Backward();
    auto host_output = output->To(Device(DeviceType::kCPU, 0));
    auto host_grad_input = input->grad()->To(Device(DeviceType::kCPU, 0));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto *y = static_cast<const Eigen::half *>(host_output.DataPtr());
    const auto *grad = static_cast<const Eigen::half *>(host_grad_input.DataPtr());
    EXPECT_FLOAT_EQ(static_cast<float>(y[0]), 17.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(y[1]), 53.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad[0]), 5.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad[1]), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad[2]), 7.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad[3]), 8.0f);
}
#endif
