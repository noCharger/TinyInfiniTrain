#include <iostream>
#include <vector>
#include <memory>

#include "gtest/gtest.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/autograd/matmul.h"

using namespace infini_train;

TEST(MatmulTest, BasicMatrixMultiply) {
    // 创建输入张量 (2x3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(input_data, input_values, std::size(input_values) * sizeof(float));
    // 创建权重张量 (3x2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {7, 8, 9, 10, 11, 12};
    std::memcpy(other_data, other_values, std::size(other_values) * sizeof(float));

    autograd::Matmul matmul_op;
    auto output = matmul_op.Forward({input, other});

    // 验证输出结果 (2x2)
    float expected[] = {58, 64, 
                        139, 154};
    for (int i = 0; i < std::size(expected); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(output[0]->DataPtr())[i], expected[i]);
    }
}

TEST(MatmulTest, BatchedMatrixMultiply) {
    // 创建输入张量 (2, 2, 3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1,2,3,4,5,6,    // batch 0
                          7,8,9,10,11,12}; // batch 1
    std::memcpy(input_data, input_values, std::size(input_values) * sizeof(float));

    // 创建权重张量 (2, 3, 2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {1,2,3,4,5,6,    // batch 0
                          7,8,9,10,11,12}; // batch 1
    std::memcpy(other_data, other_values, std::size(other_values) * sizeof(float));

    // 执行正向传播
    autograd::Matmul matmul_op;
    auto output = matmul_op.Forward({input, other});

    // 验证输出结果 (2, 2, 2)
    float expected[] = {22, 28, 49, 64,    // batch 0
                        220, 244, 301, 334}; // batch 1
    for (int i = 0; i < std::size(expected); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(output[0]->DataPtr())[i], expected[i]);
    }
}

TEST(MatmulTest, BackwardPass) {
    // 创建输入张量 (2x3)
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 3},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* input_data = static_cast<float*>(input->DataPtr());
    float input_values[] = {1, 2, 3, 
                          4, 5, 6};
    std::memcpy(input_data, input_values, std::size(input_values) * sizeof(float));

    // 创建权重张量 (3x2)
    auto other = std::make_shared<Tensor>(
        std::vector<int64_t>{3, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* other_data = static_cast<float*>(other->DataPtr());
    float other_values[] = {7, 8, 
                          9, 10, 
                          11, 12};
    std::memcpy(other_data, other_values, std::size(other_values) * sizeof(float));

    // 模拟梯度输出 (2x2)
    auto grad_output = std::make_shared<Tensor>(
        std::vector<int64_t>{2, 2},
        DataType::kFLOAT32,
        Device(DeviceType::kCPU, 0)
    );
    float* grad_data = static_cast<float*>(grad_output->DataPtr());
    float grad_values[] = {0.1, 0.2, 
                         0.3, 0.4};
    std::memcpy(grad_data, grad_values, std::size(grad_values) * sizeof(float));

    auto output_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);

    autograd::Matmul matmul_op;
    matmul_op.SetupContext({input, other}, {output_tensor});
    auto output = matmul_op.Backward({grad_output});

    // 验证输入梯度 (2x3)make
    float expected_grad_input[] = {2.3, 2.9, 3.5, 
                                   5.3, 6.7, 8.1};
    for (int i = 0; i < std::size(expected_grad_input); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(output[0]->DataPtr())[i], expected_grad_input[i]);
    }

    // 验证权重梯度 (3x2)
    float expected_grad_other[] = {1.3, 1.8, 
                                   1.7, 2.4, 
                                   2.1, 3.0};
    for (int i = 0; i < std::size(expected_grad_other); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float*>(output[1]->DataPtr())[i], expected_grad_other[i]);
    }
}

TEST(MatmulTest, FP16ForwardAndAutograd) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT16);
    auto other = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT16);
    input->RequiresGrad();
    other->RequiresGrad();
    auto *a = static_cast<dtype::Float16 *>(input->DataPtr());
    auto *b = static_cast<dtype::Float16 *>(other->DataPtr());
    a[0] = dtype::Float16(1.0f); a[1] = dtype::Float16(2.0f);
    a[2] = dtype::Float16(3.0f); a[3] = dtype::Float16(4.0f);
    b[0] = dtype::Float16(5.0f); b[1] = dtype::Float16(6.0f);
    b[2] = dtype::Float16(7.0f); b[3] = dtype::Float16(8.0f);

    auto output = input->Matmul(other);
    ASSERT_EQ(output->Dtype(), DataType::kFLOAT16);
    const auto *out = static_cast<const dtype::Float16 *>(output->DataPtr());
    EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 19.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 22.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(out[2]), 43.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(out[3]), 50.0f);

    output->Backward();
    const auto *grad_a = static_cast<const dtype::Float16 *>(input->grad()->DataPtr());
    const auto *grad_b = static_cast<const dtype::Float16 *>(other->grad()->DataPtr());
    EXPECT_FLOAT_EQ(static_cast<float>(grad_a[0]), 11.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad_a[1]), 15.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad_a[2]), 11.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad_a[3]), 15.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad_b[0]), 4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad_b[1]), 4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad_b[2]), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(grad_b[3]), 6.0f);
}

TEST(MatmulTest, BF16BatchedForward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 1, 2}, DataType::kBFLOAT16);
    auto other = std::make_shared<Tensor>(std::vector<int64_t>{2, 2, 1}, DataType::kBFLOAT16);
    auto *a = static_cast<dtype::BFloat16 *>(input->DataPtr());
    auto *b = static_cast<dtype::BFloat16 *>(other->DataPtr());
    a[0] = dtype::BFloat16(1.0f); a[1] = dtype::BFloat16(2.0f);
    a[2] = dtype::BFloat16(3.0f); a[3] = dtype::BFloat16(4.0f);
    b[0] = dtype::BFloat16(5.0f); b[1] = dtype::BFloat16(6.0f);
    b[2] = dtype::BFloat16(7.0f); b[3] = dtype::BFloat16(8.0f);

    auto output = input->Matmul(other);
    ASSERT_EQ(output->Dtype(), DataType::kBFLOAT16);
    const auto *out = static_cast<const dtype::BFloat16 *>(output->DataPtr());
    EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 17.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 53.0f);
}

TEST(MatmulTest, BF16LinearForwardAndBackward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kBFLOAT16)->RequiresGrad();
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kBFLOAT16)->RequiresGrad();
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kBFLOAT16)->RequiresGrad();
    auto *x = static_cast<dtype::BFloat16 *>(input->DataPtr());
    auto *w = static_cast<dtype::BFloat16 *>(weight->DataPtr());
    x[0] = dtype::BFloat16(1.0f); x[1] = dtype::BFloat16(2.0f);
    x[2] = dtype::BFloat16(3.0f); x[3] = dtype::BFloat16(4.0f);
    w[0] = dtype::BFloat16(1.0f); w[1] = dtype::BFloat16(2.0f);
    w[2] = dtype::BFloat16(3.0f); w[3] = dtype::BFloat16(4.0f);
    bias->Fill<float>(1.0f);

    auto output = std::make_shared<autograd::Linear>()->Apply({input, weight, bias})[0];
    ASSERT_EQ(output->Dtype(), DataType::kBFLOAT16);
    const auto *y = static_cast<const dtype::BFloat16 *>(output->DataPtr());
    EXPECT_FLOAT_EQ(static_cast<float>(y[0]), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(y[1]), 12.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(y[2]), 12.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(y[3]), 26.0f);

    output->Backward();
    const auto *gx = static_cast<const dtype::BFloat16 *>(input->grad()->DataPtr());
    const auto *gw = static_cast<const dtype::BFloat16 *>(weight->grad()->DataPtr());
    const auto *gb = static_cast<const dtype::BFloat16 *>(bias->grad()->DataPtr());
    EXPECT_FLOAT_EQ(static_cast<float>(gx[0]), 4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gx[1]), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gx[2]), 4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gx[3]), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gw[0]), 4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gw[1]), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gw[2]), 4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gw[3]), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gb[0]), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(gb[1]), 2.0f);
}

TEST(MatmulTest, CreateGraphSupportsBatchedInputSecondDerivative) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 1, 1}, DataType::kFLOAT32)->RequiresGrad();
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, DataType::kFLOAT32)->RequiresGrad();
    static_cast<float *>(input->DataPtr())[0] = 1.0f;
    static_cast<float *>(input->DataPtr())[1] = 2.0f;
    static_cast<float *>(weight->DataPtr())[0] = 3.0f;

    auto output = input->Matmul(weight);
    output->Backward(nullptr, false, true);
    ASSERT_TRUE(input->grad()->requires_grad());
    weight->ZeroGrad();
    input->grad()->Backward();
    EXPECT_FLOAT_EQ(static_cast<float *>(weight->grad()->DataPtr())[0], 2.0f);
}
