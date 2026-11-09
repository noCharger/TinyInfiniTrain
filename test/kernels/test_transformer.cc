#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

TEST(TransformerKernelsTest, RMSNormFP16ForwardAndBackward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 2}, DataType::kFLOAT16)->RequiresGrad();
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT16)->RequiresGrad();
    auto *x = static_cast<dtype::Float16 *>(input->DataPtr());
    auto *w = static_cast<dtype::Float16 *>(weight->DataPtr());
    x[0] = dtype::Float16(3.0f); x[1] = dtype::Float16(4.0f);
    w[0] = dtype::Float16(1.0f); w[1] = dtype::Float16(2.0f);

    auto output = nn::function::RMSNorm(input, weight, 1e-6f);
    ASSERT_EQ(output->Dtype(), DataType::kFLOAT16);
    const auto *y = static_cast<const dtype::Float16 *>(output->DataPtr());
    EXPECT_NEAR(static_cast<float>(y[0]), 0.848528f, 2e-3f);
    EXPECT_NEAR(static_cast<float>(y[1]), 2.262742f, 2e-3f);

    output->Backward();
    const auto *gx = static_cast<const dtype::Float16 *>(input->grad()->DataPtr());
    const auto *gw = static_cast<const dtype::Float16 *>(weight->grad()->DataPtr());
    EXPECT_NEAR(static_cast<float>(gx[0]), -0.09051f, 2e-3f);
    EXPECT_NEAR(static_cast<float>(gx[1]), 0.06788f, 2e-3f);
    EXPECT_NEAR(static_cast<float>(gw[0]), 0.848528f, 2e-3f);
    EXPECT_NEAR(static_cast<float>(gw[1]), 1.131371f, 2e-3f);
}

TEST(TransformerKernelsTest, RoPESplitHalfForwardAndBackward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 4}, DataType::kFLOAT32)->RequiresGrad();
    auto cos = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);
    auto sin = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);
    float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float cos_values[] = {1, 1, 0, 0};
    float sin_values[] = {0, 0, 1, 1};
    std::memcpy(input->DataPtr(), input_values, sizeof(input_values));
    std::memcpy(cos->DataPtr(), cos_values, sizeof(cos_values));
    std::memcpy(sin->DataPtr(), sin_values, sizeof(sin_values));

    auto output = nn::function::RoPE(input, cos, sin);
    const float expected[] = {1, 2, 3, 4, -7, -8, 5, 6};
    const auto *y = static_cast<const float *>(output->DataPtr());
    for (size_t i = 0; i < std::size(expected); ++i) { EXPECT_FLOAT_EQ(y[i], expected[i]); }

    output->Backward();
    const float expected_grad[] = {1, 1, 1, 1, 1, 1, -1, -1};
    const auto *grad = static_cast<const float *>(input->grad()->DataPtr());
    for (size_t i = 0; i < std::size(expected_grad); ++i) {
        EXPECT_FLOAT_EQ(grad[i], expected_grad[i]);
    }
}

TEST(TransformerKernelsTest, SwiGLUBF16ForwardAndBackward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 4}, DataType::kBFLOAT16)->RequiresGrad();
    auto *x = static_cast<dtype::BFloat16 *>(input->DataPtr());
    x[0] = dtype::BFloat16(0.0f); x[1] = dtype::BFloat16(1.0f);
    x[2] = dtype::BFloat16(2.0f); x[3] = dtype::BFloat16(3.0f);

    auto output = nn::function::SwiGLU(input);
    ASSERT_EQ(output->Dims(), std::vector<int64_t>({1, 2}));
    ASSERT_EQ(output->Dtype(), DataType::kBFLOAT16);
    const auto *y = static_cast<const dtype::BFloat16 *>(output->DataPtr());
    EXPECT_NEAR(static_cast<float>(y[0]), 0.0f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(y[1]), 2.19318f, 2e-2f);

    output->Backward();
    const auto *grad = static_cast<const dtype::BFloat16 *>(input->grad()->DataPtr());
    EXPECT_NEAR(static_cast<float>(grad[0]), 1.0f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(grad[1]), 2.783f, 3e-2f);
    EXPECT_NEAR(static_cast<float>(grad[2]), 0.0f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(grad[3]), 0.731f, 2e-2f);
}
