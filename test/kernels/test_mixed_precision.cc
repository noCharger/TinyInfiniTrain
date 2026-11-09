#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/dtype.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
namespace {

TEST(FP8Test, E4M3EncodingCoversNormalsSubnormalsAndSaturation) {
    EXPECT_EQ(dtype::EncodeFloat8E4M3(1.0f), 0x38);
    EXPECT_EQ(dtype::EncodeFloat8E4M3(0.5f), 0x30);
    EXPECT_EQ(dtype::EncodeFloat8E4M3(std::ldexp(1.0f, -9)), 0x01);
    EXPECT_EQ(dtype::EncodeFloat8E4M3(448.0f), 0x7e);
    EXPECT_EQ(dtype::EncodeFloat8E4M3(INFINITY), 0x7e);
    EXPECT_FLOAT_EQ(dtype::DecodeFloat8E4M3(0x38), 1.0f);
    EXPECT_FLOAT_EQ(dtype::DecodeFloat8E4M3(0xfe), -448.0f);
    EXPECT_TRUE(std::isnan(dtype::DecodeFloat8E4M3(0x7f)));
}

TEST(FP8Test, TensorCastUsesOneBytePhysicalStorage) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32);
    auto *values = static_cast<float *>(input->DataPtr());
    values[0] = 1.0f;
    values[1] = -0.5f;
    values[2] = 0.001953125f;
    values[3] = 500.0f;

    auto fp8 = input->Cast(DataType::kFLOAT8_E4M3);
    EXPECT_EQ(fp8->SizeInBytes(), 4u);
    auto restored = fp8->Cast(DataType::kFLOAT32);
    const auto *actual = static_cast<const float *>(restored->DataPtr());
    EXPECT_FLOAT_EQ(actual[0], 1.0f);
    EXPECT_FLOAT_EQ(actual[1], -0.5f);
    EXPECT_FLOAT_EQ(actual[2], 0.001953125f);
    EXPECT_FLOAT_EQ(actual[3], 448.0f);
}

TEST(FP8Test, DynamicScalingAndStraightThroughBackward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kBFLOAT16)->RequiresGrad();
    dtype::StoreFromFloat(input->DataPtr(), input->Dtype(), 0, 1.1f);
    dtype::StoreFromFloat(input->DataPtr(), input->Dtype(), 1, -0.7f);
    dtype::StoreFromFloat(input->DataPtr(), input->Dtype(), 2, 100.0f);

    EXPECT_FLOAT_EQ(nn::function::FP8DynamicScale(input), 4.0f);
    auto quantized = nn::function::FP8FakeQuantize(input);
    EXPECT_EQ(quantized->Dtype(), DataType::kBFLOAT16);
    EXPECT_NEAR(dtype::LoadAsFloat(quantized->DataPtr(), quantized->Dtype(), 0), 1.125f, 0.01f);
    EXPECT_NEAR(dtype::LoadAsFloat(quantized->DataPtr(), quantized->Dtype(), 2), 96.0f, 0.01f);

    quantized->Backward();
    ASSERT_TRUE(input->grad());
    for (size_t i = 0; i < input->NumElements(); ++i) {
        EXPECT_FLOAT_EQ(dtype::LoadAsFloat(input->grad()->DataPtr(), input->grad()->Dtype(), i), 1.0f);
    }
}

TEST(FP8Test, MatmulUsesQuantizedOperandsAndFP32Accumulation) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 2}, DataType::kFLOAT32)->RequiresGrad();
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{2, 1}, DataType::kFLOAT32)->RequiresGrad();
    auto *input_values = static_cast<float *>(input->DataPtr());
    input_values[0] = 1.1f;
    input_values[1] = -0.7f;
    auto *weight_values = static_cast<float *>(weight->DataPtr());
    weight_values[0] = 0.3f;
    weight_values[1] = 2.0f;

    auto output = nn::function::FP8Matmul(input, weight);
    EXPECT_EQ(output->Dtype(), DataType::kFLOAT32);
    EXPECT_NEAR(static_cast<const float *>(output->DataPtr())[0], -1.07f, 0.08f);
    output->Backward();
    EXPECT_TRUE(input->grad());
    EXPECT_TRUE(weight->grad());
}

TEST(FP8Test, FakeQuantizedBF16MatmulTrainsWithFP32MasterWeight) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, DataType::kBFLOAT16);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, DataType::kBFLOAT16)->RequiresGrad();
    input->Fill<float>(1.0f);
    weight->Fill<float>(2.0f);
    optimizers::Adam optimizer({weight}, 0.1f);

    float initial_abs_output = 0.0f;
    float final_abs_output = 0.0f;
    for (int step = 0; step < 20; ++step) {
        optimizer.ZeroGrad();
        auto output = nn::function::FP8Matmul(input, weight);
        ASSERT_EQ(output->Dtype(), DataType::kBFLOAT16);
        const float abs_output = std::fabs(
            dtype::LoadAsFloat(output->DataPtr(), output->Dtype(), 0));
        if (step == 0) initial_abs_output = abs_output;
        final_abs_output = abs_output;
        output->Pow(2.0f)->Backward();
        optimizer.Step();
    }

    EXPECT_LT(final_abs_output, initial_abs_output * 0.25f);
    EXPECT_TRUE(weight->grad());
}

} // namespace
} // namespace infini_train
