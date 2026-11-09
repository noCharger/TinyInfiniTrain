#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"

using namespace infini_train;

TEST(TensorAutogradTest, BackwardComputesGradient) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(2.0f);

    auto y = x->Pow(2); 

    y->Backward();

    // dy/dx = 2x = [4.0] (当x=2时)
    float* grad = static_cast<float*>(x->grad()->DataPtr());
    EXPECT_FLOAT_EQ(grad[0], 4.0f);
}

TEST(TensorAutogradTest, BackwardWithMultipleOutputs) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(1.0f);  // [1.0, 1.0, 1.0]

    auto y1 = x->Mul(2.0f);  // [2.0, 2.0, 2.0]
    auto y2 = x->Pow(3);     // [1.0, 1.0, 1.0]

    auto grad1 = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    grad1->Fill(1.0f);
    auto grad2 = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    grad2->Fill(2.0f);
    y1->Backward(grad1);
    y2->Backward(grad2);

    // dy1/dx = 2, dy2/dx = 3x^2 = 3  总梯度 = 1 * 2 + 2 * 3 = 8
    float* grad = static_cast<float*>(x->grad()->DataPtr());
    EXPECT_FLOAT_EQ(grad[0], 8.0f);
    EXPECT_FLOAT_EQ(grad[1], 8.0f);
    EXPECT_FLOAT_EQ(grad[2], 8.0f);
}

TEST(TensorAutogradTest, TopologicalBackwardAccumulatesDiamondGraphOnce) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(3.0f);

    auto square = x * x;
    auto y = square + square;
    y->Backward();

    EXPECT_FLOAT_EQ(static_cast<float *>(x->grad()->DataPtr())[0], 12.0f);
}

TEST(TensorAutogradTest, RetainGraphAllowsSecondBackward) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(2.0f);

    auto y = x->Pow(2.0f);
    y->Backward(nullptr, true);
    y->Backward();

    EXPECT_FLOAT_EQ(static_cast<float *>(x->grad()->DataPtr())[0], 8.0f);
}

TEST(TensorAutogradTest, ReleasedGraphRejectsSecondBackward) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(2.0f);
    auto y = x->Pow(2.0f);
    y->Backward();

    EXPECT_DEATH(y->Backward(), "retain_graph=true");
}

TEST(TensorAutogradTest, CreateGraphSupportsSecondDerivative) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(2.0f);

    auto y = x->Pow(3.0f);
    y->Backward(nullptr, false, true);
    auto first_derivative = x->grad();
    ASSERT_TRUE(first_derivative->requires_grad());
    EXPECT_FLOAT_EQ(static_cast<float *>(first_derivative->DataPtr())[0], 12.0f);

    x->ZeroGrad();
    first_derivative->Backward();
    EXPECT_FLOAT_EQ(static_cast<float *>(x->grad()->DataPtr())[0], 12.0f);
}

TEST(TensorAutogradTest, CreateGraphSupportsUnarySecondDerivative) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    x->RequiresGrad();
    x->Fill(0.5f);

    auto y = x->Sin();
    y->Backward(nullptr, false, true);
    auto first_derivative = x->grad();
    EXPECT_NEAR(static_cast<float *>(first_derivative->DataPtr())[0], std::cos(0.5f), 1e-6f);

    x->ZeroGrad();
    first_derivative->Backward();
    EXPECT_NEAR(static_cast<float *>(x->grad()->DataPtr())[0], -std::sin(0.5f), 1e-6f);
}

TEST(TensorAutogradTest, CreateGraphPreservesBroadcastReduction) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    x->RequiresGrad();
    auto *x_data = static_cast<float *>(x->DataPtr());
    x_data[0] = 1.0f;
    x_data[1] = 2.0f;
    x_data[2] = 3.0f;

    auto scale = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    scale->RequiresGrad();
    scale->Fill(2.0f);

    auto y = x * scale;
    y->Backward(nullptr, false, true);
    auto scale_derivative = scale->grad();
    ASSERT_TRUE(scale_derivative->requires_grad());
    EXPECT_FLOAT_EQ(static_cast<float *>(scale_derivative->DataPtr())[0], 6.0f);

    x->ZeroGrad();
    scale->ZeroGrad();
    scale_derivative->Backward();
    const auto *x_grad = static_cast<const float *>(x->grad()->DataPtr());
    EXPECT_FLOAT_EQ(x_grad[0], 1.0f);
    EXPECT_FLOAT_EQ(x_grad[1], 1.0f);
    EXPECT_FLOAT_EQ(x_grad[2], 1.0f);
}

TEST(TensorTransformTest, Flatten2DTo1D) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32);
    float* data = static_cast<float*>(t->DataPtr());
    for (int i = 0; i < 12; ++i) {
        data[i] = i + 1;  // 1-12
    }

    auto flattened = t->Flatten(0, 1);

    EXPECT_EQ(flattened->Dims(), std::vector<int64_t>({12}));
    float* flat_data = static_cast<float*>(flattened->DataPtr());
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(flat_data[i], i + 1);
    }
}

TEST(TensorTransformTest, FlattenWithRange) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32);

    auto result = t->Flatten(1, -1);
    
    EXPECT_EQ(result->Dims(), (std::vector<int64_t>{2, 12}));
}

TEST(TensorTransformTest, FlattenNonContiguous) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32);
    auto transposed = t->Transpose(0, 1);  
    auto flattened = transposed->Flatten(0, -1);
    
    EXPECT_EQ(flattened->Dims(), std::vector<int64_t>{12});

    EXPECT_EQ(flattened->NumElements(), 12);
}

TEST(TensorTransformTest, NestedViewPreservesStorageOffset) {
    auto base = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32);
    auto *data = static_cast<float *>(base->DataPtr());
    data[0] = 10.0f;
    data[1] = 20.0f;
    data[2] = 30.0f;
    data[3] = 40.0f;

    auto subview = std::make_shared<Tensor>(*base, sizeof(float), std::vector<int64_t>{3});
    auto reshaped = subview->View({3, 1});
    auto same_device = reshaped->To(Device());

    EXPECT_EQ(same_device.DataPtr(), reshaped->DataPtr());
    const auto *view_data = static_cast<const float *>(same_device.DataPtr());
    EXPECT_FLOAT_EQ(view_data[0], 20.0f);
    EXPECT_FLOAT_EQ(view_data[1], 30.0f);
    EXPECT_FLOAT_EQ(view_data[2], 40.0f);
}

TEST(TensorTransformTest, StorageCanBeReleasedAndRematerializedWithoutLosingMetadata) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    tensor->Fill<float>(1.0f);
    ASSERT_TRUE(tensor->HasStorage());

    tensor->ReleaseStorage();
    EXPECT_FALSE(tensor->HasStorage());
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensor->Dtype(), DataType::kFLOAT32);
    EXPECT_EQ(tensor->GetDevice(), Device());

    tensor->AllocateStorage();
    ASSERT_TRUE(tensor->HasStorage());
    tensor->Fill<float>(2.0f);
    for (size_t i = 0; i < tensor->NumElements(); ++i) {
        EXPECT_FLOAT_EQ(static_cast<const float *>(tensor->DataPtr())[i], 2.0f);
    }
}

TEST(TensorDTypeTest, FillAndElementwisePreserveFP16) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT16);
    x->Fill(1.5f);
    auto y = -x;

    EXPECT_EQ(y->Dtype(), DataType::kFLOAT16);
    const auto *data = static_cast<const dtype::Float16 *>(y->DataPtr());
    for (int i = 0; i < 3; ++i) { EXPECT_FLOAT_EQ(static_cast<float>(data[i]), -1.5f); }
}

TEST(TensorDTypeTest, FillAndElementwisePreserveBF16) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kBFLOAT16);
    x->Fill(2.0f);
    auto y = x * 3.0f;

    EXPECT_EQ(y->Dtype(), DataType::kBFLOAT16);
    const auto *data = static_cast<const dtype::BFloat16 *>(y->DataPtr());
    for (int i = 0; i < 3; ++i) { EXPECT_FLOAT_EQ(static_cast<float>(data[i]), 6.0f); }
}

TEST(TensorDTypeTest, CastRoundTripPreservesValuesAndAutograd) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    x->RequiresGrad();
    auto *data = static_cast<float *>(x->DataPtr());
    data[0] = 1.25f;
    data[1] = -2.5f;
    data[2] = 4.0f;

    auto fp16 = x->Cast(DataType::kFLOAT16);
    auto round_trip = fp16->Cast(DataType::kFLOAT32);
    ASSERT_EQ(round_trip->Dtype(), DataType::kFLOAT32);
    const auto *round_trip_data = static_cast<const float *>(round_trip->DataPtr());
    EXPECT_FLOAT_EQ(round_trip_data[0], 1.25f);
    EXPECT_FLOAT_EQ(round_trip_data[1], -2.5f);
    EXPECT_FLOAT_EQ(round_trip_data[2], 4.0f);

    round_trip->Backward();
    const auto *grad = static_cast<const float *>(x->grad()->DataPtr());
    EXPECT_FLOAT_EQ(grad[0], 1.0f);
    EXPECT_FLOAT_EQ(grad[1], 1.0f);
    EXPECT_FLOAT_EQ(grad[2], 1.0f);
}
