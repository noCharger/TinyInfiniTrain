#include<iostream>
#include<vector>

#include "gtest/gtest.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/optimizer.h"

using namespace infini_train;

TEST(AdamOptimizerTest, BasicParameterUpdate) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32);
    param->Fill(1.0f); // 初始参数值 [1.0, 1.0, 1.0]
    param->RequiresGrad();
    
    auto grad = std::make_shared<Tensor>(param->Dims(), param->Dtype());
    grad->Fill(1.0f);
    float* grad_data = static_cast<float*>(param->grad()->DataPtr());
    std::memcpy(grad_data, grad->DataPtr(), grad->SizeInBytes());

    optimizers::Adam optimizer({param}, 0.001f, 0.9f, 0.999f, 1e-8);

    optimizer.Step();

    float* param_data = static_cast<float*>(param->DataPtr());
    for (int i = 0; i < 3; ++i) {
        EXPECT_LT(param_data[i], 1.0f); // 参数值应该减小
    }
}

TEST(AdamOptimizerTest, MomentumAccumulation) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32);
    param->Fill(1.0f);
    param->RequiresGrad();
    param->grad()->Fill(0.5f);

    float learning_rate = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

    optimizers::Adam optimizer({param}, learning_rate, beta1, beta2, eps);
    
    std::vector<float> param_history;
    for (int i = 0; i < 3; ++i) {
        optimizer.Step();
        param_history.push_back(static_cast<float*>(param->DataPtr())[0]);
    }

    EXPECT_LT(param_history[1], param_history[0]);
    EXPECT_LT(param_history[2], param_history[1]);

    float m = 0, v = 0, expected_update = 0;
    for (int t = 1; t <= 3; ++t) {
        m = beta1 * m + (1 - beta1) * 0.5f;       // 一阶动量
        v = beta2 * v + (1 - beta2) * 0.25f;      // 二阶动量
        float m_hat = m / (1.0f - std::pow(beta1, t));  // 动态校正因子
        float v_hat = v / (1.0f - std::pow(beta2, t));

        expected_update -= learning_rate * m_hat / (std::sqrt(v_hat) + 1e-8f);
        EXPECT_NEAR(param_history[t-1] - 1.0f, expected_update, 1e-5);
    }
}

TEST(AdamOptimizerTest, FP16ParameterUsesFP32State) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT16);
    param->Fill(1.0f);
    param->RequiresGrad();
    param->grad()->Fill(1.0f);

    optimizers::Adam optimizer({param}, 0.01f);
    optimizer.Step();

    const float value = static_cast<float>(static_cast<dtype::Float16 *>(param->DataPtr())[0]);
    EXPECT_LT(value, 1.0f);
    EXPECT_NEAR(value, 0.99f, 1e-3f);
}

TEST(AdamOptimizerTest, BF16MasterWeightAccumulatesSubUlpUpdates) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kBFLOAT16);
    param->Fill(1.0f);
    param->RequiresGrad();
    param->grad()->Fill(1.0f);

    optimizers::Adam optimizer({param}, 0.001f);
    for (int step = 0; step < 8; ++step) { optimizer.Step(); }

    const float value = static_cast<float>(static_cast<dtype::BFloat16 *>(param->DataPtr())[0]);
    EXPECT_LT(value, 1.0f);
    EXPECT_NEAR(value, 0.992f, 0.004f);
}

TEST(AdamOptimizerTest, FP8MasterWeightAccumulatesSubUlpUpdates) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT8_E4M3);
    param->Fill(1.0f);
    param->RequiresGrad();
    param->grad()->Fill(1.0f);

    optimizers::Adam optimizer({param}, 0.01f);
    // A direct E4M3 update would round 0.99 back to 1.0 on every step. The
    // FP32 master accumulates updates until the model copy crosses an E4M3 bin.
    for (int step = 0; step < 8; ++step) { optimizer.Step(); }

    EXPECT_LT(dtype::LoadAsFloat(param->DataPtr(), param->Dtype(), 0), 1.0f);
}
