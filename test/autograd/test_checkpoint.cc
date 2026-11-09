#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/checkpoint.h"
#include "infini_train/include/autograd/function.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
namespace {

class CountingSquare final : public Function {
public:
    CountingSquare() : Function("CountingSquare") {}

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        ++forward_calls;
        auto output = inputs.at(0)->Mul(inputs.at(0));
        inner_forward_recorded_grad.push_back(output->requires_grad());
        return {output};
    }

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &) override {
        ++setup_calls;
        saved_tensors_ = {inputs.at(0)};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {saved_tensors_.at(0)->Mul(2.0f)->Mul(grad_outputs.at(0))};
    }

    int forward_calls = 0;
    int setup_calls = 0;
    std::vector<bool> inner_forward_recorded_grad;
};

class CountingScaleModule final : public nn::Module {
public:
    CountingScaleModule() {
        auto scale = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32)->RequiresGrad();
        scale->Fill<float>(2.0f);
        parameters_["scale"] = std::move(scale);
    }

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        ++forward_calls;
        return {inputs.at(0)->Mul(parameters_.at("scale"))};
    }

    int forward_calls = 0;
};

TEST(CheckpointTest, RecomputesWithoutRetainingInnerGraph) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32)->RequiresGrad();
    input->Fill<float>(3.0f);
    auto square = std::make_shared<CountingSquare>();
    auto checkpoint = std::make_shared<CheckpointFunction>(square);

    auto output = checkpoint->Apply({input})[0];
    ASSERT_TRUE(output->requires_grad());
    EXPECT_EQ(square->forward_calls, 1);
    EXPECT_EQ(square->setup_calls, 0);
    ASSERT_EQ(square->inner_forward_recorded_grad.size(), 1u);
    EXPECT_FALSE(square->inner_forward_recorded_grad[0]);

    output->Backward();

    ASSERT_TRUE(input->grad());
    EXPECT_FLOAT_EQ(static_cast<const float *>(input->grad()->DataPtr())[0], 6.0f);
    EXPECT_EQ(square->forward_calls, 2);
    EXPECT_EQ(square->setup_calls, 1);
    ASSERT_EQ(square->inner_forward_recorded_grad.size(), 2u);
    EXPECT_FALSE(square->inner_forward_recorded_grad[1]);
    EXPECT_TRUE(checkpoint->GraphReleased());
}

TEST(CheckpointTest, NoGradGuardIsNestableAndRestoresState) {
    EXPECT_TRUE(IsGradEnabled());
    {
        NoGradGuard outer;
        EXPECT_FALSE(IsGradEnabled());
        {
            NoGradGuard inner;
            EXPECT_FALSE(IsGradEnabled());
        }
        EXPECT_FALSE(IsGradEnabled());
    }
    EXPECT_TRUE(IsGradEnabled());
}

TEST(CheckpointTest, ModuleRecomputationAccumulatesParameterAndInputGradientsOnce) {
    CountingScaleModule module;
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32)->RequiresGrad();
    input->Fill<float>(3.0f);

    auto output = CheckpointModule(&module, input);
    EXPECT_EQ(module.forward_calls, 1);
    EXPECT_FLOAT_EQ(static_cast<const float *>(output->DataPtr())[0], 6.0f);

    output->Backward();

    EXPECT_EQ(module.forward_calls, 2);
    ASSERT_TRUE(input->grad());
    EXPECT_FLOAT_EQ(static_cast<const float *>(input->grad()->DataPtr())[0], 2.0f);
    const auto &scale = module.parameter("scale");
    ASSERT_TRUE(scale->grad());
    EXPECT_FLOAT_EQ(static_cast<const float *>(scale->grad()->DataPtr())[0], 3.0f);
}

} // namespace
} // namespace infini_train::autograd
