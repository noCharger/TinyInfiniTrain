#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

class RMSNorm : public Function {
public:
    static constexpr char kType[] = "RMSNormFunction";
    explicit RMSNorm(float eps) : Function(kType), eps_(eps) {}

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    float eps_;
};

class RoPE : public Function {
public:
    static constexpr char kType[] = "RoPEFunction";
    RoPE() : Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;
};

class SwiGLU : public Function {
public:
    static constexpr char kType[] = "SwiGLUFunction";
    SwiGLU() : Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;
};

} // namespace infini_train::autograd
