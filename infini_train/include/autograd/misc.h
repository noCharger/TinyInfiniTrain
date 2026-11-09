#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Cast : public Function {
public:
    explicit Cast(DataType dtype) : Function("CastFunction"), output_dtype_(dtype) {}

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    DataType input_dtype_ = DataType::kFLOAT32;
    DataType output_dtype_ = DataType::kFLOAT32;
};

// Dynamically scaled FP8 fake quantization. Forward quantizes through physical
// E4M3 bytes using `scaled = input * scale`, then dequantizes back to the input
// dtype. Backward uses the standard straight-through estimator so gradients
// and optimizer master weights remain BF16/FP16/FP32.
class FP8FakeQuantize : public Function {
public:
    explicit FP8FakeQuantize(float scale) : Function("FP8FakeQuantizeFunction"), scale_(scale) {
        CHECK_GT(scale_, 0.0f);
    }

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    float scale_;
};

class Split : public Function {
public:
    static constexpr char kType[] = "SplitFunction";

    Split(int64_t split_size, int dim = 0) : Function(kType), split_size_(split_size), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const int64_t split_size_ = 0;
    const int dim_ = 0;
    std::vector<int64_t> input_dims_;
};

class NoOp : public Function {
public:
    static constexpr char kType[] = "NoOpFunction";

    explicit NoOp(const std::vector<int64_t> &output_dims) : Function(kType), output_dims_(output_dims) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const std::vector<int64_t> output_dims_;
    std::vector<int64_t> input_dims_;
};

class Slice : public Function {
public:
    static constexpr char kType[] = "SliceFunction";

    Slice(const std::vector<int64_t> &starts, const std::vector<int64_t> &ends, const std::vector<int64_t> &steps)
        : Function(kType), starts_(starts), ends_(ends), steps_(steps) {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const std::vector<int64_t> starts_;
    const std::vector<int64_t> ends_;
    const std::vector<int64_t> steps_;
};

class Stack : public Function {
public:
    static constexpr char kType[] = "StackFunction";

    Stack(int64_t dim) : Function(kType), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t dim_ = 0;
    std::vector<int64_t> input_dims_;
};
} // namespace infini_train::autograd
