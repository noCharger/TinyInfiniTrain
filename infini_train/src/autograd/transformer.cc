#include "infini_train/include/autograd/transformer.h"

#include <tuple>

#include "glog/logging.h"
#include "infini_train/include/dispatcher.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> RMSNorm::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2u);
    const auto &input = input_tensors[0];
    auto kernel = Dispatcher::Instance().GetKernel({input->GetDevice().Type(), "RMSNormForward"});
    auto [output, rstd]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
            input, input_tensors[1], eps_);
    saved_tensors_ = {rstd};
    return {output};
}

void RMSNorm::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                           const std::vector<std::shared_ptr<Tensor>> &) {
    saved_tensors_.insert(saved_tensors_.begin(), input_tensors.begin(), input_tensors.end());
}

std::vector<std::shared_ptr<Tensor>> RMSNorm::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 3u);
    CHECK_EQ(grad_outputs.size(), 1u);
    const auto &input = saved_tensors_[0];
    auto kernel = Dispatcher::Instance().GetKernel({input->GetDevice().Type(), "RMSNormBackward"});
    auto [grad_input, grad_weight]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
            input, saved_tensors_[1], saved_tensors_[2], grad_outputs[0]);
    return {grad_input, grad_weight};
}

std::vector<std::shared_ptr<Tensor>> RoPE::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3u);
    const auto &input = input_tensors[0];
    auto kernel = Dispatcher::Instance().GetKernel({input->GetDevice().Type(), "RoPEForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, input_tensors[1], input_tensors[2])};
}

void RoPE::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        const std::vector<std::shared_ptr<Tensor>> &) {
    saved_tensors_ = input_tensors;
}

std::vector<std::shared_ptr<Tensor>> RoPE::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 3u);
    CHECK_EQ(grad_outputs.size(), 1u);
    const auto &input = saved_tensors_[0];
    auto kernel = Dispatcher::Instance().GetKernel({input->GetDevice().Type(), "RoPEBackward"});
    auto [grad_input, grad_cos, grad_sin]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
            input, saved_tensors_[1], saved_tensors_[2], grad_outputs[0]);
    return {grad_input, grad_cos, grad_sin};
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1u);
    const auto &input = input_tensors[0];
    auto kernel = Dispatcher::Instance().GetKernel({input->GetDevice().Type(), "SwiGLUForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

void SwiGLU::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &) {
    saved_tensors_ = input_tensors;
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1u);
    CHECK_EQ(grad_outputs.size(), 1u);
    const auto &input = saved_tensors_[0];
    auto kernel = Dispatcher::Instance().GetKernel({input->GetDevice().Type(), "SwiGLUBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, grad_outputs[0])};
}

} // namespace infini_train::autograd
