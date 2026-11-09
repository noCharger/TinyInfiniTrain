#include "infini_train/include/autograd/matmul.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Matmul::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];

    auto device = input1->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MatmulForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input1, input2)};
}

void Matmul::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];
    const auto &output = output_tensors[0];
    saved_tensors_ = {input1, input2};
    out_features_ = output->Dims()[0];
}

std::vector<std::shared_ptr<Tensor>> Matmul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input1 = saved_tensors_[0];
    const auto &input2 = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    if (IsCreateGraphEnabled()) {
        auto grad_input1 = grad_output->Matmul(input2->Transpose(-2, -1));
        std::shared_ptr<Tensor> grad_input2;
        if (input2->Dims().size() == 2u) {
            const int64_t k = input1->Dims().back();
            const int64_t n = input2->Dims().back();
            const int64_t rows = static_cast<int64_t>(input1->NumElements()) / k;
            auto input_2d = input1->View({rows, k});
            auto grad_2d = grad_output->View({rows, n});
            grad_input2 = input_2d->Transpose(0, 1)->Matmul(grad_2d);
        } else {
            grad_input2 = input1->Transpose(-2, -1)->Matmul(grad_output);
        }
        return {grad_input1, grad_input2};
    }

    auto device = input1->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MatmulBackward"});
    auto [grad_input1, grad_input2]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(input1, input2, grad_output);
    return {grad_input1, grad_input2};
}
} // namespace infini_train::autograd
