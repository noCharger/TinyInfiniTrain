#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::vector<std::shared_ptr<Tensor>> SplitForward(const std::shared_ptr<Tensor> &input, int64_t split_size, int dim) {
    CHECK_GT(split_size, 0);
    CHECK_GE(dim, 0) << "Currently we do not support negative dimension";
    const auto &input_dims = input->Dims();
    CHECK_LT(dim, input_dims.size());

    std::vector<std::shared_ptr<Tensor>> outputs;
    for (int64_t start = 0; start < input_dims[dim]; start += split_size) {
        auto output_dims = input_dims;
        output_dims[dim] = std::min(split_size, input_dims[dim] - start);
        auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());
        const size_t element_size = input->SizeInBytes() / input->NumElements();
        const int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
        const int64_t W
            = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
        for (int64_t n = 0; n < N; ++n) {
            // out[0:N, :, 0:W] = in[0:N, start:end, 0:W]
            const int64_t H_in = input_dims[dim];
            const int64_t H_out = output_dims[dim];
            const int64_t end = std::min(start + split_size, input_dims[dim]);
            memcpy(static_cast<uint8_t *>(output->DataPtr()) + n * H_out * W * element_size,
                   static_cast<const uint8_t *>(input->DataPtr()) + (n * H_in * W + start * W) * element_size,
                   (end - start) * W * element_size);
        }
        outputs.push_back(std::move(output));
    }
    return outputs;
}

std::shared_ptr<Tensor> SplitBackward(const std::vector<int64_t> &input_dims, int64_t split_size, int dim,
                                      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_GT(split_size, 0);
    CHECK_GE(dim, 0) << "Currently we do not support negative dimension";
    CHECK_LT(dim, input_dims.size());
    CHECK_EQ(grad_outputs.size(), (input_dims[dim] + split_size - 1) / split_size);

    CHECK(!grad_outputs.empty());
    auto grad_input = std::make_shared<Tensor>(input_dims, grad_outputs[0]->Dtype(), grad_outputs[0]->GetDevice());
    grad_input->Fill<float>(0.0f);
    const size_t element_size = grad_input->SizeInBytes() / grad_input->NumElements();
    for (int64_t start = 0, idx = 0; start < input_dims[dim]; start += split_size, ++idx) {
        auto output_dims = input_dims;
        output_dims[dim] = std::min(split_size, input_dims[dim] - start);
        const auto &grad_output = grad_outputs[idx];
        CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(grad_input->Dtype()));
        CHECK_EQ(grad_output->GetDevice(), grad_input->GetDevice());
        for (int dim_idx = 0; dim_idx < grad_input->Dims().size(); ++dim_idx) {
            CHECK_EQ(output_dims[dim_idx], grad_output->Dims()[dim_idx]);
        }
        const int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
        const int64_t W
            = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
        for (int64_t n = 0; n < N; ++n) {
            // grad_input[0:N, start:end, 0:W] = grad_output[0:N, :, 0:W]
            const int64_t H_in = input_dims[dim];
            const int64_t H_out = output_dims[dim];
            const int64_t end = std::min(start + split_size, input_dims[dim]);
            memcpy(static_cast<uint8_t *>(grad_input->DataPtr()) + (n * H_in * W + start * W) * element_size,
                   static_cast<const uint8_t *>(grad_output->DataPtr()) + n * H_out * W * element_size,
                   (end - start) * W * element_size);
        }
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_SPLIT_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_SPLIT_KERNEL(SplitForward)
REGISTER_CPU_SPLIT_KERNEL(SplitBackward)

#undef REGISTER_CPU_SPLIT_KERNEL
