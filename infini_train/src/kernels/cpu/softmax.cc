#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "glog/logging.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> SoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim) {
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
    dim = dim < 0 ? input->Dims().size() + dim : dim;
    CHECK_GE(dim, 0);
    CHECK_LT(static_cast<size_t>(dim), input->Dims().size());
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const auto &dims = input->Dims();
    int64_t outer = 1;
    const int64_t axis = dims[dim];
    int64_t inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= dims[i];
    for (size_t i = dim + 1; i < dims.size(); ++i) inner *= dims[i];
    std::vector<float> values(axis);

    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
            const int64_t offset = outer_index * axis * inner + inner_index;
            float max_value = -std::numeric_limits<float>::infinity();
            for (int64_t axis_index = 0; axis_index < axis; ++axis_index) {
                max_value = std::max(max_value, dtype::LoadAsFloat(
                    input->DataPtr(), input->Dtype(), offset + axis_index * inner));
            }
            float sum = 0.0f;
            for (int64_t axis_index = 0; axis_index < axis; ++axis_index) {
                values[axis_index] = std::exp(dtype::LoadAsFloat(
                    input->DataPtr(), input->Dtype(), offset + axis_index * inner) - max_value);
                sum += values[axis_index];
            }
            for (int64_t axis_index = 0; axis_index < axis; ++axis_index) {
                dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), offset + axis_index * inner,
                                      values[axis_index] / sum);
            }
        }
    }
    return output;
}

std::shared_ptr<Tensor> SoftmaxBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &output, int64_t dim) {
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(output->Dtype()));
    CHECK_EQ(grad_output->GetDevice(), output->GetDevice());
    dim = dim < 0 ? output->Dims().size() + dim : dim;
    auto grad_input = std::make_shared<Tensor>(output->Dims(), output->Dtype(), output->GetDevice());
    const auto &dims = output->Dims();
    int64_t outer = 1;
    const int64_t axis = dims[dim];
    int64_t inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= dims[i];
    for (size_t i = dim + 1; i < dims.size(); ++i) inner *= dims[i];

    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
            const int64_t offset = outer_index * axis * inner + inner_index;
            float dot = 0.0f;
            for (int64_t axis_index = 0; axis_index < axis; ++axis_index) {
                const int64_t index = offset + axis_index * inner;
                dot += dtype::LoadAsFloat(output->DataPtr(), output->Dtype(), index)
                       * dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), index);
            }
            for (int64_t axis_index = 0; axis_index < axis; ++axis_index) {
                const int64_t index = offset + axis_index * inner;
                const float value = dtype::LoadAsFloat(output->DataPtr(), output->Dtype(), index)
                                    * (dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), index) - dot);
                dtype::StoreFromFloat(grad_input->DataPtr(), grad_input->Dtype(), index, value);
            }
        }
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_SOFTMAX_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_SOFTMAX_KERNEL(SoftmaxForward)
REGISTER_CPU_SOFTMAX_KERNEL(SoftmaxBackward)

#undef REGISTER_CPU_SOFTMAX_KERNEL
