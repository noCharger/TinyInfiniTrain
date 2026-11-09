#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {

void CheckSameFloatingTensorType(const std::shared_ptr<Tensor> &input,
                                 const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(other->Dtype()));
    CHECK_EQ(input->GetDevice(), other->GetDevice());
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
}

size_t CacheIndex(const Tensor &cache, int64_t sequence_index, int64_t pair_index,
                  int64_t half_hidden) {
    return cache.Dims().size() == 1 ? pair_index : sequence_index * half_hidden + pair_index;
}

void CheckRoPEShapes(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &cos,
                     const std::shared_ptr<Tensor> &sin) {
    CHECK_GE(input->Dims().size(), 2u);
    const int64_t hidden = input->Dims().back();
    const int64_t sequence = input->Dims()[input->Dims().size() - 2];
    CHECK_EQ(hidden % 2, 0);
    CheckSameFloatingTensorType(input, cos);
    CheckSameFloatingTensorType(input, sin);
    CHECK(cos->Dims() == sin->Dims());
    CHECK(cos->Dims().size() == 1u || cos->Dims().size() == 2u);
    if (cos->Dims().size() == 1u) {
        CHECK_EQ(cos->Dims()[0], hidden / 2);
    } else {
        CHECK_EQ(cos->Dims()[0], sequence);
        CHECK_EQ(cos->Dims()[1], hidden / 2);
    }
}

float Sigmoid(float x) { return 0.5f * (std::tanh(0.5f * x) + 1.0f); }

} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RMSNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, float eps) {
    CHECK_GE(input->Dims().size(), 1u);
    CHECK_EQ(weight->Dims().size(), 1u);
    const int64_t hidden = input->Dims().back();
    CHECK_EQ(weight->Dims()[0], hidden);
    CHECK_GT(eps, 0.0f);
    CheckSameFloatingTensorType(input, weight);

    const int64_t rows = static_cast<int64_t>(input->NumElements()) / hidden;
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>{rows}, DataType::kFLOAT32, input->GetDevice());
    auto *rstd_data = static_cast<float *>(rstd->DataPtr());
    for (int64_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0f;
        for (int64_t col = 0; col < hidden; ++col) {
            const float x = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col);
            sum_squares += x * x;
        }
        const float scale = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden) + eps);
        rstd_data[row] = scale;
        for (int64_t col = 0; col < hidden; ++col) {
            const float y = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col)
                            * scale * dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), col);
            dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), row * hidden + col, y);
        }
    }
    return {output, rstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RMSNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    CheckSameFloatingTensorType(input, weight);
    CheckSameFloatingTensorType(input, grad_output);
    const int64_t hidden = input->Dims().back();
    const int64_t rows = static_cast<int64_t>(input->NumElements()) / hidden;
    CHECK_EQ(rstd->NumElements(), static_cast<size_t>(rows));
    CHECK_EQ(static_cast<int>(rstd->Dtype()), static_cast<int>(DataType::kFLOAT32));
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), weight->Dtype(), weight->GetDevice());
    std::vector<float> weight_accumulator(hidden, 0.0f);
    const auto *rstd_data = static_cast<const float *>(rstd->DataPtr());

    for (int64_t row = 0; row < rows; ++row) {
        const float scale = rstd_data[row];
        float projection = 0.0f;
        for (int64_t col = 0; col < hidden; ++col) {
            const float x = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col);
            const float grad = dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), row * hidden + col);
            const float w = dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), col);
            projection += grad * w * x;
            weight_accumulator[col] += grad * x * scale;
        }
        const float correction = scale * scale * scale * projection / static_cast<float>(hidden);
        for (int64_t col = 0; col < hidden; ++col) {
            const float x = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col);
            const float grad = dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), row * hidden + col);
            const float w = dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), col);
            dtype::StoreFromFloat(grad_input->DataPtr(), grad_input->Dtype(), row * hidden + col,
                                  grad * w * scale - x * correction);
        }
    }
    for (int64_t col = 0; col < hidden; ++col) {
        dtype::StoreFromFloat(grad_weight->DataPtr(), grad_weight->Dtype(), col, weight_accumulator[col]);
    }
    return {grad_input, grad_weight};
}

std::shared_ptr<Tensor> RoPEForward(const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &cos,
                                    const std::shared_ptr<Tensor> &sin) {
    CheckRoPEShapes(input, cos, sin);
    const int64_t hidden = input->Dims().back();
    const int64_t half = hidden / 2;
    const int64_t sequence = input->Dims()[input->Dims().size() - 2];
    const int64_t rows = static_cast<int64_t>(input->NumElements()) / hidden;
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t token = row % sequence;
        for (int64_t pair = 0; pair < half; ++pair) {
            const size_t cache_idx = CacheIndex(*cos, token, pair, half);
            const float c = dtype::LoadAsFloat(cos->DataPtr(), cos->Dtype(), cache_idx);
            const float s = dtype::LoadAsFloat(sin->DataPtr(), sin->Dtype(), cache_idx);
            const float first = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + pair);
            const float second = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + half + pair);
            dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), row * hidden + pair,
                                  first * c - second * s);
            dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), row * hidden + half + pair,
                                  first * s + second * c);
        }
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RoPEBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &cos,
             const std::shared_ptr<Tensor> &sin, const std::shared_ptr<Tensor> &grad_output) {
    CheckRoPEShapes(input, cos, sin);
    CheckSameFloatingTensorType(input, grad_output);
    const int64_t hidden = input->Dims().back();
    const int64_t half = hidden / 2;
    const int64_t sequence = input->Dims()[input->Dims().size() - 2];
    const int64_t rows = static_cast<int64_t>(input->NumElements()) / hidden;
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto grad_cos = std::make_shared<Tensor>(cos->Dims(), cos->Dtype(), cos->GetDevice());
    auto grad_sin = std::make_shared<Tensor>(sin->Dims(), sin->Dtype(), sin->GetDevice());
    std::vector<float> cos_accumulator(cos->NumElements(), 0.0f);
    std::vector<float> sin_accumulator(sin->NumElements(), 0.0f);

    for (int64_t row = 0; row < rows; ++row) {
        const int64_t token = row % sequence;
        for (int64_t pair = 0; pair < half; ++pair) {
            const size_t cache_idx = CacheIndex(*cos, token, pair, half);
            const float c = dtype::LoadAsFloat(cos->DataPtr(), cos->Dtype(), cache_idx);
            const float s = dtype::LoadAsFloat(sin->DataPtr(), sin->Dtype(), cache_idx);
            const float first = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + pair);
            const float second = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + half + pair);
            const float grad_first
                = dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), row * hidden + pair);
            const float grad_second
                = dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), row * hidden + half + pair);
            dtype::StoreFromFloat(grad_input->DataPtr(), grad_input->Dtype(), row * hidden + pair,
                                  grad_first * c + grad_second * s);
            dtype::StoreFromFloat(grad_input->DataPtr(), grad_input->Dtype(), row * hidden + half + pair,
                                  -grad_first * s + grad_second * c);
            cos_accumulator[cache_idx] += grad_first * first + grad_second * second;
            sin_accumulator[cache_idx] += -grad_first * second + grad_second * first;
        }
    }
    for (size_t i = 0; i < cos_accumulator.size(); ++i) {
        dtype::StoreFromFloat(grad_cos->DataPtr(), grad_cos->Dtype(), i, cos_accumulator[i]);
        dtype::StoreFromFloat(grad_sin->DataPtr(), grad_sin->Dtype(), i, sin_accumulator[i]);
    }
    return {grad_input, grad_cos, grad_sin};
}

std::shared_ptr<Tensor> SwiGLUForward(const std::shared_ptr<Tensor> &input) {
    CHECK_GE(input->Dims().size(), 1u);
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
    const int64_t combined = input->Dims().back();
    CHECK_EQ(combined % 2, 0);
    const int64_t hidden = combined / 2;
    const int64_t rows = static_cast<int64_t>(input->NumElements()) / combined;
    auto output_dims = input->Dims();
    output_dims.back() = hidden;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < hidden; ++col) {
            const float gate = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * combined + col);
            const float up = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * combined + hidden + col);
            dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), row * hidden + col,
                                  gate * Sigmoid(gate) * up);
        }
    }
    return output;
}

std::shared_ptr<Tensor> SwiGLUBackward(const std::shared_ptr<Tensor> &input,
                                      const std::shared_ptr<Tensor> &grad_output) {
    CHECK_GE(input->Dims().size(), 1u);
    CheckSameFloatingTensorType(input, grad_output);
    const int64_t combined = input->Dims().back();
    CHECK_EQ(combined % 2, 0);
    const int64_t hidden = combined / 2;
    const int64_t rows = static_cast<int64_t>(input->NumElements()) / combined;
    CHECK_EQ(grad_output->NumElements(), static_cast<size_t>(rows * hidden));
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < hidden; ++col) {
            const float gate = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * combined + col);
            const float up = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * combined + hidden + col);
            const float grad = dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), row * hidden + col);
            const float sigmoid = Sigmoid(gate);
            dtype::StoreFromFloat(grad_input->DataPtr(), grad_input->Dtype(), row * combined + col,
                                  grad * up * (sigmoid + gate * sigmoid * (1.0f - sigmoid)));
            dtype::StoreFromFloat(grad_input->DataPtr(), grad_input->Dtype(), row * combined + hidden + col,
                                  grad * gate * sigmoid);
        }
    }
    return grad_input;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_TRANSFORMER_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_TRANSFORMER_KERNEL(RMSNormForward)
REGISTER_CPU_TRANSFORMER_KERNEL(RMSNormBackward)
REGISTER_CPU_TRANSFORMER_KERNEL(RoPEForward)
REGISTER_CPU_TRANSFORMER_KERNEL(RoPEBackward)
REGISTER_CPU_TRANSFORMER_KERNEL(SwiGLUForward)
REGISTER_CPU_TRANSFORMER_KERNEL(SwiGLUBackward)

#undef REGISTER_CPU_TRANSFORMER_KERNEL
