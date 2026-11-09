#include "infini_train/include/nn/functional.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/misc.h"
#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/autograd/transformer.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/nn/init.h"

namespace infini_train::nn::function {
std::shared_ptr<Tensor> Tril(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    return std::make_shared<autograd::Tril>(diagonal)->Apply({input})[0];
}

std::shared_ptr<Tensor> Triu(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    return std::make_shared<autograd::Triu>(diagonal)->Apply({input})[0];
}

std::shared_ptr<Tensor> Ones(const std::vector<int64_t> size) {
    auto ones = std::make_shared<Tensor>(size, DataType::kFLOAT32);
    return init::Ones(ones);
}

std::shared_ptr<Tensor> Reciprocal(const std::shared_ptr<Tensor> &input) { return input->Reciprocal(); }

std::shared_ptr<Tensor> Sin(const std::shared_ptr<Tensor> &input) { return input->Sin(); }

std::shared_ptr<Tensor> Cos(const std::shared_ptr<Tensor> &input) { return input->Cos(); }

std::shared_ptr<Tensor> Tanh(const std::shared_ptr<Tensor> &input) { return input->Tanh(); }

std::shared_ptr<Tensor> Pow(const std::shared_ptr<Tensor> &input, float exponent) { return input->Pow(exponent); }

std::shared_ptr<Tensor> Pow(float base, const std::shared_ptr<Tensor> &input) {
    return std::make_shared<autograd::Pow>(base, true)->Apply({input})[0];
}

std::shared_ptr<Tensor> Rsqrt(const std::shared_ptr<Tensor> &input) { return input->Rsqrt(); }

std::shared_ptr<Tensor> Mean(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Mean>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Slice(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                              const std::vector<int64_t> &ends, const std::vector<int64_t> &steps) {
    return input->Slice(starts, ends, steps);
}

std::shared_ptr<Tensor> Stack(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    return std::make_shared<autograd::Stack>(dim)->Apply(inputs)[0];
}

std::shared_ptr<Tensor> Softmax(const std::shared_ptr<Tensor> &input, int64_t dim) {
    return std::make_shared<autograd::Softmax>(dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Sigmoid(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<autograd::Sigmoid>()->Apply({input})[0];
}

std::shared_ptr<Tensor> RMSNorm(const std::shared_ptr<Tensor> &input,
                                const std::shared_ptr<Tensor> &weight, float eps) {
    return std::make_shared<autograd::RMSNorm>(eps)->Apply({input, weight})[0];
}

std::shared_ptr<Tensor> RoPE(const std::shared_ptr<Tensor> &input,
                            const std::shared_ptr<Tensor> &cos,
                            const std::shared_ptr<Tensor> &sin) {
    return std::make_shared<autograd::RoPE>()->Apply({input, cos, sin})[0];
}

std::shared_ptr<Tensor> SwiGLU(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<autograd::SwiGLU>()->Apply({input})[0];
}

float FP8DynamicScale(const std::shared_ptr<Tensor> &input, int margin) {
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
    CHECK(input->Dtype() != DataType::kFLOAT8_E4M3);
    auto host = std::make_shared<Tensor>(input->To(Device()));
    float amax = 0.0f;
    for (size_t i = 0; i < host->NumElements(); ++i) {
        amax = std::max(amax, std::fabs(dtype::LoadAsFloat(host->DataPtr(), host->Dtype(), i)));
    }
    if (amax == 0.0f || !std::isfinite(amax)) return 1.0f;
    const float exponent = std::floor(std::log2(448.0f / amax)) - static_cast<float>(margin);
    return std::exp2(std::clamp(exponent, -24.0f, 24.0f));
}

std::shared_ptr<Tensor> FP8FakeQuantize(const std::shared_ptr<Tensor> &input, float scale) {
    if (scale <= 0.0f) scale = FP8DynamicScale(input);
    return std::make_shared<autograd::FP8FakeQuantize>(scale)->Apply({input})[0];
}

std::shared_ptr<Tensor> FP8Matmul(const std::shared_ptr<Tensor> &input,
                                 const std::shared_ptr<Tensor> &other,
                                 float input_scale, float other_scale) {
    CHECK(input->Dtype() == other->Dtype());
    return FP8FakeQuantize(input, input_scale)->Matmul(FP8FakeQuantize(other, other_scale));
}
} // namespace infini_train::nn::function
