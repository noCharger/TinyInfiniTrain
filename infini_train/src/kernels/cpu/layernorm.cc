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
void CheckLayerNormInputs(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                          const std::shared_ptr<Tensor> &bias) {
    CHECK_EQ(input->Dims().size(), 3u);
    CHECK_EQ(weight->Dims().size(), 1u);
    CHECK_EQ(bias->Dims().size(), 1u);
    CHECK_EQ(input->Dims()[2], weight->Dims()[0]);
    CHECK_EQ(input->Dims()[2], bias->Dims()[0]);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(weight->Dtype()));
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(bias->Dtype()));
    CHECK_EQ(input->GetDevice(), weight->GetDevice());
    CHECK_EQ(input->GetDevice(), bias->GetDevice());
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
}
} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                 const std::shared_ptr<Tensor> &bias, float eps) {
    CheckLayerNormInputs(input, weight, bias);
    const int64_t batch = input->Dims()[0];
    const int64_t sequence = input->Dims()[1];
    const int64_t hidden = input->Dims()[2];
    const int64_t rows = batch * sequence;
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto mean = std::make_shared<Tensor>(std::vector<int64_t>{batch, sequence}, DataType::kFLOAT32,
                                         input->GetDevice());
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>{batch, sequence}, DataType::kFLOAT32,
                                         input->GetDevice());
    auto *mean_data = static_cast<float *>(mean->DataPtr());
    auto *rstd_data = static_cast<float *>(rstd->DataPtr());

    for (int64_t row = 0; row < rows; ++row) {
        float row_mean = 0.0f;
        for (int64_t col = 0; col < hidden; ++col) {
            row_mean += dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col);
        }
        row_mean /= static_cast<float>(hidden);
        float variance = 0.0f;
        for (int64_t col = 0; col < hidden; ++col) {
            const float centered
                = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col) - row_mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(hidden);
        const float scale = 1.0f / std::sqrt(variance + eps);
        mean_data[row] = row_mean;
        rstd_data[row] = scale;
        for (int64_t col = 0; col < hidden; ++col) {
            const float normalized
                = (dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col) - row_mean) * scale;
            const float value = normalized * dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), col)
                                + dtype::LoadAsFloat(bias->DataPtr(), bias->Dtype(), col);
            dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), row * hidden + col, value);
        }
    }
    return {output, mean, rstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &mean,
                  const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    CheckLayerNormInputs(input, weight, bias);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(grad_output->Dtype()));
    CHECK_EQ(input->GetDevice(), grad_output->GetDevice());
    const int64_t batch = input->Dims()[0];
    const int64_t sequence = input->Dims()[1];
    const int64_t hidden = input->Dims()[2];
    const int64_t rows = batch * sequence;
    CHECK_EQ(mean->NumElements(), static_cast<size_t>(rows));
    CHECK_EQ(rstd->NumElements(), static_cast<size_t>(rows));

    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), weight->Dtype(), weight->GetDevice());
    auto grad_bias = std::make_shared<Tensor>(bias->Dims(), bias->Dtype(), bias->GetDevice());
    std::vector<float> weight_accumulator(hidden, 0.0f);
    std::vector<float> bias_accumulator(hidden, 0.0f);
    const auto *mean_data = static_cast<const float *>(mean->DataPtr());
    const auto *rstd_data = static_cast<const float *>(rstd->DataPtr());

    for (int64_t row = 0; row < rows; ++row) {
        float grad_mean = 0.0f;
        float grad_norm_mean = 0.0f;
        for (int64_t col = 0; col < hidden; ++col) {
            const float normalized
                = (dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col) - mean_data[row])
                  * rstd_data[row];
            const float grad = dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(),
                                                  row * hidden + col);
            const float grad_normalized = dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), col) * grad;
            grad_mean += grad_normalized;
            grad_norm_mean += grad_normalized * normalized;
            bias_accumulator[col] += grad;
            weight_accumulator[col] += grad * normalized;
        }
        grad_mean /= static_cast<float>(hidden);
        grad_norm_mean /= static_cast<float>(hidden);
        for (int64_t col = 0; col < hidden; ++col) {
            const float normalized
                = (dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * hidden + col) - mean_data[row])
                  * rstd_data[row];
            const float grad = dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(),
                                                  row * hidden + col);
            const float grad_normalized = dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), col) * grad;
            const float value = (grad_normalized - grad_mean - normalized * grad_norm_mean) * rstd_data[row];
            dtype::StoreFromFloat(grad_input->DataPtr(), grad_input->Dtype(), row * hidden + col, value);
        }
    }
    for (int64_t col = 0; col < hidden; ++col) {
        dtype::StoreFromFloat(grad_weight->DataPtr(), grad_weight->Dtype(), col, weight_accumulator[col]);
        dtype::StoreFromFloat(grad_bias->DataPtr(), grad_bias->Dtype(), col, bias_accumulator[col]);
    }
    return {grad_input, grad_weight, grad_bias};
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LAYERNORM_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LAYERNORM_KERNEL(LayerNormForward)
REGISTER_CPU_LAYERNORM_KERNEL(LayerNormBackward)

#undef REGISTER_CPU_LAYERNORM_KERNEL
