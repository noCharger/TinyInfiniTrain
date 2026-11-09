#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

using RowMajorMatf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {
void GemmLowPrecision(const Tensor &a, size_t a_offset, const Tensor &b, size_t b_offset,
                      Tensor *c, size_t c_offset, int64_t m, int64_t k, int64_t n,
                      bool transpose_b = false) {
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int64_t inner = 0; inner < k; ++inner) {
                const size_t a_idx = a_offset + row * k + inner;
                const size_t b_idx = transpose_b ? b_offset + col * k + inner
                                                 : b_offset + inner * n + col;
                sum += dtype::LoadAsFloat(a.DataPtr(), a.Dtype(), a_idx)
                       * dtype::LoadAsFloat(b.DataPtr(), b.Dtype(), b_idx);
            }
            dtype::StoreFromFloat(c->DataPtr(), c->Dtype(), c_offset + row * n + col, sum);
        }
    }
}
} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(other->Dtype()));
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
    CHECK_EQ(input->GetDevice(), other->GetDevice());
    const int64_t k = *input_dims.rbegin();
    const int64_t n = *other_dims.rbegin();
    CHECK_EQ(k, *(other_dims.rbegin() + 1));

    if (other_dims.size() == 2) {
        // Broadcast: input[..., K] @ other[K, N]
        auto output_dims = input_dims;
        *output_dims.rbegin() = n;
        auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());
        if (input->Dtype() == DataType::kFLOAT32) {
            output->EigenMatrix() = input->EigenMatrix() * other->EigenMatrix();
        } else {
            const int64_t rows = static_cast<int64_t>(input->NumElements()) / k;
            GemmLowPrecision(*input, 0, *other, 0, output.get(), 0, rows, k, n);
        }
        return output;
    }

    // Batched: input[..., M, K] @ other[..., K, N]
    CHECK_EQ(input_dims.size(), other_dims.size());
    const int64_t m = *(input_dims.rbegin() + 1);
    int64_t batch = 1;
    for (size_t i = 0; i < input_dims.size() - 2; ++i) {
        CHECK_EQ(input_dims[i], other_dims[i]);
        batch *= input_dims[i];
    }
    auto output_dims = input_dims;
    *output_dims.rbegin() = n;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());
    if (input->Dtype() == DataType::kFLOAT32) {
        const float *a = static_cast<const float *>(input->DataPtr());
        const float *b = static_cast<const float *>(other->DataPtr());
        float *c = static_cast<float *>(output->DataPtr());
        for (int64_t i = 0; i < batch; ++i) {
            Eigen::Map<RowMajorMatf>(c + i * m * n, m, n).noalias() =
                Eigen::Map<const RowMajorMatf>(a + i * m * k, m, k) *
                Eigen::Map<const RowMajorMatf>(b + i * k * n, k, n);
        }
    } else {
        for (int64_t i = 0; i < batch; ++i) {
            GemmLowPrecision(*input, i * m * k, *other, i * k * n,
                             output.get(), i * m * n, m, k, n);
        }
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const int64_t k = *input_dims.rbegin();
    const int64_t n = *other_dims.rbegin();

    auto grad_input = std::make_shared<Tensor>(input_dims, input->Dtype(), input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, other->Dtype(), other->GetDevice());

    if (other_dims.size() == 2) {
        // grad_input = grad_output @ other^T
        if (input->Dtype() == DataType::kFLOAT32) {
            grad_input->EigenMatrix() = grad_output->EigenMatrix() * other->EigenMatrix().transpose();
            // grad_other = input^T @ grad_output
            grad_other->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
        } else {
            const int64_t rows = static_cast<int64_t>(input->NumElements()) / k;
            GemmLowPrecision(*grad_output, 0, *other, 0, grad_input.get(), 0,
                             rows, n, k, /*transpose_b=*/true);
            for (int64_t inner = 0; inner < k; ++inner) {
                for (int64_t col = 0; col < n; ++col) {
                    float sum = 0.0f;
                    for (int64_t row = 0; row < rows; ++row) {
                        sum += dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * k + inner)
                               * dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), row * n + col);
                    }
                    dtype::StoreFromFloat(grad_other->DataPtr(), grad_other->Dtype(), inner * n + col, sum);
                }
            }
        }
        return {grad_input, grad_other};
    }

    // Batched backward
    const int64_t m = *(input_dims.rbegin() + 1);
    int64_t batch = 1;
    for (size_t i = 0; i < input_dims.size() - 2; ++i) { batch *= input_dims[i]; }
    if (input->Dtype() == DataType::kFLOAT32) {
        const float *go = static_cast<const float *>(grad_output->DataPtr());
        const float *a = static_cast<const float *>(input->DataPtr());
        const float *b = static_cast<const float *>(other->DataPtr());
        float *ga = static_cast<float *>(grad_input->DataPtr());
        float *gb = static_cast<float *>(grad_other->DataPtr());
        for (int64_t i = 0; i < batch; ++i) {
            Eigen::Map<RowMajorMatf>(ga + i * m * k, m, k).noalias() =
                Eigen::Map<const RowMajorMatf>(go + i * m * n, m, n) *
                Eigen::Map<const RowMajorMatf>(b + i * k * n, k, n).transpose();
            Eigen::Map<RowMajorMatf>(gb + i * k * n, k, n).noalias() =
                Eigen::Map<const RowMajorMatf>(a + i * m * k, m, k).transpose() *
                Eigen::Map<const RowMajorMatf>(go + i * m * n, m, n);
        }
    } else {
        for (int64_t i = 0; i < batch; ++i) {
            GemmLowPrecision(*grad_output, i * m * n, *other, i * k * n,
                             grad_input.get(), i * m * k, m, n, k, true);
            for (int64_t inner = 0; inner < k; ++inner) {
                for (int64_t col = 0; col < n; ++col) {
                    float sum = 0.0f;
                    for (int64_t row = 0; row < m; ++row) {
                        sum += dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), i * m * k + row * k + inner)
                               * dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(),
                                                    i * m * n + row * n + col);
                    }
                    dtype::StoreFromFloat(grad_other->DataPtr(), grad_other->Dtype(),
                                          i * k * n + inner * n + col, sum);
                }
            }
        }
    }
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(weight->Dtype()));
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
    CHECK_EQ(input->GetDevice(), weight->GetDevice());
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
        CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(bias->Dtype()));
        CHECK_EQ(input->GetDevice(), bias->GetDevice());
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    if (input->Dtype() == DataType::kFLOAT32) {
        if (transpose) {
            output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
        } else {
            output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
        }

        if (bias) {
            output->EigenMatrix().rowwise() += bias->EigenVector();
        }
    } else {
        for (int64_t row = 0; row < bs; ++row) {
            for (int64_t out = 0; out < out_features; ++out) {
                float sum = bias ? dtype::LoadAsFloat(bias->DataPtr(), bias->Dtype(), out) : 0.0f;
                for (int64_t in = 0; in < in_features; ++in) {
                    const int64_t weight_idx = transpose ? out * in_features + in
                                                         : in * out_features + out;
                    sum += dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * in_features + in)
                           * dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), weight_idx);
                }
                dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), row * out_features + out, sum);
            }
        }
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(weight->Dtype()));
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(grad_output->Dtype()));
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
    CHECK_EQ(input->GetDevice(), weight->GetDevice());
    CHECK_EQ(input->GetDevice(), grad_output->GetDevice());
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, input->Dtype(), input->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, weight->Dtype(), weight->GetDevice());
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, input->Dtype(), input->GetDevice());
    }

    if (input->Dtype() == DataType::kFLOAT32) {
        if (transpose) {
            grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
            grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
        } else {
            grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
            grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
        }
        if (bias) {
            grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
        }
    } else {
        for (int64_t row = 0; row < bs; ++row) {
            for (int64_t in = 0; in < in_features; ++in) {
                float sum = 0.0f;
                for (int64_t out = 0; out < out_features; ++out) {
                    const int64_t weight_idx = transpose ? out * in_features + in
                                                         : in * out_features + out;
                    sum += dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(),
                                              row * out_features + out)
                           * dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), weight_idx);
                }
                dtype::StoreFromFloat(grad_input->DataPtr(), grad_input->Dtype(), row * in_features + in, sum);
            }
        }
        for (int64_t in = 0; in < in_features; ++in) {
            for (int64_t out = 0; out < out_features; ++out) {
                float sum = 0.0f;
                for (int64_t row = 0; row < bs; ++row) {
                    sum += dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), row * in_features + in)
                           * dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(),
                                                row * out_features + out);
                }
                const int64_t weight_idx = transpose ? out * in_features + in
                                                     : in * out_features + out;
                dtype::StoreFromFloat(grad_weight->DataPtr(), grad_weight->Dtype(), weight_idx, sum);
            }
        }
        if (bias) {
            for (int64_t out = 0; out < out_features; ++out) {
                float sum = 0.0f;
                for (int64_t row = 0; row < bs; ++row) {
                    sum += dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(),
                                              row * out_features + out);
                }
                dtype::StoreFromFloat(grad_bias->DataPtr(), grad_bias->Dtype(), out, sum);
            }
        }
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
