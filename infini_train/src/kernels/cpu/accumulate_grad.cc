#include <cmath>
#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    CHECK_EQ(gradient->NumElements(), tensor->NumElements());
    CHECK_EQ(static_cast<int>(gradient->Dtype()), static_cast<int>(tensor->Dtype()));
    CHECK(dtype::IsFloatingPoint(tensor->Dtype()));
    for (size_t idx = 0; idx < gradient->NumElements(); ++idx) {
        const float value = dtype::LoadAsFloat(tensor->DataPtr(), tensor->Dtype(), idx)
                            + rate * dtype::LoadAsFloat(gradient->DataPtr(), gradient->Dtype(), idx);
        dtype::StoreFromFloat(tensor->DataPtr(), tensor->Dtype(), idx, value);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    CHECK_EQ(grad->NumElements(), param->NumElements());
    CHECK_EQ(grad->NumElements(), m->NumElements());
    CHECK_EQ(grad->NumElements(), v->NumElements());
    CHECK(dtype::IsFloatingPoint(grad->Dtype()));
    CHECK(dtype::IsFloatingPoint(param->Dtype()));
    CHECK_EQ(static_cast<int>(m->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(v->Dtype()), static_cast<int>(DataType::kFLOAT32));
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());
    const int64_t n = grad->NumElements();
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));
    for (int64_t i = 0; i < n; ++i) {
        const float g = dtype::LoadAsFloat(grad->DataPtr(), grad->Dtype(), i);
        m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
        v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
        const float p = dtype::LoadAsFloat(param->DataPtr(), param->Dtype(), i)
                        - learning_rate * (m_ptr[i] / bc1) / (std::sqrt(v_ptr[i] / bc2) + eps);
        dtype::StoreFromFloat(param->DataPtr(), param->Dtype(), i, p);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
