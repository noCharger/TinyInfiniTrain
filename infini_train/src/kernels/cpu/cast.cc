#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> CastForward(const std::shared_ptr<Tensor> &input, DataType output_dtype) {
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
    CHECK(dtype::IsFloatingPoint(output_dtype));
    auto output = std::make_shared<Tensor>(input->Dims(), output_dtype, input->GetDevice());
    for (size_t i = 0; i < input->NumElements(); ++i) {
        dtype::StoreFromFloat(output->DataPtr(), output_dtype, i,
                              dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), i));
    }
    return output;
}

std::shared_ptr<Tensor> FP8FakeQuantizeForward(const std::shared_ptr<Tensor> &input, float scale) {
    CHECK(dtype::IsFloatingPoint(input->Dtype()));
    CHECK(input->Dtype() != DataType::kFLOAT8_E4M3);
    CHECK_GT(scale, 0.0f);
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    for (size_t i = 0; i < input->NumElements(); ++i) {
        const float value = dtype::LoadAsFloat(input->DataPtr(), input->Dtype(), i);
        const uint8_t quantized = dtype::EncodeFloat8E4M3(value * scale);
        dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), i,
                              dtype::DecodeFloat8E4M3(quantized) / scale);
    }
    return output;
}

} // namespace infini_train::kernels::cpu

REGISTER_KERNEL(infini_train::DeviceType::kCPU, CastForward,
                infini_train::kernels::cpu::CastForward)
REGISTER_KERNEL(infini_train::DeviceType::kCPU, FP8FakeQuantizeForward,
                infini_train::kernels::cpu::FP8FakeQuantizeForward)
