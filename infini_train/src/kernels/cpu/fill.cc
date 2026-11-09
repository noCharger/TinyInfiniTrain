#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void Fill(std::shared_ptr<Tensor> tensor, double value) {
    for (size_t i = 0; i < tensor->NumElements(); ++i) {
        dtype::StoreFromFloat(tensor->DataPtr(), tensor->Dtype(), i, static_cast<float>(value));
    }
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_FILL_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_FILL_KERNEL(Fill)

#undef REGISTER_CPU_FILL_KERNEL
