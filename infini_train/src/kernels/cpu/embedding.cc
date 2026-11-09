#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight) {
    /*
        x: [*]
        -> Embedding (weight: [num_embeddings, embedding_dim])
        -> o: [*, embedding_dim]
    */
    CHECK(input->Dtype() == DataType::kINT64);
    CHECK(dtype::IsFloatingPoint(weight->Dtype()));
    CHECK_EQ(input->GetDevice(), weight->GetDevice());
    const auto &input_dims = input->Dims();
    CHECK_EQ(weight->Dims().size(), 2);
    const int embedding_dim = weight->Dims()[1];
    auto output_dims = input_dims;
    output_dims.push_back(embedding_dim);
    auto output = std::make_shared<Tensor>(output_dims, weight->Dtype(), weight->GetDevice());

    for (int i = 0; i < input->NumElements(); i++) {
        int idx = static_cast<int>(static_cast<const int64_t *>(input->DataPtr())[i]);
        for (int j = 0; j < embedding_dim; j++) {
            dtype::StoreFromFloat(output->DataPtr(), output->Dtype(), i * embedding_dim + j,
                                  dtype::LoadAsFloat(weight->DataPtr(), weight->Dtype(), idx * embedding_dim + j));
        }
    }

    return output;
}

std::shared_ptr<Tensor> EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &weight_dims,
                                          const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->Dtype() == DataType::kINT64);
    CHECK_EQ(weight_dims.size(), 2);
    const int embedding_dim = weight_dims[1];
    CHECK_EQ(input->Dims().size() + 1, grad_output->Dims().size());
    for (int idx = 0; idx < input->Dims().size(); ++idx) { CHECK_EQ(input->Dims()[idx], grad_output->Dims()[idx]); }
    CHECK_EQ(*grad_output->Dims().rbegin(), embedding_dim);

    auto grad_weight = std::make_shared<Tensor>(weight_dims, grad_output->Dtype(), grad_output->GetDevice());
    std::vector<float> accumulator(grad_weight->NumElements(), 0.0f);

    for (int i = 0; i < input->NumElements(); i++) {
        int idx = static_cast<int>(static_cast<const int64_t *>(input->DataPtr())[i]);
        for (int j = 0; j < embedding_dim; j++) {
            accumulator[idx * embedding_dim + j]
                += dtype::LoadAsFloat(grad_output->DataPtr(), grad_output->Dtype(), i * embedding_dim + j);
        }
    }
    for (size_t i = 0; i < accumulator.size(); ++i) {
        dtype::StoreFromFloat(grad_weight->DataPtr(), grad_weight->Dtype(), i, accumulator[i]);
    }
    return grad_weight;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_EMBEDDING_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_EMBEDDING_KERNEL(EmbeddingForward)
REGISTER_CPU_EMBEDDING_KERNEL(EmbeddingBackward)

#undef REGISTER_CPU_EMBEDDING_KERNEL
