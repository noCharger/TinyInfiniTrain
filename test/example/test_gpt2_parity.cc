#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "example/gpt2/net.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {
constexpr uint32_t kMagic = 0x54475032;

template <typename T> T Read(std::ifstream &input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return value;
}

std::filesystem::path FixturePath() {
    if (const char *environment = std::getenv("TINYINFINITRAIN_GPT2_PARITY_FIXTURE")) return environment;
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()
           / "Data/tiny_gpt2_fp16_reference.bin";
}

std::vector<float> TensorValues(const std::shared_ptr<Tensor> &tensor) {
    std::vector<float> values(tensor->NumElements());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = dtype::LoadAsFloat(tensor->DataPtr(), tensor->Dtype(), i);
    }
    return values;
}

void ExpectParity(const std::vector<float> &actual, const std::vector<float> &reference,
                  float max_tolerance, float mean_tolerance) {
    ASSERT_EQ(actual.size(), reference.size());
    float max_error = 0.0f;
    float mean_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]));
        max_error = std::max(max_error, std::abs(actual[i] - reference[i]));
        mean_error += std::abs(actual[i] - reference[i]);
    }
    mean_error /= static_cast<float>(actual.size());
    std::cout << "logits parity: max_abs_error=" << max_error
              << " mean_abs_error=" << mean_error << '\n';
    EXPECT_LE(max_error, max_tolerance);
    EXPECT_LE(mean_error, mean_tolerance);
}
} // namespace

TEST(TinyGPT2ParityTest, PyTorchFP16LogitsMatchBeforeAndAfterSingleAdamStep) {
    const auto fixture = FixturePath();
    if (!std::filesystem::exists(fixture)) {
        GTEST_SKIP() << "Generate the PyTorch fixture with tools/generate_gpt2_reference.py";
    }
    std::ifstream input(fixture, std::ios::binary);
    ASSERT_TRUE(input.is_open());
    ASSERT_EQ(Read<uint32_t>(input), kMagic);
    ASSERT_EQ(Read<uint32_t>(input), 1u);
    const int32_t block = Read<int32_t>(input);
    const int32_t vocab = Read<int32_t>(input);
    const int32_t layers = Read<int32_t>(input);
    const int32_t heads = Read<int32_t>(input);
    const int32_t embd = Read<int32_t>(input);
    const int32_t batch = Read<int32_t>(input);
    const int32_t sequence = Read<int32_t>(input);
    const uint32_t tensor_count = Read<uint32_t>(input);

    GPT2 model(GPT2Config{
        .block_size = block,
        .vocab_size = vocab,
        .n_layer = layers,
        .n_head = heads,
        .n_embd = embd,
        .dtype = DataType::kFLOAT16,
    });
    auto state = model.StateDict();
    ASSERT_EQ(tensor_count, state.size());
    for (uint32_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
        const uint32_t name_length = Read<uint32_t>(input);
        std::string name(name_length, '\0');
        input.read(name.data(), name.size());
        const uint64_t num_elements = Read<uint64_t>(input);
        std::vector<float> values(num_elements);
        input.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float));
        ASSERT_TRUE(state.contains(name)) << name;
        auto &tensor = state.at(name);
        ASSERT_EQ(tensor->NumElements(), num_elements) << name;
        for (size_t i = 0; i < values.size(); ++i) {
            dtype::StoreFromFloat(tensor->DataPtr(), tensor->Dtype(), i, values[i]);
        }
    }

    const size_t token_count = static_cast<size_t>(batch) * sequence;
    auto tokens = std::make_shared<Tensor>(std::vector<int64_t>{batch, sequence}, DataType::kINT64);
    auto targets = std::make_shared<Tensor>(std::vector<int64_t>{batch, sequence}, DataType::kINT64);
    input.read(reinterpret_cast<char *>(tokens->DataPtr()), token_count * sizeof(int64_t));
    input.read(reinterpret_cast<char *>(targets->DataPtr()), token_count * sizeof(int64_t));
    const size_t logit_count = token_count * vocab;
    std::vector<float> reference_before(logit_count);
    std::vector<float> reference_after(logit_count);
    input.read(reinterpret_cast<char *>(reference_before.data()), logit_count * sizeof(float));
    input.read(reinterpret_cast<char *>(reference_after.data()), logit_count * sizeof(float));
    ASSERT_TRUE(input.good() || input.eof());

    auto logits_before = model.Forward({tokens})[0];
    ExpectParity(TensorValues(logits_before), reference_before, 5e-3f, 5e-4f);
    nn::CrossEntropyLoss loss_fn;
    auto loss = loss_fn.Forward({logits_before, targets})[0];
    loss->Backward();
    optimizers::Adam optimizer(model.Parameters(), 1e-3f);
    optimizer.Step();
    auto logits_after = model.Forward({tokens})[0];
    ExpectParity(TensorValues(logits_after), reference_after, 1e-2f, 1e-3f);
}
