#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "example/gpt2/net.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {
void ExpectTensorNear(const std::shared_ptr<Tensor> &actual,
                      const std::shared_ptr<Tensor> &expected, float tolerance) {
    ASSERT_EQ(actual->Dims(), expected->Dims());
    ASSERT_EQ(actual->Dtype(), DataType::kFLOAT32);
    ASSERT_EQ(expected->Dtype(), DataType::kFLOAT32);
    const auto *actual_data = static_cast<const float *>(actual->DataPtr());
    const auto *expected_data = static_cast<const float *>(expected->DataPtr());
    for (size_t i = 0; i < actual->NumElements(); ++i) {
        EXPECT_NEAR(actual_data[i], expected_data[i], tolerance) << "element " << i;
    }
}

std::filesystem::path WriteTinyDatasetBinary() {
    const auto path = std::filesystem::temp_directory_path() / "tinyinfinitrain_gpt2_train.bin";
    std::array<uint8_t, 1024> header{};
    const uint32_t magic = 20240520;
    const uint32_t version = 1;
    const uint32_t num_tokens = 32;
    std::memcpy(header.data(), &magic, sizeof(magic));
    std::memcpy(header.data() + 4, &version, sizeof(version));
    std::memcpy(header.data() + 8, &num_tokens, sizeof(num_tokens));
    std::vector<uint16_t> tokens(num_tokens);
    for (uint32_t i = 0; i < num_tokens; ++i) tokens[i] = i % 16;
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(header.data()), header.size());
    output.write(reinterpret_cast<const char *>(tokens.data()), tokens.size() * sizeof(uint16_t));
    return path;
}

std::filesystem::path WriteTinyTokenizerBinary() {
    const auto path = std::filesystem::temp_directory_path() / "tinyinfinitrain_gpt2_tokenizer.bin";
    std::array<uint8_t, 1024> header{};
    const uint32_t magic = 20240328;
    const uint32_t version = 1;
    const uint32_t vocab_size = 16;
    std::memcpy(header.data(), &magic, sizeof(magic));
    std::memcpy(header.data() + 4, &version, sizeof(version));
    std::memcpy(header.data() + 8, &vocab_size, sizeof(vocab_size));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(header.data()), header.size());
    for (uint32_t token = 0; token < vocab_size; ++token) {
        const std::string text = std::to_string(token);
        const uint8_t length = static_cast<uint8_t>(text.size());
        output.write(reinterpret_cast<const char *>(&length), sizeof(length));
        output.write(text.data(), text.size());
    }
    return path;
}
} // namespace

TEST(TinyGPT2Test, FP16SingleStepForwardBackwardAdam) {
    GPT2 model(GPT2Config{
        .block_size = 4,
        .vocab_size = 16,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 8,
        .dtype = DataType::kFLOAT16,
    });
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kINT64);
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kINT64);
    auto *input_data = static_cast<int64_t *>(input->DataPtr());
    auto *target_data = static_cast<int64_t *>(target->DataPtr());
    for (int64_t i = 0; i < 8; ++i) {
        input_data[i] = i % 16;
        target_data[i] = (i + 1) % 16;
    }

    auto logits = model.Forward({input})[0];
    ASSERT_EQ(logits->Dims(), std::vector<int64_t>({2, 4, 16}));
    ASSERT_EQ(logits->Dtype(), DataType::kFLOAT16);
    nn::CrossEntropyLoss loss_fn;
    auto loss = loss_fn.Forward({logits, target})[0];
    ASSERT_EQ(loss->Dtype(), DataType::kFLOAT32);
    loss->Backward();

    auto params = model.Parameters();
    ASSERT_FALSE(params.empty());
    ASSERT_NE(params.front()->grad(), nullptr);
    optimizers::Adam optimizer(params, 1e-3f);
    optimizer.Step();
}

TEST(TinyGPT2Test, BF16CheckpointedTrainingUsesFP32MasterWeights) {
    GPT2 model(GPT2Config{
        .block_size = 4,
        .vocab_size = 16,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 8,
        .dtype = DataType::kBFLOAT16,
        .activation_recomputation = true,
    });
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kINT64);
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kINT64);
    auto *input_data = static_cast<int64_t *>(input->DataPtr());
    auto *target_data = static_cast<int64_t *>(target->DataPtr());
    for (int64_t i = 0; i < 8; ++i) {
        input_data[i] = i % 16;
        target_data[i] = (i + 1) % 16;
    }

    nn::CrossEntropyLoss loss_fn;
    optimizers::Adam optimizer(model.Parameters(), 1e-2f);
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    for (int step = 0; step < 12; ++step) {
        optimizer.ZeroGrad();
        auto logits = model.Forward({input})[0];
        ASSERT_EQ(logits->Dtype(), DataType::kBFLOAT16);
        auto loss = loss_fn.Forward({logits, target})[0];
        const float loss_value = static_cast<const float *>(loss->DataPtr())[0];
        ASSERT_TRUE(std::isfinite(loss_value));
        if (step == 0) initial_loss = loss_value;
        final_loss = loss_value;
        loss->Backward();
        optimizer.Step();
    }
    EXPECT_LT(final_loss, initial_loss);
}

TEST(TinyGPT2Test, ActivationRecomputationMatchesFullGraphLogitsAndGradients) {
    GPT2Config reference_config{
        .block_size = 4,
        .vocab_size = 16,
        .n_layer = 2,
        .n_head = 2,
        .n_embd = 8,
        .dtype = DataType::kFLOAT32,
        .activation_recomputation = false,
    };
    auto checkpoint_config = reference_config;
    checkpoint_config.activation_recomputation = true;
    GPT2 reference(reference_config);
    GPT2 checkpointed(checkpoint_config);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kINT64);
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kINT64);
    auto *input_data = static_cast<int64_t *>(input->DataPtr());
    auto *target_data = static_cast<int64_t *>(target->DataPtr());
    for (int64_t i = 0; i < 8; ++i) {
        input_data[i] = i % 16;
        target_data[i] = (i + 1) % 16;
    }

    auto reference_logits = reference.Forward({input})[0];
    auto checkpoint_logits = checkpointed.Forward({input})[0];
    ExpectTensorNear(checkpoint_logits, reference_logits, 1e-6f);

    nn::CrossEntropyLoss loss_fn;
    loss_fn.Forward({reference_logits, target})[0]->Backward();
    loss_fn.Forward({checkpoint_logits, target})[0]->Backward();

    const auto reference_state = reference.StateDict();
    const auto checkpoint_state = checkpointed.StateDict();
    ASSERT_EQ(reference_state.size(), checkpoint_state.size());
    for (const auto &[name, reference_param] : reference_state) {
        const auto it = checkpoint_state.find(name);
        ASSERT_NE(it, checkpoint_state.end()) << name;
        ASSERT_TRUE(reference_param->grad()) << name;
        ASSERT_TRUE(it->second->grad()) << name;
        ExpectTensorNear(it->second->grad(), reference_param->grad(), 2e-5f);
    }
}

TEST(TinyGPT2Test, BinaryDataPipeAndTokenizerRunMultipleEpochs) {
    const auto dataset_path = WriteTinyDatasetBinary();
    const auto tokenizer_path = WriteTinyTokenizerBinary();
    auto dataset = std::make_shared<TinyShakespeareDataset>(dataset_path.string(), 4);
    DataLoader loader(dataset, 2);
    Tokenizer tokenizer(tokenizer_path.string());
    EXPECT_EQ(tokenizer.Decode(7), "7");

    GPT2 model(GPT2Config{
        .block_size = 4,
        .vocab_size = 16,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 8,
        .dtype = DataType::kFLOAT16,
        .activation_recomputation = true,
    });
    nn::CrossEntropyLoss loss_fn;
    optimizers::Adam optimizer(model.Parameters(), 1e-3f);
    const auto tracked_parameter = model.StateDict().at("transformer.h.0.attn.c_attn.weight");
    std::vector<uint8_t> parameter_before(tracked_parameter->SizeInBytes());
    std::memcpy(parameter_before.data(), tracked_parameter->DataPtr(), parameter_before.size());
    std::array<float, 3> epoch_loss_sums{};
    std::array<int, 3> epoch_steps{};
    int steps = 0;
    for (int epoch = 0; epoch < 3; ++epoch) {
        for (auto iterator = loader.begin(); iterator != loader.end(); ++iterator) {
            auto [input, target] = *iterator;
            optimizer.ZeroGrad();
            auto logits = model.Forward({input})[0];
            auto loss = loss_fn.Forward({logits, target})[0];
            const float loss_value = static_cast<const float *>(loss->DataPtr())[0];
            EXPECT_TRUE(std::isfinite(loss_value));
            epoch_loss_sums[epoch] += loss_value;
            ++epoch_steps[epoch];
            loss->Backward();
            optimizer.Step();
            ++steps;
        }
    }
    EXPECT_EQ(steps, 12);
    EXPECT_LT(epoch_loss_sums[2] / epoch_steps[2], epoch_loss_sums[0] / epoch_steps[0]);
    std::vector<uint8_t> parameter_after(tracked_parameter->SizeInBytes());
    std::memcpy(parameter_after.data(), tracked_parameter->DataPtr(), parameter_after.size());
    EXPECT_NE(parameter_after, parameter_before);
    std::filesystem::remove(dataset_path);
    std::filesystem::remove(tokenizer_path);
}
