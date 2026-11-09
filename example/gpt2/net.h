#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

struct GPT2Config {
    int64_t block_size = 1024;
    int64_t vocab_size = 50257;
    int64_t n_layer = 12;
    int64_t n_head = 12;
    int64_t n_embd = 768;
    infini_train::DataType dtype = infini_train::DataType::kFLOAT32;

    // Recompute each local Transformer block during backward instead of
    // retaining its attention/MLP intermediates from the original forward.
    bool activation_recomputation = false;

    // Tensor parallelism: split attention heads and MLP hidden dim across tp_size ranks.
    // TP=1 (default) disables all collective ops and is identical to the original code.
    // Megatron-LM analogue: TransformerConfig.tensor_model_parallel_size
    int tp_size = 1;

    // Pipeline parallelism: divide the n_layer transformer blocks across pp_size stages.
    // PP=1 (default) keeps all layers on one process.
    // pre_process / post_process mirror Megatron's GPTModel flags:
    //   pre_process=true  → this stage owns Embedding (wte, wpe)
    //   post_process=true → this stage owns ln_f and lm_head
    int pp_size = 1;
    bool pre_process = true;
    bool post_process = true;
};

class NewGELU : public infini_train::nn::Module {
public:
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class CausalSelfAttention : public infini_train::nn::Module {
public:
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit CausalSelfAttention(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    void To(infini_train::Device device) override;

private:
    GPT2Config config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;
    std::shared_ptr<infini_train::Tensor> bias_ = nullptr;
};

class MLP : public infini_train::nn::Module {
public:
    static constexpr char kCFclayerName[] = "c_fc";
    static constexpr char kGeluLayerName[] = "gelu";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit MLP(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    int tp_size_ = 1;
};

class Block : public infini_train::nn::Module {
public:
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit Block(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class GPT2 : public infini_train::nn::Module {
public:
    static constexpr char kWTELayerName[] = "wte";
    static constexpr char kWPELayerName[] = "wpe";
    static constexpr char kHLayerName[] = "h";
    static constexpr char kLnFLayerName[] = "ln_f";
    static constexpr char kTransformerLayerName[] = "transformer";
    static constexpr char kLMHeadLayerName[] = "lm_head";

    enum class ModelType : int8_t {
        kGPT2,
        kGPT2Medium,
        kGPT2Large,
        kGPT2XL,
    };

    explicit GPT2(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    void To(infini_train::Device device) override;

    // PP first/last stages own separate physical copies of the tied token
    // embedding / lm_head weight. These collectives establish identical
    // initial values and sum their gradients before an optimizer update.
    void SynchronizeTiedEmbeddingWeights();
    void SynchronizeTiedEmbeddingGradients();

    const GPT2Config &Config() const { return config_; }

    static std::unique_ptr<GPT2> FromPretrained(ModelType model_type);
    static std::unique_ptr<GPT2> FromLLMC(const std::string &filepath);

private:
    GPT2Config config_;
    // PP layer range owned by this stage: [stage_layer_start_, stage_layer_end_).
    // Both equal 0..n_layer when pp_size == 1.
    int64_t stage_layer_start_ = 0;
    int64_t stage_layer_end_ = 0;
};
