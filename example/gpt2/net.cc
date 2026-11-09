#include "example/gpt2/net.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/checkpoint.h"
#include "infini_train/include/autograd/linear_parallel.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/distributed/nccl_utils.h"
#include "infini_train/include/distributed/parallel_state.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

namespace {
constexpr int kRandomSeed = 42;

class GPT2Linear : public nn::Linear {
public:
    GPT2Linear(int64_t in_features, int64_t out_features, bool residual_scale = false,
               bool skip_init = false, infini_train::DataType dtype = infini_train::DataType::kFLOAT32)
        : Linear(in_features, out_features, true, infini_train::Device(), dtype),
          residual_scale_(residual_scale), skip_init_(skip_init) {}

    bool residual_scale() const { return residual_scale_; }
    bool skip_init() const { return skip_init_; }

private:
    const bool residual_scale_ = false;
    const bool skip_init_ = false;
};

} // namespace

std::vector<std::shared_ptr<infini_train::Tensor>>
NewGELU::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto &input = x[0];
    return {0.5 * input
            * (1.0 + nn::function::Tanh(std::sqrt(2.0 / M_PI) * (input + 0.044715 * nn::function::Pow(input, 3.0))))};
}

CausalSelfAttention::CausalSelfAttention(const GPT2Config &config)
    : config_(config), n_head_(config.n_head), n_embd_(config.n_embd) {
    CHECK_EQ(config.n_embd % config.n_head, 0);
    if (config.tp_size > 1) {
        // TP: each rank holds 1/TP of the output channels.
        // c_attn (ColumnParallelLinear): weight shard [3*n_embd/TP, n_embd], full input
        // c_proj (RowParallelLinear):    weight shard [n_embd, n_embd/TP],   sharded input
        CHECK_EQ(config.n_embd % config.tp_size, 0) << "n_embd must be divisible by tp_size";
        CHECK_EQ(config.n_head % config.tp_size, 0) << "n_head must be divisible by tp_size";
        modules_[kCAttnLayerName] = std::make_unique<GPT2Linear>(
            config.n_embd, config.n_embd * 3 / config.tp_size, false, false, config.dtype);
        modules_[kCProjLayerName] = std::make_unique<GPT2Linear>(
            config.n_embd / config.tp_size, config.n_embd, /*residual_scale=*/true, false, config.dtype);
    } else {
        modules_[kCAttnLayerName]
            = std::make_unique<GPT2Linear>(config.n_embd, config.n_embd * 3, false, false, config.dtype);
        modules_[kCProjLayerName]
            = std::make_unique<GPT2Linear>(config.n_embd, config.n_embd, true, false, config.dtype);
    }
    // (1, 1, block_size, block_size)
    bias_ = nn::function::Tril(nn::function::Ones({config_.block_size, config_.block_size}))
                ->View({1, 1, config_.block_size, config_.block_size});
    if (config.dtype != infini_train::DataType::kFLOAT32) bias_ = bias_->Cast(config.dtype);
}

void CausalSelfAttention::To(infini_train::Device device) {
    nn::Module::To(device);
    bias_ = std::make_shared<infini_train::Tensor>(bias_->To(device));
}

namespace {
// Pull a module's weight/(optional) bias tensors out for direct use as
// ColumnParallelLinear/RowParallelLinear::Apply() inputs — those Functions
// take raw (input, weight, bias) tensors, not an nn::Module to Forward() through.
std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
WeightAndBias(infini_train::nn::Module *linear) {
    auto weight = *linear->mutable_parameter(infini_train::nn::Linear::kParamWeightName);
    auto bias = linear->has_parameter(infini_train::nn::Linear::kParamBiasName)
                    ? *linear->mutable_parameter(infini_train::nn::Linear::kParamBiasName)
                    : nullptr;
    return {weight, bias};
}
} // namespace

std::vector<std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    const auto B = x[0]->Dims()[0]; // bs
    const auto T = x[0]->Dims()[1]; // seq_len
    const auto C = x[0]->Dims()[2]; // n_embd
    const int64_t tp_size = config_.tp_size;
    const int64_t n_head_local = n_head_ / tp_size; // heads this TP rank owns
    const int64_t n_embd_local = n_embd_ / tp_size;

    // calculate query, key, values for the heads owned by this TP rank
    // (bs, seq_len, n_embd) -> Linear(n_embd, 3 * n_embd / TP) -> (bs, seq_len, 3 * n_embd / TP)
    // -> Split -> (3, bs, seq_len, n_embd / TP)
    std::shared_ptr<infini_train::Tensor> qkv_shard;
    if (tp_size > 1) {
#ifdef USE_NCCL
        auto [weight, bias] = WeightAndBias(modules_[kCAttnLayerName].get());
        // c_attn stays sharded (gather_output=false): each TP rank computes
        // attention only for its own n_head/TP heads; c_proj's AllReduce below
        // combines the per-rank outputs, so no communication is needed here.
        qkv_shard = std::make_shared<infini_train::autograd::ColumnParallelLinear>(
                        /*gather_output=*/false, /*sequence_parallel=*/false)
                        ->Apply({x[0], weight, bias})[0];
#else
        LOG(FATAL) << "config.tp_size > 1 requires building with USE_NCCL";
#endif
    } else {
        qkv_shard = modules_[kCAttnLayerName]->Forward(x)[0];
    }
    auto qkv = qkv_shard->Split(n_embd_local, 2);
    // (bs, seq_len, n_embd / TP)
    auto q = qkv[0];
    // (bs, seq_len, n_embd / TP)
    auto k = qkv[1];
    // (bs, seq_len, n_embd / TP)
    auto v = qkv[2];
    // (bs, seq_len, n_embd/TP) -> (bs, seq_len, n_head/TP, n_embd/n_head) -> (bs, n_head/TP, seq_len, n_embd/n_head)
    k = k->View({B, T, n_head_local, C / n_head_})->Transpose(1, 2);
    // (bs, seq_len, n_embd/TP) -> (bs, seq_len, n_head/TP, n_embd/n_head) -> (bs, n_head/TP, seq_len, n_embd/n_head)
    q = q->View({B, T, n_head_local, C / n_head_})->Transpose(1, 2);
    // (bs, seq_len, n_embd/TP) -> (bs, seq_len, n_head/TP, n_embd/n_head) -> (bs, n_head/TP, seq_len, n_embd/n_head)
    v = v->View({B, T, n_head_local, C / n_head_})->Transpose(1, 2);

    // manual implementation of attention
    // this materializes the large (T,T) matrix for all the queries and keys

    // q: (bs, n_head, seq_len, n_embd / n_head)
    // k: (bs, n_head, seq_len, n_embd / n_head) -> (bs, n_head, n_embd / n_head, seq_len)
    // q matmul k: (bs, n_head, seq_len, seq_len) -> mul 1.0 / sqrt(n_embd / n_head) -> (bs, n_head, seq_len, seq_len)
    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(*k->Dims().rbegin()));
    // (1, 1, seq_len, seq_len)
    auto mask = bias_->Slice({0, 0, 0, 0}, {1, 1, T, T}, {1, 1, 1, 1});
    // (1, 1, seq_len, seq_len) -> eq 0 -> (1, 1, seq_len, seq_len) -> masked_fill -> (bs, n_head, seq_len, seq_len)
    att = att->MaskedFill(mask == 0, -std::numeric_limits<float>::infinity());
    // (bs, n_head, seq_len, seq_len)
    att = nn::function::Softmax(att, -1);
    // (bs, n_head/TP, seq_len, n_embd / n_head)
    auto y = att->Matmul(v);
    // (bs, n_head/TP, seq_len, n_embd/n_head) -> Transpose(1, 2) -> (bs, seq_len, n_head/TP, n_embd/n_head)
    // -> (bs, seq_len, n_embd/TP)
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, n_embd_local});
    // output projection: c_proj is RowParallel — local matmul on the n_embd/TP
    // shard, then AllReduce sums each rank's partial output back to (bs,T,n_embd).
    if (tp_size > 1) {
#ifdef USE_NCCL
        auto [weight, bias] = WeightAndBias(modules_[kCProjLayerName].get());
        y = std::make_shared<infini_train::autograd::RowParallelLinear>(/*sequence_parallel=*/false)
                ->Apply({y, weight, bias})[0];
#else
        LOG(FATAL) << "config.tp_size > 1 requires building with USE_NCCL";
#endif
    } else {
        // (bs, seq_len, n_embd) -> Linear(n_embd, n_embd) -> (bs, seq_len, n_embd)
        y = modules_[kCProjLayerName]->Forward({y})[0];
    }
    // (bs, seq_len, n_embd)
    return {y};
}

MLP::MLP(const GPT2Config &config) : tp_size_(config.tp_size) {
    if (config.tp_size > 1) {
        // c_fc  (ColumnParallelLinear): weight shard [4*n_embd/TP, n_embd], full input
        // c_proj (RowParallelLinear):   weight shard [n_embd, 4*n_embd/TP], sharded input
        modules_[kCFclayerName] = std::make_unique<GPT2Linear>(
            config.n_embd, config.n_embd * 4 / config.tp_size, false, false, config.dtype);
        modules_[kGeluLayerName] = std::make_unique<NewGELU>();
        modules_[kCProjLayerName] = std::make_unique<GPT2Linear>(
            config.n_embd * 4 / config.tp_size, config.n_embd, /*residual_scale=*/true, false, config.dtype);
    } else {
        modules_[kCFclayerName]
            = std::make_unique<GPT2Linear>(config.n_embd, config.n_embd * 4, false, false, config.dtype);
        modules_[kGeluLayerName] = std::make_unique<NewGELU>();
        modules_[kCProjLayerName]
            = std::make_unique<GPT2Linear>(config.n_embd * 4, config.n_embd, true, false, config.dtype);
    }
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MLP::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // c_fc stays sharded (gather_output=false): GELU is elementwise, so each TP
    // rank can apply it to its own 4*n_embd/TP shard with no communication.
    // (bs, seq_len, n_embd) -> Linear(n_embd, 4*n_embd/TP) -> (bs, seq_len, 4*n_embd/TP)
    std::shared_ptr<infini_train::Tensor> x1;
    if (tp_size_ > 1) {
#ifdef USE_NCCL
        auto [weight, bias] = WeightAndBias(modules_[kCFclayerName].get());
        x1 = std::make_shared<infini_train::autograd::ColumnParallelLinear>(
                 /*gather_output=*/false, /*sequence_parallel=*/false)
                 ->Apply({x[0], weight, bias})[0];
#else
        LOG(FATAL) << "config.tp_size > 1 requires building with USE_NCCL";
#endif
    } else {
        x1 = modules_[kCFclayerName]->Forward(x)[0];
    }
    // (bs, seq_len, 4*n_embd/TP) -> GELU -> (bs, seq_len, 4*n_embd/TP)
    auto x2 = modules_[kGeluLayerName]->Forward({x1})[0];
    // c_proj is RowParallel: local matmul on the shard, then AllReduce sums
    // each rank's partial output back to the full (bs, seq_len, n_embd).
    std::shared_ptr<infini_train::Tensor> x3;
    if (tp_size_ > 1) {
#ifdef USE_NCCL
        auto [weight, bias] = WeightAndBias(modules_[kCProjLayerName].get());
        x3 = std::make_shared<infini_train::autograd::RowParallelLinear>(/*sequence_parallel=*/false)
                 ->Apply({x2, weight, bias})[0];
#else
        LOG(FATAL) << "config.tp_size > 1 requires building with USE_NCCL";
#endif
    } else {
        // (bs, seq_len, 4 * n_embd) -> Linear(4 * n_embd, n_embd) -> (bs, seq_len, n_embd)
        x3 = modules_[kCProjLayerName]->Forward({x2})[0];
    }
    // (bs, seq_len, n_embd)
    return {x3};
}

Block::Block(const GPT2Config &config) {
    modules_[kLn1LayerName]
        = std::make_unique<nn::LayerNorm>(std::vector<int64_t>{config.n_embd}, 1e-5f,
                                          infini_train::Device(), config.dtype);
    modules_[kAttnLayerName] = std::make_unique<CausalSelfAttention>(config);
    modules_[kLn2LayerName]
        = std::make_unique<nn::LayerNorm>(std::vector<int64_t>{config.n_embd}, 1e-5f,
                                          infini_train::Device(), config.dtype);
    modules_[kMlpLayerName] = std::make_unique<MLP>(config);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
Block::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> attention -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x1 = x[0] + modules_[kAttnLayerName]->Forward(modules_[kLn1LayerName]->Forward(x))[0];
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x2 = x1 + modules_[kMlpLayerName]->Forward(modules_[kLn2LayerName]->Forward({x1}))[0];
    // (bs, seq_len, n_embd)
    return {x2};
}

GPT2::GPT2(const GPT2Config &config) : config_(config) {
    // PP stage layer range: each stage owns n_layer/pp_size consecutive blocks.
    CHECK_EQ(config.n_layer % config.pp_size, 0) << "n_layer must be divisible by pp_size";
    const int64_t layers_per_stage = config.n_layer / config.pp_size;
#ifdef USE_NCCL
    if (config.pp_size > 1) {
        // GetPPRank() silently returns 0 if InitParallelGroups() hasn't run yet
        // (see parallel_state.cc), which would otherwise make every process
        // silently build itself as stage 0 with no diagnostic.
        CHECK(infini_train::distributed::GetPPGroup())
            << "GPT2Config::pp_size > 1 requires distributed::InitParallelGroups() "
            << "to have been called before constructing GPT2 (this stage's real "
            << "pp_rank can't be determined otherwise).";
    }
    // real pipeline rank of this process within its PP group.
    const int64_t pp_rank = infini_train::distributed::GetPPRank();
#else
    // No distributed backend compiled in: every process is implicitly stage 0
    // of a single-stage pipeline (matches pp_size's default value of 1).
    const int64_t pp_rank = 0;
#endif
    CHECK_LT(pp_rank, config.pp_size);
    stage_layer_start_ = pp_rank * layers_per_stage;
    stage_layer_end_ = (pp_rank + 1) * layers_per_stage;

    {
        std::unordered_map<std::string, std::unique_ptr<nn::Module>> transformer;

        // First PP stage: owns token and position embeddings.
        if (config.pre_process) {
            transformer[kWTELayerName]
                = std::make_unique<nn::Embedding>(config.vocab_size, config.n_embd,
                                                   infini_train::Device(), config.dtype);
            transformer[kWPELayerName]
                = std::make_unique<nn::Embedding>(config.block_size, config.n_embd,
                                                   infini_train::Device(), config.dtype);
        }

        // Only construct the blocks assigned to this PP stage.
        {
            std::vector<std::unique_ptr<nn::Module>> h;
            for (int64_t i = stage_layer_start_; i < stage_layer_end_; i++) {
                h.push_back(std::make_unique<Block>(config));
            }
            transformer[kHLayerName] = std::make_unique<nn::Sequential>(std::move(h));
        }

        // Last PP stage: owns final LayerNorm.
        if (config.post_process) {
            transformer[kLnFLayerName]
                = std::make_unique<nn::LayerNorm>(std::vector<int64_t>{config.n_embd}, 1e-5f,
                                                  infini_train::Device(), config.dtype);
        }

        modules_[kTransformerLayerName] = std::make_unique<nn::ModuleDict>(std::move(transformer));
    }

    if (config.post_process) {
        // don't init this one, we will tie weights
        modules_[kLMHeadLayerName]
            = std::make_unique<GPT2Linear>(config.n_embd, config.vocab_size, false, true, config.dtype);
        // https://paperswithcode.com/method/weight-tying
        // Same-stage tying aliases one Tensor. Split PP stages are synchronized
        // by SynchronizeTiedEmbeddingWeights/Gradients (Megatron's
        // share_embeddings_and_output_weights contract).
        if (config.pre_process) {
            *mutable_module(kTransformerLayerName)
                 ->mutable_module(kWTELayerName)
                 ->mutable_parameter(GPT2Linear::kParamWeightName)
                = module(kLMHeadLayerName).parameter(GPT2Linear::kParamWeightName);
        }
    }

    // Initialize replicated embeddings identically on every TP rank, while
    // giving each logical Linear shard its own deterministic stream. DP rank
    // never enters either seed, so data-parallel replicas remain identical.
#ifdef USE_NCCL
    const int64_t tp_rank = infini_train::distributed::GetTPRank();
#else
    const int64_t tp_rank = 0;
#endif
    int64_t linear_index = 0;
    int64_t embedding_index = 0;
    Apply([&](Module *module) {
        if (module->type() == nn::Linear::kType) {
            auto *linear = static_cast<GPT2Linear *>(module);
            const float std = linear->residual_scale() ? 0.02 / std::sqrt(2 * config_.n_layer) : 0.02;
            if (!linear->skip_init()) {
                std::mt19937 module_gen(kRandomSeed + static_cast<int>(pp_rank) * 10000
                                        + static_cast<int>(tp_rank) * 100000 + static_cast<int>(linear_index++));
                nn::init::Normal(*linear->mutable_parameter(GPT2Linear::kParamWeightName), 0.0f, std, module_gen);
            }
            if (linear->has_parameter(nn::Linear::kParamBiasName)) {
                nn::init::Zeros(*linear->mutable_parameter(nn::Linear::kParamBiasName));
            }
        } else if (module->type() == nn::Embedding::kType) {
            std::mt19937 module_gen(kRandomSeed + 500000 + static_cast<int>(embedding_index++));
            nn::init::Normal(*module->mutable_parameter(nn::Embedding::kParamWeightName), 0.0f, 0.02, module_gen);
        }
    });
}

namespace {
void CopyTensorValue(const std::shared_ptr<infini_train::Tensor> &destination,
                     const std::shared_ptr<infini_train::Tensor> &source) {
    CHECK(destination->Dims() == source->Dims());
    CHECK(destination->Dtype() == source->Dtype());
    destination->Fill<float>(0.0f);
    infini_train::Dispatcher::Instance()
        .GetKernel({destination->GetDevice().Type(), "AccumulateGrad"})
        .Call<void>(source, 1.0f, destination);
}
} // namespace

void GPT2::To(infini_train::Device device) {
    nn::Module::To(device);
    SynchronizeTiedEmbeddingWeights();
}

void GPT2::SynchronizeTiedEmbeddingWeights() {
    if (config_.pp_size == 1) return;
#ifdef USE_NCCL
    auto *pp_group = infini_train::distributed::GetPPCollectiveGroup();
    CHECK(pp_group);
    CHECK_EQ(pp_group->WorldSize(), config_.pp_size);
    CHECK(device_.IsCUDA()) << "NCCL tied-weight synchronization requires CUDA model parameters";
    auto shared = std::make_shared<infini_train::Tensor>(
        std::vector<int64_t>{config_.vocab_size, config_.n_embd}, config_.dtype, device_);
    shared->Fill<float>(0.0f);
    if (config_.pre_process) {
        const auto &weight = mutable_module(kTransformerLayerName)
                                 ->mutable_module(kWTELayerName)
                                 ->parameter(nn::Embedding::kParamWeightName);
        CopyTensorValue(shared, weight);
    }
    pp_group->AllReduce(shared->DataPtr(), shared->NumElements(),
                        infini_train::distributed::ToNcclDataType(shared->Dtype()), ncclSum);
    if (config_.post_process) {
        auto weight = *mutable_module(kLMHeadLayerName)->mutable_parameter(nn::Linear::kParamWeightName);
        CopyTensorValue(weight, shared);
    }
#else
    LOG(FATAL) << "Pipeline tied-weight synchronization requires USE_NCCL";
#endif
}

void GPT2::SynchronizeTiedEmbeddingGradients() {
    if (config_.pp_size == 1) return;
#ifdef USE_NCCL
    auto *pp_group = infini_train::distributed::GetPPCollectiveGroup();
    CHECK(pp_group);
    auto shared_grad = std::make_shared<infini_train::Tensor>(
        std::vector<int64_t>{config_.vocab_size, config_.n_embd}, config_.dtype, device_);
    shared_grad->Fill<float>(0.0f);
    std::shared_ptr<infini_train::Tensor> local_grad;
    if (config_.pre_process) {
        local_grad = mutable_module(kTransformerLayerName)
                         ->mutable_module(kWTELayerName)
                         ->parameter(nn::Embedding::kParamWeightName)
                         ->grad();
    } else if (config_.post_process) {
        local_grad = mutable_module(kLMHeadLayerName)->parameter(nn::Linear::kParamWeightName)->grad();
    }
    if (local_grad) CopyTensorValue(shared_grad, local_grad);
    pp_group->AllReduce(shared_grad->DataPtr(), shared_grad->NumElements(),
                        infini_train::distributed::ToNcclDataType(shared_grad->Dtype()), ncclSum);
    if (local_grad) CopyTensorValue(local_grad, shared_grad);
#else
    LOG(FATAL) << "Pipeline tied-gradient synchronization requires USE_NCCL";
#endif
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto &transformer = modules_[kTransformerLayerName];

    std::shared_ptr<infini_train::Tensor> hidden;

    if (config_.pre_process) {
        // First PP stage: compute token + position embeddings.
        // (bs, seq_len)
        auto &idx = x[0];
        const auto device = idx->GetDevice();
        const auto t = idx->Dims()[1]; // seq_len
        CHECK_LE(t, config_.block_size) << "Cannot forward sequence of length " << t
                                        << ", block size is only " << config_.block_size;
        // (seq_len)
        auto pos = nn::init::Arange(0, t, infini_train::DataType::kINT64, device);
        // (bs, seq_len) -> Embedding -> (bs, seq_len, n_embd)
        auto tok_emb = transformer->mutable_module(kWTELayerName)->Forward({idx})[0];
        // (seq_len) -> Embedding -> (seq_len, n_embd)
        auto pos_emb = transformer->mutable_module(kWPELayerName)->Forward({pos})[0];
        // (bs, seq_len, n_embd)
        hidden = tok_emb + pos_emb;
    } else {
        // Middle / last PP stage: input is the activation received via P2P from previous stage.
        // The pipeline schedule's P2P Recv deposits the tensor here before Forward is called.
        hidden = x[0];
    }

    // Run only the transformer blocks owned by this PP stage.
    // (bs, seq_len, n_embd) -> Block x (stage_end - stage_start) -> (bs, seq_len, n_embd)
    std::vector<std::shared_ptr<infini_train::Tensor>> h_out;
    if (config_.activation_recomputation) {
        const int64_t local_layers = stage_layer_end_ - stage_layer_start_;
        auto *blocks = transformer->mutable_module(kHLayerName);
        for (int64_t local_idx = 0; local_idx < local_layers; ++local_idx) {
            hidden = infini_train::autograd::CheckpointModule(
                blocks->mutable_module(std::to_string(local_idx)), hidden);
        }
        h_out = {hidden};
    } else {
        h_out = transformer->mutable_module(kHLayerName)->Forward({hidden});
    }

    if (config_.post_process) {
        // Last PP stage: apply final LayerNorm and language model head.
        // (bs, seq_len, n_embd) -> LayerNorm -> (bs, seq_len, n_embd)
        auto x_norm = transformer->mutable_module(kLnFLayerName)->Forward(h_out);
        // (bs, seq_len, n_embd) -> Linear(n_embd, vocab_size) -> (bs, seq_len, vocab_size)
        return modules_[kLMHeadLayerName]->Forward(x_norm);
    }

    // Middle PP stages return the hidden state; the pipeline schedule sends it to the next stage.
    // The pipeline schedule sends h_out[0] to the next stage after this returns.
    return h_out;
}

std::unique_ptr<GPT2> GPT2::FromPretrained(ModelType model_type) {
    LOG(FATAL) << "Not implemented yet";
    return nullptr;
}

namespace {
std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

constexpr int32_t kHeaderMagic = 20240326;
constexpr int32_t kHeaderFP32Version = 3;
} // namespace

std::unique_ptr<GPT2> GPT2::FromLLMC(const std::string &filepath) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(256 * sizeof(int32_t), &ifs);

    const auto magic = BytesToType<uint32_t>(header, 0);
    CHECK_EQ(magic, kHeaderMagic);
    const auto version = BytesToType<uint32_t>(header, 4);
    CHECK_EQ(version, kHeaderFP32Version);

    const auto block_size = BytesToType<uint32_t>(header, 8);
    const auto vocab_size = BytesToType<uint32_t>(header, 12);
    const auto n_layer = BytesToType<uint32_t>(header, 16);
    const auto n_head = BytesToType<uint32_t>(header, 20);
    const auto n_embd = BytesToType<uint32_t>(header, 24);
    auto gpt2 = std::make_unique<GPT2>(GPT2Config{
        .block_size = block_size, .vocab_size = vocab_size, .n_layer = n_layer, .n_head = n_head, .n_embd = n_embd});

    const auto padded_vocab_size = BytesToType<uint32_t>(header, 28);
    LOG(ERROR) << "magic: " << magic << " version: " << version << " block_size: " << block_size
               << " vocab_size: " << vocab_size << " n_layer: " << n_layer << " n_head: " << n_head
               << " n_embd: " << n_embd << " padded_vocab_size: " << padded_vocab_size;

    auto state_dict = gpt2->StateDict();
    // transformer.wte.weight
    // (padded_vocab_size, n_embd) -> un_pad -> (vocab_size, n_embd)
    auto &transformer_wte_weight = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kWTELayerName,
                                                          nn::Embedding::kParamWeightName)];
    ifs.read(reinterpret_cast<char *>(transformer_wte_weight->DataPtr()), transformer_wte_weight->SizeInBytes());
    ifs.ignore((padded_vocab_size - vocab_size) * n_embd * sizeof(float));
    // transformer.wpe.weight
    auto &transformer_wpe_weight = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kWPELayerName,
                                                          nn::Embedding::kParamWeightName)];
    ifs.read(reinterpret_cast<char *>(transformer_wpe_weight->DataPtr()), transformer_wpe_weight->SizeInBytes());
    // transformer.h.{i}.ln_1.weight
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                     std::to_string(idx), Block::kLn1LayerName, nn::LayerNorm::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.ln_1.bias
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                     std::to_string(idx), Block::kLn1LayerName, nn::LayerNorm::kParamBiasName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.attn.c_attn.weight
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                              std::to_string(idx), Block::kAttnLayerName,
                                              CausalSelfAttention::kCAttnLayerName, GPT2Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.attn.c_attn.bias
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                              std::to_string(idx), Block::kAttnLayerName,
                                              CausalSelfAttention::kCAttnLayerName, GPT2Linear::kParamBiasName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.attn.c_proj.weight
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                              std::to_string(idx), Block::kAttnLayerName,
                                              CausalSelfAttention::kCProjLayerName, GPT2Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.attn.c_proj.bias
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                              std::to_string(idx), Block::kAttnLayerName,
                                              CausalSelfAttention::kCProjLayerName, GPT2Linear::kParamBiasName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.ln_2.weight
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                     std::to_string(idx), Block::kLn2LayerName, nn::LayerNorm::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.ln_2.bias
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                     std::to_string(idx), Block::kLn2LayerName, nn::LayerNorm::kParamBiasName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.mlp.c_fc.weight
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                              std::to_string(idx), Block::kMlpLayerName, MLP::kCFclayerName,
                                              GPT2Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.mlp.c_fc.bias
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                              std::to_string(idx), Block::kMlpLayerName, MLP::kCFclayerName,
                                              GPT2Linear::kParamBiasName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.mlp.c_proj.weight
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                              std::to_string(idx), Block::kMlpLayerName, MLP::kCProjLayerName,
                                              GPT2Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.h.{i}.mlp.c_proj.bias
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kHLayerName,
                                              std::to_string(idx), Block::kMlpLayerName, MLP::kCProjLayerName,
                                              GPT2Linear::kParamBiasName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }
    // transformer.ln_f.weight
    auto &transformer_ln_f_weight = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kLnFLayerName,
                                                           nn::LayerNorm::kParamWeightName)];
    ifs.read(reinterpret_cast<char *>(transformer_ln_f_weight->DataPtr()), transformer_ln_f_weight->SizeInBytes());
    // transformer.ln_f.bias
    auto &transformer_ln_f_bias = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName, GPT2::kLnFLayerName,
                                                         nn::LayerNorm::kParamBiasName)];
    ifs.read(reinterpret_cast<char *>(transformer_ln_f_bias->DataPtr()), transformer_ln_f_bias->SizeInBytes());

    return gpt2;
}
