#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train {

// ZeRO optimizer memory stages (Rajbhandari et al., SC 2020).
// Megatron-LM analogue: megatron/core/optimizer/distrib_optimizer.py
//
// Stage 0 (None):  baseline Adam — m, v, and params fully replicated on each DP rank.
// Stage 1 (ZeRO-1): shard optimizer states (m, v) across DP group.
//                    Memory: m+v reduced by 1/DP per rank.
// Stage 2 (ZeRO-2): additionally shard gradients across DP group.
//                    Memory: grad additionally reduced by 1/DP per rank.
// Stage 3 (ZeRO-3): additionally persist only a 1/DP parameter shard between
//                    iterations. MaterializeParameters() AllGathers the full
//                    tensors in place before forward; Step() updates the local
//                    shard and releases full parameter/gradient storage.
//
// Collective sequence (per Step() call, ZeRO-1/2):
//   ReduceScatter(grad)  → each rank gets its grad shard, averaged (ncclAvg)
//                          across DP ranks — NOT summed, matching standard
//                          DDP semantics (each rank's grad came from a
//                          different local mini-batch of the same batch).
//   AdamAccumulateGrad(grad_shard, param_shard, m_shard, v_shard)
//   AllGather(param_shard) → restore full param for next forward
// ZeRO-3 instead AllGathers in MaterializeParameters() immediately before the
// next forward, so full parameter storage is absent between iterations.
enum class ZeroStage : int8_t {
    kNone  = 0,
    kZero1 = 1,
    kZero2 = 2,
    kZero3 = 3,
};
class Optimizer {
public:
    explicit Optimizer(const std::vector<std::shared_ptr<Tensor>> &params);

    void ZeroGrad();

    virtual void Step() = 0;

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
};

namespace optimizers {
class SGD : public Optimizer {
public:
    SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate);

    void Step() override;

private:
    const float learning_rate_ = 0.0;
};

class Adam : public Optimizer {
public:
    Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate = 1e-3,
         float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8,
         ZeroStage zero_stage = ZeroStage::kNone);

    void Step() override;

    // ZeRO-3 lifecycle. Call MaterializeParameters() before each forward;
    // Step() automatically returns the model to its partitioned state. Calling
    // PartitionParameters() after construction removes the initial full copy.
    void MaterializeParameters();
    void PartitionParameters();
    bool ParametersMaterialized() const { return parameters_materialized_; }
    size_t FullParameterBytes() const;
    size_t LocalParameterShardBytes() const;

private:
    int64_t t_;
    const float learning_rate_;
    const float beta1_;
    const float beta2_;
    const float eps_;
    ZeroStage zero_stage_;

    // Full optimizer states (ZeroStage::kNone — original path).
    std::vector<std::shared_ptr<Tensor>> m_;
    std::vector<std::shared_ptr<Tensor>> v_;

    // Sharded optimizer states (ZeRO-1/2).
    // Shard size = ceil(param.numel() / dp_world_size) per element.
    // Megatron-LM analogue: DistributedOptimizer.model_float16_groups / model_fp32_groups
    std::vector<std::shared_ptr<Tensor>> m_shard_;
    std::vector<std::shared_ptr<Tensor>> v_shard_;

    // Persistent local parameter partitions for ZeRO-3; null at other stages.
    std::vector<std::shared_ptr<Tensor>> param_shard_;
    std::vector<std::shared_ptr<Tensor>> param_master_shard_;
    bool parameters_materialized_ = true;

    // FP32 master parameters for FP8/FP16/BF16 mixed-precision training.
    // Updated in FP32, then downcast to the model dtype for forward/backward.
    // Megatron-LM analogue: DistributedOptimizer._copy_model_params_to_main_params()
    // Entries are nullptr for native FP32 parameters and current ZeRO paths.
    std::vector<std::shared_ptr<Tensor>> params_fp32_;
};
} // namespace optimizers
} // namespace infini_train
