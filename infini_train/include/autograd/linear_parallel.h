#pragma once
#ifdef USE_NCCL

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

// Tensor-parallel linear layers following Megatron-LM's column/row split strategy.
// Reference: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language
//   Models Using Model Parallelism" (2019), Section 3.
//
// Megatron-LM analogue:
//   ColumnParallelLinear → megatron/core/tensor_parallel/layers.py:ColumnParallelLinear
//   RowParallelLinear    → megatron/core/tensor_parallel/layers.py:RowParallelLinear
// The collective insert points follow megatron/core/tensor_parallel/mappings.py:
//   _GatherFromModelParallelRegion  (AllGather  in ColPar Forward)
//   _ReduceFromModelParallelRegion  (AllReduce  in RowPar Forward)
//   _ScatterToModelParallelRegion   (split input in RowPar Forward, no comm)
//   _ReduceScatterToSequenceParallelRegion (ReduceScatter for SP mode)

namespace infini_train::autograd {

// ColumnParallelLinear
// -------------------
// Weight is split along the output dimension: each TP rank holds weight[out/TP, in].
//
// Forward:
//   local_out = input @ weight_shard^T + bias_shard   (full input, partial output)
//   if gather_output: output = AllGather(local_out) across TP group   → [*, out]
//   else:             output = local_out                               → [*, out/TP]
//
// Backward:
//   grad_weight: local matmul (no comm, each rank owns its shard)
//   grad_input:  AllReduce(grad_input_shard) across TP group
//                (or ReduceScatter if sequence_parallel=true)
//
// Inputs[0]: input tensor [*, in_features]
// Inputs[1]: weight shard [out_features/TP, in_features]
// Inputs[2]: bias shard   [out_features/TP]  (optional, nullptr if no bias)
class ColumnParallelLinear : public Function {
public:
    explicit ColumnParallelLinear(bool gather_output = true,
                                  bool sequence_parallel = false);

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &outputs) override;

    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    bool gather_output_;
    bool sequence_parallel_;
    int64_t out_features_ = 0;
    bool has_bias_ = false;
};

// RowParallelLinear
// -----------------
// Weight is split along the input dimension: each TP rank holds weight[out, in/TP].
//
// Forward:
//   input_shard = input[*, in/TP]  (input already sharded; caller slices or scatters)
//   local_out = input_shard @ weight_shard^T   → [*, out]
//   if not sequence_parallel: output = AllReduce(local_out) across TP group
//   else:                      output = ReduceScatter(local_out) → [*/TP, out]
//   output = output + replicated_bias (after the collective)
//
// Backward:
//   grad_input_shard: local matmul (no comm; input was already sharded)
//   grad_weight:      local matmul (no comm)
//
// Inputs[0]: input shard [*, in_features/TP]  (pre-scattered by caller)
// Inputs[1]: weight shard [out_features, in_features/TP]
// Inputs[2]: bias         [out_features]  (optional, added on rank 0 only)
class RowParallelLinear : public Function {
public:
    explicit RowParallelLinear(bool sequence_parallel = false);

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &outputs) override;

    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    bool sequence_parallel_;
    int64_t out_features_ = 0;
    bool bias_provided_ = false;
};

} // namespace infini_train::autograd
#endif // USE_NCCL
