#ifdef USE_NCCL
#include "infini_train/include/autograd/linear_parallel.h"

#include "glog/logging.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/distributed/nccl_utils.h"
#include "infini_train/include/distributed/parallel_state.h"

namespace infini_train::autograd {

namespace {

// ncclAllGather stacks each rank's whole shard as a NEW leading axis:
// shard [*, out/TP] -> raw [TP, *, out/TP]. We actually want the TP axis
// concatenated onto the FEATURE (last) axis: [*, out] where
// out = TP * out/TP, matching Megatron's `_gather` (all_gather + cat(dim=-1)).
// Reuse the existing Transpose/Contiguous/View idiom (already used e.g. in
// CausalSelfAttention::Forward) to walk the TP axis from position 0 to just
// before the last axis, then flatten the trailing (TP, out/TP) pair.
std::shared_ptr<Tensor> GatherAlongLastDim(const std::shared_ptr<Tensor> &shard,
                                           distributed::ProcessGroup *tp_group) {
    const int tp_size = tp_group->WorldSize();
    const auto &shard_dims = shard->Dims();

    std::vector<int64_t> raw_dims = {tp_size};
    raw_dims.insert(raw_dims.end(), shard_dims.begin(), shard_dims.end());
    auto raw = std::make_shared<Tensor>(raw_dims, shard->Dtype(), shard->GetDevice());
    tp_group->AllGather(shard->DataPtr(), raw->DataPtr(), shard->NumElements(),
                        distributed::ToNcclDataType(shard->Dtype()));

    // [TP, d0, d1, ..., d_{k-1}] -> [d0, d1, ..., TP, d_{k-1}] via k-1 adjacent swaps.
    std::shared_ptr<Tensor> permuted = raw;
    for (int dim = 0; dim + 2 < static_cast<int>(raw_dims.size()); ++dim) {
        permuted = permuted->Transpose(dim, dim + 1);
    }

    auto out_dims = shard_dims;
    out_dims.back() *= tp_size;
    return permuted->Contiguous()->View(out_dims);
}

} // namespace

// ─── ColumnParallelLinear ────────────────────────────────────────────────────

ColumnParallelLinear::ColumnParallelLinear(bool gather_output, bool sequence_parallel)
    : Function("ColumnParallelLinear"),
      gather_output_(gather_output),
      sequence_parallel_(sequence_parallel) {}

std::vector<std::shared_ptr<Tensor>> ColumnParallelLinear::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
    // inputs[0]: input [*, in_features]  (replicated across the TP group)
    // inputs[1]: weight shard [out/TP, in_features]
    // inputs[2]: bias shard [out/TP]  (optional)
    CHECK_GE(inputs.size(), 2u);
    const auto &input  = inputs[0];
    const auto &weight = inputs[1];
    has_bias_ = (inputs.size() >= 3 && inputs[2] != nullptr);

    out_features_ = weight->Dims()[0];

    // Local matmul: output_shard [*, out/TP]. `input` is identical on every TP
    // rank (it was never sharded), so no communication is needed before this.
    auto output_shard = Dispatcher::Instance()
        .GetKernel({input->GetDevice().Type(), "LinearForward"})
        .Call<std::shared_ptr<Tensor>>(input, weight, true,
                                        has_bias_ ? inputs[2] : nullptr);

    if (!gather_output_) {
        return {output_shard};
    }

    auto *tp_group = distributed::GetTPGroup();
    CHECK(tp_group) << "ColumnParallelLinear(gather_output=true) requires an initialized TP group";
    if (tp_group->WorldSize() == 1) {
        return {output_shard};
    }
    if (sequence_parallel_) {
        LOG(FATAL) << "ColumnParallelLinear(sequence_parallel=true) is not implemented: "
                   << "the ReduceScatter-to-sequence-shards variant requires a sequence-first "
                   << "([seq, batch, hidden]) tensor layout across the whole model. "
                   << "TinyInfiniTrain's GPT-2 example uses batch-first [B, T, C]; out of scope.";
    }
    return {GatherAlongLastDim(output_shard, tp_group)};
}

void ColumnParallelLinear::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &inputs,
    const std::vector<std::shared_ptr<Tensor>> &/*outputs*/) {
    // Save input and weight for backward.
    saved_tensors_ = {inputs[0], inputs[1]};
    if (has_bias_) {
        saved_tensors_.push_back(inputs[2]);
    }
}

std::vector<std::shared_ptr<Tensor>> ColumnParallelLinear::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // grad_outputs[0]: grad w.r.t. output [*, out/TP] (or [*, out] if gathered)
    const auto &input  = saved_tensors_[0];
    const auto &weight = saved_tensors_[1];
    auto grad = grad_outputs[0];

    auto *tp_group = distributed::GetTPGroup();
    const int tp_size = tp_group ? tp_group->WorldSize() : 1;

    if (gather_output_ && tp_size > 1) {
        // Backward of "gather" (all_gather + cat) is "split": each rank only
        // needs the gradient slice covering its own out/TP contribution
        // (Megatron mappings.py: _GatherFromModelParallelRegion.backward == split,
        // no communication — every rank already has the full grad_outputs[0]).
        const int tp_rank = distributed::GetTPRank();
        const auto &grad_dims = grad->Dims();
        std::vector<int64_t> starts(grad_dims.size(), 0);
        std::vector<int64_t> ends = grad_dims;
        std::vector<int64_t> steps(grad_dims.size(), 1);
        starts.back() = static_cast<int64_t>(tp_rank) * out_features_;
        ends.back() = static_cast<int64_t>(tp_rank + 1) * out_features_;
        grad = grad->Slice(starts, ends, steps);
    }

    auto [grad_input_shard, grad_weight, grad_bias] =
        Dispatcher::Instance()
            .GetKernel({input->GetDevice().Type(), "LinearBackward"})
            .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                              std::shared_ptr<Tensor>>>(
                input, weight, true, out_features_, grad, has_bias_);

    // grad_input_shard is this rank's contribution to d(loss)/d(input), computed
    // only through this rank's weight shard. Since `input` was replicated (the
    // SAME tensor on every rank), the true d(loss)/d(input) is the SUM of every
    // rank's contribution — AllReduce (Megatron mappings.py:
    // _ReduceFromModelParallelRegion, used as ColumnParallel's input-identity
    // backward).
    if (tp_size > 1) {
        if (sequence_parallel_) {
            LOG(FATAL) << "ColumnParallelLinear(sequence_parallel=true) backward is not implemented "
                       << "(see Forward for the same sequence-first-layout caveat).";
        }
        tp_group->AllReduce(grad_input_shard->DataPtr(), grad_input_shard->NumElements(),
                            distributed::ToNcclDataType(grad_input_shard->Dtype()), ncclSum);
    }

    std::vector<std::shared_ptr<Tensor>> grads = {grad_input_shard, grad_weight};
    if (has_bias_) {
        grads.push_back(grad_bias);
    }
    return grads;
}

// ─── RowParallelLinear ───────────────────────────────────────────────────────

RowParallelLinear::RowParallelLinear(bool sequence_parallel)
    : Function("RowParallelLinear"), sequence_parallel_(sequence_parallel) {}

std::vector<std::shared_ptr<Tensor>> RowParallelLinear::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
    // inputs[0]: input shard [*, in/TP]  (pre-scattered)
    // inputs[1]: weight shard [out, in/TP]
    // inputs[2]: bias [out]  (optional, added on tp_rank==0 only)
    CHECK_GE(inputs.size(), 2u);
    const auto &input  = inputs[0];
    const auto &weight = inputs[1];
    out_features_ = weight->Dims()[0];

    auto *tp_group = distributed::GetTPGroup();
    const int tp_size = tp_group ? tp_group->WorldSize() : 1;
    bias_provided_ = (inputs.size() >= 3 && inputs[2] != nullptr);

    // Local matmul: local_out [*, out]. `input` is already this rank's shard
    // (pre-scattered by the caller), so no communication before this.
    auto local_out = Dispatcher::Instance()
        .GetKernel({input->GetDevice().Type(), "LinearForward"})
        .Call<std::shared_ptr<Tensor>>(input, weight, true, nullptr);

    if (tp_size > 1 && !sequence_parallel_) {
        // AllReduce across TP group → output [*, out]. local_out on every rank
        // is a PARTIAL sum over that rank's slice of the input/contraction
        // dimension; summing across ranks gives the correct full output
        // (Megatron mappings.py: _ReduceFromModelParallelRegion).
        tp_group->AllReduce(local_out->DataPtr(), local_out->NumElements(),
                            distributed::ToNcclDataType(local_out->Dtype()), ncclSum);
    } else if (tp_size > 1) {
        LOG(FATAL) << "RowParallelLinear(sequence_parallel=true) is not implemented: "
                   << "ReduceScatter along the sequence axis requires a sequence-first "
                   << "([seq, batch, hidden]) tensor layout across the whole model. "
                   << "TinyInfiniTrain's GPT-2 example uses batch-first [B, T, C]; out of scope.";
    }

    // Megatron's row-parallel bias is replicated and added after the partial
    // products have been reduced. This avoids multiplying the bias by TP and
    // gives every replica the same bias gradient/update.
    if (bias_provided_) {
        local_out = Dispatcher::Instance()
                        .GetKernel({input->GetDevice().Type(), "AddForward"})
                        .Call<std::shared_ptr<Tensor>>(local_out, inputs[2]);
    }

    return {local_out};
}

void RowParallelLinear::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &inputs,
    const std::vector<std::shared_ptr<Tensor>> &/*outputs*/) {
    saved_tensors_ = {inputs[0], inputs[1]};
    // Save the bias tensor whenever one was PROVIDED (not just when this rank
    // added it) — Backward needs its shape/dtype/device on every rank to
    // return a same-shaped gradient slot, even a zero one.
    if (bias_provided_) {
        saved_tensors_.push_back(inputs[2]);
    }
}

std::vector<std::shared_ptr<Tensor>> RowParallelLinear::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // No communication on grad_input: `input` was already sharded (pre-scattered
    // by the caller, not produced by an identity/gather region), and grad_output
    // arrives already-correct-and-full on every rank (backward of the AllReduce
    // sum in Forward is itself an identity: every rank's downstream computation
    // on the replicated `out` independently produces the same full gradient).
    // grad_weight and grad_input are therefore purely local matmuls.
    const auto &input  = saved_tensors_[0];
    const auto &weight = saved_tensors_[1];
    const auto &grad   = grad_outputs[0];

    if (sequence_parallel_) {
        LOG(FATAL) << "RowParallelLinear(sequence_parallel=true) backward is not implemented "
                   << "(see Forward for the same sequence-first-layout caveat).";
    }

    auto [grad_input, grad_weight, grad_bias] =
        Dispatcher::Instance()
            .GetKernel({input->GetDevice().Type(), "LinearBackward"})
            .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                              std::shared_ptr<Tensor>>>(
                input, weight, true, out_features_, grad, bias_provided_);

    std::vector<std::shared_ptr<Tensor>> grads = {grad_input, grad_weight};
    if (bias_provided_) {
        grads.push_back(grad_bias);
    }
    return grads;
}

} // namespace infini_train::autograd
#endif // USE_NCCL
