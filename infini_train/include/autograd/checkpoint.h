#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Module;
}

// Activation recomputation (gradient checkpointing).
//
// Problem: storing all intermediate activations for a 12-layer GPT-2 Block costs:
//   per Block:  attention weights [B, H, T, T] + MLP intermediate [B, T, 4C]
//   total:      O(n_layer) tensors  →  ~12 × 12 × (768² + 3072×768) ≈ hundreds of MB
//
// Solution: CheckpointFunction wraps any Function and trades compute for memory.
//   Forward:   run wrapped_->Forward(inputs) but discard all intermediate activations;
//              save only the inputs (cheap: [B, T, C]).
//   Backward:  re-run wrapped_->Forward(saved_inputs) to reconstruct activations
//              (recompute), then call wrapped_->Backward(grad_outputs).
//
// Memory: each checkpointed Block keeps its [B, T, C] boundary instead of
// [B,H,T,T]+[B,T,4C] intermediates. TinyInfiniTrain currently checkpoints each
// Block independently; it does not implement the segmented O(sqrt(L)) scheme
// from Chen et al. (2016).
//
// Usage:
//   auto block_fn = std::make_shared<BlockFunction>(block);
//   auto ckpt = std::make_shared<CheckpointFunction>(block_fn);
//   x = ckpt->Apply({x})[0];
//
// Megatron-LM analogue: megatron/core/transformer/transformer_layer.py
//   uses torch.utils.checkpoint.checkpoint() on TransformerLayer.forward(),
//   enabled by TransformerConfig.recompute_granularity.
//   Modes: "full" (whole block), "selective" (only attention softmax).
namespace infini_train::autograd {

class CheckpointFunction : public Function {
public:
    // Wraps any Function subclass.  The wrapped function must be stateless
    // between calls (no side effects in Forward that affect Backward state).
    explicit CheckpointFunction(std::shared_ptr<Function> fn);

    // Forward: run fn->Forward(inputs) once without SetupContext.
    //   Only inputs are retained; all intermediate activations are freed.
    //   Returns the same outputs as fn->Forward would return.
    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs) override;

    // SetupContext: save inputs for later recomputation.
    //   Deliberately does NOT call wrapped_->SetupContext() — that is deferred
    //   to the recomputation inside Backward.
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &outputs) override;

    // Backward: recompute forward activations, then run fn->Backward.
    //   1. Re-run wrapped_->Forward(saved_inputs_) — this time call SetupContext
    //      so saved_tensors_ (attention weights, MLP activations) are populated.
    //   2. Call wrapped_->Backward(grad_outputs) using those recomputed tensors.
    //   Memory peak during recompute = one Block's activations (ephemeral).
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

    void ReleaseVariables() override;

private:
    std::shared_ptr<Function> wrapped_;
    std::vector<std::shared_ptr<Tensor>> saved_inputs_;
};

// Checkpoints a complete nn::Module with one Tensor input and one Tensor
// output. Unlike CheckpointFunction, whose wrapped Function supplies a manual
// Backward implementation, this variant rebuilds the module's ordinary
// autograd graph during backward. Module parameters participate in that
// recomputed graph and accumulate their gradients exactly once; the detached
// recomputation input's gradient is returned to the outer graph.
//
// The Module must outlive the backward pass. The input must require gradients,
// which is naturally true for Transformer block activations (including
// received pipeline-stage activations).
class ModuleCheckpointFunction final : public Function {
public:
    explicit ModuleCheckpointFunction(nn::Module *module);

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &outputs) override;
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;
    void ReleaseVariables() override;

private:
    nn::Module *module_ = nullptr;
    std::shared_ptr<Tensor> saved_input_;
};

std::shared_ptr<Tensor> CheckpointModule(nn::Module *module,
                                         const std::shared_ptr<Tensor> &input);

} // namespace infini_train::autograd
