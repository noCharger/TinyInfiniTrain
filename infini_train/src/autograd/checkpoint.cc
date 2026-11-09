#include "infini_train/include/autograd/checkpoint.h"

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::autograd {

CheckpointFunction::CheckpointFunction(std::shared_ptr<Function> fn)
    : Function("CheckpointFunction"), wrapped_(std::move(fn)) {
    CHECK(wrapped_) << "CheckpointFunction requires a non-null wrapped Function";
}

std::vector<std::shared_ptr<Tensor>> CheckpointFunction::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
    // Do not call wrapped_->SetupContext() here. Inner Apply calls also see the
    // no-grad guard, so their temporary contexts and graph edges are discarded.
    NoGradGuard no_grad;
    return wrapped_->Forward(inputs);
}

void CheckpointFunction::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &inputs,
    const std::vector<std::shared_ptr<Tensor>> &/*outputs*/) {
    // Save only the inputs.  All intermediate activations from wrapped_->Forward
    // are intentionally discarded here — that is the memory saving.
    saved_inputs_ = inputs;
}

std::vector<std::shared_ptr<Tensor>> CheckpointFunction::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // ── Step 1: recompute activations ────────────────────────────────────────
    std::vector<std::shared_ptr<Tensor>> recomputed_outputs;
    {
        NoGradGuard no_grad;
        recomputed_outputs = wrapped_->Forward(saved_inputs_);
    }
    wrapped_->SetupContext(saved_inputs_, recomputed_outputs);

    // ── Step 2: backward through the recomputed graph ────────────────────────
    return wrapped_->Backward(grad_outputs);
}

void CheckpointFunction::ReleaseVariables() {
    saved_inputs_.clear();
    wrapped_->ReleaseVariables();
    Function::ReleaseVariables();
}

ModuleCheckpointFunction::ModuleCheckpointFunction(nn::Module *module)
    : Function("ModuleCheckpointFunction"), module_(module) {
    CHECK_NOTNULL(module_);
}

std::vector<std::shared_ptr<Tensor>> ModuleCheckpointFunction::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 1u) << "Module checkpointing currently supports one input";
    CHECK(inputs[0]->requires_grad())
        << "Module checkpoint input must require gradients so the outer graph can invoke recomputation";
    NoGradGuard no_grad;
    auto outputs = module_->Forward(inputs);
    CHECK_EQ(outputs.size(), 1u) << "Module checkpointing currently supports one output";
    return outputs;
}

void ModuleCheckpointFunction::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &inputs,
    const std::vector<std::shared_ptr<Tensor>> &/*outputs*/) {
    saved_input_ = inputs.at(0);
}

std::vector<std::shared_ptr<Tensor>> ModuleCheckpointFunction::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1u);
    CHECK(saved_input_);
    CHECK(!IsCreateGraphEnabled())
        << "Module activation checkpointing does not yet support create_graph=true";

    // Detach from the outer graph, then rebuild only this module's graph. Its
    // real parameter tensors are leaves in the recomputed graph, so their
    // gradients are accumulated by the nested backward. The clone's gradient
    // is the sole value returned to the outer graph.
    auto recomputed_input = saved_input_->Clone()->RequiresGrad();
    auto recomputed_outputs = module_->Forward({recomputed_input});
    CHECK_EQ(recomputed_outputs.size(), 1u);
    recomputed_outputs[0]->Backward(grad_outputs[0]);
    CHECK(recomputed_input->grad());
    return {recomputed_input->grad()};
}

void ModuleCheckpointFunction::ReleaseVariables() {
    saved_input_.reset();
    Function::ReleaseVariables();
}

std::shared_ptr<Tensor> CheckpointModule(nn::Module *module,
                                         const std::shared_ptr<Tensor> &input) {
    return std::make_shared<ModuleCheckpointFunction>(module)->Apply({input})[0];
}

} // namespace infini_train::autograd
