#include "infini_train/include/autograd/function.h"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
namespace {
thread_local bool g_create_graph_enabled = false;
thread_local bool g_grad_enabled = true;

class CreateGraphGuard {
public:
    explicit CreateGraphGuard(bool enabled) : previous_(g_create_graph_enabled) {
        g_create_graph_enabled = enabled;
    }
    ~CreateGraphGuard() { g_create_graph_enabled = previous_; }

private:
    bool previous_;
};

class AccumulateGrad final : public Function {
public:
    explicit AccumulateGrad(const std::shared_ptr<Tensor> &tensor)
        : Function("AccumulateGrad"), tensor_(tensor) {
        SetOutputMetadata({OutputMetadata{tensor->Dims(), tensor->Dtype(), tensor->GetDevice()}});
    }

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &) override {
        LOG(FATAL) << "AccumulateGrad::Forward shall not be called directly!";
        return {};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        CHECK_EQ(grad_outputs.size(), 1u);
        auto tensor = tensor_.lock();
        CHECK(tensor) << "Leaf tensor was destroyed before gradient accumulation";
        tensor->AccumulateGrad(grad_outputs[0], IsCreateGraphEnabled());
        return {};
    }

    void BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int) override {
        if (grad_output) { Backward({grad_output}); }
    }

private:
    std::weak_ptr<Tensor> tensor_;
};

using GradSlots = std::vector<std::shared_ptr<Tensor>>;

std::shared_ptr<Tensor> Accumulate(const std::shared_ptr<Tensor> &current,
                                   const std::shared_ptr<Tensor> &incoming,
                                   bool create_graph) {
    if (!current) { return incoming; }
    if (!incoming) { return current; }
    if (create_graph) { return current->Add(incoming); }

    auto kernel = Dispatcher::Instance().GetKernel({current->GetDevice().Type(), "AccumulateGrad"});
    kernel.Call<void>(incoming, 1.0f, current);
    return current;
}

std::shared_ptr<Tensor> ZerosFrom(const OutputMetadata &metadata) {
    auto result = std::make_shared<Tensor>(metadata.dims, metadata.dtype, metadata.device);
    result->Fill<float>(0.0f);
    return result;
}
} // namespace

std::vector<std::shared_ptr<Tensor>> Function::Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto output_tensors = Forward(input_tensors);

    bool output_requires_grad = false;
    for (const auto &input_tensor : input_tensors) {
        output_requires_grad |= input_tensor->requires_grad();
    }
    output_requires_grad &= IsGradEnabled();

    if (!output_requires_grad) {
        for (auto &output_tensor : output_tensors) {
            output_tensor->set_requires_grad(false);
            output_tensor->set_is_leaf(true);
            output_tensor->set_grad_fn(nullptr);
            output_tensor->set_output_idx(0);
        }
        return output_tensors;
    }

    SetupContext(input_tensors, output_tensors);
    for (int idx = 0; idx < input_tensors.size(); ++idx) {
        const auto &input_tensor = input_tensors[idx];
        if (input_tensor->requires_grad() && input_tensor->is_leaf()) {
            next_functions_.emplace_back(std::make_shared<AccumulateGrad>(input_tensor), 0);
        } else {
            next_functions_.emplace_back(input_tensor->grad_fn(), input_tensor->output_idx());
        }
    }

    output_metadata_.clear();
    output_metadata_.reserve(output_tensors.size());
    for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
        auto &output_tensor = output_tensors[output_idx];
        output_metadata_.push_back(
            OutputMetadata{output_tensor->Dims(), output_tensor->Dtype(), output_tensor->GetDevice()});
        output_tensor->set_requires_grad(output_requires_grad);
        output_tensor->set_is_leaf(false);
        output_tensor->set_grad_fn(output_requires_grad ? shared_from_this() : nullptr);
        output_tensor->set_output_idx(output_idx);
    }

    return output_tensors;
}

NoGradGuard::NoGradGuard() : previous_(g_grad_enabled) { g_grad_enabled = false; }

NoGradGuard::~NoGradGuard() { g_grad_enabled = previous_; }

bool IsGradEnabled() { return g_grad_enabled; }

void Function::BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int grad_output_idx) {
    RunBackward(shared_from_this(), grad_output, grad_output_idx, false, false);
}

void Function::IncreaseDependenciesNumber() {}

void Function::ReleaseVariables() {
    saved_tensors_.clear();
    graph_released_ = true;
}

bool IsCreateGraphEnabled() { return g_create_graph_enabled; }

void RunBackward(const std::shared_ptr<Function> &root, const std::shared_ptr<Tensor> &gradient,
                 int output_idx, bool retain_graph, bool create_graph) {
    CHECK(root) << "Backward requires a valid grad_fn";
    CHECK(!root->GraphReleased())
        << "Trying to backward through the graph a second time after saved tensors were released. "
           "Pass retain_graph=true to the first backward call.";
    CHECK_GE(output_idx, 0);
    CHECK_LT(static_cast<size_t>(output_idx), root->OutputMetadataList().size());

    std::vector<std::shared_ptr<Function>> postorder;
    std::unordered_set<Function *> visited;
    std::function<void(const std::shared_ptr<Function> &)> visit =
        [&](const std::shared_ptr<Function> &fn) {
            if (!fn || !visited.insert(fn.get()).second) { return; }
            for (const auto &[next, unused_output_idx] : fn->NextFunctions()) {
                (void)unused_output_idx;
                visit(next);
            }
            postorder.push_back(fn);
        };
    visit(root);

    std::unordered_map<Function *, GradSlots> gradients;
    gradients[root.get()] = GradSlots(root->OutputMetadataList().size(), nullptr);
    gradients[root.get()][output_idx] = gradient;

    CreateGraphGuard create_graph_guard(create_graph);
    for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
        const auto &fn = *it;
        auto grad_it = gradients.find(fn.get());
        if (grad_it == gradients.end()) { continue; }

        auto &grad_outputs = grad_it->second;
        const auto &metadata = fn->OutputMetadataList();
        CHECK_EQ(grad_outputs.size(), metadata.size());
        bool has_gradient = false;
        for (const auto &grad : grad_outputs) { has_gradient |= static_cast<bool>(grad); }
        if (!has_gradient) { continue; }
        for (size_t i = 0; i < grad_outputs.size(); ++i) {
            if (!grad_outputs[i]) { grad_outputs[i] = ZerosFrom(metadata[i]); }
        }

        auto grad_inputs = fn->Backward(grad_outputs);
        const auto &edges = fn->NextFunctions();
        CHECK_EQ(grad_inputs.size(), edges.size());
        for (size_t i = 0; i < grad_inputs.size(); ++i) {
            const auto &[next, next_output_idx] = edges[i];
            if (!grad_inputs[i] || !next) { continue; }
            auto &slots = gradients[next.get()];
            if (slots.empty()) { slots.resize(next->OutputMetadataList().size(), nullptr); }
            CHECK_GE(next_output_idx, 0);
            CHECK_LT(static_cast<size_t>(next_output_idx), slots.size());
            slots[next_output_idx] = Accumulate(slots[next_output_idx], grad_inputs[i], create_graph);
        }
    }

    if (!retain_graph && !create_graph) {
        for (const auto &fn : postorder) { fn->ReleaseVariables(); }
    }
}
} // namespace infini_train::autograd
