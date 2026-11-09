#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
struct OutputMetadata {
    std::vector<int64_t> dims;
    DataType dtype;
    Device device;
};

class Function : public std::enable_shared_from_this<Function> {
public:
    using Edge = std::pair<std::shared_ptr<Function>, int>;

    static constexpr char kUndefinedType[] = "Undefined";

    Function() : type_(kUndefinedType) {}
    explicit Function(const std::string &type) : type_(type) {}

    virtual ~Function() = default;

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;
    virtual void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                              const std::vector<std::shared_ptr<Tensor>> &output_tensors) {}
    virtual std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) = 0;

    std::vector<std::shared_ptr<Tensor>> Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors);
    virtual void BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int idx);

    const std::vector<Edge> &NextFunctions() const { return next_functions_; }
    const std::vector<OutputMetadata> &OutputMetadataList() const { return output_metadata_; }
    bool GraphReleased() const { return graph_released_; }
    virtual void ReleaseVariables();

    void IncreaseDependenciesNumber();

protected:
    std::vector<std::shared_ptr<Tensor>> saved_tensors_;
    void SetOutputMetadata(std::vector<OutputMetadata> metadata) { output_metadata_ = std::move(metadata); }

private:
    std::vector<Edge> next_functions_;
    std::vector<OutputMetadata> output_metadata_;
    bool graph_released_ = false;
    const std::string type_ = kUndefinedType;
};

// Thread-local gradient recording switch. Operations executed under
// NoGradGuard still run their kernels but do not retain contexts or connect an
// autograd graph. CheckpointFunction uses this during the original forward and
// recomputation passes so only its explicit input checkpoint remains live.
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard &) = delete;
    NoGradGuard &operator=(const NoGradGuard &) = delete;

private:
    bool previous_;
};

bool IsGradEnabled();

// Runs a reverse-mode pass from one Function output. The engine explicitly
// topologically sorts the reachable graph, accumulates all incoming gradients
// before executing a node, and releases saved tensors unless retain_graph is set.
void RunBackward(const std::shared_ptr<Function> &root, const std::shared_ptr<Tensor> &gradient,
                 int output_idx, bool retain_graph, bool create_graph);

// Backward implementations use this to select differentiable formulas when a
// higher-order graph is requested.
bool IsCreateGraphEnabled();
} // namespace infini_train::autograd
