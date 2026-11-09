#pragma once
#ifdef USE_NCCL

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

#include "infini_train/include/tensor.h"

// Forward declaration to avoid pulling in the full GPT2 header.
class GPT2;
namespace infini_train {
class Optimizer;
}

namespace infini_train::distributed {

// ─── DataBatch ───────────────────────────────────────────────────────────────

// A single training batch split into num_microbatches micro-batches.
// shape of each shard: [global_batch / num_microbatches, seq_len]
struct DataBatch {
    std::vector<std::shared_ptr<Tensor>> inputs;   // token index tensors
    std::vector<std::shared_ptr<Tensor>> targets;  // label tensors
};

struct ScheduleMetrics {
    float elapsed_ms = 0.0f;
    float compute_ms = 0.0f;
    int peak_inflight_activations = 0;
};

// Asynchronous CUDA-event profiler. Compute intervals exclude time queued in
// P2P Recv/Send, while elapsed_ms spans the entire local schedule. Aggregating
// compute_ms across ranks and dividing by PP * max(elapsed_ms) gives measured
// pipeline utilization (and 1-utilization is the measured bubble fraction).
class ScheduleProfiler {
public:
    ~ScheduleProfiler();
    void Start(int device_id);
    cudaEvent_t BeginCompute();
    void EndCompute(cudaEvent_t start);
    void ActivationForwarded();
    void ActivationBackwarded();
    void Finish();
    const ScheduleMetrics &Metrics() const { return metrics_; }

private:
    void DestroyEvents();
    int device_id_ = 0;
    int inflight_activations_ = 0;
    cudaEvent_t schedule_start_ = nullptr;
    cudaEvent_t schedule_stop_ = nullptr;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> compute_intervals_;
    ScheduleMetrics metrics_;
};

// ─── GPipeSchedule ───────────────────────────────────────────────────────────
//
// GPipe (Huang et al., NeurIPS 2019):
//   1. Forward all M micro-batches through every stage sequentially (F phase).
//   2. Then backward all M micro-batches in reverse order (B phase).
//
// Megatron-LM analogue: megatron/core/pipeline_parallel/schedules.py:forward_backward_no_pipelining
//
// Bubble ratio: (PP - 1) / (PP - 1 + M)
//   - Worst when M is small (M=1 → bubble = (PP-1)/PP).
//   - Disappears as M → ∞.
//   - Peak activation memory: proportional to M (all micro-batch activations kept for B phase).
class GPipeSchedule {
public:
    GPipeSchedule(int num_stages, int num_microbatches, GPT2 *stage_model);

    // Execute one training step (forward all + backward all).
    // Internally calls stage_model_->Forward() for each micro-batch,
    // exchanges activations/gradients via GetPPGroup()->Send/Recv(),
    // then calls loss.Backward() on the last stage.
    void Run(DataBatch batch);

    // Analytical bubble ratio: fraction of time stages are idle.
    static float BubbleRatio(int pp, int m) {
        return static_cast<float>(pp - 1) / static_cast<float>(pp - 1 + m);
    }

    // Peak activation memory relative to a single micro-batch (GPipe keeps all M).
    static int PeakActivationScale(int /*pp*/, int m) { return m; }
    const ScheduleMetrics &Metrics() const { return profiler_.Metrics(); }

private:
    int num_stages_;
    int num_microbatches_;
    GPT2 *stage_model_;  // non-owning pointer; caller manages lifetime
    ScheduleProfiler profiler_;
};

// ─── OneFOneBSchedule ────────────────────────────────────────────────────────
//
// 1F1B (Narayanan et al., SOSP 2019 — PipeDream; steady-state from Megatron-LM):
//   Warmup phase:   PP - 1 forward micro-batches (fills the pipeline).
//   Steady state:   alternating 1 forward, 1 backward per step.
//   Cooldown phase: PP - 1 backward micro-batches (drains the pipeline).
//
// Megatron-LM analogue: megatron/core/pipeline_parallel/schedules.py:
//   forward_backward_pipelining_without_interleaving (non-interleaved)
//   forward_backward_pipelining_with_interleaving   (interleaved, V stages)
//
// Non-interleaved bubble ratio: (PP - 1) / (PP - 1 + M)
//   — Same formula as GPipe, but peak activation memory = PP (not M).
//
// Interleaved (V virtual stages per physical stage):
//   bubble ratio = (PP - 1) / (PP - 1 + M * V)
//   — Smaller bubble at the cost of more P2P messages per step.
//
// Weight stashing (PipeDream):
//   Each micro-batch uses the parameter version that was current at its forward pass.
//   Stash: copy params before each forward.
//   Restore: load stashed params before corresponding backward.
//   Evict: discard after backward completes.
//   TODO: evict policy — staleness bound (PipeDream-2BW uses 2 versions max).
class OneFOneBSchedule {
public:
    OneFOneBSchedule(int num_stages, int num_microbatches, GPT2 *stage_model,
                     bool interleaved = false, int num_virtual_stages = 1);

    // Execute one training step using 1F1B schedule.
    void Run(DataBatch batch);

    // PipeDream mode: perform an optimizer update after every micro-batch
    // backward. Each forward parameter version is stashed and restored before
    // its matching backward, then evicted after the update.
    OneFOneBSchedule(int num_stages, int num_microbatches, GPT2 *stage_model,
                     Optimizer *optimizer);

    int OptimizerSteps() const { return optimizer_steps_; }
    size_t StashedVersionCount() const;
    const ScheduleMetrics &Metrics() const { return profiler_.Metrics(); }

    // Non-interleaved and interleaved bubble ratio formulas.
    static float BubbleRatio(int pp, int m, int v = 1) {
        return static_cast<float>(pp - 1) / static_cast<float>(pp - 1 + m * v);
    }

    // Peak activation memory: 1F1B keeps at most PP micro-batch activations
    // (vs GPipe which keeps all M).  Returns the number of in-flight activations.
    static int PeakActivationScale(int pp, int /*m*/) { return pp; }

private:
    int num_stages_;
    int num_microbatches_;
    GPT2 *stage_model_;
    bool interleaved_;
    int num_virtual_stages_;
    Optimizer *optimizer_ = nullptr;
    int optimizer_steps_ = 0;
    ScheduleProfiler profiler_;

    // Weight stashing buffer (PipeDream-style):
    //   weight_stash_[k] = parameter snapshot taken before the k-th micro-batch forward.
    //   Indexed by microbatch id; entries evicted after the corresponding backward.
    std::vector<std::unordered_map<std::string, std::shared_ptr<Tensor>>> weight_stash_;

    // Capture a snapshot of stage_model_->Parameters() into weight_stash_[microbatch_id].
    void StashWeights(int microbatch_id);

    // Restore the parameter snapshot for microbatch_id, run backward, then evict.
    void RestoreAndEvictWeights(int microbatch_id);
};

} // namespace infini_train::distributed
#endif // USE_NCCL
