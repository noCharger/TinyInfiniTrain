#ifdef USE_NCCL
#include "infini_train/include/distributed/pipeline_schedule.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "glog/logging.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "infini_train/include/cuda_utils.h"
#endif

#include "example/gpt2/net.h"
#include "infini_train/include/distributed/nccl_utils.h"
#include "infini_train/include/distributed/parallel_state.h"
#include "infini_train/include/distributed/process_group.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"

namespace infini_train::distributed {

namespace {

namespace nn = infini_train::nn;

// Allocates a fresh Tensor of the given shape/dtype/device and blocks on a P2P
// Recv into it. Used both for cross-stage activations (Forward direction) and
// cross-stage gradients (Backward direction) — the two differ only in shape.
std::shared_ptr<Tensor> RecvLike(ProcessGroup *pp_group, int peer, const std::vector<int64_t> &dims, DataType dtype,
                                 Device device) {
    auto tensor = std::make_shared<Tensor>(dims, dtype, device);
    pp_group->Recv(tensor->DataPtr(), tensor->NumElements(), ToNcclDataType(dtype), peer);
    return tensor;
}

void SendTensor(ProcessGroup *pp_group, int peer, const std::shared_ptr<Tensor> &tensor) {
    pp_group->Send(tensor->DataPtr(), tensor->NumElements(), ToNcclDataType(tensor->Dtype()), peer);
}

std::shared_ptr<Tensor> SendRecvLike(ProcessGroup *pp_group, int peer, const std::shared_ptr<Tensor> &send,
                                    const std::vector<int64_t> &recv_dims, DataType recv_dtype, Device device) {
    auto recv = std::make_shared<Tensor>(recv_dims, recv_dtype, device);
    pp_group->SendRecv(send->DataPtr(), send->NumElements(), ToNcclDataType(send->Dtype()), peer,
                       recv->DataPtr(), recv->NumElements(), ToNcclDataType(recv_dtype), peer);
    return recv;
}

// Deep-copies src's bytes into dst's EXISTING buffer in place (unlike
// Tensor::Clone(), which allocates a new tensor). Used by weight restore,
// where dst is the live parameter shared_ptr that the model graph already
// holds references to — the buffer must be overwritten, not replaced.
void CopyInto(const std::shared_ptr<Tensor> &dst, const std::shared_ptr<Tensor> &src) {
    CHECK_EQ(dst->NumElements(), src->NumElements());
    switch (dst->GetDevice().Type()) {
    case DeviceType::kCPU:
        std::memcpy(dst->DataPtr(), src->DataPtr(), src->SizeInBytes());
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        cudaMemcpyAsync(dst->DataPtr(), src->DataPtr(), src->SizeInBytes(), cudaMemcpyDeviceToDevice,
                        infini_train::ThreadLocalStream(dst->GetDevice().Index()));
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(dst->GetDevice().Type());
    }
}

} // namespace

ScheduleProfiler::~ScheduleProfiler() { DestroyEvents(); }

void ScheduleProfiler::DestroyEvents() {
    if (schedule_start_) cudaEventDestroy(schedule_start_);
    if (schedule_stop_) cudaEventDestroy(schedule_stop_);
    for (const auto &[start, stop] : compute_intervals_) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    schedule_start_ = nullptr;
    schedule_stop_ = nullptr;
    compute_intervals_.clear();
}

void ScheduleProfiler::Start(int device_id) {
    DestroyEvents();
    device_id_ = device_id;
    inflight_activations_ = 0;
    metrics_ = {};
    CHECK(cudaSetDevice(device_id_) == cudaSuccess);
    CHECK(cudaEventCreate(&schedule_start_) == cudaSuccess);
    CHECK(cudaEventCreate(&schedule_stop_) == cudaSuccess);
    CHECK(cudaEventRecord(schedule_start_, infini_train::ThreadLocalStream(device_id_)) == cudaSuccess);
}

cudaEvent_t ScheduleProfiler::BeginCompute() {
    cudaEvent_t start = nullptr;
    CHECK(cudaEventCreate(&start) == cudaSuccess);
    CHECK(cudaEventRecord(start, infini_train::ThreadLocalStream(device_id_)) == cudaSuccess);
    return start;
}

void ScheduleProfiler::EndCompute(cudaEvent_t start) {
    cudaEvent_t stop = nullptr;
    CHECK(cudaEventCreate(&stop) == cudaSuccess);
    CHECK(cudaEventRecord(stop, infini_train::ThreadLocalStream(device_id_)) == cudaSuccess);
    compute_intervals_.emplace_back(start, stop);
}

void ScheduleProfiler::ActivationForwarded() {
    ++inflight_activations_;
    metrics_.peak_inflight_activations = std::max(metrics_.peak_inflight_activations, inflight_activations_);
}

void ScheduleProfiler::ActivationBackwarded() {
    CHECK_GT(inflight_activations_, 0);
    --inflight_activations_;
}

void ScheduleProfiler::Finish() {
    CHECK_EQ(inflight_activations_, 0);
    CHECK(cudaEventRecord(schedule_stop_, infini_train::ThreadLocalStream(device_id_)) == cudaSuccess);
    CHECK(cudaEventSynchronize(schedule_stop_) == cudaSuccess);
    CHECK(cudaEventElapsedTime(&metrics_.elapsed_ms, schedule_start_, schedule_stop_) == cudaSuccess);
    metrics_.compute_ms = 0.0f;
    for (const auto &[start, stop] : compute_intervals_) {
        float interval_ms = 0.0f;
        CHECK(cudaEventElapsedTime(&interval_ms, start, stop) == cudaSuccess);
        metrics_.compute_ms += interval_ms;
    }
}

// ─── GPipeSchedule ───────────────────────────────────────────────────────────

GPipeSchedule::GPipeSchedule(int num_stages, int num_microbatches, GPT2 *stage_model)
    : num_stages_(num_stages),
      num_microbatches_(num_microbatches),
      stage_model_(stage_model) {
    CHECK_GT(num_microbatches_, 0);
    CHECK_GT(num_stages_, 0);
    CHECK_NOTNULL(stage_model_);
}

void GPipeSchedule::Run(DataBatch batch) {
    const int pp_rank = GetPPRank();
    const bool is_first_stage = pp_rank == 0;
    const bool is_last_stage = pp_rank == num_stages_ - 1;
    auto *pp_group = GetPPGroup();
    CHECK(pp_group) << "GPipeSchedule requires InitParallelGroups() to have been called";

    const auto &config = stage_model_->Config();
    // The schedule's own idea of "which stage am I" (derived from GetPPRank()
    // and num_stages_) and GPT2Config's idea (pp_size/pre_process/post_process,
    // set independently by the caller at model-construction time) must agree —
    // otherwise a rank could compute a loss on a model that never built
    // lm_head, or skip a Send the next stage is blocking on, with no diagnostic.
    CHECK_EQ(num_stages_, config.pp_size) << "GPipeSchedule's num_stages_ must match GPT2Config::pp_size";
    CHECK_EQ(is_first_stage, config.pre_process)
        << "GPipeSchedule's is_first_stage (pp_rank==0) must match GPT2Config::pre_process";
    CHECK_EQ(is_last_stage, config.post_process)
        << "GPipeSchedule's is_last_stage (pp_rank==num_stages_-1) must match GPT2Config::post_process";

    nn::CrossEntropyLoss loss_fn;

    // batch.inputs[m] is expected to carry a correctly-shaped (micro_batch, seq_len)
    // tensor on EVERY rank, not just the first stage: the first stage reads it as
    // real token ids, while non-first stages only use its shape/device to size the
    // P2P Recv buffer for the activation coming from the previous stage. Likewise
    // batch.targets[m] must be present on every rank but is only read on the last
    // stage. This keeps DataBatch's shape unchanged rather than adding a
    // stage-conditional field.
    CHECK_GE(batch.inputs.size(), static_cast<size_t>(num_microbatches_))
        << "DataBatch.inputs has fewer entries than num_microbatches_";
    CHECK_GE(batch.targets.size(), static_cast<size_t>(num_microbatches_))
        << "DataBatch.targets has fewer entries than num_microbatches_";
    profiler_.Start(batch.inputs[0]->GetDevice().Index());

    std::vector<std::shared_ptr<Tensor>> activations(num_microbatches_);
    std::vector<std::shared_ptr<Tensor>> outputs(num_microbatches_);
    std::vector<std::shared_ptr<Tensor>> losses(num_microbatches_);

    // --- Forward phase (all M micro-batches; GPipe keeps every activation live) ---
    for (int m = 0; m < num_microbatches_; ++m) {
        std::shared_ptr<Tensor> activation;
        if (is_first_stage) {
            activation = batch.inputs[m];
        } else {
            const auto &tok_dims = batch.inputs[m]->Dims(); // (micro_batch, seq_len)
            activation = RecvLike(pp_group, pp_rank - 1, {tok_dims[0], tok_dims[1], config.n_embd},
                                  config.dtype, batch.inputs[m]->GetDevice());
            // Received activations are leaves w.r.t. this stage's local graph;
            // RequiresGrad() wires an AccumulateGrad edge so Backward() below
            // populates activation->grad() for us to Send upstream.
            activation->RequiresGrad();
        }
        activations[m] = activation;

        const cudaEvent_t compute_start = profiler_.BeginCompute();
        auto output = stage_model_->Forward({activation})[0];
        outputs[m] = output;

        if (!is_last_stage) {
            profiler_.EndCompute(compute_start);
            SendTensor(pp_group, pp_rank + 1, output);
        } else {
            losses[m] = loss_fn.Forward({output, batch.targets[m]})[0];
            profiler_.EndCompute(compute_start);
        }
        profiler_.ActivationForwarded();
    }

    // --- Backward phase (all M micro-batches, reverse order) ---
    for (int m = num_microbatches_ - 1; m >= 0; --m) {
        std::shared_ptr<Tensor> grad;
        if (!is_last_stage) {
            grad = RecvLike(pp_group, pp_rank + 1, outputs[m]->Dims(), outputs[m]->Dtype(),
                            outputs[m]->GetDevice());
        }
        const cudaEvent_t compute_start = profiler_.BeginCompute();
        if (is_last_stage) {
            losses[m]->Backward();
        } else {
            outputs[m]->Backward(grad);
        }
        profiler_.EndCompute(compute_start);

        if (!is_first_stage) {
            SendTensor(pp_group, pp_rank - 1, activations[m]->grad());
        }
        profiler_.ActivationBackwarded();
    }
    stage_model_->SynchronizeTiedEmbeddingGradients();

    profiler_.Finish();

    // optimizer.Step() is called by the training loop after Run() returns.
}

// ─── OneFOneBSchedule ────────────────────────────────────────────────────────

OneFOneBSchedule::OneFOneBSchedule(int num_stages, int num_microbatches, GPT2 *stage_model, bool interleaved,
                                   int num_virtual_stages)
    : num_stages_(num_stages),
      num_microbatches_(num_microbatches),
      stage_model_(stage_model),
      interleaved_(interleaved),
      num_virtual_stages_(num_virtual_stages) {
    CHECK_GT(num_microbatches_, 0);
    CHECK_GT(num_stages_, 0);
    CHECK_NOTNULL(stage_model_);
    CHECK_GE(num_virtual_stages_, 1);
    weight_stash_.resize(num_microbatches_);
}

OneFOneBSchedule::OneFOneBSchedule(int num_stages, int num_microbatches, GPT2 *stage_model,
                                   Optimizer *optimizer)
    : OneFOneBSchedule(num_stages, num_microbatches, stage_model, false, 1) {
    CHECK_NOTNULL(optimizer);
    optimizer_ = optimizer;
}

size_t OneFOneBSchedule::StashedVersionCount() const {
    return std::count_if(weight_stash_.begin(), weight_stash_.end(),
                         [](const auto &stash) { return !stash.empty(); });
}

void OneFOneBSchedule::StashWeights(int microbatch_id) {
    auto &stash = weight_stash_[microbatch_id];
    stash.clear();
    // Keyed by StateDict()'s dotted parameter name, NOT by position in
    // Parameters()'s vector: Parameters() walks unordered_maps, so a
    // positional index is only meaningful if nothing between Stash and
    // Restore ever rebuilds those maps (e.g. Module::To() does exactly that).
    // Name-keying makes this correct regardless of map iteration order.
    std::unordered_map<const Tensor *, std::shared_ptr<Tensor>> unique_snapshots;
    for (const auto &[name, param] : stage_model_->StateDict()) {
        auto [it, inserted] = unique_snapshots.emplace(param.get(), nullptr);
        if (inserted) it->second = param->Clone();
        stash[name] = it->second;
    }
}

void OneFOneBSchedule::RestoreAndEvictWeights(int microbatch_id) {
    auto &stash = weight_stash_[microbatch_id];
    CHECK(!stash.empty()) << "No stashed parameter version for micro-batch " << microbatch_id;
    for (const auto &[name, param] : stage_model_->StateDict()) {
        const auto it = stash.find(name);
        if (it == stash.end()) {
            continue;
        }
        CopyInto(param, it->second);
    }
    stash.clear();
}

void OneFOneBSchedule::Run(DataBatch batch) {
    CHECK(!interleaved_) << "OneFOneBSchedule with interleaved virtual pipeline stages "
                         << "(num_virtual_stages > 1) is not implemented: each physical stage "
                         << "would need to own multiple non-contiguous layer ranges, which "
                         << "GPT2's single stage_layer_start_/stage_layer_end_ range doesn't "
                         << "support. Only the non-interleaved (num_virtual_stages == 1) "
                         << "schedule is implemented.";

    const int pp_rank = GetPPRank();
    const bool is_first_stage = pp_rank == 0;
    const bool is_last_stage = pp_rank == num_stages_ - 1;
    auto *pp_group = GetPPGroup();
    CHECK(pp_group) << "OneFOneBSchedule requires InitParallelGroups() to have been called";

    const auto &config = stage_model_->Config();
    CHECK_EQ(num_stages_, config.pp_size) << "OneFOneBSchedule's num_stages_ must match GPT2Config::pp_size";
    CHECK_EQ(is_first_stage, config.pre_process)
        << "OneFOneBSchedule's is_first_stage (pp_rank==0) must match GPT2Config::pre_process";
    CHECK_EQ(is_last_stage, config.post_process)
        << "OneFOneBSchedule's is_last_stage (pp_rank==num_stages_-1) must match GPT2Config::post_process";

    nn::CrossEntropyLoss loss_fn;

    const int M = num_microbatches_;
    CHECK_EQ(StashedVersionCount(), 0u) << "A previous 1F1B run left unevicted weight versions";
    optimizer_steps_ = 0;
    CHECK_GE(batch.inputs.size(), static_cast<size_t>(M)) << "DataBatch.inputs has fewer entries than num_microbatches_";
    CHECK_GE(batch.targets.size(), static_cast<size_t>(M))
        << "DataBatch.targets has fewer entries than num_microbatches_";
    profiler_.Start(batch.inputs[0]->GetDevice().Index());
    // Clamped to M: with fewer micro-batches than pipeline slots, every
    // micro-batch is forward-only-then-immediately-backward (degenerates
    // toward GPipe's shape), not an out-of-range/negative warmup count.
    const int warmup_steps = std::min(num_stages_ - pp_rank - 1, M);

    std::vector<std::shared_ptr<Tensor>> activations(M);
    std::vector<std::shared_ptr<Tensor>> outputs(M);
    std::vector<std::shared_ptr<Tensor>> losses(M);

    auto receive_activation = [&](int m) {
        if (is_first_stage) return batch.inputs[m];
        const auto &tok_dims = batch.inputs[m]->Dims();
        auto activation = RecvLike(pp_group, pp_rank - 1, {tok_dims[0], tok_dims[1], config.n_embd}, config.dtype,
                                   batch.inputs[m]->GetDevice());
        activation->RequiresGrad();
        return activation;
    };

    auto forward_compute = [&](int m, const std::shared_ptr<Tensor> &activation) {
        if (optimizer_) StashWeights(m);
        activations[m] = activation;
        const cudaEvent_t compute_start = profiler_.BeginCompute();
        outputs[m] = stage_model_->Forward({activation})[0];
        if (is_last_stage) losses[m] = loss_fn.Forward({outputs[m], batch.targets[m]})[0];
        profiler_.EndCompute(compute_start);
        profiler_.ActivationForwarded();
    };

    auto backward_compute = [&](int m, const std::shared_ptr<Tensor> &grad) {
        if (optimizer_) {
            RestoreAndEvictWeights(m);
            optimizer_->ZeroGrad();
        }
        const cudaEvent_t compute_start = profiler_.BeginCompute();
        if (is_last_stage) losses[m]->Backward();
        else outputs[m]->Backward(grad);
        profiler_.EndCompute(compute_start);
        profiler_.ActivationBackwarded();
    };

    auto finish_backward = [&] {
        if (!optimizer_) return;
        stage_model_->SynchronizeTiedEmbeddingGradients();
        optimizer_->Step();
        ++optimizer_steps_;
    };

    // Warmup: fill this stage's downstream distance with forward-only work.
    for (int m = 0; m < warmup_steps; ++m) {
        auto activation = receive_activation(m);
        forward_compute(m, activation);
        if (!is_last_stage) SendTensor(pp_group, pp_rank + 1, outputs[m]);
    }

    const int steady_steps = M - warmup_steps;
    std::shared_ptr<Tensor> next_activation;
    if (steady_steps > 0 && !is_first_stage) next_activation = receive_activation(warmup_steps);

    // Steady state. The two directions crossing each pipeline link are issued
    // in one NCCL group: send forward F(m) while receiving backward dF(b),
    // then send dInput(b) while receiving the next forward activation. Plain
    // send-then-recv calls on one stream form a circular wait here.
    for (int i = 0; i < steady_steps; ++i) {
        const int forward_id = warmup_steps + i;
        const int backward_id = i;
        auto activation = is_first_stage ? batch.inputs[forward_id] : next_activation;
        CHECK(activation);
        forward_compute(forward_id, activation);

        std::shared_ptr<Tensor> grad;
        if (!is_last_stage) {
            grad = SendRecvLike(pp_group, pp_rank + 1, outputs[forward_id], outputs[backward_id]->Dims(),
                                outputs[backward_id]->Dtype(), outputs[backward_id]->GetDevice());
        }
        backward_compute(backward_id, grad);

        if (!is_first_stage) {
            auto input_grad = activations[backward_id]->grad();
            CHECK(input_grad);
            if (i + 1 < steady_steps) {
                const auto &next_dims = batch.inputs[forward_id + 1]->Dims();
                next_activation = SendRecvLike(pp_group, pp_rank - 1, input_grad,
                                               {next_dims[0], next_dims[1], config.n_embd}, config.dtype,
                                               batch.inputs[forward_id + 1]->GetDevice());
                next_activation->RequiresGrad();
            } else {
                SendTensor(pp_group, pp_rank - 1, input_grad);
            }
        }
        finish_backward();
    }

    // Cooldown: only the warmup forwards remain, so drain their gradients in
    // order without scheduling another forward receive.
    for (int backward_id = steady_steps; backward_id < M; ++backward_id) {
        std::shared_ptr<Tensor> grad;
        if (!is_last_stage) {
            grad = RecvLike(pp_group, pp_rank + 1, outputs[backward_id]->Dims(), outputs[backward_id]->Dtype(),
                            outputs[backward_id]->GetDevice());
        }
        backward_compute(backward_id, grad);
        if (!is_first_stage) SendTensor(pp_group, pp_rank - 1, activations[backward_id]->grad());
        finish_backward();
    }

    if (!optimizer_) stage_model_->SynchronizeTiedEmbeddingGradients();

    CHECK_EQ(StashedVersionCount(), 0u) << "1F1B did not evict every completed weight version";
    profiler_.Finish();
    // If optimizer_ is null, the caller performs one accumulated optimizer
    // step after Run(). If supplied, this is PipeDream mode and all M updates
    // have already happened inside their matching backward steps.
}

} // namespace infini_train::distributed
#endif // USE_NCCL
