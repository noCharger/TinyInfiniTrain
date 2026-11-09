#include "infini_train/include/optimizer.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "glog/logging.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "infini_train/include/cuda_utils.h"
#endif
#ifdef USE_NCCL
#include "infini_train/include/distributed/nccl_utils.h"
#include "infini_train/include/distributed/parallel_state.h"
#endif

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params) : params_(params) {}

void Optimizer::ZeroGrad() {
    for (auto param : params_) { param->ZeroGrad(); }
}

namespace optimizers {

namespace {

std::shared_ptr<Tensor> CastWithoutAutograd(const std::shared_ptr<Tensor> &tensor, DataType dtype) {
    auto kernel = Dispatcher::Instance().GetKernel({tensor->GetDevice().Type(), "CastForward"});
    return kernel.Call<std::shared_ptr<Tensor>>(tensor, dtype);
}

void CopyTensorData(const std::shared_ptr<Tensor> &dst, const std::shared_ptr<Tensor> &src) {
    CHECK(dst->Dims() == src->Dims());
    CHECK_EQ(static_cast<int>(dst->Dtype()), static_cast<int>(src->Dtype()));
    CHECK_EQ(dst->GetDevice(), src->GetDevice());
    switch (dst->GetDevice().Type()) {
    case DeviceType::kCPU:
        std::memcpy(dst->DataPtr(), src->DataPtr(), dst->SizeInBytes());
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        cudaMemcpyAsync(dst->DataPtr(), src->DataPtr(), dst->SizeInBytes(), cudaMemcpyDeviceToDevice,
                        infini_train::ThreadLocalStream(dst->GetDevice().Index()));
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(dst->GetDevice().Type());
    }
}

#ifdef USE_NCCL
// Copies count_elems elements of src (starting at src_offset_elems) into dst
// (starting at dst_offset_elems). dst and src may have different total sizes
// (e.g. a small shard vs. a padded scratch buffer) — only the requested range
// is touched, so callers must ensure both ranges fit within their tensor.
void CopyRange(const std::shared_ptr<Tensor> &dst, size_t dst_offset_elems, const std::shared_ptr<Tensor> &src,
               size_t src_offset_elems, size_t count_elems) {
    if (count_elems == 0) {
        return;
    }
    const size_t elem_size = dst->SizeInBytes() / dst->NumElements();
    const size_t nbytes = count_elems * elem_size;
    auto *dst_ptr = static_cast<uint8_t *>(dst->DataPtr()) + dst_offset_elems * elem_size;
    const auto *src_ptr = static_cast<const uint8_t *>(src->DataPtr()) + src_offset_elems * elem_size;
    switch (dst->GetDevice().Type()) {
    case DeviceType::kCPU:
        std::memcpy(dst_ptr, src_ptr, nbytes);
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, cudaMemcpyDeviceToDevice,
                        infini_train::ThreadLocalStream(dst->GetDevice().Index()));
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(dst->GetDevice().Type());
    }
}

// How many of this rank's shard_numel slots fall within the real (unpadded)
// [0, numel) range. shard_numel is a ceil()'d, uniform-across-ranks shard
// size, so the last rank's shard commonly straddles the numel boundary when
// numel isn't evenly divisible by dp_size.
size_t ShardValidCount(size_t numel, size_t dp_rank, size_t shard_numel) {
    const size_t start = dp_rank * shard_numel;
    return start >= numel ? 0 : std::min(shard_numel, numel - start);
}
#endif // USE_NCCL

} // namespace

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate)
    : Optimizer(params), learning_rate_(learning_rate) {}

void SGD::Step() {
    for (auto param : params_) {
        auto device = param->GetDevice().Type();
        auto kernel = Dispatcher::Instance().GetKernel({device, "AccumulateGrad"});
        kernel.Call<void>(param->grad(), -learning_rate_, param);
    }
}

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate,
           float beta1, float beta2, float eps, ZeroStage zero_stage)
    : Optimizer(params), t_(0), learning_rate_(learning_rate),
      beta1_(beta1), beta2_(beta2), eps_(eps), zero_stage_(zero_stage) {

    for (const auto &param : params_) {
        if (zero_stage_ == ZeroStage::kNone) {
            // Original path: full m/v on every rank.
            m_.emplace_back(std::make_shared<Tensor>(param->Dims(), DataType::kFLOAT32, param->GetDevice()));
            v_.emplace_back(std::make_shared<Tensor>(param->Dims(), DataType::kFLOAT32, param->GetDevice()));
            m_.back()->Fill<float>(0.0f);
            v_.back()->Fill<float>(0.0f);
        } else {
            // ZeRO-1/2/3: each DP rank owns a 1/DP shard of m and v.
            // Shard size = ceil(numel / dp_world_size) elements.
#ifdef USE_NCCL
            const int dp_size = distributed::GetDPGroup()
                                     ? distributed::GetDPGroup()->WorldSize()
                                     : 1;
#else
            const int dp_size = 1;
#endif
            const size_t numel = param->NumElements();
            const size_t shard_numel = (numel + dp_size - 1) / dp_size;
            // 1-D shard shape — this is the whole point of ZeRO's memory saving:
            // each rank only ever allocates 1/dp_size of the optimizer state,
            // not the full param->Dims(). The trailing padding slots (when numel
            // isn't evenly divisible by dp_size) always see zero gradient in
            // Step(), so they stay inert at their zero-init value forever.
            m_shard_.emplace_back(
                std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(shard_numel)}, DataType::kFLOAT32,
                                         param->GetDevice()));
            v_shard_.emplace_back(
                std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(shard_numel)}, DataType::kFLOAT32,
                                         param->GetDevice()));
            m_shard_.back()->Fill<float>(0.0f);
            v_shard_.back()->Fill<float>(0.0f);
        }
        if (zero_stage_ == ZeroStage::kZero3) {
#ifdef USE_NCCL
            auto *dp_group = distributed::GetDPGroup();
            CHECK(dp_group) << "ZeRO-3 requires an initialized DP process group";
            const size_t shard_numel = m_shard_.back()->NumElements();
            const size_t valid = ShardValidCount(param->NumElements(), dp_group->Rank(), shard_numel);
            auto shard = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(shard_numel)},
                                                  param->Dtype(), param->GetDevice());
            shard->Fill<float>(0.0f);
            CopyRange(shard, 0, param, dp_group->Rank() * shard_numel, valid);
            param_shard_.push_back(std::move(shard));
#else
            LOG(FATAL) << "ZeRO-3 requires building with USE_NCCL";
#endif
        } else {
            param_shard_.push_back(nullptr);
        }
        if (zero_stage_ != ZeroStage::kNone && param->Dtype() != DataType::kFLOAT32) {
#ifdef USE_NCCL
            CHECK(param->Dtype() == DataType::kFLOAT16 || param->Dtype() == DataType::kBFLOAT16)
                << "NCCL ZeRO master shards support FP16 and BF16 model parameters";
            auto *dp_group = distributed::GetDPGroup();
            CHECK(dp_group);
            std::shared_ptr<Tensor> model_shard = param_shard_.back();
            if (!model_shard) {
                const size_t shard_numel = m_shard_.back()->NumElements();
                const size_t valid = ShardValidCount(param->NumElements(), dp_group->Rank(), shard_numel);
                model_shard = std::make_shared<Tensor>(
                    std::vector<int64_t>{static_cast<int64_t>(shard_numel)}, param->Dtype(), param->GetDevice());
                model_shard->Fill<float>(0.0f);
                CopyRange(model_shard, 0, param, dp_group->Rank() * shard_numel, valid);
            }
            param_master_shard_.push_back(CastWithoutAutograd(model_shard, DataType::kFLOAT32));
#else
            LOG(FATAL) << "ZeRO mixed precision requires building with USE_NCCL";
#endif
        } else {
            param_master_shard_.push_back(nullptr);
        }
        if (zero_stage_ == ZeroStage::kNone && param->Dtype() != DataType::kFLOAT32) {
            CHECK(param->Dtype() == DataType::kFLOAT8_E4M3 || param->Dtype() == DataType::kFLOAT16
                  || param->Dtype() == DataType::kBFLOAT16)
                << "Adam master weights support FP8, FP16, and BF16 model parameters";
            params_fp32_.emplace_back(CastWithoutAutograd(param, DataType::kFLOAT32));
        } else {
            params_fp32_.emplace_back(nullptr);
        }
    }
}

void Adam::Step() {
    ++t_;

    if (zero_stage_ == ZeroStage::kNone) {
        // ── Original path ────────────────────────────────────────────────────
        for (size_t i = 0; i < params_.size(); ++i) {
            auto &param = params_[i];
            const auto &grad = param->grad();
            if (!grad) continue;
            auto device = param->GetDevice().Type();
            auto update_param = params_fp32_[i] ? params_fp32_[i] : param;
            Dispatcher::Instance()
                .GetKernel({device, "AdamAccumulateGrad"})
                .Call<void>(grad, update_param, m_[i], v_[i], learning_rate_, beta1_, beta2_, eps_, t_);
            if (params_fp32_[i]) {
                CopyTensorData(param, CastWithoutAutograd(update_param, param->Dtype()));
            }
        }
        return;
    }

    // ── ZeRO paths ───────────────────────────────────────────────────────────
#ifdef USE_NCCL
    auto *dp_group = distributed::GetDPGroup();
    CHECK(dp_group) << "ZeRO requires an initialized DP process group";
    CHECK(zero_stage_ != ZeroStage::kZero3 || parameters_materialized_)
        << "ZeRO-3 Step() requires MaterializeParameters() before forward/backward";
    const size_t dp_size = dp_group->WorldSize();
    const size_t dp_rank = dp_group->Rank();

    for (size_t i = 0; i < params_.size(); ++i) {
        auto &param = params_[i];
        const auto &grad = param->grad();
        if (!grad) continue;
        const auto device = param->GetDevice().Type();
        const auto dtype = param->Dtype();
        const auto nccl_dtype = distributed::ToNcclDataType(dtype);

        const size_t numel = param->NumElements();
        const size_t shard_numel = m_shard_[i]->NumElements();
        // shard_numel was fixed at Adam construction time from whatever
        // dp_size was live then; dp_size above is re-read fresh every Step().
        // If the DP group was ever destroyed/reinitialized at a different
        // world size in between, CopyRange below (which has no bounds
        // checking of its own) would read/write past grad_padded/param_shard's
        // actual allocation — catch the mismatch here with a clear diagnostic
        // instead of silently corrupting memory.
        CHECK_EQ(shard_numel, (numel + dp_size - 1) / dp_size)
            << "DP process group world size changed between Adam construction and this Step() call";

        // AdamAccumulateGrad has zero shape validation between its four tensor
        // args (it trusts grad->NumElements() for the loop bound and indexes
        // param/m/v the same way) — grad_shard/param_shard MUST therefore be
        // exactly shard_numel-sized, matching m_shard_[i]/v_shard_[i].
        std::shared_ptr<Tensor> grad_shard;
        if (zero_stage_ >= ZeroStage::kZero2) {
            // ZeRO-2/3: reduce-scatter the gradient. ReduceScatter requires a
            // send buffer of exactly shard_numel * dp_size elements on every
            // rank, but grad only has `numel` (<= shard_numel * dp_size when
            // numel isn't evenly divisible) — pad with zeros first; padding
            // contributes nothing to the sum.
            auto grad_padded = std::make_shared<Tensor>(
                std::vector<int64_t>{static_cast<int64_t>(shard_numel * dp_size)}, dtype, param->GetDevice());
            grad_padded->Fill<float>(0.0f);
            CopyRange(grad_padded, 0, grad, 0, numel);

            grad_shard = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(shard_numel)}, dtype,
                                                  param->GetDevice());
            // ncclAvg (not ncclSum): each DP rank computed its gradient from a
            // DIFFERENT local mini-batch of the same global batch, so the
            // optimizer must see the AVERAGE across ranks, matching standard
            // DDP semantics — ncclSum would feed the optimizer a gradient
            // dp_size times too large.
            dp_group->ReduceScatter(grad_padded->DataPtr(), grad_shard->DataPtr(), shard_numel, nccl_dtype, ncclAvg);
        } else {
            // ZeRO-1: full AllReduce gradient first (same as DDP — averaged,
            // not summed, across ranks), then each rank slices out only its
            // own shard for the optimizer update.
            dp_group->AllReduce(grad->DataPtr(), grad->NumElements(), nccl_dtype, ncclAvg);

            grad_shard = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(shard_numel)}, dtype,
                                                  param->GetDevice());
            grad_shard->Fill<float>(0.0f);
            const size_t valid = ShardValidCount(numel, dp_rank, shard_numel);
            CopyRange(grad_shard, 0, grad, dp_rank * shard_numel, valid);
        }

        const size_t valid = ShardValidCount(numel, dp_rank, shard_numel);
        std::shared_ptr<Tensor> param_shard;
        if (zero_stage_ == ZeroStage::kZero3) {
            param_shard = param_shard_[i];
        } else {
            // ZeRO-1/2 materialize this rank's update shard from the resident
            // full tensor. ZeRO-3 already owns a persistent local shard.
            param_shard = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(shard_numel)}, dtype,
                                                   param->GetDevice());
            param_shard->Fill<float>(0.0f);
            CopyRange(param_shard, 0, param, dp_rank * shard_numel, valid);
        }

        // Run Adam on this rank's shard only — this is ZeRO's memory saving:
        // every rank updates 1/dp_size of the optimizer state and parameters.
        auto update_param = param_master_shard_[i] ? param_master_shard_[i] : param_shard;
        Dispatcher::Instance()
            .GetKernel({device, "AdamAccumulateGrad"})
            .Call<void>(grad_shard, update_param, m_shard_[i], v_shard_[i], learning_rate_, beta1_, beta2_, eps_, t_);
        if (param_master_shard_[i]) {
            CopyTensorData(param_shard, CastWithoutAutograd(update_param, dtype));
        }

        if (zero_stage_ != ZeroStage::kZero3) {
            // ZeRO-1/2 restore the full tensor immediately. ZeRO-3 leaves only
            // param_shard resident and gathers immediately before next forward.
            auto param_gathered = std::make_shared<Tensor>(
                std::vector<int64_t>{static_cast<int64_t>(shard_numel * dp_size)}, dtype, param->GetDevice());
            dp_group->AllGather(param_shard->DataPtr(), param_gathered->DataPtr(), shard_numel, nccl_dtype);
            CopyRange(param, 0, param_gathered, 0, numel);
        }
        // ZeRO optimizer steps retain only the local gradient shard while the
        // update is in flight; the full autograd gradient is no longer needed
        // after this parameter has been reduced and updated.
        param->ReleaseGrad();
    }
    if (zero_stage_ == ZeroStage::kZero3) PartitionParameters();
#else
    LOG(FATAL) << "ZeroStage != kNone requires building with USE_NCCL";
#endif // USE_NCCL
}

void Adam::MaterializeParameters() {
    if (zero_stage_ != ZeroStage::kZero3 || parameters_materialized_) return;
#ifdef USE_NCCL
    auto *dp_group = distributed::GetDPGroup();
    CHECK(dp_group) << "ZeRO-3 requires an initialized DP process group";
    for (size_t i = 0; i < params_.size(); ++i) {
        auto &param = params_[i];
        const auto &shard = param_shard_[i];
        const size_t shard_numel = shard->NumElements();
        auto gathered = std::make_shared<Tensor>(
            std::vector<int64_t>{static_cast<int64_t>(shard_numel * dp_group->WorldSize())}, param->Dtype(),
            param->GetDevice());
        dp_group->AllGather(shard->DataPtr(), gathered->DataPtr(), shard_numel,
                            distributed::ToNcclDataType(param->Dtype()));
        param->AllocateStorage();
        CopyRange(param, 0, gathered, 0, param->NumElements());
    }
    parameters_materialized_ = true;
#else
    LOG(FATAL) << "ZeRO-3 requires building with USE_NCCL";
#endif
}

void Adam::PartitionParameters() {
    if (zero_stage_ != ZeroStage::kZero3 || !parameters_materialized_) return;
    for (auto &param : params_) {
        param->ReleaseGrad();
        param->ReleaseStorage();
    }
    parameters_materialized_ = false;
}

size_t Adam::FullParameterBytes() const {
    size_t bytes = 0;
    for (const auto &param : params_) bytes += param->SizeInBytes();
    return bytes;
}

size_t Adam::LocalParameterShardBytes() const {
    size_t bytes = 0;
    for (const auto &shard : param_shard_) {
        if (shard) bytes += shard->SizeInBytes();
    }
    return bytes;
}
} // namespace optimizers
} // namespace infini_train
