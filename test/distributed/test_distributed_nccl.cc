#if defined(USE_CUDA) && defined(USE_NCCL)

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "example/gpt2/net.h"
#include "infini_train/include/autograd/linear_parallel.h"
#include "infini_train/include/cuda_utils.h"
#include "infini_train/include/device.h"
#include "infini_train/include/distributed/nccl_utils.h"
#include "infini_train/include/distributed/parallel_state.h"
#include "infini_train/include/distributed/pipeline_schedule.h"
#include "infini_train/include/dtype.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {
int EnvironmentInt(const char *name) {
    const char *value = std::getenv(name);
    CHECK(value) << "Missing environment variable " << name;
    return std::stoi(value);
}

std::string EnvironmentString(const char *name) {
    const char *value = std::getenv(name);
    CHECK(value) << "Missing environment variable " << name;
    return value;
}

Device CudaDevice() { return Device(DeviceType::kCUDA, EnvironmentInt("LOCAL_RANK")); }

void CopyFromHost(const std::shared_ptr<Tensor> &tensor, const std::vector<float> &values) {
    ASSERT_EQ(tensor->Dtype(), DataType::kFLOAT32);
    ASSERT_EQ(tensor->NumElements(), values.size());
    const cudaStream_t stream = ThreadLocalStream(tensor->GetDevice().Index());
    ASSERT_EQ(cudaMemcpyAsync(tensor->DataPtr(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice,
                              stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}

std::vector<float> CopyToHost(const std::shared_ptr<Tensor> &tensor) {
    auto host = tensor->To(Device());
    EXPECT_EQ(cudaStreamSynchronize(ThreadLocalStream(tensor->GetDevice().Index())), cudaSuccess);
    std::vector<float> values(tensor->NumElements());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = dtype::LoadAsFloat(host.DataPtr(), host.Dtype(), i);
    }
    return values;
}

std::shared_ptr<Tensor> TokensOnCuda(const std::vector<int64_t> &values, int64_t batch, int64_t sequence) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{batch, sequence}, DataType::kINT64, CudaDevice());
    const cudaStream_t stream = ThreadLocalStream(tensor->GetDevice().Index());
    EXPECT_EQ(cudaMemcpyAsync(tensor->DataPtr(), values.data(), values.size() * sizeof(int64_t), cudaMemcpyHostToDevice,
                              stream),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    return tensor;
}

struct ParallelGroupsGuard {
    ~ParallelGroupsGuard() { distributed::DestroyParallelGroups(); }
};

void RunTensorParallelTest(int rank, int world_size) {
    ASSERT_EQ(world_size, 2);
    distributed::InitParallelGroups(rank, world_size, 2, 1, 1);
    ParallelGroupsGuard guard;
    const Device device = CudaDevice();

    auto collective = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, device);
    CopyFromHost(collective, {static_cast<float>(rank + 1)});
    distributed::GetTPGroup()->AllReduce(collective->DataPtr(), 1, ncclFloat32, ncclSum);
    EXPECT_FLOAT_EQ(CopyToHost(collective)[0], 3.0f);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 2}, DataType::kFLOAT32, device)->RequiresGrad();
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 2}, DataType::kFLOAT32, device)->RequiresGrad();
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, device)->RequiresGrad();
    CopyFromHost(input, {1.0f, 2.0f});
    CopyFromHost(weight, rank == 0 ? std::vector<float>{1.0f, 0.0f} : std::vector<float>{0.0f, 1.0f});
    bias->Fill<float>(0.0f);
    auto gathered = std::make_shared<autograd::ColumnParallelLinear>(true)->Apply({input, weight, bias})[0];
    const auto gathered_values = CopyToHost(gathered);
    ASSERT_EQ(gathered_values.size(), 2u);
    EXPECT_FLOAT_EQ(gathered_values[0], 1.0f);
    EXPECT_FLOAT_EQ(gathered_values[1], 2.0f);
    gathered->Backward();
    const auto input_grad = CopyToHost(input->grad());
    EXPECT_FLOAT_EQ(input_grad[0], 1.0f);
    EXPECT_FLOAT_EQ(input_grad[1], 1.0f);

    auto input_shard = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, DataType::kFLOAT32, device)->RequiresGrad();
    auto row_weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, DataType::kFLOAT32, device)->RequiresGrad();
    auto row_bias = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, device)->RequiresGrad();
    CopyFromHost(input_shard, {rank == 0 ? 1.0f : 2.0f});
    CopyFromHost(row_weight, {rank == 0 ? 3.0f : 4.0f});
    CopyFromHost(row_bias, {5.0f});
    auto reduced = std::make_shared<autograd::RowParallelLinear>()->Apply({input_shard, row_weight, row_bias})[0];
    EXPECT_FLOAT_EQ(CopyToHost(reduced)[0], 16.0f);
    reduced->Backward();
    EXPECT_FLOAT_EQ(CopyToHost(input_shard->grad())[0], rank == 0 ? 3.0f : 4.0f);
    EXPECT_FLOAT_EQ(CopyToHost(row_bias->grad())[0], rank == 0 ? 1.0f : 0.0f);
}

void RunZeroTest(int rank, int world_size, ZeroStage stage) {
    ASSERT_EQ(world_size, 2);
    distributed::InitParallelGroups(rank, world_size, 1, 1, 2);
    ParallelGroupsGuard guard;
    auto parameter = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, CudaDevice())->RequiresGrad();
    CopyFromHost(parameter, {1.0f, 2.0f, 3.0f, 4.0f});
    parameter->grad()->Fill<float>(rank == 0 ? 1.0f : 3.0f);
    optimizers::Adam optimizer({parameter}, 0.1f, 0.9f, 0.999f, 1e-8f, stage);
    optimizer.Step();
    EXPECT_EQ(parameter->grad(), nullptr);
    if (stage == ZeroStage::kZero3) {
        EXPECT_FALSE(optimizer.ParametersMaterialized());
        EXPECT_FALSE(parameter->HasStorage());
        EXPECT_LT(optimizer.LocalParameterShardBytes(), optimizer.FullParameterBytes());
        optimizer.MaterializeParameters();
        EXPECT_TRUE(optimizer.ParametersMaterialized());
        EXPECT_TRUE(parameter->HasStorage());
    }
    const auto values = CopyToHost(parameter);
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(values[i], static_cast<float>(i + 1) - 0.1f, 1e-5f);
    }
}

void RunZeroBF16Test(int rank, int world_size) {
    ASSERT_EQ(world_size, 2);
    distributed::InitParallelGroups(rank, world_size, 1, 1, 2);
    ParallelGroupsGuard guard;
    auto parameter = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kBFLOAT16, CudaDevice())
                         ->RequiresGrad();
    parameter->Fill<float>(1.0f);
    parameter->grad()->Fill<float>(rank == 0 ? 1.0f : 3.0f);
    optimizers::Adam optimizer({parameter}, 0.001f, 0.9f, 0.999f, 1e-8f, ZeroStage::kZero2);
    for (int step = 0; step < 8; ++step) {
        if (!parameter->grad()) parameter->RequiresGrad();
        parameter->grad()->Fill<float>(rank == 0 ? 1.0f : 3.0f);
        optimizer.Step();
    }
    for (float value : CopyToHost(parameter)) {
        EXPECT_LT(value, 1.0f);
        EXPECT_NEAR(value, 0.992f, 0.004f);
    }
}

bool HasNonzeroGradient(GPT2 &model) {
    for (const auto &parameter : model.Parameters()) {
        if (!parameter->grad()) continue;
        for (float value : CopyToHost(parameter->grad())) {
            if (std::abs(value) > 1e-8f) return true;
        }
    }
    return false;
}

void ExpectReplicatedAcrossTP(const std::shared_ptr<Tensor> &tensor) {
    const auto local = CopyToHost(tensor);
    auto sum = tensor->Clone();
    auto *group = distributed::GetTPGroup();
    ASSERT_NE(group, nullptr);
    group->AllReduce(sum->DataPtr(), sum->NumElements(), distributed::ToNcclDataType(sum->Dtype()), ncclSum);
    const auto global_sum = CopyToHost(sum);
    ASSERT_EQ(local.size(), global_sum.size());
    for (size_t i = 0; i < local.size(); ++i) {
        EXPECT_NEAR(local[i], global_sum[i] / group->WorldSize(), 1e-5f) << "at flat index " << i;
    }
}

void RunTensorParallelGPT2Test(int rank, int world_size) {
    ASSERT_EQ(world_size, 2);
    distributed::InitParallelGroups(rank, world_size, 2, 1, 1);
    ParallelGroupsGuard guard;

    GPT2 model(GPT2Config{
        .block_size = 4,
        .vocab_size = 16,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 8,
        .dtype = DataType::kFLOAT32,
        .tp_size = 2,
    });
    model.To(CudaDevice());
    const auto state = model.StateDict();
    ASSERT_TRUE(state.contains("transformer.wte.weight"));
    ASSERT_TRUE(state.contains("lm_head.weight"));
    EXPECT_EQ(state.at("transformer.wte.weight").get(), state.at("lm_head.weight").get());

    auto tokens = TokensOnCuda({0, 1, 2, 3}, 1, 4);
    auto targets = TokensOnCuda({1, 2, 3, 4}, 1, 4);
    auto logits = model.Forward({tokens})[0];
    EXPECT_EQ(logits->Dims(), std::vector<int64_t>({1, 4, 16}));
    ExpectReplicatedAcrossTP(logits);

    nn::CrossEntropyLoss loss_fn;
    auto loss = loss_fn.Forward({logits, targets})[0];
    ASSERT_TRUE(std::isfinite(CopyToHost(loss)[0]));
    loss->Backward();
    EXPECT_TRUE(HasNonzeroGradient(model));

    optimizers::Adam optimizer(model.Parameters(), 1e-3f);
    optimizer.Step();
    optimizer.ZeroGrad();
    auto logits_after_step = model.Forward({tokens})[0];
    ExpectReplicatedAcrossTP(logits_after_step);
}

void ExpectPipelineTiedWeightEqual(GPT2 &model, int rank) {
    const auto state = model.StateDict();
    const auto key = rank == 0 ? "transformer.wte.weight" : "lm_head.weight";
    ASSERT_TRUE(state.contains(key));
    float checksum = 0.0f;
    for (float value : CopyToHost(state.at(key))) checksum += value;
    auto reduced = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, CudaDevice());
    CopyFromHost(reduced, {checksum});
    distributed::GetPPCollectiveGroup()->AllReduce(reduced->DataPtr(), 1, ncclFloat32, ncclSum);
    EXPECT_NEAR(CopyToHost(reduced)[0] / 2.0f, checksum, 1e-5f);
}

distributed::DataBatch MakePipelineBatch(int microbatches) {
    distributed::DataBatch batch;
    for (int microbatch = 0; microbatch < microbatches; ++microbatch) {
        std::vector<int64_t> tokens(4), targets(4);
        for (int i = 0; i < 4; ++i) {
            tokens[i] = (microbatch * 4 + i) % 16;
            targets[i] = (tokens[i] + 1) % 16;
        }
        batch.inputs.push_back(TokensOnCuda(tokens, 1, 4));
        batch.targets.push_back(TokensOnCuda(targets, 1, 4));
    }
    return batch;
}

std::unique_ptr<GPT2> MakePipelineStage(int rank, bool activation_recomputation) {
    auto model = std::make_unique<GPT2>(GPT2Config{
        .block_size = 4,
        .vocab_size = 16,
        .n_layer = 2,
        .n_head = 2,
        .n_embd = 8,
        .dtype = DataType::kFLOAT32,
        .activation_recomputation = activation_recomputation,
        .pp_size = 2,
        .pre_process = rank == 0,
        .post_process = rank == 1,
    });
    model->To(CudaDevice());
    return model;
}

void RunPipelineTest(int rank, int world_size) {
    ASSERT_EQ(world_size, 2);
    distributed::InitParallelGroups(rank, world_size, 1, 2, 1);
    ParallelGroupsGuard guard;
    constexpr int kMicrobatches = 4;
    {
        auto model = MakePipelineStage(rank, /*activation_recomputation=*/false);
        ExpectPipelineTiedWeightEqual(*model, rank);
        distributed::GPipeSchedule schedule(2, kMicrobatches, model.get());
        schedule.Run(MakePipelineBatch(kMicrobatches));
        ASSERT_EQ(cudaStreamSynchronize(ThreadLocalStream(CudaDevice().Index())), cudaSuccess);
        EXPECT_TRUE(HasNonzeroGradient(*model));
        EXPECT_GT(schedule.Metrics().elapsed_ms, 0.0f);
        EXPECT_GT(schedule.Metrics().compute_ms, 0.0f);
        EXPECT_EQ(schedule.Metrics().peak_inflight_activations, kMicrobatches);
    }
    {
        // Exercise the hardest checkpoint composition: PipeDream restores the
        // matching stashed weight version before checkpoint backward rebuilds
        // this micro-batch's block graph.
        auto model = MakePipelineStage(rank, /*activation_recomputation=*/true);
        ExpectPipelineTiedWeightEqual(*model, rank);
        const auto parameter_before = CopyToHost(model->Parameters().front());
        optimizers::Adam optimizer(model->Parameters(), 1e-3f);
        distributed::OneFOneBSchedule schedule(2, kMicrobatches, model.get(), &optimizer);
        schedule.Run(MakePipelineBatch(kMicrobatches));
        ASSERT_EQ(cudaStreamSynchronize(ThreadLocalStream(CudaDevice().Index())), cudaSuccess);
        EXPECT_TRUE(HasNonzeroGradient(*model));
        EXPECT_EQ(schedule.OptimizerSteps(), kMicrobatches);
        EXPECT_EQ(schedule.StashedVersionCount(), 0u);
        EXPECT_GT(schedule.Metrics().elapsed_ms, 0.0f);
        EXPECT_GT(schedule.Metrics().compute_ms, 0.0f);
        EXPECT_LE(schedule.Metrics().peak_inflight_activations, 2);
        const auto parameter_after = CopyToHost(model->Parameters().front());
        ASSERT_EQ(parameter_before.size(), parameter_after.size());
        EXPECT_TRUE(std::mismatch(parameter_before.begin(), parameter_before.end(), parameter_after.begin()).first
                    != parameter_before.end());
        ExpectPipelineTiedWeightEqual(*model, rank);
    }
}
} // namespace

TEST(DistributedNcclTest, SelectedMode) {
    const int rank = EnvironmentInt("RANK");
    const int world_size = EnvironmentInt("WORLD_SIZE");
    const std::string mode = EnvironmentString("TINYINFINITRAIN_DISTRIBUTED_TEST");
    if (mode == "tp") return RunTensorParallelTest(rank, world_size);
    if (mode == "tp_gpt2") return RunTensorParallelGPT2Test(rank, world_size);
    if (mode == "zero1") return RunZeroTest(rank, world_size, ZeroStage::kZero1);
    if (mode == "zero2") return RunZeroTest(rank, world_size, ZeroStage::kZero2);
    if (mode == "zero3") return RunZeroTest(rank, world_size, ZeroStage::kZero3);
    if (mode == "zero2_bf16") return RunZeroBF16Test(rank, world_size);
    if (mode == "pipeline") return RunPipelineTest(rank, world_size);
    FAIL() << "Unknown distributed test mode " << mode;
}

#endif
