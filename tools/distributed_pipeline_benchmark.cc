#if defined(USE_CUDA) && defined(USE_NCCL)

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "example/gpt2/net.h"
#include "infini_train/include/cuda_utils.h"
#include "infini_train/include/distributed/nccl_utils.h"
#include "infini_train/include/distributed/parallel_state.h"
#include "infini_train/include/distributed/pipeline_schedule.h"
#include "infini_train/include/tensor.h"

namespace {
using namespace infini_train;

int EnvironmentInt(const char *name) {
    const char *value = std::getenv(name);
    CHECK(value) << "Missing environment variable " << name;
    return std::stoi(value);
}

struct Arguments {
    std::string schedule = "1f1b";
    int microbatches = 4;
    bool activation_recomputation = false;
    int micro_batch_size = 2;
    int sequence_length = 128;
    int layers_per_stage = 2;
    int n_head = 8;
    int n_embd = 256;
    int vocab_size = 512;
};

Arguments ParseArguments(int argc, char **argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        CHECK_LT(i + 1, argc) << "Missing value for " << argument;
        if (argument == "--schedule") args.schedule = argv[++i];
        else if (argument == "--microbatches") args.microbatches = std::stoi(argv[++i]);
        else if (argument == "--activation-recomputation") {
            const std::string value = argv[++i];
            CHECK(value == "off" || value == "on");
            args.activation_recomputation = value == "on";
        } else if (argument == "--micro-batch-size") args.micro_batch_size = std::stoi(argv[++i]);
        else if (argument == "--sequence-length") args.sequence_length = std::stoi(argv[++i]);
        else if (argument == "--layers-per-stage") args.layers_per_stage = std::stoi(argv[++i]);
        else if (argument == "--n-head") args.n_head = std::stoi(argv[++i]);
        else if (argument == "--n-embd") args.n_embd = std::stoi(argv[++i]);
        else if (argument == "--vocab-size") args.vocab_size = std::stoi(argv[++i]);
        else LOG(FATAL) << "Unknown argument " << argument;
    }
    CHECK(args.schedule == "gpipe" || args.schedule == "1f1b");
    CHECK_GT(args.microbatches, 0);
    CHECK_GT(args.micro_batch_size, 0);
    CHECK_GT(args.sequence_length, 0);
    CHECK_GT(args.layers_per_stage, 0);
    CHECK_GT(args.n_head, 0);
    CHECK_GT(args.n_embd, 0);
    CHECK_EQ(args.n_embd % args.n_head, 0);
    CHECK_GT(args.vocab_size, 1);
    return args;
}

Device CudaDevice() { return Device(DeviceType::kCUDA, EnvironmentInt("LOCAL_RANK")); }

template <typename T>
void CopyFromHost(const std::shared_ptr<Tensor> &tensor, const std::vector<T> &values) {
    CHECK_EQ(tensor->NumElements(), values.size());
    const cudaStream_t stream = ThreadLocalStream(tensor->GetDevice().Index());
    CHECK(cudaMemcpyAsync(tensor->DataPtr(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream)
          == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
}

std::shared_ptr<Tensor> TokensOnCuda(const std::vector<int64_t> &values,
                                    int micro_batch_size, int sequence_length) {
    auto tensor = std::make_shared<Tensor>(
        std::vector<int64_t>{micro_batch_size, sequence_length}, DataType::kINT64, CudaDevice());
    CopyFromHost(tensor, values);
    return tensor;
}

distributed::DataBatch MakeBatch(const Arguments &args) {
    distributed::DataBatch batch;
    const int tokens_per_microbatch = args.micro_batch_size * args.sequence_length;
    for (int microbatch = 0; microbatch < args.microbatches; ++microbatch) {
        std::vector<int64_t> tokens(tokens_per_microbatch), targets(tokens_per_microbatch);
        for (int token = 0; token < tokens_per_microbatch; ++token) {
            tokens[token] = (microbatch * tokens_per_microbatch + token) % args.vocab_size;
            targets[token] = (tokens[token] + 1) % args.vocab_size;
        }
        batch.inputs.push_back(TokensOnCuda(tokens, args.micro_batch_size, args.sequence_length));
        batch.targets.push_back(TokensOnCuda(targets, args.micro_batch_size, args.sequence_length));
    }
    return batch;
}

std::unique_ptr<GPT2> MakeStage(int rank, int world_size, const Arguments &args) {
    auto model = std::make_unique<GPT2>(GPT2Config{
        .block_size = args.sequence_length,
        .vocab_size = args.vocab_size,
        .n_layer = world_size * args.layers_per_stage,
        .n_head = args.n_head,
        .n_embd = args.n_embd,
        .dtype = DataType::kFLOAT32,
        .activation_recomputation = args.activation_recomputation,
        .pp_size = world_size,
        .pre_process = rank == 0,
        .post_process = rank == world_size - 1,
    });
    model->To(CudaDevice());
    return model;
}

void Barrier() {
    auto value = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, CudaDevice());
    value->Fill<float>(1.0f);
    distributed::GetPPCollectiveGroup()->AllReduce(value->DataPtr(), 1, ncclFloat32, ncclSum);
}

float ReduceFloat(float value, ncclRedOp_t operation) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, CudaDevice());
    CopyFromHost(tensor, std::vector<float>{value});
    distributed::GetPPCollectiveGroup()->AllReduce(tensor->DataPtr(), 1, ncclFloat32, operation);
    auto host = tensor->To(Device());
    return static_cast<const float *>(host.DataPtr())[0];
}

uint64_t ReduceUint64(uint64_t value, ncclRedOp_t operation) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kUINT64, CudaDevice());
    CopyFromHost(tensor, std::vector<uint64_t>{value});
    distributed::GetPPCollectiveGroup()->AllReduce(tensor->DataPtr(), 1, ncclUint64, operation);
    auto host = tensor->To(Device());
    return static_cast<const uint64_t *>(host.DataPtr())[0];
}

uint64_t MemoryPoolAttribute(cudaMemPool_t pool, cudaMemPoolAttr attribute) {
    uint64_t value = 0;
    CHECK(cudaMemPoolGetAttribute(pool, attribute, &value) == cudaSuccess);
    return value;
}

distributed::ScheduleMetrics RunSchedule(const Arguments &args, GPT2 *model,
                                         distributed::DataBatch batch, int world_size) {
    if (args.schedule == "gpipe") {
        distributed::GPipeSchedule schedule(world_size, args.microbatches, model);
        schedule.Run(std::move(batch));
        return schedule.Metrics();
    }
    distributed::OneFOneBSchedule schedule(world_size, args.microbatches, model);
    schedule.Run(std::move(batch));
    return schedule.Metrics();
}
} // namespace

int main(int argc, char **argv) {
    const Arguments args = ParseArguments(argc, argv);
    const int rank = EnvironmentInt("RANK");
    const int world_size = EnvironmentInt("WORLD_SIZE");
    CHECK_GE(world_size, 2);

    infini_train::distributed::InitParallelGroups(rank, world_size, 1, world_size, 1);
    {
        auto model = MakeStage(rank, world_size, args);
        auto batch = MakeBatch(args);
        Barrier();
        CHECK(cudaStreamSynchronize(infini_train::ThreadLocalStream(CudaDevice().Index())) == cudaSuccess);

        cudaMemPool_t pool = nullptr;
        CHECK(cudaDeviceGetDefaultMemPool(&pool, CudaDevice().Index()) == cudaSuccess);
        const uint64_t baseline_bytes = MemoryPoolAttribute(pool, cudaMemPoolAttrUsedMemCurrent);
        uint64_t reset_high_watermark = 0;
        CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &reset_high_watermark) == cudaSuccess);

        const auto local = RunSchedule(args, model.get(), std::move(batch), world_size);
        const uint64_t peak_bytes = MemoryPoolAttribute(pool, cudaMemPoolAttrUsedMemHigh);
        const float compute_sum_ms = ReduceFloat(local.compute_ms, ncclSum);
        const float elapsed_max_ms = ReduceFloat(local.elapsed_ms, ncclMax);
        const uint64_t baseline_max_bytes = ReduceUint64(baseline_bytes, ncclMax);
        const uint64_t peak_max_bytes = ReduceUint64(peak_bytes, ncclMax);
        CHECK_GE(peak_bytes, baseline_bytes);
        const uint64_t activation_delta_max_bytes = ReduceUint64(peak_bytes - baseline_bytes, ncclMax);
        const float inflight_max = ReduceFloat(static_cast<float>(local.peak_inflight_activations), ncclMax);
        const float utilization = std::clamp(compute_sum_ms / (world_size * elapsed_max_ms), 0.0f, 1.0f);

        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "METRIC schedule=" << args.schedule
                      << " pp=" << world_size
                      << " microbatches=" << args.microbatches
                      << " activation_recomputation="
                      << (args.activation_recomputation ? "on" : "off")
                      << " micro_batch_size=" << args.micro_batch_size
                      << " sequence_length=" << args.sequence_length
                      << " layers_per_stage=" << args.layers_per_stage
                      << " n_head=" << args.n_head
                      << " n_embd=" << args.n_embd
                      << " vocab_size=" << args.vocab_size
                      << " elapsed_ms=" << elapsed_max_ms
                      << " compute_sum_ms=" << compute_sum_ms
                      << " measured_bubble=" << (1.0f - utilization)
                      << " baseline_bytes=" << baseline_max_bytes
                      << " peak_bytes=" << peak_max_bytes
                      << " activation_delta_bytes=" << activation_delta_max_bytes
                      << " peak_inflight=" << static_cast<int>(inflight_max) << '\n';
        }
    }
    infini_train::distributed::DestroyParallelGroups();
    return 0;
}

#endif
