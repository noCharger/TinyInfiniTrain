#ifdef USE_NCCL
#include "infini_train/include/distributed/parallel_state.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

#include "glog/logging.h"
#include <nccl.h>

#define NCCL_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const ncclResult_t result = (call);                                                                            \
        LOG_IF(FATAL, result != ncclSuccess)                                                                           \
            << "NCCL Error: " << ncclGetErrorString(result) << " at " << __FILE__ << ':' << __LINE__;                \
    } while (0)

namespace infini_train::distributed {
namespace {

std::unique_ptr<ProcessGroup> g_tp_group;
std::unique_ptr<ProcessGroup> g_pp_group;
std::unique_ptr<ProcessGroup> g_pp_collective_group;
std::unique_ptr<ProcessGroup> g_dp_group;

std::string EnvironmentOr(const char *name, std::string fallback) {
    const char *value = std::getenv(name);
    return value && *value ? value : std::move(fallback);
}

int EnvironmentIntOr(const char *name, int fallback) {
    const char *value = std::getenv(name);
    return value && *value ? std::stoi(value) : fallback;
}

std::filesystem::path RendezvousPath(const char *axis, int group_id) {
    const auto directory = std::filesystem::path(
        EnvironmentOr("TINYINFINITRAIN_RENDEZVOUS_DIR", std::filesystem::temp_directory_path().string()));
    const std::string run_id
        = EnvironmentOr("TINYINFINITRAIN_RUN_ID", "parent-" + std::to_string(static_cast<long long>(getppid())));
    std::filesystem::create_directories(directory);
    return directory / ("tinyinfinitrain-" + run_id + '-' + axis + '-' + std::to_string(group_id) + ".ncclid");
}

void WriteUniqueIdAtomically(const std::filesystem::path &path, const ncclUniqueId &uid) {
    const auto temporary = path.string() + ".rank0-" + std::to_string(static_cast<long long>(getpid()));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        CHECK(output.is_open()) << "Cannot create NCCL rendezvous file " << temporary;
        output.write(reinterpret_cast<const char *>(&uid), sizeof(uid));
        output.flush();
        CHECK(output.good()) << "Failed writing NCCL rendezvous file " << temporary;
    }
    std::filesystem::rename(temporary, path);
}

ncclUniqueId MakeUniqueId(const char *axis, int group_id, int rank_in_group) {
    const auto path = RendezvousPath(axis, group_id);
    if (rank_in_group == 0) {
        ncclUniqueId uid{};
        NCCL_CHECK(ncclGetUniqueId(&uid));
        WriteUniqueIdAtomically(path, uid);
        return uid;
    }

    const int timeout_seconds = EnvironmentIntOr("TINYINFINITRAIN_RENDEZVOUS_TIMEOUT_SEC", 60);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        std::error_code error;
        if (std::filesystem::file_size(path, error) == sizeof(ncclUniqueId) && !error) {
            ncclUniqueId uid{};
            std::ifstream input(path, std::ios::binary);
            input.read(reinterpret_cast<char *>(&uid), sizeof(uid));
            if (input.gcount() == sizeof(uid)) return uid;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    LOG(FATAL) << "Timed out waiting for NCCL " << axis << " group " << group_id
               << " rendezvous file " << path
               << ". Launch all ranks with the same TINYINFINITRAIN_RUN_ID and TINYINFINITRAIN_RENDEZVOUS_DIR.";
    return {};
}

int ResolveDeviceId(int world_rank) {
    return EnvironmentIntOr("TINYINFINITRAIN_LOCAL_RANK",
                            EnvironmentIntOr("LOCAL_RANK", world_rank));
}

} // namespace

void InitParallelGroups(int world_rank, int world_size, int tp_size, int pp_size, int dp_size) {
    CHECK_GE(world_rank, 0);
    CHECK_LT(world_rank, world_size);
    CHECK_EQ(tp_size * pp_size * dp_size, world_size)
        << "tp_size * pp_size * dp_size must equal world_size";
    CHECK(!g_tp_group && !g_pp_group && !g_pp_collective_group && !g_dp_group)
        << "InitParallelGroups may only be called once before DestroyParallelGroups";

    const int tp_rank = world_rank % tp_size;
    const int dp_rank = (world_rank / tp_size) % dp_size;
    const int pp_rank = world_rank / (tp_size * dp_size);
    const int device_id = ResolveDeviceId(world_rank);

    const int tp_group_id = dp_rank + pp_rank * dp_size;
    g_tp_group = std::make_unique<ProcessGroup>(
        tp_rank, tp_size, MakeUniqueId("tp", tp_group_id, tp_rank), device_id);

    const int dp_group_id = tp_rank + pp_rank * tp_size;
    g_dp_group = std::make_unique<ProcessGroup>(
        dp_rank, dp_size, MakeUniqueId("dp", dp_group_id, dp_rank), device_id);

    const int pp_group_id = tp_rank + dp_rank * tp_size;
    g_pp_group = std::make_unique<ProcessGroup>(
        pp_rank, pp_size, MakeUniqueId("pp", pp_group_id, pp_rank), device_id);
    g_pp_collective_group = std::make_unique<ProcessGroup>(
        pp_rank, pp_size, MakeUniqueId("pp-collective", pp_group_id, pp_rank), device_id);
}

ProcessGroup *GetTPGroup() { return g_tp_group.get(); }
ProcessGroup *GetPPGroup() { return g_pp_group.get(); }
ProcessGroup *GetDPGroup() { return g_dp_group.get(); }
ProcessGroup *GetPPCollectiveGroup() { return g_pp_collective_group.get(); }

int GetTPRank() { return g_tp_group ? g_tp_group->Rank() : 0; }
int GetPPRank() { return g_pp_group ? g_pp_group->Rank() : 0; }
int GetDPRank() { return g_dp_group ? g_dp_group->Rank() : 0; }

void DestroyParallelGroups() {
    // Destroy in reverse creation order so every rank tears communicators down
    // consistently after its outstanding work has synchronized.
    g_pp_collective_group.reset();
    g_pp_group.reset();
    g_dp_group.reset();
    g_tp_group.reset();
}

} // namespace infini_train::distributed
#endif // USE_NCCL
