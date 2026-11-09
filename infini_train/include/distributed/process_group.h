#pragma once
#ifdef USE_NCCL
#include <cstddef>
#include <cuda_runtime_api.h>
#include <nccl.h>

namespace infini_train::distributed {

// NCCL-backed process group for a single parallel dimension (TP, PP, or DP).
// One ProcessGroup instance corresponds to one ncclComm_t.
// Megatron-LM analogue: ProcessGroup from torch.distributed, one per
//   parallel axis, created in parallel_state.py:initialize_model_parallel().
class ProcessGroup {
public:
    // uid must be identical across all ranks in the group (broadcast from rank 0).
    ProcessGroup(int rank, int world_size, ncclUniqueId uid, int device_id);
    ~ProcessGroup();

    // Prevent accidental copies — ncclComm_t is not copyable.
    ProcessGroup(const ProcessGroup &) = delete;
    ProcessGroup &operator=(const ProcessGroup &) = delete;

    int Rank() const { return rank_; }
    int WorldSize() const { return world_size_; }
    ncclComm_t Comm() const { return comm_; }

    // Collectives below synchronize before returning. P2P Send/Recv only
    // enqueue work: subsequent compute on the same stream observes the data in
    // order, which is required to avoid host-side 1F1B send/send deadlocks.
    // A null stream selects TinyInfiniTrain's per-thread non-blocking stream.

    // AllReduce: buf = op(buf across all ranks)
    void AllReduce(void *buf, size_t count, ncclDataType_t dtype, ncclRedOp_t op,
                   cudaStream_t s = nullptr);

    // ReduceScatter: recv[i] = op(send[i] across all ranks), recv size = count/world_size
    void ReduceScatter(const void *send, void *recv, size_t recv_count,
                       ncclDataType_t dtype, ncclRedOp_t op, cudaStream_t s = nullptr);

    // AllGather: recv = concat(send across all ranks in rank order)
    void AllGather(const void *send, void *recv, size_t send_count,
                   ncclDataType_t dtype, cudaStream_t s = nullptr);

    // Point-to-point send to peer rank within this group
    void Send(const void *buf, size_t count, ncclDataType_t dtype, int peer,
              cudaStream_t s = nullptr);

    // Point-to-point recv from peer rank within this group
    void Recv(void *buf, size_t count, ncclDataType_t dtype, int peer,
              cudaStream_t s = nullptr);

    // Atomically enqueue a bidirectional exchange. Grouping the matching
    // ncclSend/ncclRecv pair is required by 1F1B when a forward activation and
    // an older backward gradient cross the same pipeline link concurrently.
    void SendRecv(const void *send_buf, size_t send_count, ncclDataType_t send_dtype, int send_peer,
                  void *recv_buf, size_t recv_count, ncclDataType_t recv_dtype, int recv_peer,
                  cudaStream_t s = nullptr);

private:
    int rank_;
    int world_size_;
    int device_id_;
    ncclComm_t comm_;
};

} // namespace infini_train::distributed
#endif // USE_NCCL
