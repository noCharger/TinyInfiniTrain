#ifdef USE_NCCL
#include "infini_train/include/distributed/process_group.h"

#include <cuda_runtime_api.h>
#include "glog/logging.h"
#include "infini_train/include/cuda_utils.h"

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t r = (call);                                               \
        LOG_IF(FATAL, r != ncclSuccess)                                        \
            << "NCCL Error: " << ncclGetErrorString(r)                         \
            << " at " << __FILE__ << ":" << __LINE__;                          \
    } while (0)

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t result = (call);                                                                             \
        LOG_IF(FATAL, result != cudaSuccess)                                                                           \
            << "CUDA Error: " << cudaGetErrorString(result) << " at " << __FILE__ << ':' << __LINE__;              \
    } while (0)

namespace infini_train::distributed {

ProcessGroup::ProcessGroup(int rank, int world_size, ncclUniqueId uid, int device_id)
    : rank_(rank), world_size_(world_size), device_id_(device_id) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, uid, rank_));
}

ProcessGroup::~ProcessGroup() {
    cudaSetDevice(device_id_);
    const ncclResult_t result = ncclCommDestroy(comm_);
    LOG_IF(ERROR, result != ncclSuccess) << "ncclCommDestroy failed: " << ncclGetErrorString(result);
}

namespace {
cudaStream_t ResolveStream(cudaStream_t stream, int device_id) {
    return stream ? stream : infini_train::ThreadLocalStream(device_id);
}
} // namespace

void ProcessGroup::AllReduce(void *buf, size_t count, ncclDataType_t dtype,
                              ncclRedOp_t op, cudaStream_t s) {
    s = ResolveStream(s, device_id_);
    NCCL_CHECK(ncclAllReduce(buf, buf, count, dtype, op, comm_, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void ProcessGroup::ReduceScatter(const void *send, void *recv, size_t recv_count,
                                  ncclDataType_t dtype, ncclRedOp_t op,
                                  cudaStream_t s) {
    s = ResolveStream(s, device_id_);
    NCCL_CHECK(ncclReduceScatter(send, recv, recv_count, dtype, op, comm_, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void ProcessGroup::AllGather(const void *send, void *recv, size_t send_count,
                              ncclDataType_t dtype, cudaStream_t s) {
    s = ResolveStream(s, device_id_);
    NCCL_CHECK(ncclAllGather(send, recv, send_count, dtype, comm_, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void ProcessGroup::Send(const void *buf, size_t count, ncclDataType_t dtype,
                        int peer, cudaStream_t s) {
    s = ResolveStream(s, device_id_);
    NCCL_CHECK(ncclSend(buf, count, dtype, peer, comm_, s));
}

void ProcessGroup::Recv(void *buf, size_t count, ncclDataType_t dtype,
                        int peer, cudaStream_t s) {
    s = ResolveStream(s, device_id_);
    NCCL_CHECK(ncclRecv(buf, count, dtype, peer, comm_, s));
}

void ProcessGroup::SendRecv(const void *send_buf, size_t send_count, ncclDataType_t send_dtype, int send_peer,
                            void *recv_buf, size_t recv_count, ncclDataType_t recv_dtype, int recv_peer,
                            cudaStream_t s) {
    s = ResolveStream(s, device_id_);
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(send_buf, send_count, send_dtype, send_peer, comm_, s));
    NCCL_CHECK(ncclRecv(recv_buf, recv_count, recv_dtype, recv_peer, comm_, s));
    NCCL_CHECK(ncclGroupEnd());
}

} // namespace infini_train::distributed
#endif // USE_NCCL
