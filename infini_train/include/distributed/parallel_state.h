#pragma once
#ifdef USE_NCCL

#include "infini_train/include/distributed/process_group.h"

namespace infini_train::distributed {

// Global TP/PP/DP process group registry.
// Call InitParallelGroups() once at program startup before using any collective.
// Megatron-LM analogue: megatron/core/parallel_state.py:initialize_model_parallel()
//
// Rank layout (tp-dp-pp, innermost first):
//   global_rank = tp_rank
//               + dp_rank * tp_size
//               + pp_rank * tp_size * dp_size
//
// Example: world_size=8, tp_size=2, dp_size=2, pp_size=2
//   TP groups: {0,1}, {2,3}, {4,5}, {6,7}
//   DP groups: {0,2}, {1,3}, {4,6}, {5,7}
//   PP groups: {0,4}, {1,5}, {2,6}, {3,7}
void InitParallelGroups(int world_rank, int world_size,
                        int tp_size, int pp_size, int dp_size);

// Access the process group for this rank's parallel axis.
// Returns nullptr if InitParallelGroups() has not been called.
ProcessGroup *GetTPGroup();
ProcessGroup *GetPPGroup();
ProcessGroup *GetDPGroup();

// A second communicator over the same PP ranks, reserved for collectives such
// as tied-embedding synchronization. Keeping those collectives off the P2P
// communicator prevents operation-order interference with 1F1B exchanges.
ProcessGroup *GetPPCollectiveGroup();

// Convenience rank accessors within each parallel dimension.
int GetTPRank();
int GetPPRank();
int GetDPRank();

// Tear down all process groups. Call before MPI/NCCL finalization.
void DestroyParallelGroups();

} // namespace infini_train::distributed
#endif // USE_NCCL
