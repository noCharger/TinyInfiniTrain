#pragma once
#ifdef USE_NCCL

#include <nccl.h>

#include "glog/logging.h"
#include "infini_train/include/tensor.h"

// Maps TinyInfiniTrain DataType to the NCCL collective dtype.
// Mirrors cuda_utils.h::ToCudaDtype's pattern. Only types NCCL natively
// supports are mapped — ncclDataType_t has no 16-bit integer variants, so
// kUINT16/kINT16 fall into the LOG(FATAL) default branch.

namespace infini_train::distributed {

inline ncclDataType_t ToNcclDataType(infini_train::DataType dt) {
    switch (dt) {
    case infini_train::DataType::kUINT8:
        return ncclUint8;
    case infini_train::DataType::kINT8:
        return ncclInt8;
    case infini_train::DataType::kUINT32:
        return ncclUint32;
    case infini_train::DataType::kINT32:
        return ncclInt32;
    case infini_train::DataType::kUINT64:
        return ncclUint64;
    case infini_train::DataType::kINT64:
        return ncclInt64;
    case infini_train::DataType::kBFLOAT16:
        return ncclBfloat16;
    case infini_train::DataType::kFLOAT16:
        return ncclFloat16;
    case infini_train::DataType::kFLOAT32:
        return ncclFloat32;
    case infini_train::DataType::kFLOAT64:
        return ncclFloat64;
    default:
        LOG(FATAL) << "DataType has no NCCL collective equivalent: " << static_cast<int>(dt);
        return ncclFloat32; // unreachable
    }
}

} // namespace infini_train::distributed
#endif // USE_NCCL
