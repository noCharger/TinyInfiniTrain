#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "Eigen/Core"

#include "infini_train/include/tensor.h"

namespace infini_train::dtype {

using Float16 = Eigen::half;
using BFloat16 = Eigen::bfloat16;

// NVIDIA/ONNX-compatible finite E4M3 encoding: 1 sign, 4 exponent, 3
// significand bits, exponent bias 7, subnormal step 2^-9, max finite 448.
// The all-ones exponent/significand payload (0x7f/0xff) is NaN; infinities are
// saturated to max finite because E4M3FN has no infinity representation.
inline uint8_t EncodeFloat8E4M3(float value) {
    if (std::isnan(value)) return 0x7f;
    const uint8_t sign = std::signbit(value) ? 0x80 : 0x00;
    float magnitude = std::fabs(value);
    if (magnitude == 0.0f) return sign;
    if (!std::isfinite(magnitude) || magnitude >= 448.0f) return sign | 0x7e;

    constexpr float kMinNormal = 0.015625f;  // 2^-6
    constexpr float kSubnormalStep = 0.001953125f;  // 2^-9
    if (magnitude < kMinNormal) {
        const int mantissa = std::clamp(static_cast<int>(std::nearbyint(magnitude / kSubnormalStep)), 0, 7);
        return sign | static_cast<uint8_t>(mantissa);
    }

    int exponent = static_cast<int>(std::floor(std::log2(magnitude)));
    int exponent_field = exponent + 7;
    const float base = std::ldexp(1.0f, exponent);
    int mantissa = static_cast<int>(std::nearbyint((magnitude / base - 1.0f) * 8.0f));
    if (mantissa == 8) {
        mantissa = 0;
        ++exponent_field;
    }
    if (exponent_field >= 15) {
        exponent_field = 15;
        mantissa = std::min(mantissa, 6);
    }
    return sign | static_cast<uint8_t>((exponent_field << 3) | mantissa);
}

inline float DecodeFloat8E4M3(uint8_t bits) {
    const float sign = (bits & 0x80) ? -1.0f : 1.0f;
    const int exponent_field = (bits >> 3) & 0x0f;
    const int mantissa = bits & 0x07;
    if (exponent_field == 15 && mantissa == 7) return std::numeric_limits<float>::quiet_NaN();
    if (exponent_field == 0) return sign * static_cast<float>(mantissa) * 0.001953125f;
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent_field - 7);
}

inline bool IsFloatingPoint(DataType dtype) {
    return dtype == DataType::kFLOAT8_E4M3 || dtype == DataType::kFLOAT16 || dtype == DataType::kBFLOAT16
           || dtype == DataType::kFLOAT32 || dtype == DataType::kFLOAT64;
}

inline float LoadAsFloat(const void *data, DataType dtype, size_t index) {
    switch (dtype) {
    case DataType::kUINT8: return static_cast<const uint8_t *>(data)[index];
    case DataType::kINT8: return static_cast<const int8_t *>(data)[index];
    case DataType::kUINT16: return static_cast<const uint16_t *>(data)[index];
    case DataType::kINT16: return static_cast<const int16_t *>(data)[index];
    case DataType::kUINT32: return static_cast<float>(static_cast<const uint32_t *>(data)[index]);
    case DataType::kINT32: return static_cast<float>(static_cast<const int32_t *>(data)[index]);
    case DataType::kUINT64: return static_cast<float>(static_cast<const uint64_t *>(data)[index]);
    case DataType::kINT64: return static_cast<float>(static_cast<const int64_t *>(data)[index]);
    case DataType::kFLOAT8_E4M3: return DecodeFloat8E4M3(static_cast<const uint8_t *>(data)[index]);
    case DataType::kBFLOAT16: return static_cast<float>(static_cast<const BFloat16 *>(data)[index]);
    case DataType::kFLOAT16: return static_cast<float>(static_cast<const Float16 *>(data)[index]);
    case DataType::kFLOAT32: return static_cast<const float *>(data)[index];
    case DataType::kFLOAT64: return static_cast<float>(static_cast<const double *>(data)[index]);
    }
    throw std::runtime_error("Unsupported DataType in LoadAsFloat");
}

inline void StoreFromFloat(void *data, DataType dtype, size_t index, float value) {
    switch (dtype) {
    case DataType::kUINT8: static_cast<uint8_t *>(data)[index] = static_cast<uint8_t>(value); break;
    case DataType::kINT8: static_cast<int8_t *>(data)[index] = static_cast<int8_t>(value); break;
    case DataType::kUINT16: static_cast<uint16_t *>(data)[index] = static_cast<uint16_t>(value); break;
    case DataType::kINT16: static_cast<int16_t *>(data)[index] = static_cast<int16_t>(value); break;
    case DataType::kUINT32: static_cast<uint32_t *>(data)[index] = static_cast<uint32_t>(value); break;
    case DataType::kINT32: static_cast<int32_t *>(data)[index] = static_cast<int32_t>(value); break;
    case DataType::kUINT64: static_cast<uint64_t *>(data)[index] = static_cast<uint64_t>(value); break;
    case DataType::kINT64: static_cast<int64_t *>(data)[index] = static_cast<int64_t>(value); break;
    case DataType::kFLOAT8_E4M3: static_cast<uint8_t *>(data)[index] = EncodeFloat8E4M3(value); break;
    case DataType::kBFLOAT16: static_cast<BFloat16 *>(data)[index] = BFloat16(value); break;
    case DataType::kFLOAT16: static_cast<Float16 *>(data)[index] = Float16(value); break;
    case DataType::kFLOAT32: static_cast<float *>(data)[index] = value; break;
    case DataType::kFLOAT64: static_cast<double *>(data)[index] = static_cast<double>(value); break;
    }
}

} // namespace infini_train::dtype
