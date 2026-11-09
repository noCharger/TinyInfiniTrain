// Parallel configuration analysis tool.
// Computes bubble ratio and memory consumption for different (TP, PP, DP, ZeRO, M, V)
// configurations, using GPT-2 Small (124M params) as the reference model.
//
// Build (after setting up CMake with USE_CUDA=OFF):
//   g++ -std=c++20 -O2 tools/parallel_analysis.cc -o parallel_analysis
// Run:
//   ./parallel_analysis

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─── Model constants (GPT-2 Small) ──────────────────────────────────────────
static constexpr int64_t kVocabSize    = 50257;
static constexpr int64_t kBlockSize    = 1024;
static constexpr int64_t kNLayer       = 12;
static constexpr int64_t kNHead        = 12;
static constexpr int64_t kNEmbd        = 768;
static constexpr int64_t kParamCount   = 124'000'000; // ≈ 124M
static constexpr int64_t kBatchSize    = 8;
static constexpr int64_t kSeqLen       = 1024;
static constexpr int64_t kBytesPerFP32 = 4;
static constexpr int64_t kBytesPerBF16 = 2;

// ─── Bubble ratio ────────────────────────────────────────────────────────────
//
// Non-interleaved 1F1B / GPipe:
//   bubble_ratio = (PP - 1) / (PP - 1 + M)
//   Source: Narayanan et al. (SOSP 2019), Eq. 2.
//
// Interleaved 1F1B (V virtual stages per physical stage):
//   bubble_ratio = (PP - 1) / (PP - 1 + M * V)
//   Source: Narayanan et al. (SC 2021) "Efficient Large-Scale Language Model Training".
//
// At M → ∞, bubble_ratio → 0 in both cases.
// At M = 1 (no micro-batching), bubble_ratio → (PP-1)/PP (terrible).
double BubbleRatio(int pp, int m, int v = 1) {
    return static_cast<double>(pp - 1) / static_cast<double>(pp - 1 + m * v);
}

// ─── Activation memory per GPU ───────────────────────────────────────────────
//
// For a single Transformer layer with full activations:
//   Attention: 2 × B × T × C  (q, k projections for reuse)
//              + B × H × T × T (attention weights)  where H = n_head
//   MLP:       B × T × 4C     (intermediate after c_fc)
//              + B × T × C    (after GELU, before c_proj)
//
// Simplified formula used here (per layer, per micro-batch, in FP32 bytes):
//   attn_acts  = B * T * C * 2  (q, k)  + B * H * T * T
//   mlp_acts   = B * T * (4*C + C)
//   total_per_layer = attn_acts + mlp_acts
//
// With:
//   - PP: divide layers across pp_size stages → n_layer / pp_size per GPU
//   - TP: attention weights split across tp_size (each rank holds n_head/tp_size)
//   - Gradient checkpointing: TinyInfiniTrain checkpoints every local Block,
//     retaining one [B,T,C] input boundary per Block and recomputing the larger
//     attention/MLP intermediates during backward.
//
// Note: this is an approximation; actual Megatron accounting is more detailed.
// Reference: Korthikanti et al. (2022) "Reducing Activation Recomputation in
//   Large Transformer Models", Table 1.
int64_t ActivationMemoryBytes(int pp, int tp,
                               int64_t b, int64_t t, int64_t c,
                               int64_t n_head, int64_t n_layer,
                               bool gradient_checkpoint) {
    const int64_t layers_per_gpu = n_layer / pp;
    const int64_t heads_per_gpu  = n_head  / tp;

    const int64_t attn_per_layer =
        b * t * c * 2 * kBytesPerFP32           // q, k, v projections
        + b * heads_per_gpu * t * t * kBytesPerFP32; // attention weights (TP-split)
    const int64_t mlp_per_layer =
        b * t * (4 * c + c) * kBytesPerFP32;    // c_fc act + gelu act

    const int64_t acts_per_layer = attn_per_layer + mlp_per_layer;

    if (gradient_checkpoint) {
        // Matches GPT2Config::activation_recomputation: one checkpointed input
        // boundary per local block. This is a constant-factor reduction, not a
        // segmented O(sqrt(L)) implementation.
        const int64_t ckpt_bytes = layers_per_gpu * b * t * c * kBytesPerFP32;
        return ckpt_bytes;
    }

    return layers_per_gpu * acts_per_layer;
}

// ─── Optimizer memory per GPU ────────────────────────────────────────────────
//
// Baseline (ZeRO-0): each GPU stores full parameters + gradients + m + v.
//   = P * (dtype + dtype + 4 + 4)  bytes   (m/v always FP32)
//
// ZeRO-1: shard m and v across DP ranks.
//   m+v savings: (DP-1)/DP × 2 × P × 4 bytes
//
// ZeRO-2: additionally shard gradients.
//   grad savings: (DP-1)/DP × P × dtype_bytes
//
// ZeRO-3: additionally shard parameters.
//   param savings: (DP-1)/DP × P × dtype_bytes
//   (but each forward must AllGather params → transient full param copy exists)
//
// References:
//   Rajbhandari et al. (SC 2020) "ZeRO: Memory Optimizations Toward Training
//     Trillion Parameter Models", Table 1.
struct MemoryBreakdown {
    int64_t param_bytes;
    int64_t grad_bytes;
    int64_t optimizer_state_bytes;  // m + v (always FP32)
    int64_t activation_bytes;
    int64_t total_bytes;
};

MemoryBreakdown TotalMemoryBytes(int pp, int tp, int dp, int zero_stage,
                                  int64_t b, int64_t t, int64_t c,
                                  int64_t n_head, int64_t n_layer,
                                  bool gradient_checkpoint,
                                  bool bf16_params) {
    const int64_t dtype_bytes = bf16_params ? kBytesPerBF16 : kBytesPerFP32;
    const int64_t P = kParamCount / tp;  // TP shards params across heads/hidden

    // Parameters
    int64_t param_bytes = P * dtype_bytes;
    if (zero_stage >= 3) {
        param_bytes = param_bytes / dp;
    }

    // Gradients (same dtype as params in backward)
    int64_t grad_bytes = P * dtype_bytes;
    if (zero_stage >= 2) {
        grad_bytes = grad_bytes / dp;
    }

    // Optimizer states m + v (always FP32 for numerical stability)
    int64_t opt_bytes = P * kBytesPerFP32 * 2; // m + v
    if (bf16_params) {
        opt_bytes += P * kBytesPerFP32; // FP32 master copy
    }
    if (zero_stage >= 1) {
        opt_bytes = opt_bytes / dp;
    }

    // Activation memory
    int64_t act_bytes = ActivationMemoryBytes(pp, tp, b, t, c, n_head, n_layer,
                                               gradient_checkpoint);

    MemoryBreakdown mb;
    mb.param_bytes           = param_bytes;
    mb.grad_bytes            = grad_bytes;
    mb.optimizer_state_bytes = opt_bytes;
    mb.activation_bytes      = act_bytes;
    mb.total_bytes           = param_bytes + grad_bytes + opt_bytes + act_bytes;
    return mb;
}

static std::string FormatGB(int64_t bytes) {
    double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << gb << " GB";
    return oss.str();
}

// ─── Main: sweep (PP, M) for bubble ratio, then (TP, DP, ZeRO) for memory ──

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  TinyInfiniTrain Parallel Configuration Analysis\n";
    std::cout << "  Model: GPT-2 Small (124M params, L=12, C=768, H=12, T=1024)\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    // ── Table 1: Bubble ratio sweep ──────────────────────────────────────────
    std::cout << "Table 1: Bubble ratio  =  (PP - 1) / (PP - 1 + M*V)\n";
    std::cout << "         Non-interleaved (V=1), M = micro-batches\n\n";
    std::cout << std::setw(6) << "PP" << std::setw(6) << "V";
    for (int m : {1, 2, 4, 8, 16, 32}) {
        std::cout << std::setw(10) << ("M=" + std::to_string(m));
    }
    std::cout << "\n" << std::string(66, '-') << "\n";

    for (int pp : {2, 4, 8, 16}) {
        for (int v : {1, 2, 4}) {
            if (v > 1 && pp < 4) continue; // interleaved only makes sense for larger PP
            std::cout << std::setw(6) << pp << std::setw(6) << v;
            for (int m : {1, 2, 4, 8, 16, 32}) {
                double br = BubbleRatio(pp, m, v);
                std::cout << std::setw(9) << std::fixed << std::setprecision(3) << br << " ";
            }
            std::cout << "\n";
        }
    }

    // ── Table 2: Memory sweep ────────────────────────────────────────────────
    std::cout << "\nTable 2: GPU memory breakdown (FP32 params, B=8, T=1024, no gradient checkpoint)\n\n";
    std::cout << std::setw(4) << "TP" << std::setw(4) << "PP" << std::setw(4) << "DP"
              << std::setw(7) << "ZeRO"
              << std::setw(10) << "Param" << std::setw(10) << "Grad"
              << std::setw(12) << "Opt(m+v)" << std::setw(12) << "Activ"
              << std::setw(12) << "Total" << "\n";
    std::cout << std::string(75, '-') << "\n";

    struct Config { int tp, pp, dp, z; };
    std::vector<Config> configs = {
        {1,  1,  1, 0},
        {1,  1,  8, 0},
        {1,  1,  8, 1},
        {1,  1,  8, 2},
        {1,  1,  8, 3},
        {2,  1,  4, 1},
        {2,  2,  2, 1},
        {4,  2,  2, 2},
        {8,  4,  4, 3},
    };

    for (const auto &cfg : configs) {
        auto mb = TotalMemoryBytes(cfg.pp, cfg.tp, cfg.dp, cfg.z,
                                   kBatchSize, kSeqLen, kNEmbd, kNHead, kNLayer,
                                   /*gradient_checkpoint=*/false, /*bf16_params=*/false);
        std::cout << std::setw(4) << cfg.tp << std::setw(4) << cfg.pp
                  << std::setw(4) << cfg.dp << std::setw(7) << cfg.z
                  << std::setw(10) << FormatGB(mb.param_bytes)
                  << std::setw(10) << FormatGB(mb.grad_bytes)
                  << std::setw(12) << FormatGB(mb.optimizer_state_bytes)
                  << std::setw(12) << FormatGB(mb.activation_bytes)
                  << std::setw(12) << FormatGB(mb.total_bytes) << "\n";
    }

    // ── Table 3: BF16 + gradient checkpoint ─────────────────────────────────
    std::cout << "\nTable 3: BF16 params + gradient checkpoint, TP=2, PP=2, DP=4, ZeRO-1\n\n";
    {
        auto mb_fp32 = TotalMemoryBytes(2, 2, 4, 1, kBatchSize, kSeqLen, kNEmbd, kNHead, kNLayer,
                                         false, false);
        auto mb_bf16 = TotalMemoryBytes(2, 2, 4, 1, kBatchSize, kSeqLen, kNEmbd, kNHead, kNLayer,
                                         false, true);
        auto mb_ckpt = TotalMemoryBytes(2, 2, 4, 1, kBatchSize, kSeqLen, kNEmbd, kNHead, kNLayer,
                                         true, true);

        std::cout << "  Baseline (FP32, no ckpt): " << FormatGB(mb_fp32.total_bytes) << "\n";
        std::cout << "  BF16 params, no ckpt:     " << FormatGB(mb_bf16.total_bytes) << "\n";
        std::cout << "  BF16 + grad checkpoint:   " << FormatGB(mb_ckpt.total_bytes) << "\n";
    }

    std::cout << "\n[Note] These are analytical estimates. Actual GPU memory includes\n"
              << "       CUDA context overhead, framework buffers, and communication buffers.\n";
    return 0;
}
