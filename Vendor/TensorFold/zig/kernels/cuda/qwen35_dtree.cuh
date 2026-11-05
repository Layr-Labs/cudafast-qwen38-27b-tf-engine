// TF_DTREE: the drafter's best-first tree (cuda_draft_tree.zig grow, the history copy of cuda_draft.zig selectHold)
// on the device, one warp. The host's f64 arithmetic is reproduced operation by operation: every double add, multiply
// and divide is an explicit round-to-nearest intrinsic (no contraction), and exp/log are ports of the Zig compiler_rt
// routines the host links (exp_f64, log_f64), so each node, parent and log-chance is the host's bits.

#include "qwen35_dtree_logtab.cuh"

namespace tf_dtree {

__device__ __forceinline__ double A(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double S(double a, double b) { return __dsub_rn(a, b); }
__device__ __forceinline__ double M(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double D(double a, double b) { return __ddiv_rn(a, b); }
__device__ __forceinline__ uint64_t bits(double x) { return (uint64_t)__double_as_longlong(x); }
__device__ __forceinline__ double dbl(uint64_t u) { return __longlong_as_double((long long)u); }
__device__ __forceinline__ double bfd(uint16_t h) { return (double)__uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ float bff(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
// bfd by integer moves (the FP64 pipe is GB10's narrowest): normal and zero values rebiased, others through F2F
__device__ __forceinline__ double bfd_int(uint16_t h) {
    const uint32_t e = (h >> 7) & 0xFF, m = h & 0x7F;
    const unsigned long long sgn = (unsigned long long)(h >> 15) << 63;
    if (e != 0 && e != 0xFF) return __longlong_as_double((long long)(sgn | ((unsigned long long)(e - 127 + 1023) << 52) | ((unsigned long long)m << 45)));
    if (e == 0 && m == 0) return __longlong_as_double((long long)sgn);
    return bfd(h);
}

// compiler_rt exp_f64
__device__ double zexp(double x) {
    const double ln2hi = 6.93147180369123816490e-01, ln2lo = 1.90821492927058770002e-10, invln2 = 1.44269504088896338700e+00;
    const double P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03, P3 = 6.61375632143793436117e-05,
                 P4 = -1.65339022054652515390e-06, P5 = 4.13813679705723846039e-08;
    const uint64_t ux = bits(x);
    uint32_t hx = (uint32_t)(ux >> 32);
    const int sign = (int)(hx >> 31);
    hx &= 0x7FFFFFFFu;
    if (x != x) return x;
    if (hx >= 0x4086232Bu) {
        if (hx > 0x7FF00000u) return x;
        if (x > 709.782712893383973096) return __longlong_as_double(0x7FF0000000000000LL);
        if (x < -708.39641853226410622) {
            if (x < -745.13321910194110842) return 0.0;
        }
    }
    int k;
    double hi, lo;
    if (hx > 0x3FD62E42u) {
        if (hx > 0x3FF0A2B2u) {
            k = (int)A(M(invln2, x), sign ? -0.5 : 0.5);
        } else {
            k = 1 - sign - sign;
        }
        const double dk = (double)k;
        hi = S(x, M(dk, ln2hi));
        lo = M(dk, ln2lo);
        x = S(hi, lo);
    } else if (hx > 0x3E300000u) {
        k = 0;
        hi = x;
        lo = 0;
    } else {
        return A(1.0, x);
    }
    const double xx = M(x, x);
    const double c = S(x, M(xx, A(P1, M(xx, A(P2, M(xx, A(P3, M(xx, A(P4, M(xx, P5))))))))));
    const double y = A(1.0, A(S(D(M(x, c), S(2.0, c)), lo), hi));
    if (k == 0) return y;
    return scalbn(y, k);
}

// compiler_rt log_f64
__device__ double zlog(double x) {
    using namespace tf_dtree_log;
    int64_t ix = (int64_t)bits(x);
    const int64_t LO = (int64_t)bits(1.0 - 0x1p-4);
    const int64_t HI = (int64_t)bits(1.0 + 0x1.09p-4);
    if (LO <= ix && ix < HI) {
        if (ix == (int64_t)bits(1.0)) return 0;
        const double r = S(x, 1.0);
        const double r2 = M(r, r);
        const double r3 = M(r, r2);
        // poly1[7] + r*poly1[8] + r2*poly1[9] + r3*poly1[10], left to right
        const double t2 = A(A(A(poly1[7], M(r, poly1[8])), M(r2, poly1[9])), M(r3, poly1[10]));
        const double t1 = A(A(A(poly1[4], M(r, poly1[5])), M(r2, poly1[6])), M(r3, t2));
        const double t0 = A(A(A(poly1[1], M(r, poly1[2])), M(r2, poly1[3])), M(r3, t1));
        const double y = M(r3, t0);
        double w = M(r, 0x1p27);
        const double rhi = S(A(r, w), w);
        const double rlo = S(r, rhi);
        w = M(M(rhi, rhi), poly1[0]);
        const double hi = A(r, w);
        const double lo = A(A(S(r, hi), w), M(M(poly1[0], rlo), A(rhi, r)));
        return A(A(y, lo), hi);
    }
    const uint64_t top = (uint64_t)ix >> 48;
    if (top < 0x0010 || 0x7ff0 <= top) {
        if ((uint64_t)ix << 1 == 0) return __longlong_as_double((long long)0xFFF0000000000000ULL);
        if (ix == (int64_t)0x7FF0000000000000LL) return x;
        if ((top & 0x8000) != 0 || (top & 0x7ff0) == 0x7ff0) return __longlong_as_double(0x7FF8000000000000LL);
        ix = (int64_t)bits(M(x, 0x1p52)) - ((int64_t)52 << 52);
    }
    const int64_t tmp = ix - 0x3fe6000000000000LL;
    const int i = (int)(((uint64_t)tmp >> 45) & 127);
    const int64_t k = tmp >> 52;
    const int64_t iz = ix - (k << 52);
    const double z = dbl((uint64_t)iz);
    const double r = M(S(S(z, chi[i]), clo[i]), invc[i]);
    const double kd = (double)k;
    const double w = A(M(kd, 0x1.62e42fefa3800p-1), logc[i]);
    const double hi = A(w, r);
    const double lo = A(A(S(w, hi), r), M(kd, 0x1.ef35793c76730p-45));
    const double r2 = M(r, r);
    const double inner = A(A(poly[1], M(r, poly[2])), M(r2, A(poly[3], M(r, poly[4]))));
    return A(A(A(lo, M(r2, poly[0])), M(M(r, r2), inner)), hi);
}

constexpr int LP_TOP = 20;
constexpr int LP_WORDS = 3 + 2 * LP_TOP;
constexpr int MAX_OPEN = 32 * 12;
constexpr int MAX_ROWS = 15;

struct Args {
    const uint32_t* words;   // (rows, LP_WORDS): row r's ids at 2.., f32 logits at 2 + LP_TOP..
    const uint16_t* hp;      // (rows, rank) bf16 selector projection
    const uint16_t* pred;    // (vocab, rank) bf16 codebooks
    const uint16_t* succ;
    const double* landing;   // (16, 2) the stream's Landing table: offered, landed
    const uint32_t* hist;    // the stream's tokens
    uint32_t* held;          // out: nodes' tokens, parents, own log-chances, count
    int* parents;
    double* own;
    int* grown;
    double edge_w, scale, trust, lam;
    int rank, topk, rows, spanned, most, branch, hist_len, copy;
    uint32_t pending;
    const DroundState* st;  // non-null: pending and the history's length from the device round
    long long* prof;         // non-null (TF_DTPROF): cycles in [stage, edge, softmax, order, pick, chain, scan]
    double* tab_lc;          // non-null: tf_q35d_dtree_pairs' tables (every (row, pred) expansion ahead of the grow)
    unsigned char* tab_place;
    int* tab_cut;
    float* tab_sc;
};

// The expansions a grow can ask for: row 0 after the pending token, row r >= 1 after each of row r - 1's candidates.
__host__ __device__ __forceinline__ int pair_of(int r, int p) { return r == 0 ? 0 : 1 + (r - 1) * LP_TOP + p; }

// A finite or infinite double's order as an unsigned key (no NaN reaches it): x < y exactly when key(x) < key(y),
// except -0 below +0 (a softmax-normalized log-chance and a non-negative path cost never meet both).
__device__ __forceinline__ unsigned long long okey(double x) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(x);
    return (b >> 63) ? ~b : (b | 0x8000000000000000ULL);
}

__device__ __forceinline__ uint32_t cand_id(const Args& a, int r, int j) {
    uint32_t id = a.words[r * LP_WORDS + 2 + j];
    if (a.spanned && id >= 98304u) id += 248032u - 98304u;
    return id;
}

__device__ __forceinline__ double landing_chance(const Args& a, double own) {
    const double q = D(-own, 0.5);
    const double f = floor(q);
    const double m = fmax(0.0, f);
    int b = (int)(unsigned long long)m;
    if (m >= 15.0) b = 15;
    b = min(15, b);
    const double x0 = a.landing[b * 2], x1 = a.landing[b * 2 + 1];
    return D(A(x1, M(2.0, zexp(own))), A(x0, 2.0));
}

}  // namespace tf_dtree

// The rows an expansion reads, staged into shared memory with 16-byte loads by the whole warp (the codebooks and
// history live in mapped host memory); a padded row stride keeps the lanes' reads of one k on distinct banks.
constexpr int SROW = 258;
__device__ __forceinline__ void stage_row(const uint16_t* __restrict__ src, uint16_t* dst, int rank, int lane) {
    const uint4* s = reinterpret_cast<const uint4*>(src);
    for (int v = lane; v < rank / 8; v += 32) {
        const uint4 x = s[v];
        uint32_t* d = reinterpret_cast<uint32_t*>(dst + v * 8);
        d[0] = x.x;
        d[1] = x.y;
        d[2] = x.z;
        d[3] = x.w;
    }
}

// An expansion's rows (pred, hp, then the candidates') staged at once: rank 256 is one 16-byte word a lane a row, so
// every lane issues all its loads before the first store (one memory latency an expansion, not one a row).
__device__ __forceinline__ void stage_all(const tf_dtree::Args& a, uint32_t pred, int row, const uint32_t* cid, uint16_t (*succ)[SROW],
                                          uint16_t* pr, uint16_t* hp, int lane) {
    uint4 v[tf_dtree::LP_TOP + 2];
    v[0] = reinterpret_cast<const uint4*>(a.pred + (size_t)pred * 256)[lane];
    v[1] = reinterpret_cast<const uint4*>(a.hp + (size_t)row * 256)[lane];
#pragma unroll
    for (int j = 0; j < tf_dtree::LP_TOP; ++j)
        if (j < a.topk) v[2 + j] = reinterpret_cast<const uint4*>(a.succ + (size_t)cid[j] * 256)[lane];
    auto put = [&](uint16_t* dst, const uint4& x) {
        uint32_t* d = reinterpret_cast<uint32_t*>(dst + lane * 8);
        d[0] = x.x;
        d[1] = x.y;
        d[2] = x.z;
        d[3] = x.w;
    };
    put(pr, v[0]);
    put(hp, v[1]);
#pragma unroll
    for (int j = 0; j < tf_dtree::LP_TOP; ++j)
        if (j < a.topk) put(succ[j], v[2 + j]);
}

// Every (row, pred) expansion of a grow at once, a warp each (cuda_draft_tree.zig chancesAfter's bits): the
// log-chances, each candidate's place in grow's order and the cut at the branch or the first non-finite one, and the
// chain's f32 scores (choice()). Grid: 1 + (rows - 1) * LP_TOP.
extern "C" __global__ void __launch_bounds__(32) tf_q35d_dtree_pairs(const tf_dtree::Args a) {
    using namespace tf_dtree;
    __shared__ __align__(16) uint16_t s_succ[LP_TOP][SROW];
    __shared__ __align__(16) uint16_t s_pr[SROW];
    __shared__ __align__(16) uint16_t s_hp[SROW];
    __shared__ uint32_t cid[LP_TOP];
    __shared__ double ex[LP_TOP];
    __shared__ unsigned long long keys[LP_TOP];
    __shared__ double s_ph[256];
    const int lane = threadIdx.x;
    const int q = blockIdx.x;
    const int r = q == 0 ? 0 : 1 + (q - 1) / LP_TOP;
    const int p = q == 0 ? 0 : (q - 1) % LP_TOP;
    if (r >= a.rows || (r > 0 && p >= a.topk)) return;
    const uint32_t pending = a.st ? a.st->pending : a.pending;
    const uint32_t pred = r == 0 ? pending : cand_id(a, r - 1, p);
    if (lane < a.topk) cid[lane] = cand_id(a, r, lane);
    __syncwarp();
    stage_all(a, pred, r, cid, s_succ, s_pr, s_hp, lane);
    __syncwarp();
    // pred x selector once a k for every candidate (the same exact product each lane formed)
    for (int k = lane; k < 256; k += 32) s_ph[k] = M(bfd_int(s_pr[k]), bfd_int(s_hp[k]));
    __syncwarp();
    double l = -__longlong_as_double(0x7FF0000000000000LL);
    float sc = -__int_as_float(0x7f800000);
    if (lane < a.topk) {
        double e = 0;
        float e32 = 0;
#pragma unroll 8
        for (int k = 0; k < 256; ++k) {
            const uint16_t sv = s_succ[lane][k];
            e = A(e, M(s_ph[k], bfd_int(sv)));
            e32 = __fadd_rn(e32, __fmul_rn(__fmul_rn(bff(s_pr[k]), bff(s_hp[k])), bff(sv)));
        }
        const uint32_t lw = a.words[r * LP_WORDS + 2 + LP_TOP + lane];
        l = D(A((double)__uint_as_float(lw), M(a.edge_w, e)), a.scale);
        sc = __fadd_rn(__uint_as_float(lw), e32);
    }
    double most = l;
    for (int m = 16; m; m >>= 1) most = fmax(most, __shfl_xor_sync(0xffffffffu, most, m));
    if (lane < a.topk) ex[lane] = zexp(S(l, most));
    __syncwarp();
    double log_total = 0;
    if (lane == 0) {
        double total = 0;
        for (int j = 0; j < a.topk; ++j) total = A(total, ex[j]);
        log_total = A(most, zlog(total));
    }
    log_total = __shfl_sync(0xffffffffu, log_total, 0);
    if (lane < a.topk) {
        l = S(l, log_total);
        keys[lane] = okey(l);
    }
    __syncwarp();
    int place = 0;
    if (lane < a.topk) {
        const unsigned long long mk = keys[lane];
        for (int j = 0; j < a.topk; ++j) place += (keys[j] > mk || (keys[j] == mk && j < lane)) ? 1 : 0;
    }
    // the cut: the branch, or the place of the first non-finite log-chance in order before it
    const int nb = min(a.branch, a.topk);
    int c = (lane < a.topk && place < nb && !isfinite(l)) ? place : nb;
    for (int m = 16; m; m >>= 1) c = min(c, __shfl_xor_sync(0xffffffffu, c, m));
    const size_t base = (size_t)q * LP_TOP;
    if (lane < a.topk) {
        a.tab_lc[base + lane] = l;
        a.tab_place[base + lane] = (unsigned char)place;
        a.tab_sc[base + lane] = sc;
    }
    if (lane == 0) a.tab_cut[q] = c;
}

extern "C" __global__ void __launch_bounds__(32) tf_q35d_dtree_grow(const tf_dtree::Args a) {
    using namespace tf_dtree;
    long long pc[7] = {0, 0, 0, 0, 0, 0, 0};
    long long t0 = clock64();
#define PROF_MARK(i) do { if (a.prof) { const long long t1 = clock64(); pc[i] += t1 - t0; t0 = t1; } } while (0)
    constexpr int HIST_SH = 4096;
    __shared__ __align__(16) uint16_t s_succ[LP_TOP][SROW];
    __shared__ __align__(16) uint16_t s_pr[SROW];
    __shared__ __align__(16) uint16_t s_hp[SROW];
    __shared__ double lc[LP_TOP];
    __shared__ uint32_t cid[LP_TOP];
    __shared__ double o_cost[MAX_OPEN];
    __shared__ double o_own[MAX_OPEN];
    __shared__ int o_parent[MAX_OPEN];
    __shared__ uint32_t o_token[MAX_OPEN];
    __shared__ int o_row[MAX_OPEN];
    __shared__ int o_idx[MAX_OPEN];  // the node's place in its row's candidates (the next expansion's pair)
    __shared__ uint32_t s_held[32];
    __shared__ int s_parents[32];
    __shared__ double s_own[32];
    __shared__ uint32_t ch[MAX_ROWS];
    __shared__ __align__(16) uint32_t s_hist[HIST_SH];
    __shared__ int s_count;
    const int lane = threadIdx.x;
    const uint32_t pending = a.st ? a.st->pending : a.pending;
    const int hist_len = a.st ? a.st->hist_len : a.hist_len;
    const bool rank_ok = a.rank <= 256 && (a.rank % 8) == 0;
    const bool rank256 = a.rank == 256;
    const bool tabs = a.tab_lc != nullptr;
    const bool keyed = a.trust == 0 && a.lam == 0;  // costs are path - log-chance >= 0: compared by their bits
    int n_open = 0, count = 0, parent = -1, row = 0, pidx = 0;
    uint32_t pred = pending;
    double path = 0;
    const int stop = min(a.most, 64);
    for (;;) {
        // the host also expands the last node taken, but nothing reads that expansion
        if (tabs && row < a.rows && count < stop) {
            const int q = pair_of(row, pidx);
            const size_t base = (size_t)q * LP_TOP;
            const int cut = a.tab_cut[q];
            const int place_l = lane < a.topk ? (int)a.tab_place[base + lane] : 255;
            const double l_l = lane < a.topk ? a.tab_lc[base + lane] : 0.0;
            const uint32_t id_l = lane < a.topk ? cand_id(a, row, lane) : 0;
            if (lane < a.topk) {
                const int place = place_l;
                if (place < cut) {
                    const double l = l_l;
                    double cost = S(path, l);
                    if (a.trust > 0) {
                        const double pm = A(M(S(1.0, a.trust), zexp(l)), M(a.trust, fmax(landing_chance(a, l), 1e-300)));
                        cost = S(path, zlog(fmax(pm, 1e-300)));
                    }
                    if (a.lam != 0) cost = A(cost, M(a.lam, (double)row));
                    const int at = n_open + place;
                    o_cost[at] = cost;
                    o_parent[at] = parent;
                    o_token[at] = id_l;
                    o_row[at] = row;
                    o_own[at] = l;
                    o_idx[at] = lane;
                }
            }
            n_open += cut;
            __syncwarp();
            PROF_MARK(3);
        } else if (row < a.rows && count < stop) {
            const uint32_t my_id = lane < a.topk ? cand_id(a, row, lane) : 0;
            if (lane < a.topk) cid[lane] = my_id;
            __syncwarp();
            PROF_MARK(4);
            if (rank256) {
                stage_all(a, pred, row, cid, s_succ, s_pr, s_hp, lane);
            } else if (rank_ok) {
                stage_row(a.pred + (size_t)pred * a.rank, s_pr, a.rank, lane);
                stage_row(a.hp + (size_t)row * a.rank, s_hp, a.rank, lane);
                for (int j = 0; j < a.topk; ++j) stage_row(a.succ + (size_t)cid[j] * a.rank, s_succ[j], a.rank, lane);
            }
            __syncwarp();
            // the expansion's log-chances: each candidate's edge sum over k in order (the products are exact)
            double l = -__longlong_as_double(0x7FF0000000000000LL);
            if (lane < a.topk) {
                double e = 0;
                if (rank_ok) {
                    for (int k = 0; k < a.rank; ++k) e = A(e, M(M(bfd(s_pr[k]), bfd(s_hp[k])), bfd(s_succ[lane][k])));
                } else {
                    const uint16_t* pr = a.pred + (size_t)pred * a.rank;
                    const uint16_t* hp = a.hp + (size_t)row * a.rank;
                    const uint16_t* sv = a.succ + (size_t)my_id * a.rank;
                    for (int k = 0; k < a.rank; ++k) e = A(e, M(M(bfd(pr[k]), bfd(hp[k])), bfd(sv[k])));
                }
                const double logit = (double)__uint_as_float(a.words[row * LP_WORDS + 2 + LP_TOP + lane]);
                l = D(A(logit, M(a.edge_w, e)), a.scale);
            }
            PROF_MARK(0);
            // the softmax: the max is order-free; each exp on its lane; their sum in index order on lane 0
            double most = l;
            for (int m = 16; m; m >>= 1) most = fmax(most, __shfl_xor_sync(0xffffffffu, most, m));
            const double ex = lane < a.topk ? zexp(S(l, most)) : 0.0;
            if (lane < a.topk) lc[lane] = ex;
            __syncwarp();
            double log_total = 0;
            if (lane == 0) {
                double total = 0;
                for (int j = 0; j < a.topk; ++j) total = A(total, lc[j]);
                log_total = A(most, zlog(total));
            }
            log_total = __shfl_sync(0xffffffffu, log_total, 0);
            __syncwarp();
            if (lane < a.topk) {
                l = S(l, log_total);
                lc[lane] = l;
            }
            __syncwarp();
            PROF_MARK(2);
            // each candidate's place in the order (log-chance desc, index asc), the open list's pushes in that order up
            // to the branch or the first non-finite one
            int place = 0;
            if (lane < a.topk)
                for (int j = 0; j < a.topk; ++j) place += (lc[j] > l || (lc[j] == l && j < lane)) ? 1 : 0;
            const int nb = min(a.branch, a.topk);
            int cut = nb;
            for (int j = 0; j < a.topk; ++j) {
                int pj = 0;
                for (int q = 0; q < a.topk; ++q) pj += (lc[q] > lc[j] || (lc[q] == lc[j] && q < j)) ? 1 : 0;
                if (pj < cut && !isfinite(lc[j])) cut = pj;
            }
            if (lane < a.topk && place < cut) {
                double cost = S(path, l);
                if (a.trust > 0) {
                    const double pm = A(M(S(1.0, a.trust), zexp(l)), M(a.trust, fmax(landing_chance(a, l), 1e-300)));
                    cost = S(path, zlog(fmax(pm, 1e-300)));
                }
                if (a.lam != 0) cost = A(cost, M(a.lam, (double)row));
                const int at = n_open + place;
                o_cost[at] = cost;
                o_parent[at] = parent;
                o_token[at] = my_id;
                o_row[at] = row;
                o_own[at] = l;
                o_idx[at] = lane;
            }
            n_open += cut;
            __syncwarp();
            PROF_MARK(3);
        }
        if (count == stop || n_open == 0) break;
        // the open node of least (cost, parent, token, row): a total order, so any reduction finds the host's pick
        int best = -1;
        for (int i = lane; i < n_open; i += 32) {
            if (best < 0) {
                best = i;
                continue;
            }
            if (keyed) {
                const unsigned long long oc = (unsigned long long)__double_as_longlong(o_cost[i]), bc = (unsigned long long)__double_as_longlong(o_cost[best]);
                const int op = o_parent[i], bp = o_parent[best];
                const uint32_t ot = o_token[i], bt = o_token[best];
                if (oc < bc || (oc == bc && (op < bp || (op == bp && (ot < bt || (ot == bt && o_row[i] < o_row[best])))))) best = i;
                continue;
            }
            const double oc = o_cost[i], bc = o_cost[best];
            const int op = o_parent[i], bp = o_parent[best];
            const uint32_t ot = o_token[i], bt = o_token[best];
            if (oc < bc || (oc == bc && (op < bp || (op == bp && (ot < bt || (ot == bt && o_row[i] < o_row[best])))))) best = i;
        }
        for (int m = 16; m; m >>= 1) {
            const int ob = __shfl_xor_sync(0xffffffffu, best, m);
            if (ob < 0) continue;
            if (best < 0) {
                best = ob;
                continue;
            }
            if (keyed) {
                const unsigned long long oc = (unsigned long long)__double_as_longlong(o_cost[ob]), bc = (unsigned long long)__double_as_longlong(o_cost[best]);
                const int op = o_parent[ob], bp = o_parent[best];
                const uint32_t ot = o_token[ob], bt = o_token[best];
                if (oc < bc || (oc == bc && (op < bp || (op == bp && (ot < bt || (ot == bt && o_row[ob] < o_row[best])))))) best = ob;
                continue;
            }
            const double oc = o_cost[ob], bc = o_cost[best];
            const int op = o_parent[ob], bp = o_parent[best];
            const uint32_t ot = o_token[ob], bt = o_token[best];
            if (oc < bc || (oc == bc && (op < bp || (op == bp && (ot < bt || (ot == bt && o_row[ob] < o_row[best])))))) best = ob;
        }
        const double nc = o_cost[best];
        const int np = o_parent[best], nr = o_row[best];
        const uint32_t nt = o_token[best];
        const double no = o_own[best];
        const int ni = o_idx[best];
        __syncwarp();
        if (lane == 0) {
            o_cost[best] = o_cost[n_open - 1];
            o_parent[best] = o_parent[n_open - 1];
            o_token[best] = o_token[n_open - 1];
            o_row[best] = o_row[n_open - 1];
            o_own[best] = o_own[n_open - 1];
            o_idx[best] = o_idx[n_open - 1];
            s_held[count] = nt;
            s_parents[count] = np;
            s_own[count] = no;
        }
        --n_open;
        pred = nt;
        pidx = ni;
        parent = count;
        row = nr + 1;
        path = nc;
        ++count;
        __syncwarp();
    }
    PROF_MARK(4);
    if (lane == 0) s_count = count;
    __syncwarp();
    // the history copy: the chain of rows' best candidates, then the longest match of history and chain
    if (a.copy && hist_len >= 4) {
        uint32_t cp = pending;
        int cpi = 0;
        for (int r = 0; tabs && r < a.rows; ++r) {
            const float sc = lane < a.topk ? a.tab_sc[(size_t)pair_of(r, cpi) * LP_TOP + lane] : -__int_as_float(0x7f800000);
            int pick = (lane < a.topk && sc > -__int_as_float(0x7f800000)) ? lane : 1 << 30;
            float best = (pick < (1 << 30)) ? sc : -__int_as_float(0x7f800000);
            for (int m = 16; m; m >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, best, m);
                const int op = __shfl_xor_sync(0xffffffffu, pick, m);
                if (ob > best || (ob == best && op < pick)) {
                    best = ob;
                    pick = op;
                }
            }
            if (pick >= (1 << 30)) pick = 0;
            cpi = pick;
            cp = cand_id(a, r, pick);
            if (lane == 0) ch[r] = cp;
            __syncwarp();
        }
        for (int r = 0; !tabs && r < a.rows; ++r) {
            const uint32_t my_id = lane < a.topk ? cand_id(a, r, lane) : 0;
            if (lane < a.topk) cid[lane] = my_id;
            __syncwarp();
            if (rank256) {
                stage_all(a, cp, r, cid, s_succ, s_pr, s_hp, lane);
            } else if (rank_ok) {
                stage_row(a.pred + (size_t)cp * a.rank, s_pr, a.rank, lane);
                stage_row(a.hp + (size_t)r * a.rank, s_hp, a.rank, lane);
                for (int j = 0; j < a.topk; ++j) stage_row(a.succ + (size_t)cid[j] * a.rank, s_succ[j], a.rank, lane);
            }
            __syncwarp();
            float sc = -__int_as_float(0x7f800000);
            if (lane < a.topk) {
                float e = 0;
                if (rank_ok) {
                    for (int k = 0; k < a.rank; ++k) e = __fadd_rn(e, __fmul_rn(__fmul_rn(bff(s_pr[k]), bff(s_hp[k])), bff(s_succ[lane][k])));
                } else {
                    const uint16_t* pr = a.pred + (size_t)cp * a.rank;
                    const uint16_t* hp = a.hp + (size_t)r * a.rank;
                    const uint16_t* sv = a.succ + (size_t)my_id * a.rank;
                    for (int k = 0; k < a.rank; ++k) e = __fadd_rn(e, __fmul_rn(__fmul_rn(bff(pr[k]), bff(hp[k])), bff(sv[k])));
                }
                sc = __fadd_rn(__uint_as_float(a.words[r * LP_WORDS + 2 + LP_TOP + lane]), e);
            }
            // the first candidate with the greatest score (strictly greater replaces): pick 0 if none beats -inf
            int pick = (lane < a.topk && sc > -__int_as_float(0x7f800000)) ? lane : 1 << 30;
            float best = (pick < (1 << 30)) ? sc : -__int_as_float(0x7f800000);
            for (int m = 16; m; m >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, best, m);
                const int op = __shfl_xor_sync(0xffffffffu, pick, m);
                if (ob > best || (ob == best && op < pick)) {
                    best = ob;
                    pick = op;
                }
            }
            if (pick >= (1 << 30)) pick = 0;
            cp = cand_id(a, r, pick);
            if (lane == 0) ch[r] = cp;
            __syncwarp();
        }
        __syncwarp();
        PROF_MARK(5);
        const int L = hist_len;
        const bool in_sh = L <= HIST_SH;
        if (in_sh) {
            const int words = (L + 3) / 4;  // the history block is 16-byte aligned and at least HIST_SH words long
            const uint4* src = reinterpret_cast<const uint4*>(a.hist);
            uint4* dst = reinterpret_cast<uint4*>(s_hist);
            uint4 t[HIST_SH / 4 / 32];
#pragma unroll
            for (int q = 0; q < HIST_SH / 4 / 32; ++q)
                if (lane + q * 32 < words) t[q] = src[lane + q * 32];
#pragma unroll
            for (int q = 0; q < HIST_SH / 4 / 32; ++q)
                if (lane + q * 32 < words) dst[lane + q * 32] = t[q];
        }
        __syncwarp();
        const uint32_t* H = in_sh ? s_hist : a.hist;
        int bsum = 0, bdm = 0, bidx = -1, bhm = 0;
        for (int idx = L - 2 - lane; idx >= 0; idx -= 32) {
            if (H[idx] != pending) continue;
            int hm = 1;
            while (hm <= idx && hm < 64 && H[idx - hm] == H[L - 1 - hm]) ++hm;
            int dm = 0;
            while (dm < a.rows && idx + 1 + dm < L && H[idx + 1 + dm] == ch[dm]) ++dm;
            if (dm >= 3 && hm + dm >= 6) {
                const int sm = hm + dm;
                if (bidx < 0 || sm > bsum || (sm == bsum && (dm > bdm || (dm == bdm && idx > bidx)))) {
                    bsum = sm;
                    bdm = dm;
                    bidx = idx;
                    bhm = hm;
                }
            }
        }
        for (int m = 16; m; m >>= 1) {
            const int os = __shfl_xor_sync(0xffffffffu, bsum, m), od = __shfl_xor_sync(0xffffffffu, bdm, m);
            const int oi = __shfl_xor_sync(0xffffffffu, bidx, m), oh = __shfl_xor_sync(0xffffffffu, bhm, m);
            if (oi >= 0 && (bidx < 0 || os > bsum || (os == bsum && (od > bdm || (od == bdm && oi > bidx))))) {
                bsum = os;
                bdm = od;
                bidx = oi;
                bhm = oh;
            }
        }
        if (bidx >= 0) {
            const int avail = L - (bidx + 1);
            const int copy_rows = (bdm >= 10 && avail > a.rows) ? min(31, avail) : a.rows;
            if (lane == 0) {
                for (int r = 0; r < copy_rows; ++r) {
                    s_held[r] = (bidx + 1 + r < L) ? H[bidx + 1 + r] : ch[min(r, a.rows - 1)];
                    s_parents[r] = r - 1;
                    s_own[r] = 0.0;
                }
                s_count = copy_rows;
            }
        }
    }
    __syncwarp();
    PROF_MARK(6);
    if (a.prof && lane == 0) for (int i = 0; i < 7; ++i) a.prof[i] = pc[i];
#undef PROF_MARK
    const int n = s_count;
    if (lane < n) {
        a.held[lane] = s_held[lane];
        a.parents[lane] = s_parents[lane];
        a.own[lane] = s_own[lane];
    }
    if (lane == 0) *a.grown = n;
}
