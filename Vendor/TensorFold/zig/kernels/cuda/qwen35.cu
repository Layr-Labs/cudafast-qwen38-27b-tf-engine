// Qwen3.5's hybrids (2B, 27B) on CUDA: the Metal kernels' arithmetic and rounding; a segment is one stream's window.

#include "qwen35.cuh"
#include "qwen35_attn_tc.cuh"

using namespace tf_q35;

// P (device counts, one graph for every shape): chunks, width, rows, windows, recorded; rowpar/rowanc give the tree.
__device__ __forceinline__ bool past_windows(const int* P, int s) { return P != nullptr && s >= P[3]; }
__device__ __forceinline__ bool recorded(const int* P) { return P != nullptr && P[4] != 0; }

// One 64-entry group's sum, its entries added in order (x 16-byte aligned).
__device__ __forceinline__ float group_sum(const __nv_bfloat16* x) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; j += 8) {
        const uint4 w = *reinterpret_cast<const uint4*>(x + j);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int k = 0; k < 8; k++) s += bf(b[k]);
    }
    return s;
}

// MLX affine 4-bit token rows: out = bf16(s * q + b), D wide.
extern "C" __global__ void __launch_bounds__(256) tf_q35_embed(const int* __restrict__ ids, const uint8_t* __restrict__ W,
                                                                 const __nv_bfloat16* __restrict__ S, const __nv_bfloat16* __restrict__ B,
                                                                 __nv_bfloat16* __restrict__ out, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t id = ids[blockIdx.x];
    for (int b = threadIdx.x; b < D / 2; b += blockDim.x) {
        const uint8_t v = W[id * (D / 2) + b];
        const size_t g = id * (D / 64) + (2 * b) / 64;
        const float s = bf(S[g]), z = bf(B[g]);
        out[(size_t)blockIdx.x * D + 2 * b] = __float2bfloat16_rn(s * (float)(v & 15) + z);
        out[(size_t)blockIdx.x * D + 2 * b + 1] = __float2bfloat16_rn(s * (float)(v >> 4) + z);
    }
}

// MLX affine 8-bit token rows (a mixed checkpoint's, or widened to 8 at load): out = bf16(s * q + b), D wide.
extern "C" __global__ void __launch_bounds__(256) tf_q35_embed8(const int* __restrict__ ids, const uint8_t* __restrict__ W,
                                                                  const __nv_bfloat16* __restrict__ S, const __nv_bfloat16* __restrict__ B,
                                                                  __nv_bfloat16* __restrict__ out, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t id = ids[blockIdx.x];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const size_t g = id * (D / 64) + d / 64;
        out[(size_t)blockIdx.x * D + d] = __float2bfloat16_rn(bf(S[g]) * (float)W[id * D + d] + bf(B[g]));
    }
}

// Image rows into a prompt chunk's embeddings: row m of H (D wide) takes feature row src[m] where src[m] >= 0.
extern "C" __global__ void __launch_bounds__(256) tf_q35_splice(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ F,
                                                                  const int* __restrict__ src, int D, const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const int s = src[blockIdx.x];
    if (s < 0) return;
    for (int i = threadIdx.x; i < D; i += blockDim.x) H[(size_t)blockIdx.x * D + i] = F[(size_t)s * D + i];
}

__device__ __forceinline__ void release_next() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void await_prior() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// griddepcontrol.wait's "memory" clobber does not order loads through __restrict__ (noalias) pointers, and ptxas
// (13.0, sm_121) moves ld.global.nc loads across griddepcontrol.wait (ACQBULK) in either direction while it keeps plain
// loads in place. So a pointer to anything an earlier kernel of the forward writes (or reads, for an output) is passed
// through this opaque asm after the wait and used only through the returned local (no .nc, and its loads and stores
// depend on the wait as data).
template <typename T>
__device__ __forceinline__ T* after_wait(T* p) {
    asm volatile("" : "+l"(p)::"memory");
    return p;
}

// A weight or table prefetch meant to overlap the kernel before goes through plain(p) (no __restrict__, so no .nc load
// ptxas could sink below the wait) with its values used by an asm before the wait.
template <typename T>
__device__ __forceinline__ T* plain(T* p) {
    asm volatile("" : "+l"(p));
    return p;
}
__device__ __forceinline__ void keep4(const uint4& v) { asm volatile("" ::"r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)); }

// h = bf16(h + r) (r may be null), x = bf16(w * (h * inv)), XS (may be null) x's group sums; D 256k up to 8192.
extern "C" __global__ void __launch_bounds__(256) tf_q35_norm(__nv_bfloat16* __restrict__ H0, const __nv_bfloat16* __restrict__ R0,
                                                               const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ X0, float eps, int D,
                                                               const int* __restrict__ P, float* __restrict__ XS0) {
    constexpr int MAXE = 32;
    __shared__ float red[8];
    __shared__ __align__(16) __nv_bfloat16 xrow[8192];
    release_next();
    const int t = threadIdx.x;
    const int E = D / 256;
    // a programmatic launch (cuda_engine.zig's norm): the weights stream in while the kernel before finishes,
    // then griddepcontrol.wait before any input (H, R, P) is read; a plain launch's wait returns at once. Every load
    // is straight-line (an index past D repeats the row's last 8 entries, unused), so H, R and P are in flight together
    uint4 wv[4];
    if ((E & 3) == 0) {
#pragma unroll
        for (int j = 0; j < 4; j++) wv[j] = *reinterpret_cast<const uint4*>(plain(W) + min((t + 256 * j) * 8, D - 8));
#pragma unroll
        for (int j = 0; j < 4; j++) keep4(wv[j]);
    }
    await_prior();
    __nv_bfloat16* H = after_wait(H0);
    const __nv_bfloat16* R = after_wait(R0);
    __nv_bfloat16* X = after_wait(X0);
    float* XS = after_wait(XS0);
    if ((E & 3) == 0) {
        const size_t row = blockIdx.x;
        uint4 hw[4], rw[4];
#pragma unroll
        for (int j = 0; j < 4; j++) hw[j] = *reinterpret_cast<const uint4*>(H + row * D + min((t + 256 * j) * 8, D - 8));
        if (R != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; j++) rw[j] = *reinterpret_cast<const uint4*>(R + row * D + min((t + 256 * j) * 8, D - 8));
        }
        if (past_rows(P, blockIdx.x)) return;
        // h + r and w * (h * inv) move 8 entries a lane through xrow; the squares keep a thread's E entries in order
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d >= D) break;
            if (R != nullptr) {
                __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(&hw[j]);
                const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rw[j]);
#pragma unroll
                for (int k = 0; k < 8; k++) hb[k] = __float2bfloat16_rn(bf(hb[k]) + bf(rb[k]));
                *reinterpret_cast<uint4*>(H + row * D + d) = hw[j];
            }
            *reinterpret_cast<uint4*>(xrow + d) = hw[j];
        }
        __syncthreads();
        float sq = 0.0f;
        const uint2* mine = reinterpret_cast<const uint2*>(xrow) + t * (E / 4);
#pragma unroll
        for (int i = 0; i < MAXE / 4; i++) {
            if (4 * i >= E) break;
            const uint2 v = mine[i];
            const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const float h = bf(vb[k]);
                sq = fmaf(h, h, sq);
            }
        }
        sq = warp_sum(sq);
        if ((t & 31) == 0) red[t >> 5] = sq;
        __syncthreads();
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += red[i];
        const float scale = rsqrtf(sum / (float)D + eps);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d >= D) break;
            const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hw[j]);
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv[j]);
            uint4 xw;
            __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(&xw);
#pragma unroll
            for (int k = 0; k < 8; k++) xb[k] = __float2bfloat16_rn(bf(wb[k]) * (bf(hb[k]) * scale));
            *reinterpret_cast<uint4*>(X + row * D + d) = xw;
            *reinterpret_cast<uint4*>(xrow + d) = xw;
        }
        if (XS == nullptr) return;
        __syncthreads();
        for (int g = t; g < D / 64; g += blockDim.x) XS[row * (D / 64) + g] = group_sum(xrow + g * 64);
        return;
    }
    if (past_rows(P, blockIdx.x)) return;
    const size_t row = blockIdx.x;
    // E not a multiple of 4: a thread's entries one at a time
    float hv[MAXE];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXE; i++) {
        if (i >= E) break;
        const size_t at = row * D + t * E + i;
        float h = bf(H[at]);
        if (R != nullptr) {
            const __nv_bfloat16 hb = __float2bfloat16_rn(h + bf(R[at]));
            H[at] = hb;
            h = bf(hb);
        }
        hv[i] = h;
        ss = fmaf(h, h, ss);
    }
    ss = warp_sum(ss);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < 8; i++) total += red[i];
    const float inv = rsqrtf(total / (float)D + eps);
#pragma unroll
    for (int i = 0; i < MAXE; i++) {
        if (i >= E) break;
        const int d = t * E + i;
        const __nv_bfloat16 x = __float2bfloat16_rn(bf(W[d]) * (hv[i] * inv));
        X[row * D + d] = x;
        xrow[d] = x;
    }
    if (XS == nullptr) return;
    __syncthreads();
    for (int g = t; g < D / 64; g += blockDim.x) XS[row * (D / 64) + g] = group_sum(xrow + g * 64);
}

// 64-group sums of bf16 rows D wide (the decode projections' input sums). Grid (rows, groups / 64), a thread a group.
extern "C" __global__ void __launch_bounds__(64) tf_q35_sums(const __nv_bfloat16* __restrict__ X0, float* __restrict__ XS0, int D, const int* __restrict__ P0) {
    release_next();
    await_prior();
    const __nv_bfloat16* X = after_wait(X0);
    float* XS = after_wait(XS0);
    const int* P = after_wait(P0);
    if (past_rows(P, blockIdx.x)) return;
    const size_t row = blockIdx.x;
    const int g = blockIdx.y * blockDim.x + threadIdx.x;
    if (g < D / 64) XS[row * (D / 64) + g] = group_sum(X + row * D + g * 64);
}

// Conv, silu, q/k norms, decay and write: blockDim.y independent row warps per head, taps from the parents.
extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_pre(Geo geo, const __nv_bfloat16* __restrict__ QKV0, const Seg* __restrict__ segs,
                                                                  const int* __restrict__ rowseg, const int* __restrict__ rowpar, int layer,
                                                                  const __nv_bfloat16* __restrict__ CW,
                                                                  const __nv_bfloat16* __restrict__ Ain0, const __nv_bfloat16* __restrict__ Bin0,
                                                                  const float* __restrict__ ALOG, const __nv_bfloat16* __restrict__ DT,
                                                                  __nv_bfloat16* __restrict__ Q0, __nv_bfloat16* __restrict__ K0,
                                                                  __nv_bfloat16* __restrict__ V0, float* __restrict__ G0, __nv_bfloat16* __restrict__ BETA0,
                                                                  const int* __restrict__ P) {
    release_next();
    await_prior();
    const __nv_bfloat16* QKV = after_wait(QKV0);
    const __nv_bfloat16* Ain = after_wait(Ain0);
    const __nv_bfloat16* Bin = after_wait(Bin0);
    __nv_bfloat16* Q = after_wait(Q0);
    __nv_bfloat16* K = after_wait(K0);
    __nv_bfloat16* V = after_wait(V0);
    float* G = after_wait(G0);
    __nv_bfloat16* BETA = after_wait(BETA0);
    const int lane = threadIdx.x;
    const int head = blockIdx.y;
    const int w = blockIdx.z * blockDim.y + threadIdx.y;
    if (past_rows(P, w)) return;
    const int NK = geo.nk, NV = geo.nv, C = geo.conv();
    const Seg sg = segs[rowseg[w]];
    const __nv_bfloat16* CS = reinterpret_cast<const __nv_bfloat16*>(sg.conv) + (size_t)layer * geo.conv_rows();
    const bool isq = head < NK, isk = !isq && head < 2 * NK;
    const int c0 = head * DK;  // q, k and v channels are consecutive heads of DK
    int src[TAPS];  // tap t reads the row TAPS - 1 - t tokens back: a forward row, or -1 - (tail row)
#pragma unroll
    for (int tap = 0; tap < TAPS; tap++) src[tap] = back(rowpar, w, TAPS - 1 - tap);
    constexpr int PER = DK / 32;
    // a recorded window's row logs the tail after it (taps 1..TAPS-1) here, so conv_commit only moves the stream's
    __nv_bfloat16* tail = recorded(P) ? log_row(geo, sg, layer, w - sg.row0).conv : nullptr;
    // a lane's PER channels in 8-byte loads: each tap's row, the weights (PER x TAPS in a row), the logged tail
    static_assert(PER == 4 && TAPS == 4, "a tap's channels and a channel's taps are one uint2");
    const int cl = c0 + lane * PER;
    uint2 xw[TAPS], ww[TAPS];
#pragma unroll
    for (int tap = 0; tap < TAPS; tap++) {
        const int at = src[tap];
        xw[tap] = *reinterpret_cast<const uint2*>(at < 0 ? CS + (size_t)(-1 - at) * C + cl : QKV + (size_t)at * C + cl);
        ww[tap] = *reinterpret_cast<const uint2*>(CW + (size_t)cl * TAPS + tap * PER);
        if (tail != nullptr && tap > 0) *reinterpret_cast<uint2*>(tail + (size_t)(tap - 1) * C + cl) = xw[tap];
    }
    const __nv_bfloat16* cw = reinterpret_cast<const __nv_bfloat16*>(ww);
    float vals[PER];
#pragma unroll
    for (int j = 0; j < PER; j++) {
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < TAPS; tap++) acc += bf(cw[j * TAPS + tap]) * bf(reinterpret_cast<const __nv_bfloat16*>(&xw[tap])[j]);
        const float conv = rbf(acc);
        const float sig = rbf(1.0f / (1.0f + expf(-conv)));
        vals[j] = rbf(conv * sig);
    }
    if (isq || isk) {
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; j++) ss += vals[j] * vals[j];
        ss = warp_sum(ss);
        const float inv = rsqrtf(ss / (float)DK + (1e-6f / (float)DK));
        const float scale = isq ? rbf(1.0f / (float)DK) : rbf(rsqrtf((float)DK));
        __nv_bfloat16* out = isq ? Q + ((size_t)w * NK + head) * DK : K + ((size_t)w * NK + head - NK) * DK;
        uint2 ow;
        __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&ow);
#pragma unroll
        for (int j = 0; j < PER; j++) op[j] = __float2bfloat16_rn(scale * rbf(vals[j] * inv));
        *reinterpret_cast<uint2*>(out + lane * PER) = ow;
    } else {
        const int hv = head - 2 * NK;
        uint2 vw;
        __nv_bfloat16* vp = reinterpret_cast<__nv_bfloat16*>(&vw);
#pragma unroll
        for (int j = 0; j < PER; j++) vp[j] = __float2bfloat16_rn(vals[j]);
        *reinterpret_cast<uint2*>(V + ((size_t)w * NV + hv) * DK + lane * PER) = vw;
        if (lane == 0) {
            const float s = rbf(bf(Ain[w * NV + hv]) + bf(DT[hv]));
            const float sp = rbf(fmaxf(s, 0.0f) + logf(1.0f + expf(-fabsf(s))));
            G[w * NV + hv] = expf(-expf(ALOG[hv]) * sp);
            BETA[w * NV + hv] = __float2bfloat16_rn(1.0f / (1.0f + expf(-bf(Bin[w * NV + hv]))));
        }
    }
}

// A segment's conv tail: a recorded window leaves the stream's (gdn_pre logged its rows'); others update it in place.
extern "C" __global__ void tf_q35_conv_commit(Geo geo, const __nv_bfloat16* __restrict__ QKV0, const Seg* __restrict__ segs,
                                                const int* __restrict__ rowpar, int layer, const int* __restrict__ P) {
    release_next();
    await_prior();
    const __nv_bfloat16* QKV = after_wait(QKV0);
    const int C = geo.conv();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || past_windows(P, blockIdx.y)) return;
    const Seg sg = segs[blockIdx.y];
    __nv_bfloat16* CS = reinterpret_cast<__nv_bfloat16*>(sg.conv) + (size_t)layer * geo.conv_rows();
    if (recorded(P)) {
        // gdn_pre logged each row's tail
        if (sg.rows != 1) return;
        __nv_bfloat16* before = reinterpret_cast<__nv_bfloat16*>(sg.before + geo.nl * geo.rec() * sizeof(float)) + (size_t)layer * geo.conv_rows();
        for (int r = 0; r < TAPS - 1; ++r) before[(size_t)r * C + c] = CS[(size_t)r * C + c];
    }
    // ascending: slot r takes [r + rows], which is either a later state slot (not yet written) or a row
    for (int r = 0; r < TAPS - 1; r++) {
        const int at = r + sg.rows;
        CS[(size_t)r * C + c] = at < TAPS - 1 ? CS[(size_t)at * C + c] : QKV[(size_t)(sg.row0 + at - (TAPS - 1)) * C + c];
    }
}

// Gated delta rule, each row's state its parent's: a chain keeps it in registers, a branch rebuilds it from the log.
constexpr bool chain_root_state = true;
extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_chain(Geo geo, const __nv_bfloat16* __restrict__ q0, const __nv_bfloat16* __restrict__ k0,
                                                                     const __nv_bfloat16* __restrict__ v0, const float* __restrict__ g0,
                                                                     const __nv_bfloat16* __restrict__ beta0, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowpar, int layer, __nv_bfloat16* __restrict__ y0,
                                                                     const int* __restrict__ P) {
    release_next();
    await_prior();
    const __nv_bfloat16* q = after_wait(q0);
    const __nv_bfloat16* k = after_wait(k0);
    const __nv_bfloat16* v = after_wait(v0);
    const float* g = after_wait(g0);
    const __nv_bfloat16* beta = after_wait(beta0);
    __nv_bfloat16* y = after_wait(y0);
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV;
    const int hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    const int dk = threadIdx.x;
    const int dv = blockIdx.y * 4 + threadIdx.y;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv) * DK + PT * dk;
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    const bool record = recorded(P);
    float state[PT], base[PT], root[PT];
#pragma unroll
    for (int i = 0; i < PT; i++) state[i] = base[i] = root[i] = st[i];
    int prev = -1;  // the row the registers' state follows
    // row r's inputs, loaded while row r - 1 computes
    float kn[PT], qn[PT], vn = 0.0f, gn = 0.0f, bn = 0.0f;
    int pn = -1;
    auto load = [&](int row) {
        const uint2 kw = *reinterpret_cast<const uint2*>(k + ((size_t)row * NK + hk) * DK + PT * dk);
        const uint2 qw = *reinterpret_cast<const uint2*>(q + ((size_t)row * NK + hk) * DK + PT * dk);
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
        const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&qw);
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kn[i] = bf(kb[i]);
            qn[i] = bf(qb[i]);
        }
        vn = bf(v[((size_t)row * NV + hv) * DK + dv]);
        gn = g[row * NV + hv];
        bn = bf(beta[row * NV + hv]);
        pn = rowpar[row];
    };
    if (sg.rows > 0) load(sg.row0);
    for (int r = 0; r < sg.rows; r++) {
        const int row = sg.row0 + r;
        float kc[PT], qc[PT];
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kc[i] = kn[i];
            qc[i] = qn[i];
        }
        const float vc = vn, gc = gn, bc = bn;
        const int parent = pn;
        if (r + 1 < sg.rows) load(row + 1);
        if (parent != prev) {
            // a branch: its parent's state from the stream's and the ancestors' logged updates, oldest first
            int anc[MAXW];
            int n = 0;
            int a = parent;
            for (; a >= 0 && a != sg.row0; a = rowpar[a]) anc[n++] = a;
            const bool from_root = a == sg.row0;
#pragma unroll
            for (int i = 0; i < PT; i++) state[i] = from_root ? root[i] : base[i];
            for (int j = n - 1; j >= 0; j--) {
                const int u = anc[j];
                const float ga = g[u * NV + hv], da = log_row(geo, sg, layer, u - sg.row0).delta[hv * DK + dv];
#pragma unroll
                for (int i = 0; i < PT; i++) {
                    state[i] = state[i] * ga;
                    state[i] = fmaf(bf(k[((size_t)u * NK + hk) * DK + PT * dk + i]), da, state[i]);
                }
            }
        }
        prev = row;
        float kv_mem = 0.0f;
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = state[i] * gc;
            kv_mem += state[i] * kc[i];
        }
        kv_mem = warp_sum(kv_mem);
        const float delta = (vc - kv_mem) * bc;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = fmaf(kc[i], delta, state[i]);
            out += state[i] * qc[i];
        }
        if (chain_root_state && r == 0)
#pragma unroll
            for (int i = 0; i < PT; i++) root[i] = state[i];
        out = warp_sum(out);
        if (dk == 0) y[((size_t)row * NV + hv) * DK + dv] = __float2bfloat16_rn(out);
        if (record) {
            const LogRow l = log_row(geo, sg, layer, r);
            if (dk == 0) {
                l.delta[hv * DK + dv] = delta;
                if (dv == 0) l.g[hv] = gc;
            }
            if (dv == 0 && hv % (NV / NK) == 0)
#pragma unroll
                for (int i = 0; i < PT; i++) l.k[hk * DK + PT * dk + i] = k[((size_t)row * NK + hk) * DK + PT * dk + i];
        }
    }
    if (record && sg.rows != 1) return;
#pragma unroll
    for (int i = 0; i < PT; i++) st[i] = state[i];
}

// out = bf16(silu(z) * bf16(w * rmsnorm(y))) for row m's value head hv, a warp; yw is the lane's four y values.
__device__ __forceinline__ void q35_post_row(Geo geo, uint2 yw, const __nv_bfloat16* __restrict__ Z, const __nv_bfloat16* __restrict__ NW,
                                             float eps, __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, size_t m, int hv) {
    const int lane = threadIdx.x;
    const size_t W = (size_t)geo.nv * DK;
    constexpr int PER = DK / 32;
    const size_t at = m * W + hv * DK + lane * PER;
    const uint2 zw = *reinterpret_cast<const uint2*>(Z + at);
    const uint2 nw = *reinterpret_cast<const uint2*>(NW + lane * PER);
    const __nv_bfloat16* yp = reinterpret_cast<const __nv_bfloat16*>(&yw);
    const __nv_bfloat16* zp = reinterpret_cast<const __nv_bfloat16*>(&zw);
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&nw);
    float yv[PER], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        yv[j] = bf(yp[j]);
        ss += yv[j] * yv[j];
    }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / (float)DK + eps);
    uint2 ow;
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&ow);
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const float x = rbf(bf(wp[j]) * (yv[j] * inv));
        const float zf = bf(zp[j]);
        op[j] = __float2bfloat16_rn(zf / (1.0f + expf(-zf)) * x);
    }
    *reinterpret_cast<uint2*>(OUT + at) = ow;
    if (XS == nullptr) return;
    // group_sum's order over each 64-wide group, its 16 lanes' values shuffled to the group's lane (lanes 0 and 1)
    static_assert(DK == 128 && PER == 4, "two 64-wide groups of sixteen lanes");
    float gs = 0.0f;
#pragma unroll
    for (int src = 0; src < 16; src++) {
        uint2 w;
        w.x = __shfl_sync(0xffffffffu, ow.x, (lane & 1) * 16 + src);
        w.y = __shfl_sync(0xffffffffu, ow.y, (lane & 1) * 16 + src);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int j = 0; j < PER; j++) gs += bf(b[j]);
    }
    if (lane < DK / 64) XS[m * (W / 64) + hv * (DK / 64) + lane] = gs;
}

// out = bf16(silu(z) * bf16(w * rmsnorm(y))) per value head; blockDim.y independent row warps, at most four.
extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_post(Geo geo, const __nv_bfloat16* __restrict__ Y0, const __nv_bfloat16* __restrict__ Z0,
                                                                   const __nv_bfloat16* __restrict__ NW, float eps, __nv_bfloat16* __restrict__ OUT0,
                                                                   const int* __restrict__ P, float* __restrict__ XS0) {
    const int hv = blockIdx.y;
    release_next();
    await_prior();
    const __nv_bfloat16* Y = after_wait(Y0);
    const size_t m = blockIdx.z * blockDim.y + threadIdx.y;
    if (past_rows(P, m)) return;
    const uint2 yw = *reinterpret_cast<const uint2*>(Y + m * geo.nv * DK + hv * DK + threadIdx.x * (DK / 32));
    q35_post_row(geo, yw, after_wait(Z0), NW, eps, after_wait(OUT0), after_wait(XS0), m, hv);
}

// gdn_post fused into a recorded window's slot scan (count != nullptr): the last of a value head's dv-block CTAs to
// finish runs the head's post on the window's rows, a warp a row, with gdn_post's own arithmetic.
struct GdnPost { const __nv_bfloat16* z; const __nv_bfloat16* nw; __nv_bfloat16* out; float* xs; unsigned* count; float eps; };

__device__ __noinline__ void q35_post_tail(Geo geo, const GdnPost pa, const Seg* segs, const __nv_bfloat16* y, const int* P) {
    if (pa.count == nullptr || past_windows(P, blockIdx.z / geo.nv)) return;
    __threadfence();  // this CTA's y rows, then its arrival
    __syncthreads();
    int last = 0;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned* c = pa.count + blockIdx.z;
        last = atomicAdd(c, 1u) == gridDim.y - 1;
        if (last) *c = 0;  // every CTA of the head has arrived: ready for the next layer's scan
    }
    if (!__syncthreads_or(last)) return;
    __threadfence();
    const int hv = blockIdx.z % geo.nv;
    const Seg sg = segs[blockIdx.z / geo.nv];
    for (int r = threadIdx.y; r < sg.rows; r += blockDim.y) {
        const size_t m = sg.row0 + r;
        if (past_rows(P, m)) continue;
        // the other CTAs' y rows, from L2
        const uint2 yw = __ldcg(reinterpret_cast<const uint2*>(y + m * geo.nv * DK + hv * DK + threadIdx.x * (DK / 32)));
        q35_post_row(geo, yw, pa.z, pa.nw, pa.eps, pa.out, pa.xs, m, hv);
    }
}

#include "qwen35_head_norm.cuh"
#include "qwen35_head_rope.cuh"

// Queries rotated in place, keys rotated and values copied into the row's cache slot; grid (rows), HD threads.
extern "C" __global__ void __launch_bounds__(256) tf_q35_rope(Geo geo, __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Kn,
                                                                const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                                                const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                const int* __restrict__ P, const int* __restrict__ M) {
    release_next();
    await_prior();
    rope_rows(geo, after_wait(Qn), after_wait(Kn), after_wait(Vr), segs, rowseg, rowanc, P, M, StoreBf16{layer, geo.kvh});
}

// Causal attention in chunks of chunk_keys(the row's keys), a warp a query head; partials by (row, head, chunk).
#include "qwen35_attn.cuh"

// A row's chunks' partials into each (row, query head)'s output, rows qh * HD wide. Grid (qh, rows), a warp each.
extern "C" __global__ void __launch_bounds__(32) tf_q35_attn_merge(Geo geo, const float* __restrict__ PM0, const float* __restrict__ PL0,
                                                                     const float* __restrict__ PO0, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc,
                                                                     const int* __restrict__ P, __nv_bfloat16* __restrict__ OUT0) {
    release_next();
    await_prior();
    const float* PM = after_wait(PM0);
    const float* PL = after_wait(PL0);
    const float* PO = after_wait(PO0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    constexpr int DPL = HD / 32;
    const int QH = geo.qh;
    const int lane = threadIdx.x, qh = blockIdx.x;
    if (past_rows(P, blockIdx.y)) return;
    const size_t m = blockIdx.y;
    const int keys = segs[rowseg[m]].len + rows_seen(rowanc[m]);
    const int ck = chunk_keys(keys), nch = (keys + ck - 1) / ck;
    const size_t base = (m * QH + qh) * MAXCH;
    float mx = -INFINITY;
    for (int c = 0; c < nch; c++)
        if (PL[base + c] > 0.0f) mx = fmaxf(mx, PM[base + c]);
    float lsum = 0.0f, acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
    for (int c = 0; c < nch; c++) {
        if (!(PL[base + c] > 0.0f)) continue;
        const float e = __expf(PM[base + c] - mx);
        lsum = fmaf(PL[base + c], e, lsum);
#pragma unroll
        for (int i = 0; i < DPL; i++) acc[i] = fmaf(PO[(base + c) * HD + lane * DPL + i], e, acc[i]);
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) OUT[(m * QH + qh) * HD + lane * DPL + i] = __float2bfloat16_rn(acc[i] / lsum);
}

// A 256-thread block's outputs' 64-entry group sums into XS (null: none); rows are whole groups, so a block's are too.
__device__ __forceinline__ void block_sums(__nv_bfloat16* done, size_t total, float* XS) {
    if (XS == nullptr) return;
    __syncthreads();
    const size_t first = (size_t)blockIdx.x * 256 + threadIdx.x * 64;
    if (threadIdx.x < 4 && first < total) XS[first / 64] = group_sum(done + threadIdx.x * 64);
}

// out = bf16(y * bf16(sigmoid(gate))), the gate the second half of each query head's q_proj rows; rows qh * HD wide.
extern "C" __global__ void tf_q35_attn_gate(Geo geo, const __nv_bfloat16* __restrict__ Y0, const __nv_bfloat16* __restrict__ QRAW0,
                                              __nv_bfloat16* __restrict__ OUT0, int rows, const int* __restrict__ P, float* __restrict__ XS0) {
    __shared__ __align__(16) __nv_bfloat16 done[256];
    release_next();
    await_prior();
    const __nv_bfloat16* Y = after_wait(Y0);
    const __nv_bfloat16* QRAW = after_wait(QRAW0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    float* XS = after_wait(XS0);
    const size_t W = (size_t)geo.qh * HD;
    if (P != nullptr) rows = P[2];
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)rows * W) {
        const size_t r = i / W;
        const int d = (int)(i % W), head = d / HD;
        const float gate = bf(QRAW[r * 2 * W + head * 2 * HD + HD + d % HD]);
        const float s = rbf(1.0f / (1.0f + expf(-gate)));
        OUT[i] = done[threadIdx.x] = __float2bfloat16_rn(bf(Y[i]) * s);
    }
    block_sums(done, (size_t)rows * W, XS);
}

// tf_q35_attn_merge then tf_q35_attn_gate in one warp a (row, query head): the merge's bf16 output gated as attn_gate
// does, and the head's four 64-entry group sums of the gated row, each as block_sums adds them (null XS: none).
extern "C" __global__ void __launch_bounds__(32) tf_q35_attn_merge_gate(Geo geo, const float* __restrict__ PM0, const float* __restrict__ PL0,
                                                                          const float* __restrict__ PO0, const Seg* __restrict__ segs,
                                                                          const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc,
                                                                          const int* __restrict__ P, const __nv_bfloat16* __restrict__ QRAW0,
                                                                          __nv_bfloat16* __restrict__ OUT0, float* __restrict__ XS0) {
    __shared__ __align__(16) __nv_bfloat16 done[HD];
    release_next();
    await_prior();
    const float* PM = after_wait(PM0);
    const float* PL = after_wait(PL0);
    const float* PO = after_wait(PO0);
    const __nv_bfloat16* QRAW = after_wait(QRAW0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    float* XS = after_wait(XS0);
    constexpr int DPL = HD / 32;
    const int QH = geo.qh;
    const int lane = threadIdx.x, qh = blockIdx.x;
    if (past_rows(P, blockIdx.y)) return;
    const size_t m = blockIdx.y;
    const int keys = segs[rowseg[m]].len + rows_seen(rowanc[m]);
    const int ck = chunk_keys(keys), nch = (keys + ck - 1) / ck;
    const size_t base = (m * QH + qh) * MAXCH;
    float mx = -INFINITY;
    for (int c = 0; c < nch; c++)
        if (PL[base + c] > 0.0f) mx = fmaxf(mx, PM[base + c]);
    float lsum = 0.0f, acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
    for (int c = 0; c < nch; c++) {
        if (!(PL[base + c] > 0.0f)) continue;
        const float e = __expf(PM[base + c] - mx);
        lsum = fmaf(PL[base + c], e, lsum);
#pragma unroll
        for (int i = 0; i < DPL; i++) acc[i] = fmaf(PO[(base + c) * HD + lane * DPL + i], e, acc[i]);
    }
    const size_t W = (size_t)QH * HD;
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        const int d = lane * DPL + i;
        const __nv_bfloat16 y = __float2bfloat16_rn(acc[i] / lsum);
        const float gate = bf(QRAW[m * 2 * W + qh * 2 * HD + HD + d]);
        const float s = rbf(1.0f / (1.0f + expf(-gate)));
        OUT[m * W + qh * HD + d] = done[d] = __float2bfloat16_rn(bf(y) * s);
    }
    if (XS == nullptr) return;
    __syncwarp();
    if (lane < HD / 64) XS[m * (W / 64) + qh * (HD / 64) + lane] = group_sum(done + lane * 64);
}

// tf_q35_head_norm of a (row, head) on threads 0-63, then tf_q35_rope's rotation of it: a query head's into Qn, a key
// head's with its value into the cache slot. Grid (rows, query heads + key heads), HD threads.
template <typename Store>
__device__ __forceinline__ void norm_rope_rows(const Geo& geo, const __nv_bfloat16* __restrict__ QRAW, const __nv_bfloat16* __restrict__ KRAW,
                                               const __nv_bfloat16* __restrict__ WQ, const __nv_bfloat16* __restrict__ WK,
                                               __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                               const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, const int* __restrict__ P,
                                               const int* __restrict__ M, float eps, Store store) {
    __shared__ float sums[2];
    __shared__ __align__(16) __nv_bfloat16 xs[HD];
    const int QH = geo.qh, KVH = geo.kvh;
    if (past_rows(P, blockIdx.x)) return;
    const size_t m = blockIdx.x;
    const int h = blockIdx.y, t = threadIdx.x;
    const bool query = h < QH;
    const int head = query ? h : h - QH;
    const __nv_bfloat16* x = query ? QRAW + m * (2 * QH * HD) + head * (2 * HD) : KRAW + m * (KVH * HD) + head * HD;
    const __nv_bfloat16* W = query ? WQ : WK;
    float v[4], acc = 0.0f;
    if (t < 64) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = bf(x[4 * t + i]);
            acc += v[i] * v[i];
        }
        acc = warp_sum(acc);
        if ((t & 31) == 0) sums[t >> 5] = acc;
    }
    __syncthreads();
    if (t < 64) {
        const float inv = rsqrtf((sums[0] + sums[1]) / (float)HD + eps);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int d = 4 * t + i;
            xs[d] = __float2bfloat16_rn(bf(W[d]) * (v[i] * inv));
        }
    }
    __syncthreads();
    const Seg sg = segs[rowseg[m]];
    int pos[3];
    row_positions(sg, rowanc[m], m, M, pos);
    const int slot = sg.len + (int)m - sg.row0;
    if (query) Qn[(m * QH + head) * HD + t] = rotate(xs, t, pos);
    else store(sg, head, slot, t, rotate(xs, t, pos), Vr[(m * KVH + head) * HD + t]);
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_norm_rope(Geo geo, const __nv_bfloat16* __restrict__ QRAW, const __nv_bfloat16* __restrict__ KRAW,
                                                                     const __nv_bfloat16* __restrict__ WQ, const __nv_bfloat16* __restrict__ WK,
                                                                     __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Vr,
                                                                     const Seg* __restrict__ segs, const int* __restrict__ rowseg,
                                                                     const Ancestor* __restrict__ rowanc, int layer, const int* __restrict__ P,
                                                                     const int* __restrict__ M, float eps) {
    release_next();
    await_prior();
    norm_rope_rows(geo, after_wait(QRAW), after_wait(KRAW), WQ, WK, after_wait(Qn), after_wait(Vr), segs, rowseg, rowanc, P, M, eps, StoreBf16{layer, geo.kvh});
}

#include "qwen35_rows.cuh"
#include "qwen35_slots.cuh"
#include "qwen35_slots16.cuh"
#include "qwen35_argmax.cuh"


extern "C" __global__ void __launch_bounds__(128) tf_q35_attn_partial(Geo geo, const __nv_bfloat16* __restrict__ Q0, const Seg* __restrict__ segs,
                                                                        const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                        const int* __restrict__ P, float scale,
                                                                        float* __restrict__ PM0, float* __restrict__ PL0, float* __restrict__ PO0) {
    release_next();
    await_prior();
    const __nv_bfloat16* Q = after_wait(Q0);
    float* PM = after_wait(PM0);
    float* PL = after_wait(PL0);
    float* PO = after_wait(PO0);
    extern __shared__ __align__(16) uint4 attn_tiles[];
    // A block takes every gridDim.y-th chunk, each chunk's arithmetic as its own block's: a short stream's rows need few
    // of the MAXCH chunks, and a block of these tiles holds its SM alone, so an idle chunk's block cost a dispatch.
    // A chunk past the row's keys only marks itself empty (attn_one's k0 >= k1 case), here without its loads.
    const size_t m = blockIdx.x;
    if (past_rows(P, (int)m)) return;
    const int G = geo.qh / geo.kvh;
    const bool own = segs[rowseg[m]].len < tf_q35_tc::GROUP_LEN;
    const int keys = segs[rowseg[m]].len + rows_seen(rowanc[m]), ck = chunk_keys(keys), nch = (keys + ck - 1) / ck;
    bool first = true;
    for (int c = blockIdx.y; c < MAXCH; c += gridDim.y) {
        if (c >= nch) {
            if (own && (int)threadIdx.x < G) {
                const size_t at = (m * geo.qh + (size_t)blockIdx.z * G + threadIdx.x) * MAXCH + c;
                PM[at] = -INFINITY;
                PL[at] = 0.0f;
            }
            continue;
        }
        if (!first) __syncthreads();
        first = false;
        tf_q35_tc::attn_one<tf_q35_tc::Bf16Tiles1<128>>({attn_tiles, nullptr, nullptr, nullptr}, geo, Q, segs, rowseg, rowanc, layer, P, scale, PM, PL, PO, c);
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_attn_rows(Geo geo, const __nv_bfloat16* __restrict__ Q0, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                     const int* __restrict__ P, float scale, int total,
                                                                     float* __restrict__ PM0, float* __restrict__ PL0, float* __restrict__ PO0) {
    release_next();
    await_prior();
    const __nv_bfloat16* Q = after_wait(Q0);
    float* PM = after_wait(PM0);
    float* PL = after_wait(PL0);
    float* PO = after_wait(PO0);
    extern __shared__ __align__(16) uint4 attn_tiles[];
    // every gridDim.y-th chunk, as tf_q35_attn_partial's; a group without a long stream's row has nothing at any chunk
    __shared__ int long_rows;
    const int RB = 64 / (geo.qh / geo.kvh), m0 = blockIdx.x * RB;
    if (threadIdx.x == 0) long_rows = 0;
    __syncthreads();
    if ((int)threadIdx.x < RB) {
        const int m = m0 + threadIdx.x;
        if (m < total && !past_rows(P, m) && segs[rowseg[m]].len >= tf_q35_tc::GROUP_LEN) atomicOr(&long_rows, 1);
    }
    __syncthreads();
    if (!long_rows) return;
    for (int c = blockIdx.y; c < MAXCH; c += gridDim.y) {
        if (c != (int)blockIdx.y) __syncthreads();
        tf_q35_tc::attn_rows<4, tf_q35_tc::Bf16Tiles<256>>({attn_tiles, nullptr, nullptr, nullptr}, geo, Q, segs, rowseg, rowanc, layer, P, scale, total, PM, PL, PO, c);
    }
}
