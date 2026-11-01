// The 17..32-row windows' scan of carried plans (tree_slots.carried: read_slot -2, write_slot -1), every value's float
// arithmetic q35_slots<SLOTS, 4, true>'s (tf_q35_gdn_slots_3's): its loop's addressing hoisted (each lane's input,
// output and log bases fixed before the loop, a row a stride further), a lane's four value-row entries one 8-byte
// load, the record's lane predicates and log strides computed once.
#pragma once
template <int SLOTS>
__device__ __forceinline__ void q35_slots_lean(Geo geo, const __nv_bfloat16* q, const __nv_bfloat16* k,
        const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs,
        const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) {
    constexpr int R = 4;
    __shared__ float4 kept[4][SLOTS * R * 32];
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV, hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    const int dk = threadIdx.x, dv0 = (blockIdx.y * 4 + threadIdx.y) * R;
    constexpr int PT = DK / 32;
    static_assert(PT == 4, "a lane's four state entries are one float4 slot");
    const size_t at = ((size_t)hv * DK + dv0) * DK + PT * dk;  // row r's entries are r * DK further
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    float4* slots = kept[threadIdx.y];
    float root[R][PT], state[R][PT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) root[r][i] = st[r * DK + i];
    if (sg.pn > 0) {
        Seg prev = sg;
        prev.log = sg.plog;
        // the kept prefix's rows four at a time: their decays, strengths and keys loaded together, then applied in
        // order with q35_slots's arithmetic
        for (int j0 = 0; j0 < sg.pn; j0 += 4) {
            float pg[4];
            float4 pd[4];
            uint2 pk[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (j0 + u >= sg.pn) break;
                const LogRow l = log_row(geo, prev, layer, j0 + u);
                pg[u] = l.g[hv];
                pd[u] = *reinterpret_cast<const float4*>(l.delta + hv * DK + dv0);
                pk[u] = *reinterpret_cast<const uint2*>(l.k + hk * DK + PT * dk);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (j0 + u >= sg.pn) break;
                const float pdr[R] = {pd[u].x, pd[u].y, pd[u].z, pd[u].w};
                const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&pk[u]);
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int i = 0; i < PT; i++) {
                        root[r][i] = root[r][i] * pg[u];
                        root[r][i] = fmaf(bf(kb[i]), pdr[r], root[r][i]);
                    }
                }
            }
        }
    }
    if (sg.rows == 1)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < PT; ++i) reinterpret_cast<float*>(sg.before)[(size_t)layer * geo.rec() + at + r * DK + i] = root[r][i];
    const bool record = recorded(P);
    // each lane's row-0 inputs: a row's are `row` strides further
    const unsigned kstride = (unsigned)NK * DK, vstride = (unsigned)NV * DK;
    const __nv_bfloat16* kl = k + (size_t)hk * DK + PT * dk;
    const __nv_bfloat16* ql = q + (size_t)hk * DK + PT * dk;
    const __nv_bfloat16* vl = v + (size_t)hv * DK + dv0;
    const float* gl = g + hv;
    const __nv_bfloat16* bl = beta + hv;
    // row inputs as stored, the next row's loaded while this one computes
    struct In { uint2 k, q, v; __nv_bfloat16 beta; float g; };
    auto load = [&](unsigned row, In& x) {
        x.k = *reinterpret_cast<const uint2*>(kl + (size_t)row * kstride);
        x.q = *reinterpret_cast<const uint2*>(ql + (size_t)row * kstride);
        x.v = *reinterpret_cast<const uint2*>(vl + (size_t)row * vstride);
        x.g = gl[(size_t)row * NV];
        x.beta = bl[(size_t)row * NV];
    };
    constexpr int SH = 5 - q35_log2<2 * R>();
    const bool ylane = (dk & ((1 << SH) - 1)) == 0 && (dk >> SH) >= R;
    const int yr = (dk >> SH) - R;
    __nv_bfloat16* yl = y + (size_t)hv * DK + dv0 + yr;
    // the record: lane 0 the window rows' strengths (and decay), a key head's first value block its key
    const size_t lstride = (size_t)geo.nl * geo.log_bytes();
    char* lb = reinterpret_cast<char*>(sg.log) + (size_t)layer * geo.log_bytes();
    const bool rec_d = record && dk == 0, rec_g = rec_d && dv0 == 0, rec_k = record && dv0 == 0 && hv % (NV / NK) == 0;
    const size_t d_off = (size_t)NV * 4 + ((size_t)hv * DK + dv0) * 4, g_off = (size_t)hv * 4;
    const size_t k_off = (size_t)NV * 4 + (size_t)NV * DK * 4 + ((size_t)hk * DK + PT * dk) * 2;
    float pend[R];
    int pend_row = -1;
#pragma unroll
    for (int r = 0; r < R; ++r) pend[r] = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) state[r][i] = root[r][i];
    // the window's steps staged once (a block's threads, up to 64 rows), so a step's inputs load without its step's
    __shared__ SlotStep sp[64];
    {
        const int tid = threadIdx.y * 32 + threadIdx.x;
        if (tid < sg.rows && tid < 64) sp[tid] = steps[sg.row0 + tid];
        __syncthreads();
    }
    int live_slot = -2;
    auto run = [&](const SlotStep& step, const In& in) {
        const int row = step.row, rr = row - sg.row0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (step.read_slot >= 0 && step.read_slot != live_slot) {
                const float4 t = slots[(step.read_slot * R + r) * 32 + dk];
                state[r][0] = t.x;
                state[r][1] = t.y;
                state[r][2] = t.z;
                state[r][3] = t.w;
            } else if (step.read_slot == -1) {
#pragma unroll
                for (int i = 0; i < PT; ++i) state[r][i] = root[r][i];
            }  // -2: the step before was the parent's, its state is in the registers
        }
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&in.k);
        const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&in.q);
        const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&in.v);
        float kc[PT], qc[PT];
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kc[i] = bf(kb[i]);
            qc[i] = bf(qb[i]);
        }
        const float gc = in.g;
        const float bc = bf(in.beta);
        float both[2 * R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float kv_mem = 0.0f;
#pragma unroll
            for (int i = 0; i < PT; i++) {
                state[r][i] = state[r][i] * gc;
                kv_mem += state[r][i] * kc[i];
            }
            both[r] = kv_mem;
            both[R + r] = pend[r];
        }
        const float tot = spread_sum<2 * R>(both, dk);
        if (pend_row >= 0 && ylane) yl[(size_t)pend_row * vstride] = __float2bfloat16_rn(tot);
        float delta[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            delta[r] = (bf(vb[r]) - __shfl_sync(0xffffffffu, tot, r << SH)) * bc;
            pend[r] = 0.0f;
#pragma unroll
            for (int i = 0; i < PT; i++) {
                state[r][i] = fmaf(kc[i], delta[r], state[r][i]);
                pend[r] += state[r][i] * qc[i];
            }
            if (step.write_slot >= 0) slots[(step.write_slot * R + r) * 32 + dk] = make_float4(state[r][0], state[r][1], state[r][2], state[r][3]);
        }
        live_slot = step.write_slot >= 0 ? step.write_slot : -2;
        pend_row = row;
        if (record) {
            char* lr = lb + (size_t)rr * lstride;
            if (rec_d) *reinterpret_cast<float4*>(lr + d_off) = make_float4(delta[0], delta[1], delta[2], delta[3]);
            if (rec_g) *reinterpret_cast<float*>(lr + g_off) = gc;
            if (rec_k) *reinterpret_cast<uint2*>(lr + k_off) = in.k;
        }
        __syncwarp();
    };
    In ia, ib;
    SlotStep sa, sb;
    if (sg.rows > 0) {
        sa = sp[0];
        load(sa.row, ia);
    }
    if (sg.rows > 1) {
        sb = sp[1];
        load(sb.row, ib);
    }
    for (int item = 0; item < sg.rows; item += 2) {
        {
            const SlotStep step = sa;
            const In in = ia;
            if (item + 2 < sg.rows) {
                sa = sp[item + 2];
                load(sa.row, ia);
            }
            run(step, in);
        }
        if (item + 1 >= sg.rows) break;
        {
            const SlotStep step = sb;
            const In in = ib;
            if (item + 3 < sg.rows) {
                sb = sp[item + 3];
                load(sb.row, ib);
            }
            run(step, in);
        }
    }
    if (pend_row >= 0) {
        constexpr int SR = 5 - q35_log2<R>();
        const float tot = spread_sum<R>(pend, dk);
        if ((dk & ((1 << SR) - 1)) == 0) y[((size_t)pend_row * NV + hv) * DK + dv0 + (dk >> SR)] = __float2bfloat16_rn(tot);
    }
    if (record && sg.rows != 1) {
        if (sg.pn > 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int i = 0; i < PT; ++i) st[r * DK + i] = root[r][i];
        return;
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) st[r * DK + i] = state[r][i];
}

#define TF_Q35_LEAN(N) \
extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_lean_##N(Geo geo, const __nv_bfloat16* q, \
 const __nv_bfloat16* k, const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs, \
 const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) { \
  release_next(); await_prior(); \
  q35_slots_lean<N>(geo, after_wait(q), after_wait(k), after_wait(v), after_wait(g), after_wait(beta), segs, steps, layer, after_wait(y), P); }
TF_Q35_LEAN(2)
TF_Q35_LEAN(3)