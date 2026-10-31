// Adjacent leaf siblings share a parent load and independent, unchanged recurrence chains.
#pragma once
template<int SLOTS, int R, int B = 2, bool LEAF_ONLY = true>
__device__ __forceinline__ void q35_siblings(Geo geo, const __nv_bfloat16* q, const __nv_bfloat16* k,
 const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs,
 const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) {
    static_assert(B == 2 && R <= 2, "two independent leaf nodes per warp");
    __shared__ float4 kept[4][SLOTS * R * 32];
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV, hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    const int dk = threadIdx.x, dv0 = (blockIdx.y * 4 + threadIdx.y) * R;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv0) * DK + PT * dk;
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    float4* slots = kept[threadIdx.y];
    float root[R][PT], state[B][R][PT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) root[r][i] = st[r * DK + i];
    if (sg.pn > 0) {
        Seg prev = sg;
        prev.log = sg.plog;
        for (int j = 0; j < sg.pn; ++j) {
            const LogRow l = log_row(geo, prev, layer, j);
            const float pg = l.g[hv];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float pd = l.delta[hv * DK + dv0 + r];
#pragma unroll
                for (int i = 0; i < PT; ++i) {
                    root[r][i] = root[r][i] * pg;
                    root[r][i] = fmaf(bf(l.k[hk * DK + PT * dk + i]), pd, root[r][i]);
                }
            }
        }
    }
    if (sg.rows == 1)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < PT; ++i)
                reinterpret_cast<float*>(sg.before)[(size_t)layer * geo.rec() + at + r * DK + i] = root[r][i];
    const bool record = recorded(P);
    struct In { uint2 k, q; __nv_bfloat16 v[R], beta; float g; };
    auto load = [&](unsigned row, In& input) {
        const size_t key = ((size_t)row * NK + hk) * DK + PT * dk;
        input.k = *reinterpret_cast<const uint2*>(k + key);
        input.q = *reinterpret_cast<const uint2*>(q + key);
#pragma unroll
        for (int r = 0; r < R; ++r) input.v[r] = v[((size_t)row * NV + hv) * DK + dv0 + r];
        input.g = g[row * NV + hv];
        input.beta = beta[row * NV + hv];
    };
    float pend[B * R];
    int2 pend_row = make_int2(-1, -1);
#pragma unroll
    for (int j = 0; j < B; ++j) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            pend[j * R + r] = 0.0f;
#pragma unroll
            for (int i = 0; i < PT; ++i) state[j][r][i] = root[r][i];
        }
    }
    constexpr int SH = 5 - q35_log2<2 * B * R>();
    int last_count = 1;
    for (int item = 0; item < sg.rows;) {
        SlotStep step[B];
        step[0] = steps[sg.row0 + item];
        int count = 1;
        if (item + 1 < sg.rows && (!LEAF_ONLY || step[0].write_slot < 0)) {
            step[1] = steps[sg.row0 + item + 1];
            if (step[1].read_slot == step[0].read_slot &&
                (LEAF_ONLY ? step[1].write_slot < 0 :
                    (step[0].write_slot < 0 || step[0].write_slot != step[1].read_slot))) count = 2;
        }
        In input[B];
#pragma unroll
        for (int j = 0; j < B; ++j) if (j < count) load(step[j].row, input[j]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (step[0].read_slot >= 0) {
                const float4 t = slots[(step[0].read_slot * R + r) * 32 + dk];
                state[0][r][0] = t.x; state[0][r][1] = t.y;
                state[0][r][2] = t.z; state[0][r][3] = t.w;
            } else {
#pragma unroll
                for (int i = 0; i < PT; ++i) state[0][r][i] = root[r][i];
            }
#pragma unroll
            for (int i = 0; i < PT; ++i) state[1][r][i] = state[0][r][i];
        }
        float kc[B][PT], qc[B][PT], both[2 * B * R];
#pragma unroll
        for (int j = 0; j < B; ++j) {
            if (j < count) {
                const auto* kb = reinterpret_cast<const __nv_bfloat16*>(&input[j].k);
                const auto* qb = reinterpret_cast<const __nv_bfloat16*>(&input[j].q);
#pragma unroll
                for (int i = 0; i < PT; ++i) { kc[j][i] = bf(kb[i]); qc[j][i] = bf(qb[i]); }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float kv_mem = 0.0f;
                if (j < count) {
#pragma unroll
                    for (int i = 0; i < PT; ++i) {
                        state[j][r][i] = state[j][r][i] * input[j].g;
                        kv_mem += state[j][r][i] * kc[j][i];
                    }
                }
                both[j * R + r] = kv_mem;
                both[B * R + j * R + r] = pend[j * R + r];
            }
        }
        const float total = spread_sum<2 * B * R>(both, dk);
        const int output = (dk >> SH) - B * R;
        if ((dk & ((1 << SH) - 1)) == 0 && output >= 0) {
            const int j = output / R, r = output % R;
            const int row = j == 0 ? pend_row.x : pend_row.y;
            if (row >= 0) y[((size_t)row * NV + hv) * DK + dv0 + r] = __float2bfloat16_rn(total);
        }
        pend_row = make_int2(step[0].row, count == 2 ? (int)step[1].row : -1);
#pragma unroll
        for (int j = 0; j < B; ++j) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                pend[j * R + r] = 0.0f;
                if (j < count) {
                    const float delta = (bf(input[j].v[r]) - __shfl_sync(0xffffffffu, total, (j * R + r) << SH)) * bf(input[j].beta);
#pragma unroll
                    for (int i = 0; i < PT; ++i) {
                        state[j][r][i] = fmaf(kc[j][i], delta, state[j][r][i]);
                        pend[j * R + r] += state[j][r][i] * qc[j][i];
                    }
                    if (step[j].write_slot >= 0)
                        slots[(step[j].write_slot * R + r) * 32 + dk] = make_float4(state[j][r][0], state[j][r][1], state[j][r][2], state[j][r][3]);
                    if (record && dk == 0) {
                        const LogRow log = log_row(geo, sg, layer, step[j].row - sg.row0);
                        log.delta[hv * DK + dv0 + r] = delta;
                        if (dv0 == 0 && r == 0) log.g[hv] = input[j].g;
                    }
                }
            }
            if (j < count && record && dv0 == 0 && hv % (NV / NK) == 0) {
                const LogRow log = log_row(geo, sg, layer, step[j].row - sg.row0);
                *reinterpret_cast<uint2*>(log.k + hk * DK + PT * dk) = input[j].k;
            }
        }
        __syncwarp();
        item += count;
        last_count = count;
    }
    constexpr int SR = 5 - q35_log2<B * R>();
    const float total = spread_sum<B * R>(pend, dk);
    if ((dk & ((1 << SR) - 1)) == 0) {
        const int output = dk >> SR, j = output / R, r = output % R;
        const int row = j == 0 ? pend_row.x : pend_row.y;
        if (row >= 0) y[((size_t)row * NV + hv) * DK + dv0 + r] = __float2bfloat16_rn(total);
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
    for (int j = 0; j < B; ++j) if (j == last_count - 1)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < PT; ++i) st[r * DK + i] = state[j][r][i];
}
