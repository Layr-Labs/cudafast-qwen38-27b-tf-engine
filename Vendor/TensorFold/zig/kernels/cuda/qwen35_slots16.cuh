// Exact three-slot narrow scan with four value rows per warp.
#pragma once
template <int SLOTS, int R, int AHEAD, bool SHARED, bool REUSE = false>
__device__ __forceinline__ void q35_slots16_ready(Geo geo, const __nv_bfloat16* q, const __nv_bfloat16* k,
        const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs,
        const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) {
    __shared__ float4 kept[4][SLOTS * R * 32];
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV, hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    __shared__ SlotStep cached_steps[16];
    if constexpr(SHARED) {
        const int tid = threadIdx.y * 32 + threadIdx.x;
        if(tid < sg.rows && sg.rows <= 16) cached_steps[tid] = steps[sg.row0 + tid];
        __syncthreads();
    }
    auto get_step = [&](int item) {
        if constexpr(SHARED) { if(sg.rows <= 16) return cached_steps[item]; }
        return steps[sg.row0 + item];
    };
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
    // the carried prefix (pn2) first, then the newer one (pn), each with apply's arithmetic, row by row
#pragma unroll 1
    for (int set = 0; set < 2; ++set) {
        const int pn = set ? sg.pn : sg.pn2;
        if (pn <= 0) continue;
        Seg prev = sg;
        prev.log = set ? sg.plog : sg.plog2;
        for (int j = 0; j < pn; ++j) {
            const LogRow l = log_row(geo, prev, layer, j);
            const float pg = l.g[hv];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float pd = l.delta[hv * DK + dv0 + r];
#pragma unroll
                for (int i = 0; i < PT; i++) {
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
            for (int i = 0; i < PT; ++i) reinterpret_cast<float*>(sg.before)[(size_t)layer * geo.rec() + at + r * DK + i] = root[r][i];
    const bool record = recorded(P);
    // row inputs as stored, the next row's loaded while this one computes
    struct In { uint2 k, q; __nv_bfloat16 v[R], beta; float g; };
    auto load = [&](unsigned row, In& x) {
        const size_t key = ((size_t)row * NK + hk) * DK + PT * dk;
        x.k = *reinterpret_cast<const uint2*>(k + key);
        x.q = *reinterpret_cast<const uint2*>(q + key);
#pragma unroll
        for (int r = 0; r < R; ++r) x.v[r] = v[((size_t)row * NV + hv) * DK + dv0 + r];
        x.g = g[row * NV + hv];
        x.beta = beta[row * NV + hv];
    };
    constexpr int SH = 5 - q35_log2<2 * R>();
    const bool ylane = (dk & ((1 << SH) - 1)) == 0 && (dk >> SH) >= R;
    const int yr = (dk >> SH) - R;
    float pend[R];
    int pend_row = -1;
#pragma unroll
    for (int r = 0; r < R; ++r) pend[r] = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) state[r][i] = root[r][i];
    In ready[AHEAD]{};
    SlotStep ready_steps[AHEAD]{};
#pragma unroll
    for(int j=0;j<AHEAD;++j) if(j < sg.rows) {
        ready_steps[j] = get_step(j);
        load(ready_steps[j].row, ready[j]);
    }
    int live_slot = -2;
    for (int item = 0; item < sg.rows; ++item) {
        const SlotStep step = ready_steps[0];
        const In in = ready[0];
#pragma unroll
        for(int j=0;j<AHEAD-1;++j) {
            ready[j] = ready[j+1];
            ready_steps[j] = ready_steps[j+1];
        }
        if(item + AHEAD < sg.rows) {
            ready_steps[AHEAD-1] = get_step(item + AHEAD);
            load(ready_steps[AHEAD-1].row, ready[AHEAD-1]);
        }
        const int row = step.row, rr = row - sg.row0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (step.read_slot >= 0) { if (!REUSE || step.read_slot != live_slot) {
                const float4 t = slots[(step.read_slot * R + r) * 32 + dk];
                state[r][0] = t.x;
                state[r][1] = t.y;
                state[r][2] = t.z;
                state[r][3] = t.w;
            } } else if (step.read_slot == -1) {
#pragma unroll
                for (int i = 0; i < PT; ++i) state[r][i] = root[r][i];
            }  // -2 (tree_slots.in_registers, carried plans): the parent's state is in the registers
        }
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&in.k);
        const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&in.q);
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
        if (pend_row >= 0 && ylane) y[((size_t)pend_row * NV + hv) * DK + dv0 + yr] = __float2bfloat16_rn(tot);
        float delta[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            delta[r] = (bf(in.v[r]) - __shfl_sync(0xffffffffu, tot, r << SH)) * bc;
            pend[r] = 0.0f;
#pragma unroll
            for (int i = 0; i < PT; i++) {
                state[r][i] = fmaf(kc[i], delta[r], state[r][i]);
                pend[r] += state[r][i] * qc[i];
            }
            if (step.write_slot >= 0) slots[(step.write_slot * R + r) * 32 + dk] = make_float4(state[r][0], state[r][1], state[r][2], state[r][3]);
        }
        if constexpr(REUSE) live_slot = step.write_slot >= 0 ? step.write_slot : -2;
        pend_row = row;
        if (record) {
            const LogRow l = log_row(geo, sg, layer, rr);
            if (dk == 0) {
#pragma unroll
                for (int r = 0; r < R; ++r) l.delta[hv * DK + dv0 + r] = delta[r];
                if (dv0 == 0) l.g[hv] = gc;
            }
            if (dv0 == 0 && hv % (NV / NK) == 0) *reinterpret_cast<uint2*>(l.k + hk * DK + PT * dk) = in.k;
        }
        __syncwarp();
    }
    if (pend_row >= 0) {
        constexpr int SR = 5 - q35_log2<R>();
        const float tot = spread_sum<R>(pend, dk);
        if ((dk & ((1 << SR) - 1)) == 0) y[((size_t)pend_row * NV + hv) * DK + dv0 + (dk >> SR)] = __float2bfloat16_rn(tot);
    }
    if (record && sg.rows != 1) {
        if ((sg.pn > 0 || sg.pn2 > 0) && !sg.skip)
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

extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_slots16_3(Geo geo,
 const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const float* g,
 const __nv_bfloat16* beta,const Seg* segs,const SlotStep* steps,int layer,
 __nv_bfloat16* y,const int* P) {
 release_next(); await_prior();
 q35_slots16_ready<3,4,1,false,true>(geo,after_wait(q),after_wait(k),after_wait(v),after_wait(g),after_wait(beta),segs,steps,layer,after_wait(y),P);
}
