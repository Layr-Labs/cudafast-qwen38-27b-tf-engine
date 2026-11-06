// CAP3-03 (role 3): down/out persistent rows2, wave-balanced tile->CTA map.
//
// The crown tf_qmm_persistent_rows2::kernel walks work = blockIdx.x + k*gridDim.x over the split-cols
// tile list ([0,F) full 128-wide tiles, then [F,total) half 64-wide tiles) and, when `rotate` fires
// (group==rows, total%rows==0, gridDim%rows==0), shifts wave 2's row assignment by one. At the scored
// shape (M=496, N=5120, full_cols=4096: F=128 fulls + 64 halves over 96 CTAs, 2 tiles a CTA) that
// rotation pairs 48 CTAs as plain-full+plain-full (the two heaviest tiles) and only 8 as the cheap
// short+short; the last row band (112 rows, a short_tile body worth ~7/8 of a full tile) ends up
// beside another full instead of absorbing a heavy partner.
//
// This kernel keeps every tile body and its bits untouched (the same tf_qmm_pipe calls as the crown)
// and only reorders which work index a CTA takes, so each output's K chain and bf16 rounding are
// unchanged - any bijection of the work list is bit-exact. Wave 1 stays the crown's first 96 indices
// (complete weight panels 0..23, so each panel streams from DRAM once). Wave 2 is anti-aligned:
//   - positions c%R==R-1 (the CTAs holding the short row band of panels 0..W-1) take the wave-2
//     PLAIN fulls (panels W..PF-1, rows 0..R-2), so a short (7/8) carries a full (1) -> 1.875;
//   - the remaining positions take the wave-2 short fulls and then all halves, so a plain full
//     carries a half or a short -> <= 1.875 (half tile measured h <= 0.875).
// Every per-CTA sum is then <= 1.875 full-tile equivalents against the crown's 2.0, and both waves
// still cover whole weight panels (panels 0..23 in wave 1; 24..31 plus the half panels in wave 2),
// so the once-through weight streaming the g6b trial measured is preserved.
//
// Engagement is deliberately narrow (falls back to the crown mapping - rotate included - otherwise):
// SPLIT, rotate, total == 2*gridDim, full_cols a multiple of BN, PF = full_cols/BN with PF % R == 0,
// gridDim == PF*(R-1), and the last row band short. At the engine's slots=96 that is exactly the
// M in [481,496], N=5120 split-cols family; every other shape runs the crown schedule bit for bit.
#pragma once
namespace tf_qmm_persist_rows2v3 {
using namespace tf_qmm_prefill;

template<int BM,int WM,bool MLP,bool SPLIT>
__global__ void __launch_bounds__(WM*128) kernel(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ ManyParts parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate) {
    constexpr int BN=128, STAGES=2;
    grid_launch();
    const int G=(int)gridDim.x;
    const int R=(M+BM-1)/BM;
    const int F=R*(full_cols/BN);
    // Wave-balanced map: W0 = PF*(R-1)/R full panels open wave 1; the guards are the integer identities
    // that make the two branches below an exact bijection onto the remaining tiles.
    int W0=0, PF=0;
    bool bal=false;
    if(SPLIT && rotate && full_cols>0 && total==2*G && group==R){
        PF=full_cols/BN;
        const int num=PF*(R-1);
        if(PF>0 && full_cols%BN==0 && num%R==0 && num==G){
            W0=num/R;
            bal=W0>0 && W0<PF && tf_qmm_rows0::short_tile<BM,WM>(M-(R-1)*BM);
        }
    }
    for(int work=blockIdx.x;work<total;work+=G){
        int tile;
        if(bal){
            const int c=blockIdx.x;
            if(work<G) tile=work;                          // wave 1: the crown's own first 96 indices
            else if(c%R==R-1){                             // short carriers take the wave-2 plain fulls
                const int j=c/R;
                tile=(j%(R-1))+R*(W0+j/(R-1));
            }else{                                         // the rest take wave-2 shorts, then all halves
                const int rank=c-(c/R+(c%R>R-1?1:0));
                tile=rank<PF-W0 ? (R-1)+R*(W0+rank) : F+(rank-(PF-W0));
            }
        }else{
            // crown mapping, verbatim (rotate between complete cohorts included)
            tile=work;
            if(rotate && group==R && total%R==0 && G%R==0){
                int base=work/R*R;
                tile=base+(work%R+(work/G))%R;
            }
        }
        int q=0;
#pragma unroll
        for(int i=1;i<MANY;i++)if(i<parts.count && tile>=parts.p[i].first)q=i;
        const ManyPart& P=parts.p[q];
        const int local=tile-P.first;
        const int full=F;
        if constexpr(SPLIT){
            if(local>=full){
                const int2 at=tile_of(local-full,M,P.n-full_cols,BM,BN/2,group);
                if(tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                    tf_qmm_pipe::tile<64,BM-16*WM,BN/2,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,full_cols+at.y);
                else tf_qmm_pipe::tile<64,BM,BN/2,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,full_cols+at.y);
            }else{
                const int2 at=tile_of(local,M,full_cols,BM,BN,group);
                if(tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                    tf_qmm_pipe::tile<64,BM-16*WM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
                else tf_qmm_pipe::tile<64,BM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
            }
        }else{
            const int2 at=tile_of(local,M,P.n,BM,BN,group);
            if(tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                tf_qmm_pipe::tile<64,BM-16*WM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
            else tf_qmm_pipe::tile<64,BM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
        }
        __syncthreads();
    }
}
} // namespace tf_qmm_persist_rows2v3
