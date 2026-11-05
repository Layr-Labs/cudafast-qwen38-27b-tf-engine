// TF_DROUND: the step between a verify and the next drafter frame on the device (cuda_dround.zig). The accepted path
// (lanes/accept.zig acceptPath), the committed tokens, the drafter context's new bounds and the frame's staged ids and
// segment, all from the verify's picks in device memory; absorb's gather and ring writes read the path from there.

struct DroundState {
    int n;                 // the accepted path's rows (row 0 included)
    int rows[64];          // the path's window rows (TapRows layout: n, rows)
    unsigned int pending;  // the target's pick at the path's last row: the next frame's first id
    int hist_len;          // the stream's tokens after this round's commits
    long long begin, end;  // the drafter context after absorbing the path
};

// One thread. picks: the verify's per-row picks; wtok/wpar: the window's row tokens and row parents (-1 at row 0).
extern "C" __global__ void tf_q35d_dround_accept(const unsigned int* __restrict__ picks, const unsigned int* __restrict__ wtok,
                                                 const int* __restrict__ wpar, int wrows, long long start, long long ctx_begin,
                                                 long long ctx_end, long long window, unsigned int* __restrict__ hist, int hist_len,
                                                 DroundState* __restrict__ st, unsigned int* __restrict__ ids, long long* __restrict__ seg) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int n = 0, at = 0;
    st->rows[n++] = 0;
    for (;;) {
        const unsigned int want = picks[at];
        int next = -1;
        for (int r = at + 1; r < wrows; ++r)
            if (wpar[r] == at && wtok[r] == want) {
                next = r;
                break;
            }
        if (next < 0 || n == 64) break;
        st->rows[n++] = next;
        at = next;
    }
    st->n = n;
    // committed: the pick at each path row's parent, then the last row's pick = the picks along the path
    for (int i = 0; i < n; ++i) hist[hist_len + i] = picks[st->rows[i]];
    st->hist_len = hist_len + n;
    st->pending = picks[st->rows[n - 1]];
    // absorbRows' bounds for n rows at `start`
    long long b = ctx_begin, e = ctx_end;
    if (start < e) e = start;
    if (start > e) b = start;
    e = start + n;
    if (e > window && b < e - window) b = e - window;
    st->begin = b;
    st->end = e;
    ids[0] = st->pending;
    seg[1] = b;  // Segment {ring, begin, position}
    seg[2] = e;
}

// tf_q35d_gather_rows with the rows from device memory; blocks past the path copy its row 0 (never written to a ring).
extern "C" __global__ void tf_q35d_gather_rows_dev(const __nv_bfloat16* __restrict__ RING0, int R, long long start,
                                                   const DroundState* __restrict__ st, int width, __nv_bfloat16* __restrict__ OUT0) {
    release_next();
    await_prior();
    const __nv_bfloat16* RING = after_wait(RING0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    const int i = blockIdx.x < st->n ? (int)blockIdx.x : 0;
    const long long row = start + st->rows[i];
    copy_row(RING + (size_t)(row % R) * width, OUT + (size_t)blockIdx.x * width, width);
}

// tf_q35d_ring_put for the path's rows only.
extern "C" __global__ void tf_q35d_ring_put_dev(const __nv_bfloat16* __restrict__ K0, const __nv_bfloat16* __restrict__ V0,
                                                __nv_bfloat16* __restrict__ RK0, __nv_bfloat16* __restrict__ RV0, int width, long long pos0,
                                                int W, const DroundState* __restrict__ st) {
    release_next();
    await_prior();
    const __nv_bfloat16* K = after_wait(K0);
    const __nv_bfloat16* V = after_wait(V0);
    __nv_bfloat16* RK = after_wait(RK0);
    __nv_bfloat16* RV = after_wait(RV0);
    const int i = blockIdx.x;
    if (i >= st->n) return;
    const size_t slot = (size_t)((pos0 + i) % W);
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        RK[slot * width + c] = K[(size_t)i * width + c];
        RV[slot * width + c] = V[(size_t)i * width + c];
    }
}
