/* gpu_attn.c — a decode token's attention on an NVIDIA GPU with the CPU's bits (gpu_attn.h says what
 * and why). The numbers are in docs/MEASUREMENTS.md question 51, measured by tests/bench_gpu_attn.c:
 * the kernels run at 89-92% of a plain read of the same bytes; a layer at 2048 positions takes 185 µs
 * in the decode's pattern with the keep-warm warp and zero copy, 704 on the CPU.
 *
 * The arithmetic is tr_attention_group's (src/kernels/kernels.c), operation for operation. Every
 * float operation is written with its rounding (.rn: PTX never fuses those into an fma), none with
 * .ftz or .approx:
 *   s_t = fl(dot(q, k_t) * scale), the dot under the 16-lane contract and tr_lane_combine's tree
 *   m   = max s_t, in any order (a zero maximum's sign cannot reach the output: exp(+-0) = 1)
 *   e_t = tr_expf(fl(s_t - m)): expf_core of src/kernels/expf.c in f64, its exception table included
 *   sum = the 16-lane sum of the e_t with the same tree; a_t = fl(e_t / sum)
 *   out = +0.0f, then out = fl(out + fl(a_t * v_t)) for t increasing
 * VRAM, per layer: keys position-minor [head][dim][cap] (a warp reads 32 positions in one line),
 * then values [head][cap][dim]; cap is n_ctx rounded up to 128. The kernels:
 *   attn_scores  grid (ceil(n/128), heads): a thread per position, q in shared memory, the block's
 *                maximum. The block holding position n-1 appends the token's key and value to the
 *                cache, and the thread of that position reads its key from the input
 *   attn_exp     grid (ceil(n/128), heads): the head's maximum from the block maxima, then e_t
 *   attn_v       grid (4, heads), 32 dims a block: the 16-lane sum of the e_t (through shared memory,
 *                SUM_CHUNK positions at a time), then the values stream through shared memory with
 *                their e_t (cp.async, V_STAGES - 1 tiles in flight) and one warp adds them in position
 *                order; a_t = e_t / sum is taken per tile, a lane per position, and passed along the
 *                warp with shuffles
 *   kv_write     rows of a prefill pass into the cache, the keys transposed through shared memory
 *   keep_warm    one warp on the second stream, napping on the global timer until the next call
 *                starts or TR_GPU_WARM_NS pass: a laptop GPU keeps its clocks through the CPU's gap
 * A decode call's input and output are pinned memory the kernels read and write themselves (no copy
 * engine: ~57 µs a layer less, question 51). */
#include "gpu_attn.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../base/platform.h"
#include "../kernels/expf_table.h"

#define GPU_HEAD_DIM 128
#define GPU_CAP_ALIGN 128  /* cap is a multiple of this: the block maxima and the tiles divide it */
#define GPU_MAX_CTX (1 << 24)
#define V_CHUNK 32         /* dims per attn_v block */
#define V_TILE 32          /* positions per stage of attn_v */
#define V_TILE_LOG2 5
#define V_STAGES 6
#define V_STAGE_BYTES (V_TILE * V_CHUNK * 4 + V_TILE * 4) /* the values, then the tile's e_t */
#define SUM_CHUNK 4096     /* positions of e_t in shared memory at once while attn_v takes the sum */
#define WRITE_ROWS 512     /* rows tr_gpu_attn_write stages at once: a prefill pass */
#define WARM_NAP_NS 20000  /* keep_warm's nap between two looks at the clock */

/* ---- the driver API, opened at run time ------------------------------------------------------- */

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

enum {
    CU_CTX_SCHED_SPIN = 0x01, /* a synchronization spins: the lowest latency for the decode's wait */
    CU_CTX_MAP_HOST = 0x08,   /* pinned memory the kernels read and write */
    CU_STREAM_NON_BLOCKING = 0x1,
    CU_ATTR_CC_MAJOR = 75,
    CU_ATTR_CC_MINOR = 76,
    CU_JIT_ERROR_LOG = 5,
    CU_JIT_ERROR_LOG_SIZE = 6,
    CU_ERROR_NOT_READY = 600 /* cuStreamQuery: work still pending on the stream */
};

typedef struct {
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*CtxCreate)(CUcontext *, unsigned, CUdevice);
    CUresult (*CtxDestroy)(CUcontext);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*ModuleLoadDataEx)(CUmodule *, const void *, unsigned, int *, void **);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemAllocHost)(void **, size_t);
    CUresult (*MemFreeHost)(void *);
    CUresult (*MemHostGetDevicePointer)(CUdeviceptr *, void *, unsigned);
    CUresult (*MemsetD32)(CUdeviceptr, unsigned, size_t);
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*StreamDestroy)(CUstream);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*StreamQuery)(CUstream);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             CUstream, void **, void **);
    CUresult (*GetErrorString)(CUresult, const char **);
} cu_fns;

static const struct {
    size_t off;
    const char *name;
} cu_names[] = {
    {offsetof(cu_fns, Init), "cuInit"},
    {offsetof(cu_fns, DeviceGetCount), "cuDeviceGetCount"},
    {offsetof(cu_fns, DeviceGet), "cuDeviceGet"},
    {offsetof(cu_fns, DeviceGetName), "cuDeviceGetName"},
    {offsetof(cu_fns, DeviceGetAttribute), "cuDeviceGetAttribute"},
    {offsetof(cu_fns, CtxCreate), "cuCtxCreate_v2"},
    {offsetof(cu_fns, CtxDestroy), "cuCtxDestroy_v2"},
    {offsetof(cu_fns, CtxSetCurrent), "cuCtxSetCurrent"},
    {offsetof(cu_fns, MemGetInfo), "cuMemGetInfo_v2"},
    {offsetof(cu_fns, ModuleLoadDataEx), "cuModuleLoadDataEx"},
    {offsetof(cu_fns, ModuleUnload), "cuModuleUnload"},
    {offsetof(cu_fns, ModuleGetFunction), "cuModuleGetFunction"},
    {offsetof(cu_fns, MemAlloc), "cuMemAlloc_v2"},
    {offsetof(cu_fns, MemFree), "cuMemFree_v2"},
    {offsetof(cu_fns, MemAllocHost), "cuMemAllocHost_v2"},
    {offsetof(cu_fns, MemFreeHost), "cuMemFreeHost"},
    {offsetof(cu_fns, MemHostGetDevicePointer), "cuMemHostGetDevicePointer_v2"},
    {offsetof(cu_fns, MemsetD32), "cuMemsetD32_v2"},
    {offsetof(cu_fns, StreamCreate), "cuStreamCreate"},
    {offsetof(cu_fns, StreamDestroy), "cuStreamDestroy_v2"},
    {offsetof(cu_fns, StreamSynchronize), "cuStreamSynchronize"},
    {offsetof(cu_fns, StreamQuery), "cuStreamQuery"},
    {offsetof(cu_fns, LaunchKernel), "cuLaunchKernel"},
    {offsetof(cu_fns, GetErrorString), "cuGetErrorString"},
};

struct tr_gpu {
    tr_lib *lib;
    cu_fns cu;
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction f_scores, f_exp, f_v, f_write, f_warm;
    char name[128];
};

struct tr_gpu_attn {
    tr_gpu *g;
    int64_t n_layers, n_head, n_ctx, cap;
    uint64_t layer_bytes; /* keys then values of one layer */
    CUdeviceptr kv;       /* every layer's */
    CUdeviceptr s, mb, e; /* one call's scores, block maxima and exponentials */
    CUdeviceptr gen;      /* the call counter keep_warm watches */
    CUstream st, warm;
    float *h_in, *h_out, *h_stage; /* pinned: q, k, v of a call; its output; the rows of a write */
    CUdeviceptr d_in, d_out, d_stage;
    unsigned seq;
    unsigned long long warm_ns;
};

static void set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (err == NULL || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

/* 0 when r is success, else -1 with "<what>: <driver's message>" in err */
static int cu_check(const tr_gpu *g, CUresult r, const char *what, char *err, size_t err_len) {
    if (r == 0) return 0;
    const char *s = NULL;
    if (g->cu.GetErrorString != NULL) g->cu.GetErrorString(r, &s);
    set_err(err, err_len, "%s: CUDA error %d (%s)", what, r, s != NULL ? s : "?");
    return -1;
}

/* ---- the kernels, as PTX text ------------------------------------------------------------------- */

typedef struct {
    char *p;
    size_t len, cap;
    int bad;
} sbuf;

#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((format(printf, 2, 3)))
#endif
static void sb(sbuf *b, const char *fmt, ...) {
    while (!b->bad) {
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(b->p + b->len, b->cap - b->len, fmt, ap);
        va_end(ap);
        if (n < 0) {
            b->bad = 1;
            return;
        }
        if ((size_t)n < b->cap - b->len) {
            b->len += (size_t)n;
            return;
        }
        size_t cap = 2 * b->cap + (size_t)n + 1;
        char *p = realloc(b->p, cap);
        if (p == NULL) {
            b->bad = 1;
            return;
        }
        b->p = p;
        b->cap = cap;
    }
}

static uint64_t dbits(double d) {
    uint64_t u;
    memcpy(&u, &d, sizeof u);
    return u;
}

static uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

/* y = tr_expf(x): expf_core of src/kernels/expf.c operation for operation, in f64, the constants from
 * expf_table.h; L makes the labels unique */
static void emit_expf(sbuf *b, const char *x, const char *y, const char *L) {
    sb(b, "{\n.reg .pred %%ep<4>;\n.reg .f32 %%ef<2>;\n.reg .f64 %%ed<24>;\n.reg .b64 %%er<4>;\n.reg .b32 %%eu<4>;\n");
    sb(b, "setp.nan.f32 %%ep0, %s, %s;\n@%%ep0 bra %s_nan;\n", x, x, L);
    sb(b, "setp.gt.f32 %%ep1, %s, 0f%08X;\n@%%ep1 bra %s_ovf;\n", x, (unsigned)fbits(TR_EXPF_OVERFLOW_X), L);
    sb(b, "setp.lt.f32 %%ep2, %s, 0f%08X;\n@%%ep2 bra %s_unf;\n", x, (unsigned)fbits(TR_EXPF_UNDERFLOW_X), L);
    sb(b, "cvt.f64.f32 %%ed0, %s;\n", x);
    /* kd = (xd * 64/ln2 + 0x1.8p52) - 0x1.8p52, k = (int64_t)kd */
    sb(b, "mul.rn.f64 %%ed1, %%ed0, 0d%016llX;\n", (unsigned long long)dbits(TR_EXPF_INV_LN2_64));
    sb(b, "add.rn.f64 %%ed2, %%ed1, 0d%016llX;\n", (unsigned long long)dbits(0x1.8p52));
    sb(b, "sub.rn.f64 %%ed3, %%ed2, 0d%016llX;\n", (unsigned long long)dbits(0x1.8p52));
    sb(b, "cvt.rzi.s64.f64 %%er0, %%ed3;\n");
    /* r = (xd - kd * hi) - kd * lo */
    sb(b, "mul.rn.f64 %%ed4, %%ed3, 0d%016llX;\n", (unsigned long long)dbits(TR_EXPF_LN2_64_HI));
    sb(b, "sub.rn.f64 %%ed5, %%ed0, %%ed4;\n");
    sb(b, "mul.rn.f64 %%ed6, %%ed3, 0d%016llX;\n", (unsigned long long)dbits(TR_EXPF_LN2_64_LO));
    sb(b, "sub.rn.f64 %%ed7, %%ed5, %%ed6;\n");
    /* p = r + r * r * (0.5 + r * (C3 + r * (C4 + r * C5))) */
    sb(b, "mul.rn.f64 %%ed8, %%ed7, 0d%016llX;\n", (unsigned long long)dbits(TR_EXPF_C5));
    sb(b, "add.rn.f64 %%ed9, %%ed8, 0d%016llX;\n", (unsigned long long)dbits(TR_EXPF_C4));
    sb(b, "mul.rn.f64 %%ed10, %%ed7, %%ed9;\n");
    sb(b, "add.rn.f64 %%ed11, %%ed10, 0d%016llX;\n", (unsigned long long)dbits(TR_EXPF_C3));
    sb(b, "mul.rn.f64 %%ed12, %%ed7, %%ed11;\n");
    sb(b, "add.rn.f64 %%ed13, %%ed12, 0d%016llX;\n", (unsigned long long)dbits(0.5));
    sb(b, "mul.rn.f64 %%ed14, %%ed7, %%ed7;\nmul.rn.f64 %%ed15, %%ed14, %%ed13;\nadd.rn.f64 %%ed16, %%ed7, %%ed15;\n");
    /* t = pow2[k & 63], scale = 2^(k >> 6), y = (t + t * p) * scale */
    sb(b, "and.b64 %%er1, %%er0, 63;\nshl.b64 %%er1, %%er1, 3;\nmov.u64 %%er2, tr_pow2;\nadd.s64 %%er2, %%er2, %%er1;\n"
          "ld.global.nc.f64 %%ed17, [%%er2];\n"
          "shr.s64 %%er3, %%er0, 6;\nadd.s64 %%er3, %%er3, 1023;\nshl.b64 %%er3, %%er3, 52;\nmov.b64 %%ed18, %%er3;\n"
          "mul.rn.f64 %%ed19, %%ed17, %%ed16;\nadd.rn.f64 %%ed20, %%ed17, %%ed19;\nmul.rn.f64 %%ed21, %%ed20, %%ed18;\n");
    /* the rounding test: y * (1 -+ 2^-50) round to the same float */
    sb(b, "mul.rn.f64 %%ed22, %%ed21, 0d%016llX;\n", (unsigned long long)dbits(1.0 - TR_EXPF_MARGIN));
    sb(b, "mul.rn.f64 %%ed23, %%ed21, 0d%016llX;\n", (unsigned long long)dbits(1.0 + TR_EXPF_MARGIN));
    sb(b, "cvt.rn.f32.f64 %%ef0, %%ed22;\ncvt.rn.f32.f64 %%ef1, %%ed23;\nmov.b32 %%eu0, %%ef0;\nmov.b32 %%eu1, %%ef1;\n"
          "mov.f32 %s, %%ef0;\nsetp.eq.b32 %%ep3, %%eu0, %%eu1;\n@%%ep3 bra %s_done;\n",
       y, L);
    /* the exception table, else NaN (tests/bench_expf.c --check proves no float gets there) */
    sb(b, "mov.b32 %%eu2, %s;\nmov.b32 %%eu3, 0x7FC00000;\n", x);
    for (int i = 0; i < TR_EXPF_N_EXCEPTIONS; i++)
        sb(b, "setp.eq.b32 %%ep3, %%eu2, 0x%08X;\n@%%ep3 mov.b32 %%eu3, 0x%08X;\n", (unsigned)tr_expf_exceptions[i][0],
           (unsigned)tr_expf_exceptions[i][1]);
    sb(b, "mov.b32 %s, %%eu3;\nbra %s_done;\n", y, L);
    sb(b, "%s_nan:\nadd.rn.f32 %s, %s, %s;\nbra %s_done;\n", L, y, x, x, L);
    sb(b, "%s_ovf:\nmov.f32 %s, 0f7F800000;\nbra %s_done;\n", L, y, L);
    sb(b, "%s_unf:\nmov.f32 %s, 0f00000000;\n%s_done:\n}\n", L, y, L);
}

/* attn_scores(K, in, V, S, MB, n, scale, cap, seq, gen): grid (ceil(n/128), heads), 128 threads */
static void emit_scores(sbuf *b) {
    sb(b, ".visible .entry attn_scores(.param .u64 pK, .param .u64 pIn, .param .u64 pV, .param .u64 pS,\n"
          "    .param .u64 pMB, .param .u32 pN, .param .f32 pScale, .param .u32 pCap, .param .u32 pSeq, .param .u64 pGen)\n{\n"
          ".reg .pred %%p<5>;\n.reg .b32 %%r<32>;\n.reg .b64 %%rd<32>;\n"
          ".reg .f32 %%sc, %%s, %%t, %%q<4>, %%k<4>, %%m<4>, %%l<16>, %%w<16>;\n"
          ".shared .align 16 .f32 sq[128];\n.shared .align 16 .f32 swm[4];\n"
          "ld.param.u64 %%rd1, [pK];\nld.param.u64 %%rd2, [pIn];\nld.param.u64 %%rd3, [pV];\nld.param.u64 %%rd4, [pS];\n"
          "ld.param.u64 %%rd5, [pMB];\nld.param.u32 %%r1, [pN];\nld.param.f32 %%sc, [pScale];\nld.param.u32 %%r20, [pCap];\n"
          "ld.param.u32 %%r21, [pSeq];\nld.param.u64 %%rd20, [pGen];\n"
          "mov.u32 %%r2, %%tid.x;\nmov.u32 %%r3, %%ctaid.x;\nmov.u32 %%r4, %%ctaid.y;\nmov.u32 %%r22, %%nctaid.y;\n");
    /* block (0, 0) tells keep_warm a call has started */
    sb(b, "or.b32 %%r23, %%r2, %%r3;\nor.b32 %%r23, %%r23, %%r4;\nsetp.eq.u32 %%p4, %%r23, 0;\n"
          "@%%p4 st.volatile.global.u32 [%%rd20], %%r21;\n");
    /* q of head h into shared memory; rd21 = the bytes of one of q, k, v in the input (heads * 128 floats) */
    sb(b, "shl.b32 %%r5, %%r4, 7;\nadd.u32 %%r6, %%r5, %%r2;\nmul.wide.u32 %%rd6, %%r6, 4;\nadd.s64 %%rd7, %%rd2, %%rd6;\n"
          "ld.global.f32 %%t, [%%rd7];\nshl.b32 %%r7, %%r2, 2;\nmov.u32 %%r8, sq;\nadd.u32 %%r8, %%r8, %%r7;\n"
          "st.shared.f32 [%%r8], %%t;\nmul.wide.u32 %%rd21, %%r22, 512;\n");
    /* the block holding position n-1 appends the token's key and value of head h */
    sb(b, "sub.u32 %%r9, %%r1, 1;\nshr.u32 %%r10, %%r9, 7;\nsetp.ne.u32 %%p1, %%r3, %%r10;\n@%%p1 bra NOAPPEND;\n"
          "add.s64 %%rd22, %%rd7, %%rd21;\nld.global.f32 %%t, [%%rd22];\n"
          "mul.wide.u32 %%rd8, %%r6, %%r20;\ncvt.u64.u32 %%rd9, %%r9;\nadd.s64 %%rd8, %%rd8, %%rd9;\n"
          "shl.b64 %%rd8, %%rd8, 2;\nadd.s64 %%rd8, %%rd1, %%rd8;\nst.global.f32 [%%rd8], %%t;\n"
          "add.s64 %%rd22, %%rd22, %%rd21;\nld.global.f32 %%t, [%%rd22];\n"
          "mul.wide.u32 %%rd10, %%r4, %%r20;\nadd.s64 %%rd10, %%rd10, %%rd9;\nshl.b64 %%rd10, %%rd10, 7;\n"
          "cvt.u64.u32 %%rd11, %%r2;\nadd.s64 %%rd10, %%rd10, %%rd11;\nshl.b64 %%rd10, %%rd10, 2;\n"
          "add.s64 %%rd10, %%rd3, %%rd10;\nst.global.f32 [%%rd10], %%t;\nNOAPPEND:\nbar.sync 0;\n");
    /* the dot of position t: keys at K[h][i][t] (stride cap), or at in.k[h][i] for t == n-1 */
    sb(b, "shl.b32 %%r12, %%r3, 7;\nadd.u32 %%r12, %%r12, %%r2;\nmov.f32 %%s, 0fFF800000;\n"
          "setp.ge.u32 %%p2, %%r12, %%r1;\n@%%p2 bra NODOT;\nsetp.eq.u32 %%p3, %%r12, %%r9;\n"
          "mul.wide.u32 %%rd12, %%r5, %%r20;\ncvt.u64.u32 %%rd13, %%r12;\nadd.s64 %%rd12, %%rd12, %%rd13;\n"
          "shl.b64 %%rd12, %%rd12, 2;\nadd.s64 %%rd12, %%rd1, %%rd12;\n"
          "add.s64 %%rd14, %%rd2, %%rd21;\nmul.wide.u32 %%rd15, %%r5, 4;\nadd.s64 %%rd14, %%rd14, %%rd15;\n"
          "mul.wide.u32 %%rd16, %%r20, 4;\nselp.b64 %%rd17, %%rd14, %%rd12, %%p3;\nselp.b64 %%rd18, 4, %%rd16, %%p3;\n");
    for (int l = 0; l < 16; l++) sb(b, "mov.f32 %%l%d, 0f00000000;\n", l);
    for (int j = 0; j < GPU_HEAD_DIM / 4; j++) {
        sb(b, "ld.shared.v4.f32 {%%q0, %%q1, %%q2, %%q3}, [sq+%d];\n", 16 * j);
        for (int x = 0; x < 4; x++) sb(b, "ld.global.f32 %%k%d, [%%rd17];\nadd.s64 %%rd17, %%rd17, %%rd18;\n", x);
        for (int x = 0; x < 4; x++) {
            int lane = (4 * j + x) % 16;
            sb(b, "mul.rn.f32 %%m%d, %%q%d, %%k%d;\nadd.rn.f32 %%l%d, %%l%d, %%m%d;\n", x, x, x, lane, lane, x);
        }
    }
    /* tr_lane_combine: ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), the same for 8..15, halves added last */
    for (int half = 0; half < 2; half++) {
        int o = 8 * half, w = 7 * half;
        for (int x = 0; x < 4; x++) sb(b, "add.rn.f32 %%w%d, %%l%d, %%l%d;\n", w + x, o + 2 * x, o + 2 * x + 1);
        sb(b, "add.rn.f32 %%w%d, %%w%d, %%w%d;\n", w + 4, w + 0, w + 1);
        sb(b, "add.rn.f32 %%w%d, %%w%d, %%w%d;\n", w + 5, w + 2, w + 3);
        sb(b, "add.rn.f32 %%w%d, %%w%d, %%w%d;\n", w + 6, w + 4, w + 5);
    }
    sb(b, "add.rn.f32 %%w14, %%w6, %%w13;\nmul.rn.f32 %%s, %%w14, %%sc;\n"
          "mul.wide.u32 %%rd19, %%r4, %%r20;\nadd.s64 %%rd19, %%rd19, %%rd13;\nshl.b64 %%rd19, %%rd19, 2;\n"
          "add.s64 %%rd19, %%rd4, %%rd19;\nst.global.f32 [%%rd19], %%s;\nNODOT:\n");
    /* the block's maximum (positions past n count as -inf) */
    for (int off = 16; off >= 1; off /= 2)
        sb(b, "shfl.sync.bfly.b32 %%t, %%s, %d, 31, 0xffffffff;\nmax.f32 %%s, %%s, %%t;\n", off);
    sb(b, "and.b32 %%r14, %%r2, 31;\nshr.u32 %%r15, %%r2, 5;\nsetp.ne.u32 %%p1, %%r14, 0;\n@%%p1 bra NOWM;\n"
          "mov.u32 %%r16, swm;\nshl.b32 %%r17, %%r15, 2;\nadd.u32 %%r16, %%r16, %%r17;\nst.shared.f32 [%%r16], %%s;\n"
          "NOWM:\nbar.sync 0;\nsetp.ne.u32 %%p1, %%r2, 0;\n@%%p1 bra DONE;\n"
          "ld.shared.v4.f32 {%%q0, %%q1, %%q2, %%q3}, [swm];\n"
          "max.f32 %%q0, %%q0, %%q1;\nmax.f32 %%q2, %%q2, %%q3;\nmax.f32 %%q0, %%q0, %%q2;\n"
          "shr.u32 %%r24, %%r20, 7;\nmad.lo.u32 %%r18, %%r4, %%r24, %%r3;\nmul.wide.u32 %%rd23, %%r18, 4;\n"
          "add.s64 %%rd23, %%rd5, %%rd23;\nst.global.f32 [%%rd23], %%q0;\nDONE:\nret;\n}\n\n");
}

/* attn_exp(S, MB, E, n, cap): grid (ceil(n/128), heads), 128 threads */
static void emit_exp(sbuf *b) {
    sb(b, ".visible .entry attn_exp(.param .u64 pS, .param .u64 pMB, .param .u64 pE, .param .u32 pN, .param .u32 pCap)\n{\n"
          ".reg .pred %%p<4>;\n.reg .b32 %%r<16>;\n.reg .b64 %%rd<12>;\n.reg .f32 %%m, %%t, %%x, %%e;\n"
          "ld.param.u64 %%rd1, [pS];\nld.param.u64 %%rd2, [pMB];\nld.param.u64 %%rd3, [pE];\nld.param.u32 %%r1, [pN];\n"
          "ld.param.u32 %%r11, [pCap];\nmov.u32 %%r2, %%tid.x;\nmov.u32 %%r3, %%ctaid.x;\nmov.u32 %%r4, %%ctaid.y;\n"
          "and.b32 %%r5, %%r2, 31;\nadd.u32 %%r6, %%r1, 127;\nshr.u32 %%r6, %%r6, 7;\n"
          "shr.u32 %%r12, %%r11, 7;\nmul.lo.u32 %%r12, %%r4, %%r12;\n"
          "mov.f32 %%m, 0fFF800000;\nmov.u32 %%r7, %%r5;\n"
          "MAXLOOP:\nsetp.ge.u32 %%p1, %%r7, %%r6;\n@%%p1 bra MAXDONE;\n"
          "add.u32 %%r8, %%r12, %%r7;\nmul.wide.u32 %%rd4, %%r8, 4;\nadd.s64 %%rd4, %%rd2, %%rd4;\n"
          "ld.global.f32 %%t, [%%rd4];\nmax.f32 %%m, %%m, %%t;\nadd.u32 %%r7, %%r7, 32;\nbra MAXLOOP;\nMAXDONE:\n");
    for (int off = 16; off >= 1; off /= 2)
        sb(b, "shfl.sync.bfly.b32 %%t, %%m, %d, 31, 0xffffffff;\nmax.f32 %%m, %%m, %%t;\n", off);
    sb(b, "shl.b32 %%r8, %%r3, 7;\nadd.u32 %%r8, %%r8, %%r2;\nsetp.ge.u32 %%p2, %%r8, %%r1;\n@%%p2 bra DONE;\n"
          "mul.wide.u32 %%rd5, %%r4, %%r11;\ncvt.u64.u32 %%rd6, %%r8;\nadd.s64 %%rd5, %%rd5, %%rd6;\nshl.b64 %%rd5, %%rd5, 2;\n"
          "add.s64 %%rd6, %%rd1, %%rd5;\nld.global.f32 %%x, [%%rd6];\nsub.rn.f32 %%x, %%x, %%m;\n");
    emit_expf(b, "%x", "%e", "EXP");
    sb(b, "add.s64 %%rd7, %%rd3, %%rd5;\nst.global.f32 [%%rd7], %%e;\nDONE:\nret;\n}\n\n");
}

/* cp.async of tile (register kreg) into stage (register breg) of attn_v, its values and its e_t, then commit */
static void emit_v_issue(sbuf *b, const char *kreg, const char *breg, int id) {
    sb(b, "setp.lt.u32 %%p1, %s, %%r7;\n@!%%p1 bra ISSUE%d;\nshl.b32 %%r20, %s, %d;\nadd.u32 %%r21, %%r20, %%r13;\n"
          "mul.lo.u32 %%r22, %s, %d;\nadd.u32 %%r23, %%r22, %%r16;\n",
       kreg, id, kreg, V_TILE_LOG2, breg, V_STAGE_BYTES);
    for (int j = 0; j < V_TILE / 16; j++)
        sb(b, "add.u32 %%r24, %%r21, %d;\nsetp.lt.u32 %%p2, %%r24, %%r1;\nmul.wide.u32 %%rd10, %%r24, %d;\n"
              "add.s64 %%rd10, %%rd6, %%rd10;\n@%%p2 cp.async.cg.shared.global [%%r23+%d], [%%rd10], 16;\n",
           16 * j, GPU_HEAD_DIM * 4, 16 * j * V_CHUNK * 4);
    /* the tile's e_t: threads 0..7, 16 bytes each, after the values of the stage */
    sb(b, "setp.lt.u32 %%p3, %%r2, 8;\nshl.b32 %%r25, %%r2, 2;\nadd.u32 %%r25, %%r25, %%r20;\n"
          "setp.lt.u32 %%p4, %%r25, %%r1;\nand.pred %%p4, %%p4, %%p3;\nmul.wide.u32 %%rd11, %%r25, 4;\n"
          "add.s64 %%rd11, %%rd4, %%rd11;\nmov.u32 %%r26, sv;\nadd.u32 %%r26, %%r26, %%r22;\nshl.b32 %%r27, %%r2, 4;\n"
          "add.u32 %%r26, %%r26, %%r27;\n@%%p4 cp.async.cg.shared.global [%%r26+%d], [%%rd11], 16;\n"
          "ISSUE%d:\ncp.async.commit_group;\n",
       V_TILE * V_CHUNK * 4, id);
}

/* one position of attn_v's sum: out = fl(out + fl(a_t * v_t)) */
#define V_STEP "mul.rn.f32 %%m, %%a, %%v;\nadd.rn.f32 %%o, %%o, %%m;\n"

/* attn_v(V, E, out, n, cap): grid (4, heads), 128 threads */
static void emit_v(sbuf *b) {
    sb(b, ".visible .entry attn_v(.param .u64 pV, .param .u64 pE, .param .u64 pOut, .param .u32 pN, .param .u32 pCap)\n{\n"
          ".reg .pred %%p<8>;\n.reg .b32 %%r<64>;\n.reg .b64 %%rd<16>;\n"
          ".reg .f32 %%o, %%acc, %%t, %%sum, %%ej, %%aj, %%a, %%v, %%m, %%x<8>;\n"
          ".shared .align 16 .b8 sv[%d];\n.shared .align 16 .f32 se[%d];\n.shared .align 4 .f32 ssum;\n"
          "ld.param.u64 %%rd1, [pV];\nld.param.u64 %%rd2, [pE];\nld.param.u64 %%rd3, [pOut];\nld.param.u32 %%r1, [pN];\n"
          "ld.param.u32 %%r40, [pCap];\nmov.u32 %%r2, %%tid.x;\nmov.u32 %%r3, %%ctaid.x;\nmov.u32 %%r4, %%ctaid.y;\n"
          "and.b32 %%r5, %%r2, 31;\nshr.u32 %%r6, %%r2, 5;\nadd.u32 %%r7, %%r1, %d;\nshr.u32 %%r7, %%r7, %d;\n",
       V_STAGES * V_STAGE_BYTES, SUM_CHUNK, V_TILE - 1, V_TILE_LOG2);
    /* rd4: the head's e_t; rd6: its values at dims chunk*32 + seg*4 (seg = tid & 7); r16: where a thread's
     * 16 bytes of a value row go in stage 0 (row = tid >> 3) */
    sb(b, "mul.wide.u32 %%rd4, %%r4, %%r40;\nshl.b64 %%rd4, %%rd4, 2;\nadd.s64 %%rd4, %%rd2, %%rd4;\n"
          "and.b32 %%r12, %%r2, 7;\nshr.u32 %%r13, %%r2, 3;\n"
          "mul.wide.u32 %%rd6, %%r4, %%r40;\nshl.b64 %%rd6, %%rd6, 9;\nadd.s64 %%rd6, %%rd1, %%rd6;\n"
          "shl.b32 %%r14, %%r3, 7;\nshl.b32 %%r15, %%r12, 4;\nadd.u32 %%r14, %%r14, %%r15;\ncvt.u64.u32 %%rd7, %%r14;\n"
          "add.s64 %%rd6, %%rd6, %%rd7;\nmov.u32 %%r16, sv;\nshl.b32 %%r17, %%r13, 7;\nadd.u32 %%r16, %%r16, %%r17;\n"
          "add.u32 %%r16, %%r16, %%r15;\n");
    /* the first V_STAGES - 1 tiles: in flight while the sum is taken */
    for (int s = 0; s < V_STAGES - 1; s++) {
        sb(b, "mov.u32 %%r30, %d;\n", s);
        emit_v_issue(b, "%r30", "%r30", s);
    }
    /* the sum: lane l of warp 0 (l < 16) adds e_t for t = l, l+16, ... in order, SUM_CHUNK positions at a
     * time through se, then the lanes meet in tr_lane_combine's tree */
    sb(b, "mov.f32 %%acc, 0f00000000;\nmov.u32 %%r30, 0;\nshl.b32 %%r27, %%r2, 4;\nmov.u32 %%r38, se;\n"
          "add.u32 %%r38, %%r38, %%r27;\n"
          "SUMCHUNK:\nsetp.ge.u32 %%p1, %%r30, %%r1;\n@%%p1 bra SUMEND;\nshl.b32 %%r31, %%r2, 2;\nadd.u32 %%r31, %%r31, %%r30;\n"
          "mul.wide.u32 %%rd12, %%r31, 4;\nadd.s64 %%rd12, %%rd4, %%rd12;\n");
    for (int r = 0; r < SUM_CHUNK / 512; r++)
        sb(b, "add.u32 %%r32, %%r31, %d;\nsetp.lt.u32 %%p2, %%r32, %%r1;\n"
              "@%%p2 cp.async.cg.shared.global [%%r38+%d], [%%rd12+%d], 16;\n",
           512 * r, 2048 * r, 2048 * r);
    sb(b, "cp.async.commit_group;\ncp.async.wait_group 0;\nbar.sync 0;\nsetp.ge.u32 %%p3, %%r2, 16;\n@%%p3 bra SUMSKIP;\n"
          "sub.u32 %%r33, %%r1, %%r30;\nmin.u32 %%r33, %%r33, %d;\nmov.u32 %%r34, %%r2;\nmov.u32 %%r35, se;\n"
          "shl.b32 %%r36, %%r2, 2;\nadd.u32 %%r35, %%r35, %%r36;\n"
          "SUM8:\nadd.u32 %%r37, %%r34, 112;\nsetp.ge.u32 %%p4, %%r37, %%r33;\n@%%p4 bra SUM1;\n",
       SUM_CHUNK);
    for (int x = 0; x < 8; x++) sb(b, "ld.shared.f32 %%x%d, [%%r35+%d];\n", x, 64 * x);
    for (int x = 0; x < 8; x++) sb(b, "add.rn.f32 %%acc, %%acc, %%x%d;\n", x);
    sb(b, "add.u32 %%r34, %%r34, 128;\nadd.u32 %%r35, %%r35, 512;\nbra SUM8;\n"
          "SUM1:\nsetp.ge.u32 %%p4, %%r34, %%r33;\n@%%p4 bra SUMSKIP;\nld.shared.f32 %%x0, [%%r35];\n"
          "add.rn.f32 %%acc, %%acc, %%x0;\nadd.u32 %%r34, %%r34, 16;\nadd.u32 %%r35, %%r35, 64;\nbra SUM1;\n"
          "SUMSKIP:\nbar.sync 0;\nadd.u32 %%r30, %%r30, %d;\nbra SUMCHUNK;\n"
          "SUMEND:\nsetp.ne.u32 %%p5, %%r6, 0;\n@%%p5 bra TREESKIP;\n",
       SUM_CHUNK);
    /* lane i adds lane i+1, then i+2, i+4, i+8: lane 0 ends with ((l0+l1)+(l2+l3)) + ... as tr_lane_combine */
    for (int k = 0; k < 4; k++)
        sb(b, "shfl.sync.down.b32 %%t, %%acc, %d, 31, 0xffffffff;\nadd.rn.f32 %%acc, %%acc, %%t;\n", 1 << k);
    sb(b, "setp.ne.u32 %%p5, %%r2, 0;\n@%%p5 bra TREESKIP;\nst.shared.f32 [ssum], %%acc;\n"
          "TREESKIP:\nbar.sync 0;\nld.shared.f32 %%sum, [ssum];\n");
    /* the tiles: wait for tile i, issue tile i + V_STAGES - 1 into the stage tile i - 1 left, warp 0 adds
     * tile i: lane j takes a_t of position t0 + j, and passes it along the warp for every dim */
    sb(b, "mov.u32 %%r41, 0;\nmov.u32 %%r42, 0;\nmov.u32 %%r43, %d;\nmov.u32 %%r44, %d;\n"
          "mov.f32 %%o, 0f00000000;\n"
          "setp.ne.u32 %%p6, %%r6, 0;\n"
          "LOOP:\nsetp.ge.u32 %%p1, %%r41, %%r7;\n@%%p1 bra LOOPEND;\ncp.async.wait_group %d;\nbar.sync 0;\n",
       V_STAGES - 1, V_STAGES - 1, V_STAGES - 2);
    emit_v_issue(b, "%r43", "%r44", 99);
    sb(b, "add.u32 %%r43, %%r43, 1;\nadd.u32 %%r44, %%r44, 1;\nsetp.eq.u32 %%p7, %%r44, %d;\n@%%p7 mov.u32 %%r44, 0;\n"
          "@%%p6 bra CHAINDONE;\n"
          "shl.b32 %%r45, %%r41, %d;\nmov.u32 %%r46, sv;\nmul.lo.u32 %%r47, %%r42, %d;\nadd.u32 %%r46, %%r46, %%r47;\n"
          "shl.b32 %%r47, %%r5, 2;\nadd.u32 %%r48, %%r46, %%r47;\nld.shared.f32 %%ej, [%%r48+%d];\n"
          "div.rn.f32 %%aj, %%ej, %%sum;\nsub.u32 %%r49, %%r1, %%r45;\nsetp.lt.u32 %%p3, %%r49, %d;\n@%%p3 bra PARTIAL;\n",
       V_STAGES, V_TILE_LOG2, V_STAGE_BYTES, V_TILE * V_CHUNK * 4, V_TILE);
    for (int j = 0; j < V_TILE; j++) {
        sb(b, "shfl.sync.idx.b32 %%a, %%aj, %d, 31, 0xffffffff;\nld.shared.f32 %%v, [%%r48+%d];\n", j, V_CHUNK * 4 * j);
        sb(b, V_STEP);
    }
    sb(b, "bra CHAINDONE;\nPARTIAL:\nmov.u32 %%r50, 0;\n"
          "PLOOP:\nshfl.sync.idx.b32 %%a, %%aj, %%r50, 31, 0xffffffff;\nld.shared.f32 %%v, [%%r48];\n");
    sb(b, V_STEP);
    sb(b, "add.u32 %%r48, %%r48, %d;\nadd.u32 %%r50, %%r50, 1;\nsetp.lt.u32 %%p3, %%r50, %%r49;\n@%%p3 bra PLOOP;\n"
          "CHAINDONE:\nadd.u32 %%r41, %%r41, 1;\nadd.u32 %%r42, %%r42, 1;\nsetp.eq.u32 %%p7, %%r42, %d;\n"
          "@%%p7 mov.u32 %%r42, 0;\nbra LOOP;\n"
          "LOOPEND:\n@%%p6 bra DONE;\nshl.b32 %%r51, %%r4, 7;\nshl.b32 %%r52, %%r3, 5;\nadd.u32 %%r51, %%r51, %%r52;\n"
          "add.u32 %%r51, %%r51, %%r5;\nmul.wide.u32 %%rd8, %%r51, 4;\nadd.s64 %%rd8, %%rd3, %%rd8;\n"
          "st.global.f32 [%%rd8], %%o;\nDONE:\nret;\n}\n\n",
       V_CHUNK * 4, V_STAGES);
}

/* kv_write(src k, src v, K, V, pos0, n, cap): grid (ceil(n/32), 4, heads), 32 x 8 threads. Rows t of the
 * sources ([n][heads * 128]) go to position pos0 + t: values straight, keys through a 32 x 32 tile of
 * shared memory, so that both the reads and the position-minor writes are whole lines */
static void emit_write(sbuf *b) {
    sb(b, ".visible .entry kv_write(.param .u64 pSK, .param .u64 pSV, .param .u64 pK, .param .u64 pV,\n"
          "    .param .u32 pPos0, .param .u32 pN, .param .u32 pCap)\n{\n"
          ".reg .pred %%p<4>;\n.reg .b32 %%r<32>;\n.reg .b64 %%rd<16>;\n.reg .f32 %%f<2>;\n"
          ".shared .align 4 .f32 tile[1056];\n"
          "ld.param.u64 %%rd1, [pSK];\nld.param.u64 %%rd2, [pSV];\nld.param.u64 %%rd3, [pK];\nld.param.u64 %%rd4, [pV];\n"
          "ld.param.u32 %%r1, [pPos0];\nld.param.u32 %%r2, [pN];\nld.param.u32 %%r3, [pCap];\n"
          "mov.u32 %%r4, %%tid.x;\nmov.u32 %%r5, %%tid.y;\nmov.u32 %%r6, %%ctaid.x;\nmov.u32 %%r7, %%ctaid.y;\n"
          "mov.u32 %%r8, %%ctaid.z;\nmov.u32 %%r9, %%nctaid.z;\nshl.b32 %%r10, %%r9, 7;\n"
          "shl.b32 %%r11, %%r7, 5;\nadd.u32 %%r11, %%r11, %%r4;\nshl.b32 %%r12, %%r8, 7;\nadd.u32 %%r12, %%r12, %%r11;\n"
          "shl.b32 %%r13, %%r6, 5;\nmov.u32 %%r14, tile;\nmul.wide.u32 %%rd5, %%r8, %%r3;\n");
    /* rows t = t_block + ty + 8r: the key into tile[t][d], the value straight to V[h][pos0 + t][d] */
    for (int r = 0; r < 4; r++)
        sb(b, "add.u32 %%r15, %%r13, %%r5;\nadd.u32 %%r15, %%r15, %d;\nsetp.lt.u32 %%p1, %%r15, %%r2;\n@!%%p1 bra READ%d;\n"
              "mul.wide.u32 %%rd6, %%r15, %%r10;\ncvt.u64.u32 %%rd7, %%r12;\nadd.s64 %%rd6, %%rd6, %%rd7;\n"
              "shl.b64 %%rd6, %%rd6, 2;\nadd.s64 %%rd8, %%rd1, %%rd6;\nld.global.f32 %%f0, [%%rd8];\n"
              "mad.lo.u32 %%r16, %%r5, 33, %%r4;\nadd.u32 %%r16, %%r16, %d;\nshl.b32 %%r16, %%r16, 2;\n"
              "add.u32 %%r16, %%r16, %%r14;\nst.shared.f32 [%%r16], %%f0;\n"
              "add.s64 %%rd9, %%rd2, %%rd6;\nld.global.f32 %%f1, [%%rd9];\nadd.u32 %%r17, %%r1, %%r15;\n"
              "cvt.u64.u32 %%rd10, %%r17;\nadd.s64 %%rd10, %%rd10, %%rd5;\nshl.b64 %%rd10, %%rd10, 7;\n"
              "cvt.u64.u32 %%rd11, %%r11;\nadd.s64 %%rd10, %%rd10, %%rd11;\nshl.b64 %%rd10, %%rd10, 2;\n"
              "add.s64 %%rd10, %%rd4, %%rd10;\nst.global.f32 [%%rd10], %%f1;\nREAD%d:\n",
           8 * r, r, 8 * 33 * r, r);
    /* keys out, transposed: thread (tx, ty) writes position t_block + tx of dims d_block + ty + 8r */
    sb(b, "bar.sync 0;\nadd.u32 %%r18, %%r13, %%r4;\nsetp.ge.u32 %%p2, %%r18, %%r2;\n@%%p2 bra DONE;\n"
          "add.u32 %%r19, %%r1, %%r18;\ncvt.u64.u32 %%rd13, %%r19;\n"
          "mad.lo.u32 %%r20, %%r4, 33, %%r5;\n"
          "shl.b32 %%r20, %%r20, 2;\nadd.u32 %%r20, %%r20, %%r14;\n"
          "shl.b32 %%r21, %%r7, 5;\nadd.u32 %%r21, %%r21, %%r5;\nshl.b32 %%r22, %%r8, 7;\nadd.u32 %%r22, %%r22, %%r21;\n");
    for (int r = 0; r < 4; r++)
        sb(b, "ld.shared.f32 %%f0, [%%r20+%d];\nadd.u32 %%r23, %%r22, %d;\nmul.wide.u32 %%rd12, %%r23, %%r3;\n"
              "add.s64 %%rd12, %%rd12, %%rd13;\nshl.b64 %%rd12, %%rd12, 2;\nadd.s64 %%rd12, %%rd3, %%rd12;\n"
              "st.global.f32 [%%rd12], %%f0;\n",
           32 * r, 8 * r);
    sb(b, "DONE:\nret;\n}\n\n");
}

/* keep_warm(gen, ns): one warp naps on the global timer until *gen changes (a call has started) or ns
 * nanoseconds have passed */
static void emit_warm(sbuf *b) {
    sb(b, ".visible .entry keep_warm(.param .u64 pGen, .param .u64 pNs)\n{\n.reg .pred %%p<2>;\n.reg .b32 %%r<3>;\n"
          ".reg .b64 %%rd<5>;\nld.param.u64 %%rd1, [pGen];\nld.param.u64 %%rd2, [pNs];\n"
          "ld.volatile.global.u32 %%r1, [%%rd1];\nmov.u64 %%rd3, %%globaltimer;\nadd.u64 %%rd2, %%rd2, %%rd3;\n"
          "SPIN:\nnanosleep.u32 %d;\nld.volatile.global.u32 %%r2, [%%rd1];\nsetp.ne.u32 %%p0, %%r2, %%r1;\n"
          "@%%p0 bra DONE;\nmov.u64 %%rd4, %%globaltimer;\nsetp.lt.u64 %%p1, %%rd4, %%rd2;\n@%%p1 bra SPIN;\n"
          "DONE:\nret;\n}\n\n",
       WARM_NAP_NS);
}

/* the whole module; NULL when memory runs out */
static char *build_ptx(void) {
    sbuf b = {malloc(1 << 17), 0, 1 << 17, 0};
    if (b.p == NULL) return NULL;
    sb(&b, ".version 7.0\n.target sm_80\n.address_size 64\n\n.global .align 8 .f64 tr_pow2[64] = {");
    for (int i = 0; i < 64; i++) sb(&b, "%s0d%016llX", i ? ", " : "", (unsigned long long)dbits(tr_expf_pow2[i]));
    sb(&b, "};\n\n");
    emit_scores(&b);
    emit_exp(&b);
    emit_v(&b);
    emit_write(&b);
    emit_warm(&b);
    if (b.bad) {
        free(b.p);
        return NULL;
    }
    return b.p;
}

/* ---- the device --------------------------------------------------------------------------------- */

tr_gpu *tr_gpu_open(char *err, size_t err_len) {
#if defined(_WIN32)
    const char *lib_name = "nvcuda.dll";
#else
    const char *lib_name = "libcuda.so.1";
#endif
    tr_lib *lib = tr_lib_open(lib_name);
    if (lib == NULL) {
        set_err(err, err_len, "no NVIDIA driver (%s)", lib_name);
        return NULL;
    }
    tr_gpu *g = calloc(1, sizeof *g);
    if (g == NULL) {
        tr_lib_close(lib);
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    g->lib = lib;
    for (size_t i = 0; i < sizeof cu_names / sizeof cu_names[0]; i++) {
        void *p = tr_lib_sym(lib, cu_names[i].name);
        if (p == NULL) {
            set_err(err, err_len, "%s has no %s", lib_name, cu_names[i].name);
            tr_lib_close(lib);
            free(g);
            return NULL;
        }
        memcpy((char *)&g->cu + cu_names[i].off, &p, sizeof p); /* a symbol's address as the function pointer */
    }
    /* from here on the driver stays loaded: it keeps threads of its own for the life of the process */
    int n_dev = 0;
    if (cu_check(g, g->cu.Init(0), "cuInit", err, err_len) != 0 ||
        cu_check(g, g->cu.DeviceGetCount(&n_dev), "cuDeviceGetCount", err, err_len) != 0) {
        free(g);
        return NULL;
    }
    int found = 0;
    for (int i = 0; i < n_dev && !found; i++) {
        int major = 0, minor = 0;
        if (g->cu.DeviceGet(&g->dev, i) != 0 || g->cu.DeviceGetAttribute(&major, CU_ATTR_CC_MAJOR, g->dev) != 0 ||
            g->cu.DeviceGetAttribute(&minor, CU_ATTR_CC_MINOR, g->dev) != 0)
            continue;
        found = major >= 8;
    }
    if (!found) {
        set_err(err, err_len, "no CUDA device of compute capability 8.0 or later (%d devices)", n_dev);
        free(g);
        return NULL;
    }
    if (g->cu.DeviceGetName(g->name, (int)sizeof g->name, g->dev) != 0) snprintf(g->name, sizeof g->name, "GPU");
    if (cu_check(g, g->cu.CtxCreate(&g->ctx, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, g->dev), "cuCtxCreate", err,
                 err_len) != 0) {
        free(g);
        return NULL;
    }
    char *ptx = build_ptx();
    char *log = malloc(16384);
    if (ptx == NULL || log == NULL) {
        free(ptx);
        free(log);
        set_err(err, err_len, "out of memory");
        tr_gpu_close(g);
        return NULL;
    }
    log[0] = '\0';
    int opt[2] = {CU_JIT_ERROR_LOG, CU_JIT_ERROR_LOG_SIZE};
    void *val[2] = {log, (void *)(uintptr_t)16384};
    CUresult r = g->cu.ModuleLoadDataEx(&g->mod, ptx, 2, opt, val);
    free(ptx);
    if (r != 0) {
        set_err(err, err_len, "the driver did not compile the kernels (CUDA error %d): %s", r, log);
        free(log);
        g->mod = NULL;
        tr_gpu_close(g);
        return NULL;
    }
    free(log);
    if (cu_check(g, g->cu.ModuleGetFunction(&g->f_scores, g->mod, "attn_scores"), "attn_scores", err, err_len) != 0 ||
        cu_check(g, g->cu.ModuleGetFunction(&g->f_exp, g->mod, "attn_exp"), "attn_exp", err, err_len) != 0 ||
        cu_check(g, g->cu.ModuleGetFunction(&g->f_v, g->mod, "attn_v"), "attn_v", err, err_len) != 0 ||
        cu_check(g, g->cu.ModuleGetFunction(&g->f_write, g->mod, "kv_write"), "kv_write", err, err_len) != 0 ||
        cu_check(g, g->cu.ModuleGetFunction(&g->f_warm, g->mod, "keep_warm"), "keep_warm", err, err_len) != 0) {
        tr_gpu_close(g);
        return NULL;
    }
    return g;
}

void tr_gpu_close(tr_gpu *g) {
    if (g == NULL) return;
    if (g->ctx != NULL) {
        g->cu.CtxSetCurrent(g->ctx);
        if (g->mod != NULL) g->cu.ModuleUnload(g->mod);
        g->cu.CtxDestroy(g->ctx);
    }
    /* the library itself stays loaded (tr_gpu_open says why) */
    free(g);
}

const char *tr_gpu_name(const tr_gpu *g) {
    return g != NULL ? g->name : "";
}

/* ---- a session's cache -------------------------------------------------------------------------- */

static int64_t cap_of(int64_t n_ctx) {
    return (n_ctx + GPU_CAP_ALIGN - 1) / GPU_CAP_ALIGN * GPU_CAP_ALIGN;
}

uint64_t tr_gpu_attn_bytes(int64_t n_layers, int64_t n_head, int64_t head_dim, int64_t n_ctx) {
    if (n_layers <= 0 || n_head <= 0 || head_dim <= 0 || n_ctx <= 0) return 0;
    uint64_t cap = (uint64_t)cap_of(n_ctx), heads = (uint64_t)n_head;
    uint64_t kv = 2 * (uint64_t)n_layers * heads * cap * (uint64_t)head_dim * sizeof(float);
    uint64_t scratch = 2 * heads * cap * sizeof(float) + heads * (cap / GPU_CAP_ALIGN) * sizeof(float) + 256;
    return kv + scratch;
}

void tr_gpu_attn_free(tr_gpu_attn *a) {
    if (a == NULL) return;
    tr_gpu *g = a->g;
    g->cu.CtxSetCurrent(g->ctx);
    /* a keep_warm still napping ends by itself within TR_GPU_WARM_NS */
    if (a->warm != NULL) {
        g->cu.StreamSynchronize(a->warm);
        g->cu.StreamDestroy(a->warm);
    }
    if (a->st != NULL) {
        g->cu.StreamSynchronize(a->st);
        g->cu.StreamDestroy(a->st);
    }
    if (a->kv) g->cu.MemFree(a->kv);
    if (a->s) g->cu.MemFree(a->s);
    if (a->mb) g->cu.MemFree(a->mb);
    if (a->e) g->cu.MemFree(a->e);
    if (a->gen) g->cu.MemFree(a->gen);
    if (a->h_in != NULL) g->cu.MemFreeHost(a->h_in);
    if (a->h_out != NULL) g->cu.MemFreeHost(a->h_out);
    if (a->h_stage != NULL) g->cu.MemFreeHost(a->h_stage);
    free(a);
}

/* pinned memory of n bytes, and the address the kernels see it at */
static int alloc_pinned(tr_gpu *g, size_t n, float **host, CUdeviceptr *dev, char *err, size_t err_len) {
    void *p = NULL;
    if (cu_check(g, g->cu.MemAllocHost(&p, n), "cuMemAllocHost", err, err_len) != 0) return -1;
    *host = p;
    return cu_check(g, g->cu.MemHostGetDevicePointer(dev, p, 0), "cuMemHostGetDevicePointer", err, err_len);
}

tr_gpu_attn *tr_gpu_attn_create(tr_gpu *g, int64_t n_layers, int64_t n_head, int64_t head_dim, int64_t n_ctx,
                                char *err, size_t err_len) {
    if (g == NULL) {
        set_err(err, err_len, "no GPU");
        return NULL;
    }
    if (head_dim != GPU_HEAD_DIM) {
        set_err(err, err_len, "head_dim %lld: the GPU attention takes %d", (long long)head_dim, GPU_HEAD_DIM);
        return NULL;
    }
    if (n_layers < 1 || n_head < 1 || n_head > 65535 || n_ctx < 1 || n_ctx > GPU_MAX_CTX) {
        set_err(err, err_len, "shape not supported by the GPU attention: %lld layers, %lld heads, context %lld",
                (long long)n_layers, (long long)n_head, (long long)n_ctx);
        return NULL;
    }
    if (cu_check(g, g->cu.CtxSetCurrent(g->ctx), "cuCtxSetCurrent", err, err_len) != 0) return NULL;
    uint64_t need = tr_gpu_attn_bytes(n_layers, n_head, head_dim, n_ctx);
    size_t free_b = 0, total_b = 0;
    if (cu_check(g, g->cu.MemGetInfo(&free_b, &total_b), "cuMemGetInfo", err, err_len) != 0) return NULL;
    if ((uint64_t)free_b < need + TR_GPU_VRAM_RESERVE) {
        set_err(err, err_len, "the GPU cache needs %.0f MiB of VRAM: %.0f of %.0f MiB are free and %.0f must stay free",
                (double)need / 1048576.0, (double)free_b / 1048576.0, (double)total_b / 1048576.0,
                (double)TR_GPU_VRAM_RESERVE / 1048576.0);
        return NULL;
    }
    tr_gpu_attn *a = calloc(1, sizeof *a);
    if (a == NULL) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    a->g = g;
    a->n_layers = n_layers;
    a->n_head = n_head;
    a->n_ctx = n_ctx;
    a->cap = cap_of(n_ctx);
    a->warm_ns = TR_GPU_WARM_NS;
    size_t heads = (size_t)n_head, cap = (size_t)a->cap, row = heads * GPU_HEAD_DIM;
    a->layer_bytes = 2 * (uint64_t)heads * cap * GPU_HEAD_DIM * sizeof(float);
    if (cu_check(g, g->cu.MemAlloc(&a->kv, (size_t)(a->layer_bytes * (uint64_t)n_layers)), "cuMemAlloc", err,
                 err_len) != 0 ||
        cu_check(g, g->cu.MemAlloc(&a->s, heads * cap * sizeof(float)), "cuMemAlloc", err, err_len) != 0 ||
        cu_check(g, g->cu.MemAlloc(&a->e, heads * cap * sizeof(float)), "cuMemAlloc", err, err_len) != 0 ||
        cu_check(g, g->cu.MemAlloc(&a->mb, heads * (cap / GPU_CAP_ALIGN) * sizeof(float)), "cuMemAlloc", err, err_len) !=
            0 ||
        cu_check(g, g->cu.MemAlloc(&a->gen, 256), "cuMemAlloc", err, err_len) != 0 ||
        cu_check(g, g->cu.MemsetD32(a->gen, 0, 1), "cuMemsetD32", err, err_len) != 0 ||
        cu_check(g, g->cu.StreamCreate(&a->st, CU_STREAM_NON_BLOCKING), "cuStreamCreate", err, err_len) != 0 ||
        cu_check(g, g->cu.StreamCreate(&a->warm, CU_STREAM_NON_BLOCKING), "cuStreamCreate", err, err_len) != 0 ||
        alloc_pinned(g, 3 * row * sizeof(float), &a->h_in, &a->d_in, err, err_len) != 0 ||
        alloc_pinned(g, row * sizeof(float), &a->h_out, &a->d_out, err, err_len) != 0 ||
        alloc_pinned(g, 2 * (size_t)WRITE_ROWS * row * sizeof(float), &a->h_stage, &a->d_stage, err, err_len) != 0) {
        tr_gpu_attn_free(a);
        return NULL;
    }
    return a;
}

/* hot: begin */

int tr_gpu_attn_write(tr_gpu_attn *a, int64_t layer, int64_t pos0, int64_t n_tok, const float *k, const float *v) {
    if (a == NULL || layer < 0 || layer >= a->n_layers || pos0 < 0 || n_tok < 0 || pos0 + n_tok > a->n_ctx) return -1;
    if (n_tok == 0) return 0;
    tr_gpu *g = a->g;
    if (g->cu.CtxSetCurrent(g->ctx) != 0) return -1;
    size_t row = (size_t)a->n_head * GPU_HEAD_DIM;
    CUdeviceptr K = a->kv + a->layer_bytes * (uint64_t)layer;
    CUdeviceptr V = K + a->layer_bytes / 2;
    CUdeviceptr sk = a->d_stage, sv = a->d_stage + (CUdeviceptr)(WRITE_ROWS * row * sizeof(float));
    unsigned cap = (unsigned)a->cap;
    for (int64_t done = 0; done < n_tok; done += WRITE_ROWS) {
        int64_t m = n_tok - done < WRITE_ROWS ? n_tok - done : WRITE_ROWS;
        /* the staging buffer is free: the previous piece was waited for */
        memcpy(a->h_stage, k + (size_t)done * row, (size_t)m * row * sizeof(float));
        memcpy(a->h_stage + WRITE_ROWS * row, v + (size_t)done * row, (size_t)m * row * sizeof(float));
        unsigned p0 = (unsigned)(pos0 + done), nt = (unsigned)m;
        void *args[] = {&sk, &sv, &K, &V, &p0, &nt, &cap};
        if (g->cu.LaunchKernel(g->f_write, (nt + 31) / 32, GPU_HEAD_DIM / 32, (unsigned)a->n_head, 32, 8, 1, 0, a->st,
                               args, NULL) != 0 ||
            g->cu.StreamSynchronize(a->st) != 0)
            return -1;
    }
    return 0;
}

int tr_gpu_attn_decode(tr_gpu_attn *a, int64_t layer, int64_t pos, const float *q, const float *k, const float *v,
                       float scale, float *out) {
    if (a == NULL || layer < 0 || layer >= a->n_layers || pos < 0 || pos >= a->n_ctx) return -1;
    tr_gpu *g = a->g;
    if (g->cu.CtxSetCurrent(g->ctx) != 0) return -1;
    size_t row = (size_t)a->n_head * GPU_HEAD_DIM;
    memcpy(a->h_in, q, row * sizeof(float));
    memcpy(a->h_in + row, k, row * sizeof(float));
    memcpy(a->h_in + 2 * row, v, row * sizeof(float));
    CUdeviceptr K = a->kv + a->layer_bytes * (uint64_t)layer;
    CUdeviceptr V = K + a->layer_bytes / 2;
    unsigned n = (unsigned)(pos + 1), cap = (unsigned)a->cap, heads = (unsigned)a->n_head, nb = (n + 127) / 128;
    unsigned seq = ++a->seq;
    void *as[] = {&K, &a->d_in, &V, &a->s, &a->mb, &n, &scale, &cap, &seq, &a->gen};
    void *ae[] = {&a->s, &a->mb, &a->e, &n, &cap};
    void *av[] = {&V, &a->e, &a->d_out, &n, &cap};
    void *aw[] = {&a->gen, &a->warm_ns};
    if (g->cu.LaunchKernel(g->f_scores, nb, heads, 1, 128, 1, 1, 0, a->st, as, NULL) != 0 ||
        g->cu.LaunchKernel(g->f_exp, nb, heads, 1, 128, 1, 1, 0, a->st, ae, NULL) != 0 ||
        g->cu.LaunchKernel(g->f_v, GPU_HEAD_DIM / V_CHUNK, heads, 1, 128, 1, 1, 0, a->st, av, NULL) != 0 ||
        g->cu.StreamSynchronize(a->st) != 0)
        return -1;
    memcpy(out, a->h_out, row * sizeof(float));
    /* one warp keeps the GPU awake through the CPU's gap, until the next call starts */
    if (g->cu.LaunchKernel(g->f_warm, 1, 1, 1, 32, 1, 1, 0, a->warm, aw, NULL) != 0) return -1;
    return 0;
}

int tr_gpu_attn_warm(tr_gpu_attn *a, uint64_t ns) {
    if (a == NULL) return -1;
    tr_gpu *g = a->g;
    if (g->cu.CtxSetCurrent(g->ctx) != 0) return -1;
    /* one at a time: a warp still napping covers this call (a pass writes a layer every ~100 ms).
     * Queued warps would each nap their full time once no decode call ends them: up to a layer
     * count of them in a row, the GPU spinning and tr_gpu_attn_free waiting seconds. */
    CUresult busy = g->cu.StreamQuery(a->warm);
    if (busy == CU_ERROR_NOT_READY) return 0;
    if (busy != 0) return -1;
    unsigned long long n = ns < TR_GPU_WARM_MAX_NS ? (unsigned long long)ns : TR_GPU_WARM_MAX_NS;
    void *aw[] = {&a->gen, &n}; /* the driver copies the arguments at launch */
    return g->cu.LaunchKernel(g->f_warm, 1, 1, 1, 32, 1, 1, 0, a->warm, aw, NULL) != 0 ? -1 : 0;
}

/* hot: end */
