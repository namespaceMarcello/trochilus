/* bench_gpu_attn.c — premise benchmark: the decode attention of OLMoE-1B-7B on an NVIDIA GPU, bit for
 * bit the CPU engine's, and what one layer costs there.
 *
 * At context 2048 the CPU decode spends 11.3 of 36.8 ms per token in the attention: 16 layers of 16
 * heads of 128, f32 keys and values, 32 MiB per layer read at 48 GB/s. A laptop RTX 4070 reads its
 * GDDR6 at ~256 GB/s. This program answers two questions before any engine code is written: can the
 * GPU give the same bits, and how long does a layer take there, kernels alone and with the copies
 * and the synchronization the engine would pay.
 *
 * No build dependency: nvcuda.dll (libcuda.so.1) is opened at run time, the kernels are PTX text
 * written here and JIT-compiled by the driver. Every float operation carries .rn (PTX never fuses
 * those into an FMA), no .ftz, no .approx, so the kernels do the arithmetic of tr_attention_group
 * (src/kernels/kernels.c), operation for operation:
 *   s_t = fl(dot(q, k_t) * scale), the dot under the 16-lane contract and tr_lane_combine's tree
 *   m   = max s_t, in any order (a zero maximum's sign cannot reach the output: exp(+-0) = 1)
 *   e_t = tr_expf(fl(s_t - m)): expf_core of src/kernels/expf.c in f64, exception table included
 *   sum = the 16-lane sum of e_t with the same tree; a_t = fl(e_t / sum)
 *   out = +0.0f, then out = fl(out + fl(a_t * v_t)) for t increasing
 * In VRAM the keys are position-minor ([head][dim][pos]: a warp reads 32 positions in one line) and
 * the values [head][pos][dim]; CAP positions per head. Three kernels per layer:
 *   attn_scores  grid (positions/128, head), one thread per position, q in shared memory, the
 *                block's maximum. The block holding position n-1 appends the new key and value to
 *                the cache; the thread of that position reads its key from the input
 *   attn_exp     grid (positions/128, head): the head's maximum from the block maxima, e_t
 *   attn_v       grid (4, head), 32 dims per block: the 16-lane sum and a_t in shared memory, then
 *                the whole block streams the values through shared memory (cp.async, V_STAGES - 1
 *                tiles in flight) while one warp adds them in position order
 *
 * Modes (one run stays under 60 s, VRAM under 2 GB):
 *   info                     the device, the driver, registers and spills of every kernel
 *   check <run dir>          every query of every layer and head of a run dumped by tools/attn_probe.c
 *                            (q_Lxx_Hyy.bin records of int64 n_pos + 128 floats, k/v_Lxx_Hyy.bin rows of
 *                            128 floats), through the engine's round trip (the new key and value of each
 *                            token appended by the GPU, unwritten positions NaN), against
 *                            tr_attention_group: the 128 floats compared as bits. --tier T for the
 *                            reference (default scalar, the definition); --zero-copy for the round trip
 *                            whose kernels read the input and write the output in pinned host memory
 *   edge                     synthetic heads: n_pos 1..4096 across every tile and block border, values
 *                            of -0, exponentials that underflow, equal scores, arguments that go
 *                            through tr_expf's exception table (counted: the case fails if none does)
 *   expf                     the GPU's tr_expf against the CPU's on all 2^32 floats
 *   time [--data <run dir>] [--n N] [--only abcezdws] [--runs R] [--tokens T] [--gap-us G] [--settle S]
 *                            per layer at n_pos 2048 and 4000, 16 layers in rotation (1 GiB, so no layer
 *                            is still in the 32 MB L2 when its turn comes): (a) kernels alone, cuEvent;
 *                            (b) the round trip: pinned buffers, q and the new key and value host to
 *                            device, the kernels, out device to host, stream synchronized; (c) a plain
 *                            read of the same bytes, the ceiling; (e) the floor: an empty kernel and
 *                            the three kernels without copies, launch to synchronization; (z) the round
 *                            trip in zero copy; (d) the round trip in bursts, G us (1600) of CPU spin
 *                            between two calls as the decode leaves between two layers; (w) the same
 *                            with a keep-warm kernel (one warp spinning on another stream) through each
 *                            gap; (s) the same with the keep-warm napping. Every pattern runs S seconds
 *                            (1) unmeasured first, so the clocks settle under it; then medians of R
 *                            (>= 50) calls, or of T (>= 16) tokens x 16 layers for d, w, s. Each phase
 *                            prints its begin and end time, to read nvidia-smi's clocks against.
 *                            --zero-copy makes b, d, w and s zero copy too
 *   --mutate fma|tree|sumtree|zero|expf   a wrong kernel on purpose (no .rn in the value sum, another
 *                            dot tree, another sum tree, a -0 start, no exception table): check, edge or
 *                            expf must then fail */
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <time.h>
#endif

#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/kernels/expf_table.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

/* OLMoE-1B-7B */
#define N_HEAD 16
#define HEAD_DIM 128
#define N_LAYER 16
#define QKV (N_HEAD * HEAD_DIM)
#define CAP 4096                /* cache positions per head in VRAM */
#define MB_PER_HEAD (CAP / 128) /* block maxima of attn_scores per head */
#define IN_FLOATS (3 * QKV)     /* the input of a layer: q, the new key, the new value */
#define V_TILE 32               /* positions per shared-memory stage of attn_v */
#define V_TILE_LOG2 5
#define V_STAGES 6
#define V_CHUNK 32 /* dims per attn_v block */
#define MAX_REC 64
#define MAX_RUNS 4096
#define EXPF_CHUNK (1u << 24)

/* ---- the driver API, opened at run time ------------------------------------------------------ */

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;

#define CU_CTX_SCHED_SPIN 0x01
#define CU_CTX_MAP_HOST 0x08
#define CU_STREAM_NON_BLOCKING 0x1
enum { JIT_INFO_LOG = 3, JIT_INFO_LOG_SIZE = 4, JIT_ERROR_LOG = 5, JIT_ERROR_LOG_SIZE = 6 };
enum { ATTR_CLOCK = 13, ATTR_SMS = 16, ATTR_TCC = 35, ATTR_MEM_CLOCK = 36, ATTR_BUS = 37, ATTR_L2 = 38,
       ATTR_CC_MAJOR = 75, ATTR_CC_MINOR = 76 };
enum { FATTR_SHARED = 1, FATTR_LOCAL = 3, FATTR_REGS = 4 };

typedef struct {
    CUresult (*Init)(unsigned);
    CUresult (*DriverGetVersion)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*DeviceTotalMem)(size_t *, CUdevice);
    CUresult (*CtxCreate)(CUcontext *, unsigned, CUdevice);
    CUresult (*CtxDestroy)(CUcontext);
    CUresult (*CtxSynchronize)(void);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*ModuleLoadDataEx)(CUmodule *, const void *, unsigned, int *, void **);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*FuncGetAttribute)(int *, int, CUfunction);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemAllocHost)(void **, size_t);
    CUresult (*MemFreeHost)(void *);
    CUresult (*MemHostGetDevicePointer)(CUdeviceptr *, void *, unsigned);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*MemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (*MemsetD32)(CUdeviceptr, unsigned, size_t);
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*StreamDestroy)(CUstream);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*EventCreate)(CUevent *, unsigned);
    CUresult (*EventDestroy)(CUevent);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventElapsedTime)(float *, CUevent, CUevent);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             CUstream, void **, void **);
    CUresult (*GetErrorString)(CUresult, const char **);
} cu_api;

static cu_api cu;

static const struct {
    size_t off;
    const char *name;
} cu_syms[] = {
    {offsetof(cu_api, Init), "cuInit"},
    {offsetof(cu_api, DriverGetVersion), "cuDriverGetVersion"},
    {offsetof(cu_api, DeviceGet), "cuDeviceGet"},
    {offsetof(cu_api, DeviceGetName), "cuDeviceGetName"},
    {offsetof(cu_api, DeviceGetAttribute), "cuDeviceGetAttribute"},
    {offsetof(cu_api, DeviceTotalMem), "cuDeviceTotalMem_v2"},
    {offsetof(cu_api, CtxCreate), "cuCtxCreate_v2"},
    {offsetof(cu_api, CtxDestroy), "cuCtxDestroy_v2"},
    {offsetof(cu_api, CtxSynchronize), "cuCtxSynchronize"},
    {offsetof(cu_api, MemGetInfo), "cuMemGetInfo_v2"},
    {offsetof(cu_api, ModuleLoadDataEx), "cuModuleLoadDataEx"},
    {offsetof(cu_api, ModuleUnload), "cuModuleUnload"},
    {offsetof(cu_api, ModuleGetFunction), "cuModuleGetFunction"},
    {offsetof(cu_api, FuncGetAttribute), "cuFuncGetAttribute"},
    {offsetof(cu_api, MemAlloc), "cuMemAlloc_v2"},
    {offsetof(cu_api, MemFree), "cuMemFree_v2"},
    {offsetof(cu_api, MemAllocHost), "cuMemAllocHost_v2"},
    {offsetof(cu_api, MemFreeHost), "cuMemFreeHost"},
    {offsetof(cu_api, MemHostGetDevicePointer), "cuMemHostGetDevicePointer_v2"},
    {offsetof(cu_api, MemcpyHtoD), "cuMemcpyHtoD_v2"},
    {offsetof(cu_api, MemcpyDtoH), "cuMemcpyDtoH_v2"},
    {offsetof(cu_api, MemcpyHtoDAsync), "cuMemcpyHtoDAsync_v2"},
    {offsetof(cu_api, MemcpyDtoHAsync), "cuMemcpyDtoHAsync_v2"},
    {offsetof(cu_api, MemsetD32), "cuMemsetD32_v2"},
    {offsetof(cu_api, StreamCreate), "cuStreamCreate"},
    {offsetof(cu_api, StreamDestroy), "cuStreamDestroy_v2"},
    {offsetof(cu_api, StreamSynchronize), "cuStreamSynchronize"},
    {offsetof(cu_api, EventCreate), "cuEventCreate"},
    {offsetof(cu_api, EventDestroy), "cuEventDestroy_v2"},
    {offsetof(cu_api, EventRecord), "cuEventRecord"},
    {offsetof(cu_api, EventElapsedTime), "cuEventElapsedTime"},
    {offsetof(cu_api, LaunchKernel), "cuLaunchKernel"},
    {offsetof(cu_api, GetErrorString), "cuGetErrorString"},
};

typedef void (*any_fn)(void);

static int cu_open(void) {
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
#else
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
#endif
    if (lib == NULL) {
        fprintf(stderr, "no NVIDIA driver (nvcuda.dll / libcuda.so.1)\n");
        return -1;
    }
    for (size_t i = 0; i < sizeof cu_syms / sizeof cu_syms[0]; i++) {
#ifdef _WIN32
        FARPROC p = GetProcAddress(lib, cu_syms[i].name);
#else
        void *p = dlsym(lib, cu_syms[i].name);
#endif
        if (p == NULL) {
            fprintf(stderr, "the driver has no %s\n", cu_syms[i].name);
            return -1;
        }
        any_fn f;
        memcpy(&f, &p, sizeof f); /* every function pointer has the same representation here */
        memcpy((char *)&cu + cu_syms[i].off, &f, sizeof f);
    }
    return 0;
}

static void cu_fail(CUresult r, const char *what, int line) {
    const char *s = NULL;
    if (cu.GetErrorString != NULL) cu.GetErrorString(r, &s);
    fprintf(stderr, "bench_gpu_attn.c:%d: %s failed: %d %s\n", line, what, r, s != NULL ? s : "");
    exit(2);
}
#define CU(x) \
    do { \
        CUresult cu_r_ = (x); \
        if (cu_r_ != 0) cu_fail(cu_r_, #x, __LINE__); \
    } while (0)

/* ---- PTX, written here ----------------------------------------------------------------------- */

enum { MUT_FMA = 1, MUT_TREE = 2, MUT_SUMTREE = 4, MUT_ZERO = 8, MUT_EXPF = 16 };
static int g_mut = 0; /* the mutation asked for on the command line, 0 for the real kernels */

typedef struct {
    char *p;
    size_t len, cap;
} sbuf;

static void sb(sbuf *b, const char *fmt, ...) {
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(b->p + b->len, b->cap - b->len, fmt, ap);
        va_end(ap);
        if (n < 0) {
            fprintf(stderr, "PTX: format error\n");
            exit(2);
        }
        if ((size_t)n < b->cap - b->len) {
            b->len += (size_t)n;
            return;
        }
        size_t cap = 2 * b->cap + (size_t)n + 1;
        char *p = realloc(b->p, cap);
        if (p == NULL) {
            fprintf(stderr, "out of memory\n");
            exit(2);
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

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* y = tr_expf(x), expf_core of src/kernels/expf.c operation for operation; L makes the labels unique */
static void emit_expf(sbuf *b, const char *x, const char *y, const char *L) {
    sb(b, "{\n.reg .pred %%ep<4>;\n.reg .f32 %%ef<2>;\n.reg .f64 %%ed<24>;\n.reg .b64 %%er<8>;\n.reg .b32 %%eu<4>;\n");
    sb(b, "setp.nan.f32 %%ep0, %s, %s;\n@%%ep0 bra %s_nan;\n", x, x, L);
    sb(b, "setp.gt.f32 %%ep1, %s, 0f%08" PRIX32 ";\n@%%ep1 bra %s_ovf;\n", x, fbits(TR_EXPF_OVERFLOW_X), L);
    sb(b, "setp.lt.f32 %%ep2, %s, 0f%08" PRIX32 ";\n@%%ep2 bra %s_unf;\n", x, fbits(TR_EXPF_UNDERFLOW_X), L);
    sb(b, "cvt.f64.f32 %%ed0, %s;\n", x);
    /* kd = (xd * 64/ln2 + 0x1.8p52) - 0x1.8p52, k = (int64_t)kd */
    sb(b, "mul.rn.f64 %%ed1, %%ed0, 0d%016" PRIX64 ";\n", dbits(TR_EXPF_INV_LN2_64));
    sb(b, "add.rn.f64 %%ed2, %%ed1, 0d%016" PRIX64 ";\n", dbits(0x1.8p52));
    sb(b, "sub.rn.f64 %%ed3, %%ed2, 0d%016" PRIX64 ";\n", dbits(0x1.8p52));
    sb(b, "cvt.rzi.s64.f64 %%er0, %%ed3;\n");
    /* r = (xd - kd * hi) - kd * lo */
    sb(b, "mul.rn.f64 %%ed4, %%ed3, 0d%016" PRIX64 ";\n", dbits(TR_EXPF_LN2_64_HI));
    sb(b, "sub.rn.f64 %%ed5, %%ed0, %%ed4;\n");
    sb(b, "mul.rn.f64 %%ed6, %%ed3, 0d%016" PRIX64 ";\n", dbits(TR_EXPF_LN2_64_LO));
    sb(b, "sub.rn.f64 %%ed7, %%ed5, %%ed6;\n");
    /* p = r + r * r * (0.5 + r * (C3 + r * (C4 + r * C5))) */
    sb(b, "mul.rn.f64 %%ed8, %%ed7, 0d%016" PRIX64 ";\n", dbits(TR_EXPF_C5));
    sb(b, "add.rn.f64 %%ed9, %%ed8, 0d%016" PRIX64 ";\n", dbits(TR_EXPF_C4));
    sb(b, "mul.rn.f64 %%ed10, %%ed7, %%ed9;\n");
    sb(b, "add.rn.f64 %%ed11, %%ed10, 0d%016" PRIX64 ";\n", dbits(TR_EXPF_C3));
    sb(b, "mul.rn.f64 %%ed12, %%ed7, %%ed11;\n");
    sb(b, "add.rn.f64 %%ed13, %%ed12, 0d%016" PRIX64 ";\n", dbits(0.5));
    sb(b, "mul.rn.f64 %%ed14, %%ed7, %%ed7;\n");
    sb(b, "mul.rn.f64 %%ed15, %%ed14, %%ed13;\n");
    sb(b, "add.rn.f64 %%ed16, %%ed7, %%ed15;\n");
    /* t = pow2[k & 63], scale = 2^(k >> 6), y = (t + t * p) * scale */
    sb(b, "and.b64 %%er1, %%er0, 63;\nshl.b64 %%er1, %%er1, 3;\nmov.u64 %%er2, tr_pow2;\nadd.s64 %%er2, %%er2, %%er1;\n"
          "ld.global.nc.f64 %%ed17, [%%er2];\n"
          "shr.s64 %%er3, %%er0, 6;\nadd.s64 %%er3, %%er3, 1023;\nshl.b64 %%er3, %%er3, 52;\nmov.b64 %%ed18, %%er3;\n"
          "mul.rn.f64 %%ed19, %%ed17, %%ed16;\nadd.rn.f64 %%ed20, %%ed17, %%ed19;\nmul.rn.f64 %%ed21, %%ed20, %%ed18;\n");
    /* the rounding test: y * (1 -+ 2^-50) round to the same float */
    sb(b, "mul.rn.f64 %%ed22, %%ed21, 0d%016" PRIX64 ";\n", dbits(1.0 - TR_EXPF_MARGIN));
    sb(b, "mul.rn.f64 %%ed23, %%ed21, 0d%016" PRIX64 ";\n", dbits(1.0 + TR_EXPF_MARGIN));
    sb(b, "cvt.rn.f32.f64 %%ef0, %%ed22;\ncvt.rn.f32.f64 %%ef1, %%ed23;\nmov.b32 %%eu0, %%ef0;\nmov.b32 %%eu1, %%ef1;\n"
          "mov.f32 %s, %%ef0;\nsetp.eq.b32 %%ep3, %%eu0, %%eu1;\n@%%ep3 bra %s_done;\n",
       y, L);
    /* the exception table, else NaN (bench_expf --check proves no float gets there) */
    sb(b, "mov.b32 %%eu2, %s;\nmov.b32 %%eu3, 0x7FC00000;\n", x);
    for (int i = 0; i < TR_EXPF_N_EXCEPTIONS && !(g_mut & MUT_EXPF); i++)
        sb(b, "setp.eq.b32 %%ep3, %%eu2, 0x%08" PRIX32 ";\n@%%ep3 mov.b32 %%eu3, 0x%08" PRIX32 ";\n",
           tr_expf_exceptions[i][0], tr_expf_exceptions[i][1]);
    sb(b, "mov.b32 %s, %%eu3;\nbra %s_done;\n", y, L);
    sb(b, "%s_nan:\nadd.rn.f32 %s, %s, %s;\nbra %s_done;\n", L, y, x, x, L);
    sb(b, "%s_ovf:\nmov.f32 %s, 0f7F800000;\nbra %s_done;\n", L, y, L);
    sb(b, "%s_unf:\nmov.f32 %s, 0f00000000;\n", L, y);
    sb(b, "%s_done:\n}\n", L);
}

static void emit_header(sbuf *b) {
    sb(b, ".version 7.0\n.target sm_80\n.address_size 64\n\n.global .align 8 .f64 tr_pow2[64] = {");
    for (int i = 0; i < 64; i++) sb(b, "%s0d%016" PRIX64, i ? ", " : "", dbits(tr_expf_pow2[i]));
    sb(b, "};\n\n");
}

/* attn_scores(K, in, V, S, MB, n, scale): grid (ceil(n/128), 16), 128 threads */
static void emit_scores(sbuf *b) {
    sb(b, ".visible .entry attn_scores(.param .u64 pK, .param .u64 pIn, .param .u64 pV, .param .u64 pS,\n"
          "    .param .u64 pMB, .param .u32 pN, .param .f32 pScale)\n{\n"
          ".reg .pred %%p<4>;\n.reg .b32 %%r<24>;\n.reg .b64 %%rd<24>;\n"
          ".reg .f32 %%sc, %%s, %%t, %%q<4>, %%k<4>, %%m<4>, %%l<16>, %%w<16>;\n"
          ".shared .align 16 .f32 sq[128];\n.shared .align 16 .f32 swm[4];\n"
          "ld.param.u64 %%rd1, [pK];\nld.param.u64 %%rd2, [pIn];\nld.param.u64 %%rd3, [pV];\n"
          "ld.param.u64 %%rd4, [pS];\nld.param.u64 %%rd5, [pMB];\nld.param.u32 %%r1, [pN];\n"
          "ld.param.f32 %%sc, [pScale];\n"
          "mov.u32 %%r2, %%tid.x;\nmov.u32 %%r3, %%ctaid.x;\nmov.u32 %%r4, %%ctaid.y;\n");
    /* q of head h into shared memory */
    sb(b, "shl.b32 %%r5, %%r4, 7;\nadd.u32 %%r6, %%r5, %%r2;\nmul.wide.u32 %%rd6, %%r6, 4;\n"
          "add.s64 %%rd7, %%rd2, %%rd6;\nld.global.f32 %%t, [%%rd7];\n"
          "shl.b32 %%r7, %%r2, 2;\nmov.u32 %%r8, sq;\nadd.u32 %%r8, %%r8, %%r7;\nst.shared.f32 [%%r8], %%t;\n");
    /* the block holding position n-1 appends the new key and value of head h to the cache */
    sb(b, "sub.u32 %%r9, %%r1, 1;\nshr.u32 %%r10, %%r9, 7;\nsetp.ne.u32 %%p1, %%r3, %%r10;\n@%%p1 bra NOAPPEND;\n"
          "ld.global.f32 %%t, [%%rd7+%d];\n"
          "mul.wide.u32 %%rd8, %%r6, %d;\nadd.s64 %%rd8, %%rd1, %%rd8;\nmul.wide.u32 %%rd9, %%r9, 4;\n"
          "add.s64 %%rd8, %%rd8, %%rd9;\nst.global.f32 [%%rd8], %%t;\n"
          "ld.global.f32 %%t, [%%rd7+%d];\n"
          "mad.lo.u32 %%r11, %%r4, %d, %%r9;\nshl.b32 %%r11, %%r11, 7;\nadd.u32 %%r11, %%r11, %%r2;\n"
          "mul.wide.u32 %%rd9, %%r11, 4;\nadd.s64 %%rd9, %%rd3, %%rd9;\nst.global.f32 [%%rd9], %%t;\n"
          "NOAPPEND:\nbar.sync 0;\n",
       QKV * 4, CAP * 4, 2 * QKV * 4, CAP);
    /* the dot of position t: keys at K[h][i][t], or in.k[h][i] for t == n-1 */
    sb(b, "shl.b32 %%r12, %%r3, 7;\nadd.u32 %%r12, %%r12, %%r2;\nmov.f32 %%s, 0fFF800000;\n"
          "setp.ge.u32 %%p2, %%r12, %%r1;\n@%%p2 bra NODOT;\n"
          "setp.eq.u32 %%p3, %%r12, %%r9;\n"
          "mul.wide.u32 %%rd10, %%r5, %d;\nadd.s64 %%rd10, %%rd1, %%rd10;\nmul.wide.u32 %%rd11, %%r12, 4;\n"
          "add.s64 %%rd10, %%rd10, %%rd11;\n"
          "mul.wide.u32 %%rd12, %%r5, 4;\nadd.s64 %%rd12, %%rd2, %%rd12;\nadd.s64 %%rd12, %%rd12, %d;\n"
          "selp.b64 %%rd13, %%rd12, %%rd10, %%p3;\nselp.b64 %%rd14, 4, %d, %%p3;\n",
       CAP * 4, QKV * 4, CAP * 4);
    for (int l = 0; l < 16; l++) sb(b, "mov.f32 %%l%d, 0f00000000;\n", l);
    for (int j = 0; j < HEAD_DIM / 4; j++) {
        sb(b, "ld.shared.v4.f32 {%%q0, %%q1, %%q2, %%q3}, [sq+%d];\n", 16 * j);
        for (int x = 0; x < 4; x++) sb(b, "ld.global.f32 %%k%d, [%%rd13];\nadd.s64 %%rd13, %%rd13, %%rd14;\n", x);
        for (int x = 0; x < 4; x++) {
            int lane = (4 * j + x) % TR_LANES;
            sb(b, "mul.rn.f32 %%m%d, %%q%d, %%k%d;\nadd.rn.f32 %%l%d, %%l%d, %%m%d;\n", x, x, x, lane, lane, x);
        }
    }
    /* tr_lane_combine: ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), the same for 8..15, halves last */
    for (int half = 0; half < 2; half++) {
        int o = 8 * half, w = 7 * half;
        for (int x = 0; x < 4; x++) sb(b, "add.rn.f32 %%w%d, %%l%d, %%l%d;\n", w + x, o + 2 * x, o + 2 * x + 1);
        if (g_mut & MUT_TREE) { /* mutation: (s01+s45) + (s23+s67) */
            sb(b, "add.rn.f32 %%w%d, %%w%d, %%w%d;\n", w + 4, w + 0, w + 2);
            sb(b, "add.rn.f32 %%w%d, %%w%d, %%w%d;\n", w + 5, w + 1, w + 3);
        } else {
            sb(b, "add.rn.f32 %%w%d, %%w%d, %%w%d;\n", w + 4, w + 0, w + 1);
            sb(b, "add.rn.f32 %%w%d, %%w%d, %%w%d;\n", w + 5, w + 2, w + 3);
        }
        sb(b, "add.rn.f32 %%w%d, %%w%d, %%w%d;\n", w + 6, w + 4, w + 5);
    }
    sb(b, "add.rn.f32 %%w14, %%w6, %%w13;\nmul.rn.f32 %%s, %%w14, %%sc;\n"
          "mad.lo.u32 %%r13, %%r4, %d, %%r12;\nmul.wide.u32 %%rd15, %%r13, 4;\nadd.s64 %%rd15, %%rd4, %%rd15;\n"
          "st.global.f32 [%%rd15], %%s;\nNODOT:\n",
       CAP);
    /* the block's maximum (positions past n count as -inf) */
    for (int off = 16; off >= 1; off /= 2)
        sb(b, "shfl.sync.bfly.b32 %%t, %%s, %d, 31, 0xffffffff;\nmax.f32 %%s, %%s, %%t;\n", off);
    sb(b, "and.b32 %%r14, %%r2, 31;\nshr.u32 %%r15, %%r2, 5;\nsetp.ne.u32 %%p1, %%r14, 0;\n@%%p1 bra NOWM;\n"
          "mov.u32 %%r16, swm;\nshl.b32 %%r17, %%r15, 2;\nadd.u32 %%r16, %%r16, %%r17;\nst.shared.f32 [%%r16], %%s;\n"
          "NOWM:\nbar.sync 0;\nsetp.ne.u32 %%p1, %%r2, 0;\n@%%p1 bra DONE;\n"
          "ld.shared.v4.f32 {%%q0, %%q1, %%q2, %%q3}, [swm];\n"
          "max.f32 %%q0, %%q0, %%q1;\nmax.f32 %%q2, %%q2, %%q3;\nmax.f32 %%q0, %%q0, %%q2;\n"
          "mad.lo.u32 %%r18, %%r4, %d, %%r3;\nmul.wide.u32 %%rd16, %%r18, 4;\nadd.s64 %%rd16, %%rd5, %%rd16;\n"
          "st.global.f32 [%%rd16], %%q0;\nDONE:\nret;\n}\n\n",
       MB_PER_HEAD);
}

/* attn_exp(S, MB, E, n): grid (ceil(n/128), 16), 128 threads */
static void emit_exp(sbuf *b) {
    sb(b, ".visible .entry attn_exp(.param .u64 pS, .param .u64 pMB, .param .u64 pE, .param .u32 pN)\n{\n"
          ".reg .pred %%p<4>;\n.reg .b32 %%r<16>;\n.reg .b64 %%rd<12>;\n.reg .f32 %%m, %%t, %%x, %%e;\n"
          "ld.param.u64 %%rd1, [pS];\nld.param.u64 %%rd2, [pMB];\nld.param.u64 %%rd3, [pE];\nld.param.u32 %%r1, [pN];\n"
          "mov.u32 %%r2, %%tid.x;\nmov.u32 %%r3, %%ctaid.x;\nmov.u32 %%r4, %%ctaid.y;\n"
          "and.b32 %%r5, %%r2, 31;\nadd.u32 %%r6, %%r1, 127;\nshr.u32 %%r6, %%r6, 7;\n"
          "mov.f32 %%m, 0fFF800000;\nsetp.lt.u32 %%p1, %%r5, %%r6;\n"
          "mad.lo.u32 %%r7, %%r4, %d, %%r5;\nmul.wide.u32 %%rd4, %%r7, 4;\nadd.s64 %%rd4, %%rd2, %%rd4;\n"
          "@%%p1 ld.global.f32 %%m, [%%rd4];\n",
       MB_PER_HEAD);
    for (int off = 16; off >= 1; off /= 2)
        sb(b, "shfl.sync.bfly.b32 %%t, %%m, %d, 31, 0xffffffff;\nmax.f32 %%m, %%m, %%t;\n", off);
    sb(b, "shl.b32 %%r8, %%r3, 7;\nadd.u32 %%r8, %%r8, %%r2;\nsetp.ge.u32 %%p2, %%r8, %%r1;\n@%%p2 bra DONE;\n"
          "mad.lo.u32 %%r9, %%r4, %d, %%r8;\nmul.wide.u32 %%rd5, %%r9, 4;\nadd.s64 %%rd6, %%rd1, %%rd5;\n"
          "ld.global.f32 %%x, [%%rd6];\nsub.rn.f32 %%x, %%x, %%m;\n",
       CAP);
    emit_expf(b, "%x", "%e", "EXP");
    sb(b, "add.s64 %%rd7, %%rd3, %%rd5;\nst.global.f32 [%%rd7], %%e;\nDONE:\nret;\n}\n\n");
}

/* cp.async of tile (register kreg) into stage (register breg) of attn_v, then commit */
static void emit_v_issue(sbuf *b, const char *kreg, const char *breg, int id) {
    sb(b, "setp.lt.u32 %%p1, %s, %%r7;\n@!%%p1 bra ISSUE%d;\n"
          "shl.b32 %%r20, %s, %d;\nadd.u32 %%r20, %%r20, %%r13;\n"
          "mul.lo.u32 %%r21, %s, %d;\nadd.u32 %%r21, %%r21, %%r16;\n",
       kreg, id, kreg, V_TILE_LOG2, breg, V_TILE * V_CHUNK * 4);
    for (int j = 0; j < V_TILE / 16; j++)
        sb(b, "add.u32 %%r22, %%r20, %d;\nsetp.lt.u32 %%p2, %%r22, %%r1;\nmul.wide.u32 %%rd10, %%r22, %d;\n"
              "add.s64 %%rd10, %%rd6, %%rd10;\n@%%p2 cp.async.cg.shared.global [%%r21+%d], [%%rd10], 16;\n",
           16 * j, HEAD_DIM * 4, 16 * j * V_CHUNK * 4);
    sb(b, "ISSUE%d:\ncp.async.commit_group;\n", id);
}

/* attn_v(V, E, out, n): grid (4, 16), 128 threads */
static void emit_v(sbuf *b) {
    const char *rn = (g_mut & MUT_FMA) ? "" : ".rn";
    sb(b, ".visible .entry attn_v(.param .u64 pV, .param .u64 pE, .param .u64 pOut, .param .u32 pN)\n{\n"
          ".reg .pred %%p<8>;\n.reg .b32 %%r<48>;\n.reg .b64 %%rd<16>;\n"
          ".reg .f32 %%o, %%acc, %%t, %%sum, %%x, %%a<8>, %%v<8>, %%m<8>;\n"
          ".shared .align 16 .f32 sa[%d];\n.shared .align 16 .b8 sv[%d];\n.shared .align 4 .f32 ssum;\n"
          "ld.param.u64 %%rd1, [pV];\nld.param.u64 %%rd2, [pE];\nld.param.u64 %%rd3, [pOut];\nld.param.u32 %%r1, [pN];\n"
          "mov.u32 %%r2, %%tid.x;\nmov.u32 %%r3, %%ctaid.x;\nmov.u32 %%r4, %%ctaid.y;\n"
          "and.b32 %%r5, %%r2, 31;\nshr.u32 %%r6, %%r2, 5;\n"
          "add.u32 %%r7, %%r1, %d;\nshr.u32 %%r7, %%r7, %d;\n",
       CAP, V_STAGES * V_TILE * V_CHUNK * 4, V_TILE - 1, V_TILE_LOG2);
    /* group 0: the head's e_t into sa, 16 bytes per thread per round */
    sb(b, "mul.wide.u32 %%rd4, %%r4, %d;\nadd.s64 %%rd4, %%rd2, %%rd4;\nshl.b32 %%r8, %%r2, 4;\n"
          "mov.u32 %%r9, sa;\nadd.u32 %%r9, %%r9, %%r8;\ncvt.u64.u32 %%rd5, %%r8;\nadd.s64 %%rd5, %%rd4, %%rd5;\n"
          "shl.b32 %%r10, %%r2, 2;\n",
       CAP * 4);
    for (int r = 0; r < CAP / 512; r++)
        sb(b, "add.u32 %%r11, %%r10, %d;\nsetp.lt.u32 %%p1, %%r11, %%r1;\n"
              "@%%p1 cp.async.cg.shared.global [%%r9+%d], [%%rd5+%d], 16;\n",
           512 * r, 2048 * r, 2048 * r);
    sb(b, "cp.async.commit_group;\n");
    /* the copy geometry of a value tile: 16 bytes (seg) of a 128-byte row piece (row), 16 rows per round */
    sb(b, "and.b32 %%r12, %%r2, 7;\nshr.u32 %%r13, %%r2, 3;\n"
          "mul.wide.u32 %%rd6, %%r4, %d;\nadd.s64 %%rd6, %%rd1, %%rd6;\n"
          "shl.b32 %%r14, %%r3, %d;\nshl.b32 %%r15, %%r12, 4;\nadd.u32 %%r14, %%r14, %%r15;\n"
          "cvt.u64.u32 %%rd7, %%r14;\nadd.s64 %%rd6, %%rd6, %%rd7;\n"
          "mov.u32 %%r16, sv;\nshl.b32 %%r17, %%r13, %d;\nadd.u32 %%r16, %%r16, %%r17;\nadd.u32 %%r16, %%r16, %%r15;\n",
       CAP * HEAD_DIM * 4, 7 /* chunk * 32 dims * 4 bytes = chunk << 7 */, 7 /* row * 128 bytes */);
    /* groups 1 .. V_STAGES-1: the first tiles, in flight while the sum is taken */
    for (int s = 0; s < V_STAGES - 1; s++) {
        sb(b, "mov.u32 %%r30, %d;\n", s);
        emit_v_issue(b, "%r30", "%r30", s);
    }
    /* sum = the 16-lane sum of e_t (threads 0..15 of warp 0), then the tree across lanes */
    sb(b, "cp.async.wait_group %d;\nbar.sync 0;\nmov.f32 %%acc, 0f00000000;\n"
          "setp.ge.u32 %%p2, %%r2, 16;\n@%%p2 bra SUMSKIP;\n"
          "mov.u32 %%r18, %%r2;\nmov.u32 %%r19, sa;\nshl.b32 %%r20, %%r2, 2;\nadd.u32 %%r19, %%r19, %%r20;\n"
          "SUM8:\nadd.u32 %%r22, %%r18, 112;\nsetp.ge.u32 %%p3, %%r22, %%r1;\n@%%p3 bra SUM1;\n",
       V_STAGES - 1);
    for (int x = 0; x < 8; x++) sb(b, "ld.shared.f32 %%a%d, [%%r19+%d];\n", x, 64 * x);
    for (int x = 0; x < 8; x++) sb(b, "add.rn.f32 %%acc, %%acc, %%a%d;\n", x);
    sb(b, "add.u32 %%r18, %%r18, 128;\nadd.u32 %%r19, %%r19, 512;\nbra SUM8;\n"
          "SUM1:\nsetp.ge.u32 %%p3, %%r18, %%r1;\n@%%p3 bra SUMSKIP;\n"
          "ld.shared.f32 %%a0, [%%r19];\nadd.rn.f32 %%acc, %%acc, %%a0;\n"
          "add.u32 %%r18, %%r18, 16;\nadd.u32 %%r19, %%r19, 64;\nbra SUM1;\n"
          "SUMSKIP:\nsetp.ne.u32 %%p4, %%r6, 0;\n@%%p4 bra TREESKIP;\n");
    /* lane i adds lane i+1, then i+2, i+4, i+8: lane 0 ends with tr_lane_combine's tree */
    for (int k = 0; k < 4; k++) {
        int off = (g_mut & MUT_SUMTREE) ? 8 >> k : 1 << k;
        sb(b, "shfl.sync.down.b32 %%t, %%acc, %d, 31, 0xffffffff;\nadd.rn.f32 %%acc, %%acc, %%t;\n", off);
    }
    sb(b, "setp.ne.u32 %%p5, %%r2, 0;\n@%%p5 bra TREESKIP;\nst.shared.f32 [ssum], %%acc;\n"
          "TREESKIP:\nbar.sync 0;\nld.shared.f32 %%sum, [ssum];\n");
    /* a_t = e_t / sum, in place */
    sb(b, "mov.u32 %%r23, %%r2;\nmov.u32 %%r24, sa;\nshl.b32 %%r25, %%r2, 2;\nadd.u32 %%r24, %%r24, %%r25;\n"
          "DIV:\nsetp.ge.u32 %%p3, %%r23, %%r1;\n@%%p3 bra DIVDONE;\n"
          "ld.shared.f32 %%x, [%%r24];\ndiv.rn.f32 %%x, %%x, %%sum;\nst.shared.f32 [%%r24], %%x;\n"
          "add.u32 %%r23, %%r23, 128;\nadd.u32 %%r24, %%r24, 512;\nbra DIV;\nDIVDONE:\n");
    /* the tiles: wait for tile i, issue tile i + V_STAGES - 1 into the stage tile i-1 left, warp 0 adds tile i */
    sb(b, "mov.u32 %%r26, 0;\nmov.u32 %%r27, 0;\nmov.u32 %%r28, %d;\nmov.u32 %%r29, %d;\n"
          "mov.f32 %%o, 0f%s;\nsetp.ne.u32 %%p6, %%r6, 0;\n"
          "LOOP:\nsetp.ge.u32 %%p3, %%r26, %%r7;\n@%%p3 bra LOOPEND;\ncp.async.wait_group %d;\nbar.sync 0;\n",
       V_STAGES - 1, V_STAGES - 1, (g_mut & MUT_ZERO) ? "80000000" : "00000000", V_STAGES - 2);
    emit_v_issue(b, "%r28", "%r29", 99);
    sb(b, "add.u32 %%r28, %%r28, 1;\nadd.u32 %%r29, %%r29, 1;\nsetp.eq.u32 %%p7, %%r29, %d;\n@%%p7 mov.u32 %%r29, 0;\n"
          "@%%p6 bra CHAINDONE;\n"
          "shl.b32 %%r32, %%r26, %d;\nmov.u32 %%r33, sa;\nshl.b32 %%r34, %%r32, 2;\nadd.u32 %%r33, %%r33, %%r34;\n"
          "mov.u32 %%r35, sv;\nmul.lo.u32 %%r36, %%r27, %d;\nadd.u32 %%r35, %%r35, %%r36;\nshl.b32 %%r37, %%r5, 2;\n"
          "add.u32 %%r35, %%r35, %%r37;\nadd.u32 %%r38, %%r32, %d;\nsetp.gt.u32 %%p3, %%r38, %%r1;\n@%%p3 bra PARTIAL;\n",
       V_STAGES, V_TILE_LOG2, V_TILE * V_CHUNK * 4, V_TILE);
    for (int j0 = 0; j0 < V_TILE; j0 += 8) {
        for (int x = 0; x < 8; x++)
            sb(b, "ld.shared.f32 %%a%d, [%%r33+%d];\nld.shared.f32 %%v%d, [%%r35+%d];\n", x, 4 * (j0 + x), x,
               V_CHUNK * 4 * (j0 + x));
        for (int x = 0; x < 8; x++) sb(b, "mul%s.f32 %%m%d, %%a%d, %%v%d;\nadd%s.f32 %%o, %%o, %%m%d;\n", rn, x, x, x, rn, x);
    }
    sb(b, "bra CHAINDONE;\nPARTIAL:\nsub.u32 %%r39, %%r1, %%r32;\n"
          "PLOOP:\nld.shared.f32 %%a0, [%%r33];\nld.shared.f32 %%v0, [%%r35];\n"
          "mul%s.f32 %%m0, %%a0, %%v0;\nadd%s.f32 %%o, %%o, %%m0;\n"
          "add.u32 %%r33, %%r33, 4;\nadd.u32 %%r35, %%r35, %d;\nsub.u32 %%r39, %%r39, 1;\n"
          "setp.ne.u32 %%p3, %%r39, 0;\n@%%p3 bra PLOOP;\n"
          "CHAINDONE:\nadd.u32 %%r26, %%r26, 1;\nadd.u32 %%r27, %%r27, 1;\nsetp.eq.u32 %%p7, %%r27, %d;\n"
          "@%%p7 mov.u32 %%r27, 0;\nbra LOOP;\n"
          "LOOPEND:\n@%%p6 bra DONE;\n"
          "shl.b32 %%r40, %%r4, 7;\nshl.b32 %%r41, %%r3, 5;\nadd.u32 %%r40, %%r40, %%r41;\nadd.u32 %%r40, %%r40, %%r5;\n"
          "mul.wide.u32 %%rd8, %%r40, 4;\nadd.s64 %%rd8, %%rd3, %%rd8;\nst.global.f32 [%%rd8], %%o;\n"
          "DONE:\nret;\n}\n\n",
       rn, rn, V_CHUNK * 4, V_STAGES);
}

/* stream_read(base, row stride, 16-byte units per row, sink): grid (x, rows), 128 threads, 4 units per
 * thread: every byte of the rows read once, nothing computed */
static void emit_misc(sbuf *b) {
    sb(b, ".visible .entry stream_read(.param .u64 pBase, .param .u64 pStride, .param .u32 pLen, .param .u64 pSink)\n{\n"
          ".reg .pred %%p<4>;\n.reg .b32 %%r<32>;\n.reg .b64 %%rd<12>;\n"
          "ld.param.u64 %%rd1, [pBase];\nld.param.u64 %%rd2, [pStride];\nld.param.u32 %%r1, [pLen];\n"
          "ld.param.u64 %%rd3, [pSink];\nmov.u32 %%r2, %%ctaid.y;\ncvt.u64.u32 %%rd4, %%r2;\n"
          "mul.lo.u64 %%rd4, %%rd4, %%rd2;\nadd.s64 %%rd1, %%rd1, %%rd4;\n"
          "mov.u32 %%r3, %%ctaid.x;\nmov.u32 %%r4, %%tid.x;\nmov.u32 %%r5, %%nctaid.x;\n"
          "shl.b32 %%r6, %%r5, 7;\nmad.lo.u32 %%r7, %%r3, 128, %%r4;\nmov.u32 %%r8, 0;\n");
    for (int u = 0; u < 4; u++)
        sb(b, "mov.u32 %%r%d, 0;\nmov.u32 %%r%d, 0;\nmov.u32 %%r%d, 0;\nmov.u32 %%r%d, 0;\n"
              "setp.lt.u32 %%p1, %%r7, %%r1;\nmul.wide.u32 %%rd5, %%r7, 16;\nadd.s64 %%rd5, %%rd1, %%rd5;\n"
              "@%%p1 ld.global.cs.v4.u32 {%%r%d, %%r%d, %%r%d, %%r%d}, [%%rd5];\nadd.u32 %%r7, %%r7, %%r6;\n",
           10 + 4 * u, 11 + 4 * u, 12 + 4 * u, 13 + 4 * u, 10 + 4 * u, 11 + 4 * u, 12 + 4 * u, 13 + 4 * u);
    for (int i = 10; i < 26; i++) sb(b, "xor.b32 %%r8, %%r8, %%r%d;\n", i);
    sb(b, "setp.ne.u32 %%p2, %%r8, 0x9E3779B9;\n@%%p2 bra DONE;\nst.global.u32 [%%rd3], %%r8;\nDONE:\nret;\n}\n\n");
    /* keep_warm(ns, nap): one warp stays for ns nanoseconds, reading the global timer, with a nanosleep
     * of nap ns between two reads (0: a plain spin) */
    sb(b, ".visible .entry keep_warm(.param .u64 pNs, .param .u32 pNap)\n{\n.reg .pred %%p<2>;\n.reg .b32 %%r1;\n"
          ".reg .b64 %%rd<4>;\nld.param.u64 %%rd1, [pNs];\nld.param.u32 %%r1, [pNap];\nsetp.ne.u32 %%p1, %%r1, 0;\n"
          "mov.u64 %%rd2, %%globaltimer;\nadd.u64 %%rd1, %%rd1, %%rd2;\n"
          "SPIN:\n@%%p1 nanosleep.u32 %%r1;\nmov.u64 %%rd3, %%globaltimer;\nsetp.lt.u64 %%p0, %%rd3, %%rd1;\n"
          "@%%p0 bra SPIN;\nret;\n}\n\n");
    /* empty(): the floor of one launch */
    sb(b, ".visible .entry empty()\n{\nret;\n}\n\n");
    /* expf_sweep(first bits, out): out[i] = tr_expf(bits first + i), 256 threads */
    sb(b, ".visible .entry expf_sweep(.param .u32 pFirst, .param .u64 pOut)\n{\n"
          ".reg .b32 %%r<8>;\n.reg .b64 %%rd<4>;\n.reg .f32 %%x, %%y;\n"
          "ld.param.u32 %%r1, [pFirst];\nld.param.u64 %%rd1, [pOut];\nmov.u32 %%r2, %%ctaid.x;\nmov.u32 %%r3, %%tid.x;\n"
          "mad.lo.u32 %%r4, %%r2, 256, %%r3;\nadd.u32 %%r5, %%r1, %%r4;\nmov.b32 %%x, %%r5;\n");
    emit_expf(b, "%x", "%y", "SWEEP");
    sb(b, "mul.wide.u32 %%rd2, %%r4, 4;\nadd.s64 %%rd2, %%rd1, %%rd2;\nst.global.f32 [%%rd2], %%y;\nret;\n}\n");
}

/* ---- the device ------------------------------------------------------------------------------ */

typedef struct {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction f_scores, f_exp, f_v, f_read, f_warm, f_expf, f_empty;
    CUstream st, st2;
    CUevent ev[4];
    CUdeviceptr in, s, mb, e, out, sink;
    float *h_in, *h_out;         /* pinned */
    CUdeviceptr in_map, out_map; /* the same pinned buffers as the kernels see them (zero copy) */
    int zero_copy;               /* round_trip: the kernels read h_in and write h_out over PCIe */
    float scale;
    char name[128];
    int driver, sms, cc_major, cc_minor;
} gpu_t;

static void gpu_open(gpu_t *g) {
    memset(g, 0, sizeof *g);
    if (cu_open() != 0) exit(2);
    CU(cu.Init(0));
    CU(cu.DriverGetVersion(&g->driver));
    CU(cu.DeviceGet(&g->dev, 0));
    CU(cu.DeviceGetName(g->name, (int)sizeof g->name, g->dev));
    CU(cu.DeviceGetAttribute(&g->sms, ATTR_SMS, g->dev));
    CU(cu.DeviceGetAttribute(&g->cc_major, ATTR_CC_MAJOR, g->dev));
    CU(cu.DeviceGetAttribute(&g->cc_minor, ATTR_CC_MINOR, g->dev));
    if (g->cc_major < 8) {
        fprintf(stderr, "%s is sm_%d%d: the kernels use cp.async, sm_80 or later\n", g->name, g->cc_major, g->cc_minor);
        exit(2);
    }
    /* spin while waiting: the lowest latency for a synchronization the decode waits on */
    CU(cu.CtxCreate(&g->ctx, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, g->dev));

    sbuf b = {malloc(1 << 16), 0, 1 << 16};
    if (b.p == NULL) exit(2);
    emit_header(&b);
    emit_scores(&b);
    emit_exp(&b);
    emit_v(&b);
    emit_misc(&b);
    if (getenv("TR_GPU_PTX") != NULL) { /* TR_GPU_PTX=<file>: the PTX as sent to the driver */
        FILE *f = fopen(getenv("TR_GPU_PTX"), "wb");
        if (f != NULL) {
            fwrite(b.p, 1, b.len, f);
            fclose(f);
        }
    }
    static char err_log[1 << 16], info_log[1 << 16];
    int opt[4] = {JIT_ERROR_LOG, JIT_ERROR_LOG_SIZE, JIT_INFO_LOG, JIT_INFO_LOG_SIZE};
    void *val[4] = {err_log, (void *)(uintptr_t)sizeof err_log, info_log, (void *)(uintptr_t)sizeof info_log};
    CUresult r = cu.ModuleLoadDataEx(&g->mod, b.p, 4, opt, val);
    if (r != 0) {
        fprintf(stderr, "PTX JIT failed (%d):\n%s\n", r, err_log);
        exit(2);
    }
    free(b.p);
    CU(cu.ModuleGetFunction(&g->f_scores, g->mod, "attn_scores"));
    CU(cu.ModuleGetFunction(&g->f_exp, g->mod, "attn_exp"));
    CU(cu.ModuleGetFunction(&g->f_v, g->mod, "attn_v"));
    CU(cu.ModuleGetFunction(&g->f_read, g->mod, "stream_read"));
    CU(cu.ModuleGetFunction(&g->f_warm, g->mod, "keep_warm"));
    CU(cu.ModuleGetFunction(&g->f_expf, g->mod, "expf_sweep"));
    CU(cu.ModuleGetFunction(&g->f_empty, g->mod, "empty"));
    CU(cu.StreamCreate(&g->st, CU_STREAM_NON_BLOCKING));
    CU(cu.StreamCreate(&g->st2, CU_STREAM_NON_BLOCKING));
    for (int i = 0; i < 4; i++) CU(cu.EventCreate(&g->ev[i], 0));
    CU(cu.MemAlloc(&g->in, IN_FLOATS * sizeof(float)));
    CU(cu.MemAlloc(&g->s, (size_t)N_HEAD * CAP * sizeof(float)));
    CU(cu.MemAlloc(&g->mb, (size_t)N_HEAD * MB_PER_HEAD * sizeof(float)));
    CU(cu.MemAlloc(&g->e, (size_t)N_HEAD * CAP * sizeof(float)));
    CU(cu.MemAlloc(&g->out, QKV * sizeof(float)));
    CU(cu.MemAlloc(&g->sink, 64));
    void *p;
    CU(cu.MemAllocHost(&p, IN_FLOATS * sizeof(float)));
    g->h_in = p;
    CU(cu.MemAllocHost(&p, QKV * sizeof(float)));
    g->h_out = p;
    memset(g->h_in, 0, IN_FLOATS * sizeof(float));
    CU(cu.MemHostGetDevicePointer(&g->in_map, g->h_in, 0));
    CU(cu.MemHostGetDevicePointer(&g->out_map, g->h_out, 0));
    g->scale = 1.0f / sqrtf((float)HEAD_DIM); /* as src/models/olmoe.c */
}

static void gpu_close(gpu_t *g) {
    CU(cu.CtxSynchronize());
    CU(cu.MemFreeHost(g->h_in));
    CU(cu.MemFreeHost(g->h_out));
    CU(cu.MemFree(g->in));
    CU(cu.MemFree(g->s));
    CU(cu.MemFree(g->mb));
    CU(cu.MemFree(g->e));
    CU(cu.MemFree(g->out));
    CU(cu.MemFree(g->sink));
    for (int i = 0; i < 4; i++) CU(cu.EventDestroy(g->ev[i]));
    CU(cu.StreamDestroy(g->st));
    CU(cu.StreamDestroy(g->st2));
    CU(cu.ModuleUnload(g->mod));
    CU(cu.CtxDestroy(g->ctx));
}

/* the three kernels of one layer on the stream, reading q and the new key and value at `in` and
 * writing the output at `out`; with events around each when ev != 0 */
static void enqueue_io(gpu_t *g, CUdeviceptr K, CUdeviceptr V, int n, int ev, CUdeviceptr in, CUdeviceptr out) {
    unsigned nb = (unsigned)((n + 127) / 128);
    void *a_s[] = {&K, &in, &V, &g->s, &g->mb, &n, &g->scale};
    void *a_e[] = {&g->s, &g->mb, &g->e, &n};
    void *a_v[] = {&V, &g->e, &out, &n};
    if (ev) CU(cu.EventRecord(g->ev[0], g->st));
    CU(cu.LaunchKernel(g->f_scores, nb, N_HEAD, 1, 128, 1, 1, 0, g->st, a_s, NULL));
    if (ev) CU(cu.EventRecord(g->ev[1], g->st));
    CU(cu.LaunchKernel(g->f_exp, nb, N_HEAD, 1, 128, 1, 1, 0, g->st, a_e, NULL));
    if (ev) CU(cu.EventRecord(g->ev[2], g->st));
    CU(cu.LaunchKernel(g->f_v, HEAD_DIM / V_CHUNK, N_HEAD, 1, 128, 1, 1, 0, g->st, a_v, NULL));
    if (ev) CU(cu.EventRecord(g->ev[3], g->st));
}

static void enqueue_attention(gpu_t *g, CUdeviceptr K, CUdeviceptr V, int n, int ev) {
    enqueue_io(g, K, V, n, ev, g->in, g->out);
}

/* what the engine pays per layer: q and the new key and value in, the kernels, out back, wait. With
 * zero_copy the kernels read the pinned input and write the pinned output themselves, over PCIe:
 * no copy on the copy engine */
static void round_trip(gpu_t *g, CUdeviceptr K, CUdeviceptr V, int n, int ev) {
    if (g->zero_copy) {
        enqueue_io(g, K, V, n, ev, g->in_map, g->out_map);
    } else {
        CU(cu.MemcpyHtoDAsync(g->in, g->h_in, IN_FLOATS * sizeof(float), g->st));
        enqueue_attention(g, K, V, n, ev);
        CU(cu.MemcpyDtoHAsync(g->h_out, g->out, QKV * sizeof(float), g->st));
    }
    CU(cu.StreamSynchronize(g->st));
}

static double ev_us(gpu_t *g, int a, int b) {
    float ms = 0.0f;
    CU(cu.EventElapsedTime(&ms, g->ev[a], g->ev[b]));
    return 1000.0 * (double)ms;
}

/* ---- host side ------------------------------------------------------------------------------- */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (p == NULL) {
        fprintf(stderr, "out of memory (%.0f MiB)\n", (double)n / 1048576.0);
        exit(2);
    }
    return p;
}

static size_t read_file(const char *path, void **out) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = xmalloc(len > 0 ? (size_t)len : 1);
    if (len < 0 || fread(p, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fclose(f);
    *out = p;
    return (size_t)len;
}

/* the VRAM layouts: keys [h][d][CAP], values [h][CAP][d] */
static void put_k(float *hk, int h, int t, const float *row) {
    for (int d = 0; d < HEAD_DIM; d++) hk[((size_t)h * HEAD_DIM + (size_t)d) * CAP + (size_t)t] = row[d];
}

static void put_v(float *hv, int h, int t, const float *row) {
    memcpy(hv + ((size_t)h * CAP + (size_t)t) * HEAD_DIM, row, HEAD_DIM * sizeof(float));
}

static void fill_nan(float *x, size_t n) {
    const float nan = from_bits(0x7FC00000u);
    for (size_t i = 0; i < n; i++) x[i] = nan;
}

/* One layer of a probe run: per head the keys and values (n_rows x 128, position-major) and the
 * queries (n_rec records). */
typedef struct {
    int n_rows, n_rec;
    int64_t npos[MAX_REC];
    float *k[N_HEAD], *v[N_HEAD];
    float *q; /* [head][rec][128] */
} probe_layer;

static void probe_load(const char *dir, int layer, probe_layer *pl) {
    char path[1024];
    for (int h = 0; h < N_HEAD; h++) {
        void *p;
        snprintf(path, sizeof path, "%s/k_L%02d_H%02d.bin", dir, layer, h);
        size_t nk = read_file(path, &p);
        pl->k[h] = p;
        snprintf(path, sizeof path, "%s/v_L%02d_H%02d.bin", dir, layer, h);
        size_t nv = read_file(path, &p);
        pl->v[h] = p;
        int rows = (int)(nk / (HEAD_DIM * sizeof(float)));
        if (nk != nv || nk % (HEAD_DIM * sizeof(float)) != 0 || rows > CAP || (h > 0 && rows != pl->n_rows)) {
            fprintf(stderr, "%s: keys and values of layer %d head %d do not match\n", dir, layer, h);
            exit(2);
        }
        pl->n_rows = rows;
        snprintf(path, sizeof path, "%s/q_L%02d_H%02d.bin", dir, layer, h);
        size_t nq = read_file(path, &p);
        const size_t rec = sizeof(int64_t) + HEAD_DIM * sizeof(float);
        int n_rec = (int)(nq / rec);
        if (nq % rec != 0 || n_rec > MAX_REC || n_rec < 1 || (h > 0 && n_rec != pl->n_rec)) {
            fprintf(stderr, "%s: bad query file %s\n", dir, path);
            exit(2);
        }
        if (h == 0) {
            pl->n_rec = n_rec;
            pl->q = xmalloc((size_t)N_HEAD * (size_t)n_rec * HEAD_DIM * sizeof(float));
        }
        for (int r = 0; r < n_rec; r++) {
            int64_t np;
            memcpy(&np, (char *)p + (size_t)r * rec, sizeof np);
            if (h == 0) pl->npos[r] = np;
            if (np != pl->npos[r]) {
                fprintf(stderr, "%s: head %d of layer %d has other positions\n", dir, h, layer);
                exit(2);
            }
            memcpy(pl->q + ((size_t)h * (size_t)n_rec + (size_t)r) * HEAD_DIM, (char *)p + (size_t)r * rec + sizeof np,
                   HEAD_DIM * sizeof(float));
        }
        free(p);
    }
}

static void probe_free(probe_layer *pl) {
    for (int h = 0; h < N_HEAD; h++) {
        free(pl->k[h]);
        free(pl->v[h]);
    }
    free(pl->q);
}

/* the CPU reference: tr_attention_group for a group of one, one head per task */
typedef struct {
    const float *q;              /* [head][128] */
    const float *const *k, *const *v;
    int n;
    float scale;
    float *scores; /* [worker][CAP] */
    float *out;    /* [head][128] */
} ref_ctx;

static void ref_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    ref_ctx *c = ctx_;
    for (int64_t h = begin; h < end; h++)
        tr_attention_group(c->q + h * HEAD_DIM, HEAD_DIM, c->k[h], c->v[h], 1, c->n, HEAD_DIM, c->scale,
                           c->scores + (size_t)worker * CAP, CAP, c->out + h * HEAD_DIM, HEAD_DIM);
}

typedef struct {
    int64_t queries, heads, floats_bad, heads_bad;
    int printed;
} tally;

/* GPU out (h_out) against the reference, as bits */
static void compare(tally *t, const float *gpu, const float *ref, const char *what, int layer, int n) {
    t->queries++;
    for (int h = 0; h < N_HEAD; h++) {
        int bad = 0;
        for (int d = 0; d < HEAD_DIM; d++) {
            uint32_t a = fbits(gpu[h * HEAD_DIM + d]), r = fbits(ref[h * HEAD_DIM + d]);
            if (a == r) continue;
            bad++;
            if (t->printed < 8) {
                t->printed++;
                printf("  mismatch %s layer %d n_pos %d head %d dim %d: gpu %08" PRIX32 " (%.9g) cpu %08" PRIX32
                       " (%.9g)\n",
                       what, layer, n, h, d, a, (double)from_bits(a), r, (double)from_bits(r));
            }
        }
        t->heads++;
        t->floats_bad += bad;
        t->heads_bad += bad > 0;
    }
}

static int set_tier(const char *tier) {
    const tr_kernels *k = tr_kernels_tier(tier);
    if (k == NULL) {
        fprintf(stderr, "kernel tier %s not available on this CPU\n", tier);
        return -1;
    }
    tr_kernels_set_active(k);
    return 0;
}

/* ---- check: a probe run, query by query ------------------------------------------------------ */

static int run_check(gpu_t *g, const char *dir, tr_pool *pool) {
    const size_t layer_floats = (size_t)N_HEAD * HEAD_DIM * CAP;
    float *hk = xmalloc(layer_floats * sizeof(float)), *hv = xmalloc(layer_floats * sizeof(float));
    float *ref = xmalloc(QKV * sizeof(float));
    float *scores = xmalloc((size_t)tr_pool_size(pool) * CAP * sizeof(float));
    CUdeviceptr K, V;
    CU(cu.MemAlloc(&K, layer_floats * sizeof(float)));
    CU(cu.MemAlloc(&V, layer_floats * sizeof(float)));
    tally t = {0, 0, 0, 0, 0};
    int64_t skipped = 0, appended = 0;
    double t0 = tr_time_sec();
    for (int layer = 0; layer < N_LAYER; layer++) {
        probe_layer pl;
        memset(&pl, 0, sizeof pl);
        probe_load(dir, layer, &pl);
        int64_t filled = -1; /* positions [0, filled) are in VRAM */
        for (int r = 0; r < pl.n_rec; r++) {
            int n = (int)pl.npos[r];
            if (n > pl.n_rows || n < 1) {
                skipped++;
                continue;
            }
            if (filled != n - 1) {
                /* upload positions < n-1; the rest NaN, so a read of a position the GPU should not
                 * see, or should have appended itself, poisons the output */
                fill_nan(hk, layer_floats);
                fill_nan(hv, layer_floats);
                for (int h = 0; h < N_HEAD; h++)
                    for (int p = 0; p < n - 1; p++) {
                        put_k(hk, h, p, pl.k[h] + (size_t)p * HEAD_DIM);
                        put_v(hv, h, p, pl.v[h] + (size_t)p * HEAD_DIM);
                    }
                CU(cu.MemcpyHtoD(K, hk, layer_floats * sizeof(float)));
                CU(cu.MemcpyHtoD(V, hv, layer_floats * sizeof(float)));
            }
            for (int h = 0; h < N_HEAD; h++) {
                memcpy(g->h_in + h * HEAD_DIM, pl.q + ((size_t)h * (size_t)pl.n_rec + (size_t)r) * HEAD_DIM,
                       HEAD_DIM * sizeof(float));
                memcpy(g->h_in + QKV + h * HEAD_DIM, pl.k[h] + (size_t)(n - 1) * HEAD_DIM, HEAD_DIM * sizeof(float));
                memcpy(g->h_in + 2 * QKV + h * HEAD_DIM, pl.v[h] + (size_t)(n - 1) * HEAD_DIM, HEAD_DIM * sizeof(float));
            }
            round_trip(g, K, V, n, 0);
            filled = n;
            appended++;
            ref_ctx rc = {g->h_in, (const float *const *)pl.k, (const float *const *)pl.v, n, g->scale, scores, ref};
            tr_parallel_for(pool, N_HEAD, 1, ref_body, &rc);
            compare(&t, g->h_out, ref, dir, layer, n);
        }
        probe_free(&pl);
    }
    CU(cu.MemFree(K));
    CU(cu.MemFree(V));
    free(hk);
    free(hv);
    free(ref);
    free(scores);
    printf("check %s: %" PRId64 " queries x %d heads = %" PRId64 " head outputs compared (%" PRId64
           " tokens appended by the GPU, %" PRId64 " queries past the dump skipped), %" PRId64
           " heads differ, %" PRId64 " floats differ  [%.1f s]\n",
           dir, t.queries, N_HEAD, t.heads, appended, skipped, t.heads_bad, t.floats_bad, tr_time_sec() - t0);
    if (t.queries == 0) printf("  FAIL: nothing compared\n");
    return (t.queries > 0 && t.floats_bad == 0) ? 0 : 1;
}

/* ---- edge: synthetic heads -------------------------------------------------------------------- */

static float rnd(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(int32_t)(*s >> 32) * (1.0f / 2147483648.0f);
}

/* a key k (only k[0] set) whose score against q = (q0, 0, ...) is exactly target: s = dot * scale
 * under the lane contract, as the engine computes it */
static int find_key(uint32_t target, float scale, float *q0_out, float *k0_out) {
    static const float q0s[] = {1.0f, 1.25f, 1.5f, 1.75f, 3.0f, 5.0f, 7.0f, 0.75f};
    const tr_kernels *kt = tr_kernels_scalar();
    float q[HEAD_DIM] = {0}, key[HEAD_DIM] = {0};
    for (size_t i = 0; i < sizeof q0s / sizeof q0s[0]; i++) {
        q[0] = q0s[i];
        float guess = from_bits(target) / (q0s[i] * scale);
        float up = guess, down = guess;
        for (int step = 0; step < 256; step++) {
            for (int side = 0; side < 2; side++) {
                key[0] = side ? up : down;
                if (fbits(kt->dot_f32(q, key, HEAD_DIM) * scale) == target) {
                    *q0_out = q0s[i];
                    *k0_out = key[0];
                    return 0;
                }
            }
            up = nextafterf(up, INFINITY);
            down = nextafterf(down, -INFINITY);
        }
    }
    return -1;
}

enum { EDGE_RAND, EDGE_WIDE, EDGE_NEGZERO, EDGE_EQUAL, EDGE_EXC, N_EDGE };
static const char *const edge_name[N_EDGE] = {"random", "wide", "negzero", "equal", "exception"};

static int run_edge(gpu_t *g, tr_pool *pool) {
    static const int sizes[] = {1,   2,   3,   15,  16,   17,   31,   32,   33,   63,   64,   65,   127,
                                128, 129, 191, 192, 193,  255,  256,  257,  511,  513,  1000, 1912, 2047,
                                2048, 2049, 3999, 4000, 4001, 4094, 4095, 4096};
    const size_t layer_floats = (size_t)N_HEAD * HEAD_DIM * CAP;
    float *hk = xmalloc(layer_floats * sizeof(float)), *hv = xmalloc(layer_floats * sizeof(float));
    float *kd[N_HEAD], *vd[N_HEAD];
    for (int h = 0; h < N_HEAD; h++) {
        kd[h] = xmalloc((size_t)CAP * HEAD_DIM * sizeof(float));
        vd[h] = xmalloc((size_t)CAP * HEAD_DIM * sizeof(float));
    }
    float *ref = xmalloc(QKV * sizeof(float));
    float *scores = xmalloc((size_t)tr_pool_size(pool) * CAP * sizeof(float));
    CUdeviceptr K, V;
    CU(cu.MemAlloc(&K, layer_floats * sizeof(float)));
    CU(cu.MemAlloc(&V, layer_floats * sizeof(float)));
    tally t = {0, 0, 0, 0, 0};
    int64_t table_hits = 0, underflows = 0, subnormal_a = 0;
    uint64_t seed = 0x5EED5EEDu;
    const tr_kernels *kt = tr_kernels_scalar();
    for (int kind = 0; kind < N_EDGE; kind++) {
        int64_t bad_before = t.floats_bad;
        int cases = 0;
        for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
            int n = sizes[si];
            if (kind == EDGE_EXC && n < 16) continue;
            float *q = g->h_in;
            for (int h = 0; h < N_HEAD; h++) {
                /* wide: scores spread over hundreds, so exponentials underflow to 0 and to subnormals */
                float qs = kind == EDGE_WIDE ? 100.0f : 1.0f;
                for (int d = 0; d < HEAD_DIM; d++) q[h * HEAD_DIM + d] = qs * rnd(&seed);
                for (int p = 0; p < n; p++)
                    for (int d = 0; d < HEAD_DIM; d++) {
                        kd[h][(size_t)p * HEAD_DIM + d] = rnd(&seed);
                        vd[h][(size_t)p * HEAD_DIM + d] = rnd(&seed);
                    }
                if (kind == EDGE_EQUAL) /* every key the same: equal scores, a_t = 1/n */
                    for (int p = 1; p < n; p++) memcpy(kd[h] + (size_t)p * HEAD_DIM, kd[h], HEAD_DIM * sizeof(float));
                if (kind == EDGE_NEGZERO) {
                    /* dim 0 of every head, and all of head 15, are -0: the sum must start at +0 */
                    for (int p = 0; p < n; p++) {
                        vd[h][(size_t)p * HEAD_DIM] = -0.0f;
                        if (h == N_HEAD - 1)
                            for (int d = 0; d < HEAD_DIM; d++) vd[h][(size_t)p * HEAD_DIM + d] = -0.0f;
                    }
                }
            }
            if (kind == EDGE_EXC) {
                /* head 0: q = (q0, 0...), key 0 scores 0 (the maximum), keys 1..4 score exactly the negative
                 * arguments of tr_expf's exception table, the others below 0 */
                float *q0 = q;
                memset(q0, 0, HEAD_DIM * sizeof(float));
                int placed = 0;
                float q0v = 1.0f;
                for (int i = 0; i < TR_EXPF_N_EXCEPTIONS; i++) {
                    uint32_t xb = tr_expf_exceptions[i][0];
                    float qv, kv;
                    if (!(xb & 0x80000000u) || find_key(xb, g->scale, &qv, &kv) != 0) continue;
                    if (placed == 0) q0v = qv;
                    if (qv != q0v) continue; /* one q per head: keep the targets that share it */
                    placed++;
                    for (int d = 0; d < HEAD_DIM; d++) kd[0][(size_t)(2 * placed) * HEAD_DIM + d] = d == 0 ? kv : 0.0f;
                }
                q0[0] = q0v;
                for (int d = 0; d < HEAD_DIM; d++) kd[0][d] = 0.0f;
                for (int p = 1; p < n; p++)
                    if (p % 2 == 1 || p > 2 * placed) kd[0][(size_t)p * HEAD_DIM] = -fabsf(kd[0][(size_t)p * HEAD_DIM]);
            }
            /* positions < n-1 in VRAM, n-1 appended by the GPU, the rest NaN */
            fill_nan(hk, layer_floats);
            fill_nan(hv, layer_floats);
            for (int h = 0; h < N_HEAD; h++) {
                for (int p = 0; p < n - 1; p++) {
                    put_k(hk, h, p, kd[h] + (size_t)p * HEAD_DIM);
                    put_v(hv, h, p, vd[h] + (size_t)p * HEAD_DIM);
                }
                memcpy(g->h_in + QKV + h * HEAD_DIM, kd[h] + (size_t)(n - 1) * HEAD_DIM, HEAD_DIM * sizeof(float));
                memcpy(g->h_in + 2 * QKV + h * HEAD_DIM, vd[h] + (size_t)(n - 1) * HEAD_DIM, HEAD_DIM * sizeof(float));
            }
            CU(cu.MemcpyHtoD(K, hk, layer_floats * sizeof(float)));
            CU(cu.MemcpyHtoD(V, hv, layer_floats * sizeof(float)));
            round_trip(g, K, V, n, 0);
            ref_ctx rc = {g->h_in, (const float *const *)kd, (const float *const *)vd, n, g->scale, scores, ref};
            tr_parallel_for(pool, N_HEAD, 1, ref_body, &rc);
            compare(&t, g->h_out, ref, edge_name[kind], 0, n);
            cases++;
            /* what the case exercised, on the CPU's side: exception-table arguments, underflows, subnormal a_t */
            for (int h = 0; h < N_HEAD; h++) {
                float *s = scores; /* worker scratch is free again: recompute the row */
                float m = -INFINITY;
                for (int p = 0; p < n; p++) {
                    s[p] = kt->dot_f32(q + h * HEAD_DIM, kd[h] + (size_t)p * HEAD_DIM, HEAD_DIM) * g->scale;
                    if (s[p] > m) m = s[p];
                }
                for (int p = 0; p < n; p++) {
                    int path = tr_expf_path(s[p] - m);
                    table_hits += path == TR_EXPF_TABLE;
                    underflows += (s[p] - m) < TR_EXPF_UNDERFLOW_X;
                    float e = tr_expf(s[p] - m);
                    subnormal_a += e != 0.0f && fabsf(e) < 1.17549435e-38f;
                }
            }
        }
        printf("edge %-9s %2d cases (n_pos 1..4096), %" PRId64 " floats differ\n", edge_name[kind], cases,
               t.floats_bad - bad_before);
    }
    printf("edge: %" PRId64 " head outputs compared, %" PRId64 " heads differ, %" PRId64
           " floats differ; exercised: %" PRId64 " exponentials through the exception table, %" PRId64
           " underflows to 0, %" PRId64 " subnormal e_t\n",
           t.heads, t.heads_bad, t.floats_bad, table_hits, underflows, subnormal_a);
    int ok = t.floats_bad == 0 && table_hits > 0 && underflows > 0 && subnormal_a > 0;
    if (table_hits == 0 || underflows == 0 || subnormal_a == 0) printf("  FAIL: a branch was never exercised\n");
    CU(cu.MemFree(K));
    CU(cu.MemFree(V));
    for (int h = 0; h < N_HEAD; h++) {
        free(kd[h]);
        free(vd[h]);
    }
    free(hk);
    free(hv);
    free(ref);
    free(scores);
    return ok ? 0 : 1;
}

/* ---- expf: all 2^32 floats --------------------------------------------------------------------- */

typedef struct {
    const uint32_t *gpu;
    uint32_t first;
    int64_t bad[64];
    uint32_t first_bad[64];
} expf_ctx;

static void expf_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    expf_ctx *c = ctx_;
    int64_t bad = 0;
    for (int64_t i = begin; i < end; i++) {
        uint32_t xb = c->first + (uint32_t)i;
        float x = from_bits(xb);
        uint32_t r = fbits(tr_expf(x)), a = c->gpu[i];
        if (a == r) continue;
        if (x != x && (a & 0x7F800000u) == 0x7F800000u && (a & 0x007FFFFFu) != 0) continue; /* NaN in, NaN out */
        if (bad == 0 && c->bad[worker] == 0) c->first_bad[worker] = xb;
        bad++;
    }
    c->bad[worker] += bad;
}

static int run_expf(gpu_t *g, tr_pool *pool) {
    CUdeviceptr d;
    CU(cu.MemAlloc(&d, EXPF_CHUNK * sizeof(uint32_t)));
    void *p;
    CU(cu.MemAllocHost(&p, EXPF_CHUNK * sizeof(uint32_t)));
    expf_ctx c;
    memset(&c, 0, sizeof c);
    c.gpu = p;
    double t0 = tr_time_sec();
    int64_t exc_ok = 0;
    for (uint64_t first = 0; first < (1ull << 32); first += EXPF_CHUNK) {
        uint32_t f32 = (uint32_t)first;
        void *args[] = {&f32, &d};
        CU(cu.LaunchKernel(g->f_expf, EXPF_CHUNK / 256, 1, 1, 256, 1, 1, 0, g->st, args, NULL));
        CU(cu.MemcpyDtoHAsync(p, d, EXPF_CHUNK * sizeof(uint32_t), g->st));
        CU(cu.StreamSynchronize(g->st));
        c.first = f32;
        tr_parallel_for(pool, EXPF_CHUNK, 1 << 16, expf_body, &c);
        for (int i = 0; i < TR_EXPF_N_EXCEPTIONS; i++) {
            uint32_t xb = tr_expf_exceptions[i][0];
            if (xb >= f32 && xb - f32 < EXPF_CHUNK) exc_ok += c.gpu[xb - f32] == tr_expf_exceptions[i][1];
        }
    }
    int64_t bad = 0;
    uint32_t fb = 0;
    for (int w = 0; w < 64; w++) {
        if (c.bad[w] != 0 && bad == 0) fb = c.first_bad[w];
        bad += c.bad[w];
    }
    printf("expf: 2^32 floats, GPU against tr_expf: %" PRId64 " differ%s", bad, bad ? " (first " : "");
    if (bad) printf("%08" PRIX32 ")", fb);
    printf("; exception-table arguments right on the GPU: %" PRId64 " of %d  [%.1f s]\n", exc_ok, TR_EXPF_N_EXCEPTIONS,
           tr_time_sec() - t0);
    CU(cu.MemFreeHost(p));
    CU(cu.MemFree(d));
    return (bad == 0 && exc_ok == TR_EXPF_N_EXCEPTIONS) ? 0 : 1;
}

/* ---- time ------------------------------------------------------------------------------------- */

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef struct {
    double med, lo, hi, p90;
} stats;

static stats stat_of(double *x, int n) {
    stats s = {0, 0, 0, 0};
    if (n <= 0) return s;
    qsort(x, (size_t)n, sizeof *x, cmp_double);
    s.med = n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
    s.lo = x[0];
    s.hi = x[n - 1];
    s.p90 = x[(int)(0.9 * (n - 1))];
    return s;
}

static void stamp(const char *what) {
#ifdef _WIN32
    SYSTEMTIME st;
    GetLocalTime(&st);
    printf("[%02d:%02d:%02d.%03d] %s\n", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, what);
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv;
    localtime_r(&ts.tv_sec, &tmv);
    printf("[%02d:%02d:%02d.%03ld] %s\n", tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ts.tv_nsec / 1000000, what);
#endif
    fflush(stdout);
}

static void spin_until(double t) {
    while (tr_time_sec() < t) {
    }
}

static void line(const char *what, stats s, double bytes, const char *extra) {
    printf("  %-28s median %7.1f us  (min %7.1f, p90 %7.1f, max %7.1f)  x16 = %5.2f ms/token", what, s.med, s.lo, s.p90,
           s.hi, 16.0 * s.med / 1000.0);
    if (bytes > 0.0) printf("  %6.1f GB/s", bytes / (s.med * 1e3));
    printf("%s\n", extra);
}

/* The decode's pattern: a round trip per layer, then gap_us of CPU work (a spin, not a sleep) before
 * the next layer's, 16 layers per token. keep: 0 nothing else; 1 a keep-warm kernel (one warp on
 * another stream) spinning through the gap; 2 the same napping 20 us between two reads of the timer.
 * With x != NULL it records every call's round trip (x) and kernel time (y); returns how many. */
static int bursts(gpu_t *g, const CUdeviceptr *K, const CUdeviceptr *V, int n, int keep, int tokens, double gap_us,
                  double *x, double *y) {
    unsigned long long warm_ns = gap_us > 100.0 ? (unsigned long long)(1000.0 * (gap_us - 50.0)) : 0ull;
    unsigned nap = keep == 2 ? 20000u : 0u;
    void *aw[] = {&warm_ns, &nap};
    int m = 0;
    for (int tok = 0; tok < tokens; tok++)
        for (int layer = 0; layer < N_LAYER; layer++) {
            double t0 = tr_time_sec();
            round_trip(g, K[layer], V[layer], n, x != NULL);
            double t1 = tr_time_sec();
            if (keep) CU(cu.LaunchKernel(g->f_warm, 1, 1, 1, 32, 1, 1, 0, g->st2, aw, NULL));
            if (x != NULL && m < MAX_RUNS) {
                x[m] = 1e6 * (t1 - t0);
                y[m] = ev_us(g, 0, 3);
                m++;
            }
            spin_until(t1 + 1e-6 * gap_us);
        }
    return m;
}

static int run_time(gpu_t *g, const char *data, int only_n, const char *only, int runs, int tokens, double gap_us,
                    double settle) {
    const size_t layer_floats = (size_t)N_HEAD * HEAD_DIM * CAP;
    size_t free_b = 0, total_b = 0;
    CU(cu.MemGetInfo(&free_b, &total_b));
    size_t need = 2 * N_LAYER * layer_floats * sizeof(float);
    printf("VRAM: %.0f MiB free of %.0f; the 16 layers take %.0f MiB\n", (double)free_b / 1048576.0,
           (double)total_b / 1048576.0, (double)need / 1048576.0);
    if (need + (256u << 20) > free_b) {
        fprintf(stderr, "not enough free VRAM\n");
        return 2;
    }
    CUdeviceptr K[N_LAYER], V[N_LAYER];
    float *hk = xmalloc(layer_floats * sizeof(float)), *hv = xmalloc(layer_floats * sizeof(float));
    uint64_t seed = 12345;
    for (int layer = 0; layer < N_LAYER; layer++) {
        CU(cu.MemAlloc(&K[layer], layer_floats * sizeof(float)));
        CU(cu.MemAlloc(&V[layer], layer_floats * sizeof(float)));
        if (data != NULL) {
            probe_layer pl;
            memset(&pl, 0, sizeof pl);
            probe_load(data, layer, &pl);
            if (pl.n_rows < (only_n > 0 ? only_n : 4000)) {
                fprintf(stderr, "%s holds %d positions per head, fewer than asked\n", data, pl.n_rows);
                return 2;
            }
            fill_nan(hk, layer_floats);
            fill_nan(hv, layer_floats);
            for (int h = 0; h < N_HEAD; h++)
                for (int p = 0; p < pl.n_rows; p++) {
                    put_k(hk, h, p, pl.k[h] + (size_t)p * HEAD_DIM);
                    put_v(hv, h, p, pl.v[h] + (size_t)p * HEAD_DIM);
                }
            if (layer == 0) {
                for (int h = 0; h < N_HEAD; h++) {
                    memcpy(g->h_in + h * HEAD_DIM, pl.q + (size_t)h * (size_t)pl.n_rec * HEAD_DIM, HEAD_DIM * sizeof(float));
                    memcpy(g->h_in + QKV + h * HEAD_DIM, pl.k[h], HEAD_DIM * sizeof(float));
                    memcpy(g->h_in + 2 * QKV + h * HEAD_DIM, pl.v[h], HEAD_DIM * sizeof(float));
                }
                printf("data: %s (%d positions per head)\n", data, pl.n_rows);
            }
            probe_free(&pl);
        } else {
            if (layer == 0) {
                for (size_t i = 0; i < layer_floats; i++) {
                    hk[i] = rnd(&seed);
                    hv[i] = rnd(&seed);
                }
                for (int i = 0; i < IN_FLOATS; i++) g->h_in[i] = rnd(&seed);
                printf("data: synthetic, uniform in [-1, 1)\n");
            }
        }
        CU(cu.MemcpyHtoD(K[layer], hk, layer_floats * sizeof(float)));
        CU(cu.MemcpyHtoD(V[layer], hv, layer_floats * sizeof(float)));
    }
    free(hk);
    free(hv);
    CU(cu.MemcpyHtoD(g->in, g->h_in, IN_FLOATS * sizeof(float)));

    static const int ns[] = {2048, 4000};
    static const double cpu_us[] = {704.0, 1415.0};
    double *x = xmalloc(4 * (size_t)MAX_RUNS * sizeof(double));
    double *y = x + MAX_RUNS, *z = y + MAX_RUNS, *w = z + MAX_RUNS;
    const int warm = 16;
    for (int ni = 0; ni < 2; ni++) {
        int n = ns[ni];
        if (only_n > 0 && only_n != n) continue;
        double bytes = 2.0 * N_HEAD * HEAD_DIM * sizeof(float) * n;
        printf("\nn_pos %d: keys + values %.1f MiB per layer; CPU (48 GB/s) %.0f us per layer\n", n, bytes / 1048576.0,
               cpu_us[ni]);
        if (strpbrk(only, "abcez") != NULL) {
            /* back to back: the clocks settle under the same pattern before anything is measured */
            double t_end = tr_time_sec() + settle;
            for (int i = 0; tr_time_sec() < t_end; i++) round_trip(g, K[i % N_LAYER], V[i % N_LAYER], n, 0);
        }
        if (strchr(only, 'e')) {
            stamp("(e) floor: begin");
            int m = 0;
            for (int i = 0; i < warm + runs; i++) {
                double t0 = tr_time_sec();
                CU(cu.LaunchKernel(g->f_empty, 1, 1, 1, 32, 1, 1, 0, g->st, NULL, NULL));
                CU(cu.StreamSynchronize(g->st));
                double t1 = tr_time_sec();
                int layer = i % N_LAYER;
                enqueue_attention(g, K[layer], V[layer], n, 0);
                CU(cu.StreamSynchronize(g->st));
                double t2 = tr_time_sec();
                if (i < warm) continue;
                x[m] = 1e6 * (t1 - t0);
                y[m] = 1e6 * (t2 - t1);
                m++;
            }
            stamp("(e) floor: end");
            line("(e) empty kernel + sync", stat_of(x, m), 0.0, "");
            line("(e) kernels + sync, no copy", stat_of(y, m), bytes, "");
        }
        if (strchr(only, 'z')) {
            stamp("(z) round trip, zero copy: begin");
            int was = g->zero_copy;
            g->zero_copy = 1;
            int m = 0;
            for (int i = 0; i < warm + runs; i++) {
                int layer = i % N_LAYER;
                double t0 = tr_time_sec();
                round_trip(g, K[layer], V[layer], n, 0);
                double t1 = tr_time_sec();
                if (i >= warm) x[m++] = 1e6 * (t1 - t0);
            }
            g->zero_copy = was;
            stamp("(z) round trip, zero copy: end");
            line("(z) round trip, zero copy", stat_of(x, m), bytes, "");
        }
        if (strchr(only, 'a')) {
            stamp("(a) kernels, back to back: begin");
            int m = 0;
            for (int i = 0; i < warm + runs; i++) {
                int layer = i % N_LAYER;
                enqueue_attention(g, K[layer], V[layer], n, 1);
                CU(cu.StreamSynchronize(g->st));
                if (i < warm) continue;
                x[m] = ev_us(g, 0, 3);
                y[m] = ev_us(g, 0, 1);
                z[m] = ev_us(g, 1, 2);
                w[m] = ev_us(g, 2, 3);
                m++;
            }
            stamp("(a) kernels, back to back: end");
            stats sa = stat_of(x, m), s1 = stat_of(y, m), s2 = stat_of(z, m), s3 = stat_of(w, m);
            char extra[160];
            snprintf(extra, sizeof extra, "  [scores %.1f, exp %.1f, values %.1f]", s1.med, s2.med, s3.med);
            line("(a) kernels", sa, bytes, extra);
        }
        if (strchr(only, 'b')) {
            stamp("(b) round trip, back to back: begin");
            int m = 0;
            for (int i = 0; i < warm + runs; i++) {
                int layer = i % N_LAYER;
                double t0 = tr_time_sec();
                round_trip(g, K[layer], V[layer], n, 0);
                double t1 = tr_time_sec();
                if (i >= warm) x[m++] = 1e6 * (t1 - t0);
            }
            stamp("(b) round trip, back to back: end");
            line("(b) round trip", stat_of(x, m), bytes, "");
        }
        if (strchr(only, 'c')) {
            stamp("(c) plain read of the same bytes: begin");
            int m = 0;
            unsigned klen = (unsigned)(n / 4), vlen = (unsigned)(n * HEAD_DIM / 4);
            unsigned long long kstride = CAP * sizeof(float), vstride = (unsigned long long)CAP * HEAD_DIM * sizeof(float);
            for (int i = 0; i < warm + runs; i++) {
                int layer = i % N_LAYER;
                void *ak[] = {&K[layer], &kstride, &klen, &g->sink};
                void *av[] = {&V[layer], &vstride, &vlen, &g->sink};
                CU(cu.EventRecord(g->ev[0], g->st));
                CU(cu.LaunchKernel(g->f_read, (klen + 511) / 512, N_HEAD * HEAD_DIM, 1, 128, 1, 1, 0, g->st, ak, NULL));
                CU(cu.LaunchKernel(g->f_read, (vlen + 511) / 512, N_HEAD, 1, 128, 1, 1, 0, g->st, av, NULL));
                CU(cu.EventRecord(g->ev[1], g->st));
                CU(cu.StreamSynchronize(g->st));
                if (i >= warm) x[m++] = ev_us(g, 0, 1);
            }
            stamp("(c) plain read of the same bytes: end");
            line("(c) plain read, same bytes", stat_of(x, m), bytes, "");
        }
        static const char burst_mode[] = "dws";
        static const char *const burst_name[] = {"(d) bursts", "(w) bursts, keep-warm spin", "(s) bursts, keep-warm nap"};
        for (int kw = 0; kw < 3; kw++) {
            if (!strchr(only, burst_mode[kw])) continue;
            /* the pattern runs `settle` seconds unmeasured (at least one token), then `tokens` tokens */
            double t_end = tr_time_sec() + settle;
            do bursts(g, K, V, n, kw, 1, gap_us, NULL, NULL);
            while (tr_time_sec() < t_end);
            char msg[96];
            snprintf(msg, sizeof msg, "%s: begin", burst_name[kw]);
            stamp(msg);
            double t_start = tr_time_sec();
            int m = bursts(g, K, V, n, kw, tokens, gap_us, x, y);
            double secs = tr_time_sec() - t_start;
            snprintf(msg, sizeof msg, "%s: end", burst_name[kw]);
            stamp(msg);
            /* the kernels' own time, and the round trip per window of 16 tokens: does it drift? */
            char extra[512];
            int len = snprintf(extra, sizeof extra, "  [kernels %.1f; per 16 tokens:", stat_of(y, m).med);
            for (int w0 = 0; w0 + 16 * N_LAYER <= m && len < (int)sizeof extra - 16; w0 += 16 * N_LAYER) {
                double tmp[16 * N_LAYER];
                memcpy(tmp, x + w0, sizeof tmp);
                len += snprintf(extra + len, sizeof extra - (size_t)len, " %.0f", stat_of(tmp, 16 * N_LAYER).med);
            }
            snprintf(extra + len, sizeof extra - (size_t)len, "]");
            line(burst_name[kw], stat_of(x, m), bytes, extra);
            printf("      %d calls over %.1f s\n", m, secs);
            CU(cu.CtxSynchronize());
        }
    }
    free(x);
    for (int layer = 0; layer < N_LAYER; layer++) {
        CU(cu.MemFree(K[layer]));
        CU(cu.MemFree(V[layer]));
    }
    return 0;
}

/* ---- main ------------------------------------------------------------------------------------- */

static void run_info(gpu_t *g) {
    int v, clock = 0, mem_clock = 0, bus = 0, l2 = 0, tcc = 0;
    size_t total = 0;
    CU(cu.DeviceTotalMem(&total, g->dev));
    if (cu.DeviceGetAttribute(&v, ATTR_CLOCK, g->dev) == 0) clock = v;
    if (cu.DeviceGetAttribute(&v, ATTR_MEM_CLOCK, g->dev) == 0) mem_clock = v;
    if (cu.DeviceGetAttribute(&v, ATTR_BUS, g->dev) == 0) bus = v;
    if (cu.DeviceGetAttribute(&v, ATTR_L2, g->dev) == 0) l2 = v;
    if (cu.DeviceGetAttribute(&v, ATTR_TCC, g->dev) == 0) tcc = v;
    printf("%s, sm_%d%d, %d SMs, %.0f MiB, L2 %d KiB, bus %d bit, clock %d MHz, memory %d MHz (peak %.0f GB/s), "
           "driver %d.%d, %s\n",
           g->name, g->cc_major, g->cc_minor, g->sms, (double)total / 1048576.0, l2 / 1024, bus, clock / 1000,
           mem_clock / 1000, 2.0 * mem_clock * 1e3 * (bus / 8) / 1e9, g->driver / 1000, (g->driver % 1000) / 10,
           tcc ? "TCC" : "WDDM");
    const struct {
        const char *name;
        CUfunction f;
    } fs[] = {{"attn_scores", g->f_scores}, {"attn_exp", g->f_exp},     {"attn_v", g->f_v},
              {"stream_read", g->f_read},   {"keep_warm", g->f_warm}, {"expf_sweep", g->f_expf},
              {"empty", g->f_empty}};
    for (size_t i = 0; i < sizeof fs / sizeof fs[0]; i++) {
        int regs = 0, local = 0, shared = 0;
        CU(cu.FuncGetAttribute(&regs, FATTR_REGS, fs[i].f));
        CU(cu.FuncGetAttribute(&local, FATTR_LOCAL, fs[i].f));
        CU(cu.FuncGetAttribute(&shared, FATTR_SHARED, fs[i].f));
        printf("  %-12s %3d registers, %5d bytes shared, %d bytes local (spills)\n", fs[i].name, regs, shared, local);
    }
}

static int usage(void) {
    fprintf(stderr, "usage: bench_gpu_attn info | check <run dir> [--tier T] [--zero-copy] | edge | expf |\n"
                    "       time [--data <run dir>] [--n N] [--only abcezdws] [--runs R] [--tokens T] [--gap-us G]\n"
                    "            [--settle S] [--zero-copy]\n"
                    "       [--mutate fma|tree|sumtree|zero|expf]\n");
    return 2;
}

int main(int argc, char **argv) {
    if (argc < 2) return usage();
    const char *mode = argv[1], *dir = NULL, *tier = "scalar", *data = NULL, *only = "abcezdws";
    int only_n = 0, runs = 64, tokens = 32, zero_copy = 0;
    double gap_us = 1600.0, settle = 1.0;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (strcmp(a, "--tier") == 0 && more) tier = argv[++i];
        else if (strcmp(a, "--data") == 0 && more) data = argv[++i];
        else if (strcmp(a, "--n") == 0 && more) only_n = atoi(argv[++i]);
        else if (strcmp(a, "--only") == 0 && more) only = argv[++i];
        else if (strcmp(a, "--runs") == 0 && more) runs = atoi(argv[++i]);
        else if (strcmp(a, "--tokens") == 0 && more) tokens = atoi(argv[++i]);
        else if (strcmp(a, "--gap-us") == 0 && more) gap_us = atof(argv[++i]);
        else if (strcmp(a, "--settle") == 0 && more) settle = atof(argv[++i]);
        else if (strcmp(a, "--zero-copy") == 0) zero_copy = 1;
        else if (strcmp(a, "--mutate") == 0 && more) {
            const char *m = argv[++i];
            g_mut = strcmp(m, "fma") == 0       ? MUT_FMA
                    : strcmp(m, "tree") == 0    ? MUT_TREE
                    : strcmp(m, "sumtree") == 0 ? MUT_SUMTREE
                    : strcmp(m, "zero") == 0    ? MUT_ZERO
                    : strcmp(m, "expf") == 0    ? MUT_EXPF
                                                : -1;
            if (g_mut < 0) return usage();
        } else if (a[0] != '-' && dir == NULL) dir = a;
        else return usage();
    }
    if (runs < 50 || runs > MAX_RUNS || tokens < 16 || tokens * N_LAYER > MAX_RUNS || settle < 0.0 || settle > 5.0) {
        fprintf(stderr, "--runs 50..%d, --tokens 16..%d, --settle 0..5\n", MAX_RUNS, MAX_RUNS / N_LAYER);
        return 2;
    }
    gpu_t g;
    gpu_open(&g);
    g.zero_copy = zero_copy;
    if (g_mut) printf("MUTATED kernels (--mutate): this run must fail\n");
    if (zero_copy) printf("zero copy: the kernels read the input and write the output in pinned host memory\n");
    int rc = 0;
    if (strcmp(mode, "info") == 0) {
        run_info(&g);
    } else if (strcmp(mode, "time") == 0) {
        run_info(&g);
        rc = run_time(&g, data, only_n, only, runs, tokens, gap_us, settle);
    } else if (strcmp(mode, "check") == 0 || strcmp(mode, "edge") == 0 || strcmp(mode, "expf") == 0) {
        if (set_tier(tier) != 0) return 2;
        tr_pool *pool = tr_pool_create(0);
        if (pool == NULL || tr_pool_size(pool) > 64) return 2;
        if (strcmp(mode, "check") == 0) rc = dir != NULL ? run_check(&g, dir, pool) : usage();
        else if (strcmp(mode, "edge") == 0) rc = run_edge(&g, pool);
        else rc = run_expf(&g, pool);
        tr_pool_destroy(pool);
        printf("reference: tr_attention_group, %s kernels\n", tier);
    } else {
        rc = usage();
    }
    gpu_close(&g);
    printf("%s\n", rc == 0 ? "ok" : rc == 1 ? "FAILED" : "error");
    return rc;
}
