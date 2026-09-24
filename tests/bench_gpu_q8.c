/* bench_gpu_q8.c — premise benchmark: the decode matrix-vector product of a Q8_0 weight on an NVIDIA GPU,
 * bit for bit the CPU engine's (tr_matmul, dot_row), and at what bandwidth.
 *
 * At context 2048 the dense weights of OLMoE-1B-7B (attention Q, K, V and output projection of the 16
 * layers, 2048 x 2048 each, and the 50304 x 2048 output head: ~395 MB a token, all Q8_0) could sit in the
 * 8 GB of a laptop RTX 4070 beside the KV cache and be read at ~256 GB/s instead of the CPU's ~51. This
 * program answers two questions before any engine code is written: does the GPU give the CPU's bits,
 * and how long does one product take there, the kernel alone and with the copies and the
 * synchronization the engine would pay.
 *
 * No build dependency: nvcuda.dll (libcuda.so.1) is opened at run time, the kernels are PTX text written
 * here and JIT-compiled by the driver. Every float operation carries .rn (PTX never fuses those into an
 * FMA), no .ftz, no .approx, so the kernel does k_dot_row_q8_0's arithmetic (src/kernels/kernels.c)
 * operation for operation:
 *   d_b = the block's f16 scale widened to f32 (cvt.f32.f16: exact, subnormals kept)
 *   w_k = fl(d_b * (float)q_k), q_k the int8 quant (cvt.rn.f32.s32: exact)
 *   lane[k % 16] = fl(lane[k % 16] + fl(w_k * x_k)), k increasing, every lane starting at +0
 *   y = tr_lane_combine(lane): ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), the same for 8..15, halves last
 * Kernel q8_mv reads the GGUF layout as it is in the file (rows of 34-byte blocks, no repacking): 16
 * threads a row, thread l holds lane l, so in every block it takes element l, then element l + 16; 16
 * rows a CTA of 256 threads, 8 blocks a loop iteration with every load issued before the arithmetic.
 * The tree is four shuffles: lane i adds lane i+1, then i+2, i+4, i+8, and lane 0 ends with
 * tr_lane_combine's association. Threads of rows past the last one compute row 0 and store nothing.
 *
 * Modes (one run stays under 60 s, VRAM under 1 GB):
 *   info    the device, the driver, registers and spills of every kernel
 *   check   every output float of the GPU against the CPU's, as bits: scalar dot_row (the definition)
 *           and tr_matmul on the active tier. Synthetic matrices first: random finite f16 scales (zero
 *           and subnormal included), quants over the whole int8 range, 37 and 33 rows (a CTA with rows
 *           past the end, a warp with one row out) and 256, 1024, 2048 columns (one loop iteration, the
 *           expert shape, the attention shape); then the real ones: q, k, v and output projection of the
 *           16 layers and the output head. Six vectors each: uniform, gaussian with outliers, three real
 *           inputs (the embedding row of a token through the layer's rmsnorm: blk.0's q, k, v input for
 *           that token exactly; for the output projection W_v x, the attention output of position 0; for
 *           the head the rmsnorm with output_norm), a special one (+-0, subnormals, tiny products). The
 *           synthetic matrices also get one with huge values (sums that overflow: inf, and NaN where +inf
 *           meets -inf; two NaNs count as equal, the payload is free by kernels.h). A sentinel after the
 *           last row catches a write past the end. Counters fail the run if a covered case never occurred
 *           (subnormal and zero scales, q = -128, rows past the end, an infinite result, a NaN pair)
 *   time [--runs R]  per shape, medians of R (>= 50) after a warm-up: (a) the kernel alone, cuEvent; (b)
 *           the round trip: pinned x host to device, the kernel, y device to host, stream synchronized,
 *           host clock, taken apart in (b1) the kernel and the wait without the copies and (b2) the copies
 *           and the wait without the kernel; (b3) the round trip without the copy engine: x written to
 *           mapped pinned memory and pulled into VRAM by a one-CTA kernel, y stored by q8_mv straight into
 *           mapped pinned memory (its bits checked against (b)'s); (d) (b3) with G us of CPU spin between
 *           two calls (--gap-us G, default 1000), as the decode leaves the GPU while the CPU runs the rest of
 *           the layer; (c) a plain read of the same bytes, the ceiling; (e) a token's worth back to back
 *           (every layer's matrix of that shape), per product. Shapes: 2048 x 2048 rotating over the 64
 *           real attention matrices, 6144 x 2048 (q, k, v of a layer as one product) over the 16 layers,
 *           50304 x 2048 alternating output.weight and token_embd.weight: nothing is still in the 32 MB L2
 *           when its turn comes
 *   --mutate norn|tree|order   a wrong kernel on purpose (w*x and the lane add without .rn, so ptxas
 *           may fuse them; the tree with offsets 8, 4, 2, 1; element l + 16 before element l): check must
 *           then fail
 *   --model <file>   default models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf */
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
#endif

#include "../src/base/platform.h"
#include "../src/base/threads.h"
#include "../src/format/gguf.h"
#include "../src/kernels/kernels.h"
#include "../src/kernels/kernels_internal.h"

/* OLMoE-1B-7B */
#define N_EMBD 2048
#define N_LAYER 16
#define N_VOCAB 50304
#define N_ATTN (4 * N_LAYER) /* q, k, v, output projection of every layer */

#define ROWS_PER_CTA 16 /* 16 threads a row: 256 threads a CTA */
#define UNROLL 8        /* blocks a loop iteration of q8_mv: a row's block count must be a multiple */
#define UNROLL_LOG2 3
#define TAIL 16         /* sentinel floats after the last row */
#define SENTINEL 0x7FBADBADu
#define MAX_RUNS 4096
#define DEFAULT_MODEL "models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf"

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
#define CU_MEMHOSTALLOC_DEVICEMAP 0x02
enum { JIT_INFO_LOG = 3, JIT_INFO_LOG_SIZE = 4, JIT_ERROR_LOG = 5, JIT_ERROR_LOG_SIZE = 6 };
enum { ATTR_CLOCK = 13, ATTR_SMS = 16, ATTR_MEM_CLOCK = 36, ATTR_BUS = 37, ATTR_L2 = 38, ATTR_CC_MAJOR = 75,
       ATTR_CC_MINOR = 76 };
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
    CUresult (*MemHostAlloc)(void **, size_t, unsigned);
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
    {offsetof(cu_api, MemHostAlloc), "cuMemHostAlloc"},
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
    fprintf(stderr, "bench_gpu_q8.c:%d: %s failed: %d %s\n", line, what, r, s != NULL ? s : "");
    exit(2);
}
#define CU(x) \
    do { \
        CUresult cu_r_ = (x); \
        if (cu_r_ != 0) cu_fail(cu_r_, #x, __LINE__); \
    } while (0)

/* ---- PTX, written here ----------------------------------------------------------------------- */

enum { MUT_NORN = 1, MUT_TREE = 2, MUT_ORDER = 4 };
static int g_mut = 0; /* the mutation asked for on the command line, 0 for the real kernel */

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

/* q8_mv(W, x, y, rows, nb): grid ceil(rows / 16), 256 threads. y[r] = dot_row(row r of W, x), each row
 * nb blocks of 34 bytes (nb a multiple of UNROLL), x nb * 32 floats. */
static void emit_mv(sbuf *b) {
    const char *rn = (g_mut & MUT_NORN) ? "" : ".rn";
    sb(b, ".visible .entry q8_mv(.param .u64 pW, .param .u64 pX, .param .u64 pY, .param .u32 pRows,\n"
          "    .param .u32 pNb)\n{\n"
          ".reg .pred %%p<4>;\n.reg .b32 %%r<%d>;\n.reg .b64 %%rd<8>;\n.reg .b16 %%h<%d>;\n"
          ".reg .f32 %%acc, %%t, %%d<%d>, %%x<%d>, %%q<%d>, %%w<%d>, %%m<%d>;\n",
       16 + 2 * UNROLL, UNROLL, UNROLL, 2 * UNROLL, 2 * UNROLL, 2 * UNROLL, 2 * UNROLL);
    /* l = tid % 16, row = ctaid * 16 + tid / 16; a thread past the last row computes row 0 */
    sb(b, "ld.param.u64 %%rd1, [pW];\nld.param.u64 %%rd2, [pX];\nld.param.u64 %%rd3, [pY];\n"
          "ld.param.u32 %%r1, [pRows];\nld.param.u32 %%r2, [pNb];\n"
          "mov.u32 %%r3, %%tid.x;\nmov.u32 %%r4, %%ctaid.x;\n"
          "and.b32 %%r5, %%r3, 15;\nshr.u32 %%r6, %%r3, 4;\nmad.lo.u32 %%r7, %%r4, %d, %%r6;\n"
          "setp.lt.u32 %%p1, %%r7, %%r1;\nselp.b32 %%r8, %%r7, 0, %%p1;\n",
       ROWS_PER_CTA);
    /* rd4: the row's current block (its scale), rd5: rd4 + l (its quants), rd6: x + 4l */
    sb(b, "mul.lo.u32 %%r9, %%r2, %d;\nmul.wide.u32 %%rd4, %%r8, %%r9;\nadd.s64 %%rd4, %%rd1, %%rd4;\n"
          "cvt.u64.u32 %%rd5, %%r5;\nadd.s64 %%rd5, %%rd4, %%rd5;\n"
          "mul.wide.u32 %%rd6, %%r5, 4;\nadd.s64 %%rd6, %%rd2, %%rd6;\n"
          "mov.f32 %%acc, 0f00000000;\nshr.u32 %%r10, %%r2, %d;\n"
          "LOOP:\nsetp.eq.u32 %%p2, %%r10, 0;\n@%%p2 bra LOOPEND;\n",
       TR_Q8_0_BLOCK_BYTES, UNROLL_LOG2);
    for (int u = 0; u < UNROLL; u++) {
        int o = u * TR_Q8_0_BLOCK_BYTES;
        sb(b, "ld.global.nc.b16 %%h%d, [%%rd4+%d];\n", u, o);
        sb(b, "ld.global.nc.s8 %%r%d, [%%rd5+%d];\nld.global.nc.s8 %%r%d, [%%rd5+%d];\n", 16 + 2 * u,
           o + TR_Q8_0_SCALE_BYTES, 17 + 2 * u, o + TR_Q8_0_SCALE_BYTES + 16);
        sb(b, "ld.global.nc.f32 %%x%d, [%%rd6+%d];\nld.global.nc.f32 %%x%d, [%%rd6+%d];\n", 2 * u,
           4 * TR_Q8_0_BLOCK_ELEMS * u, 2 * u + 1, 4 * TR_Q8_0_BLOCK_ELEMS * u + 64);
    }
    for (int u = 0; u < UNROLL; u++) {
        sb(b, "cvt.f32.f16 %%d%d, %%h%d;\n", u, u);
        for (int s = 0; s < 2; s++) {
            int e = 2 * u + ((g_mut & MUT_ORDER) ? 1 - s : s);
            sb(b, "cvt.rn.f32.s32 %%q%d, %%r%d;\nmul.rn.f32 %%w%d, %%d%d, %%q%d;\n"
                  "mul%s.f32 %%m%d, %%w%d, %%x%d;\nadd%s.f32 %%acc, %%acc, %%m%d;\n",
               e, 16 + e, e, u, e, rn, e, e, e, rn, e);
        }
    }
    sb(b, "add.s64 %%rd4, %%rd4, %d;\nadd.s64 %%rd5, %%rd5, %d;\nadd.s64 %%rd6, %%rd6, %d;\n"
          "sub.u32 %%r10, %%r10, 1;\nbra LOOP;\nLOOPEND:\n",
       UNROLL * TR_Q8_0_BLOCK_BYTES, UNROLL * TR_Q8_0_BLOCK_BYTES, UNROLL * TR_Q8_0_BLOCK_ELEMS * 4);
    /* tr_lane_combine: lane i adds lane i+1, then i+2, i+4, i+8 (the mutation: 8, 4, 2, 1) */
    for (int k = 0; k < 4; k++) {
        int off = (g_mut & MUT_TREE) ? 8 >> k : 1 << k;
        sb(b, "shfl.sync.down.b32 %%t, %%acc, %d, 31, 0xffffffff;\nadd.rn.f32 %%acc, %%acc, %%t;\n", off);
    }
    sb(b, "setp.ne.u32 %%p3, %%r5, 0;\n@%%p3 bra DONE;\n@!%%p1 bra DONE;\n"
          "mul.wide.u32 %%rd7, %%r7, 4;\nadd.s64 %%rd7, %%rd3, %%rd7;\nst.global.f32 [%%rd7], %%acc;\n"
          "DONE:\nret;\n}\n\n");
}

/* stream_read(base, 16-byte units, sink): grid ceil(units / 1024), 256 threads, 4 units a thread: every
 * byte read once, nothing computed (the ceiling) */
static void emit_read(sbuf *b) {
    sb(b, ".visible .entry stream_read(.param .u64 pBase, .param .u64 pUnits, .param .u64 pSink)\n{\n"
          ".reg .pred %%p<4>;\n.reg .b32 %%r<24>;\n.reg .b64 %%rd<8>;\n"
          "ld.param.u64 %%rd1, [pBase];\nld.param.u64 %%rd2, [pUnits];\nld.param.u64 %%rd3, [pSink];\n"
          "mov.u32 %%r1, %%ctaid.x;\nmov.u32 %%r2, %%tid.x;\n"
          "mul.wide.u32 %%rd4, %%r1, 1024;\ncvt.u64.u32 %%rd5, %%r2;\nadd.s64 %%rd4, %%rd4, %%rd5;\nmov.u32 %%r3, 0;\n");
    for (int u = 0; u < 4; u++)
        sb(b, "mov.u32 %%r%d, 0;\nmov.u32 %%r%d, 0;\nmov.u32 %%r%d, 0;\nmov.u32 %%r%d, 0;\n"
              "setp.lt.u64 %%p1, %%rd4, %%rd2;\nshl.b64 %%rd6, %%rd4, 4;\nadd.s64 %%rd6, %%rd1, %%rd6;\n"
              "@%%p1 ld.global.nc.v4.u32 {%%r%d, %%r%d, %%r%d, %%r%d}, [%%rd6];\nadd.s64 %%rd4, %%rd4, 256;\n",
           4 + 4 * u, 5 + 4 * u, 6 + 4 * u, 7 + 4 * u, 4 + 4 * u, 5 + 4 * u, 6 + 4 * u, 7 + 4 * u);
    for (int i = 4; i < 20; i++) sb(b, "xor.b32 %%r3, %%r3, %%r%d;\n", i);
    sb(b, "setp.ne.u32 %%p2, %%r3, 0x9E3779B9;\n@%%p2 bra DONE;\nst.global.u32 [%%rd3], %%r3;\nDONE:\nret;\n}\n\n");
    /* copy16(src, dst, units): 16 bytes a thread; src may be mapped host memory (x pulled over PCIe by
     * the compute engine, no copy engine in the round trip) */
    sb(b, ".visible .entry copy16(.param .u64 pSrc, .param .u64 pDst, .param .u32 pUnits)\n{\n"
          ".reg .pred %%p1;\n.reg .b32 %%r<12>;\n.reg .b64 %%rd<6>;\n"
          "ld.param.u64 %%rd1, [pSrc];\nld.param.u64 %%rd2, [pDst];\nld.param.u32 %%r1, [pUnits];\n"
          "mov.u32 %%r2, %%tid.x;\nmov.u32 %%r3, %%ctaid.x;\nmov.u32 %%r4, %%ntid.x;\nmad.lo.u32 %%r5, %%r3, %%r4, %%r2;\n"
          "setp.ge.u32 %%p1, %%r5, %%r1;\n@%%p1 bra DONE;\nmul.wide.u32 %%rd3, %%r5, 16;\n"
          "add.s64 %%rd4, %%rd1, %%rd3;\nadd.s64 %%rd5, %%rd2, %%rd3;\n"
          "ld.global.v4.u32 {%%r6, %%r7, %%r8, %%r9}, [%%rd4];\nst.global.v4.u32 [%%rd5], {%%r6, %%r7, %%r8, %%r9};\n"
          "DONE:\nret;\n}\n\n");
}

/* ---- the device ------------------------------------------------------------------------------ */

typedef struct {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction f_mv, f_read, f_copy;
    CUstream st;
    CUevent ev[2];
    CUdeviceptr sink;
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
    if (g->cc_major < 7) { /* shfl.sync */
        fprintf(stderr, "%s is sm_%d%d: the kernels need sm_70 or later\n", g->name, g->cc_major, g->cc_minor);
        exit(2);
    }
    /* spin while waiting: the lowest latency for a synchronization the decode waits on */
    CU(cu.CtxCreate(&g->ctx, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, g->dev));

    sbuf b = {malloc(1 << 16), 0, 1 << 16};
    if (b.p == NULL) exit(2);
    sb(&b, ".version 7.0\n.target sm_70\n.address_size 64\n\n");
    emit_mv(&b);
    emit_read(&b);
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
    CU(cu.ModuleGetFunction(&g->f_mv, g->mod, "q8_mv"));
    CU(cu.ModuleGetFunction(&g->f_read, g->mod, "stream_read"));
    CU(cu.ModuleGetFunction(&g->f_copy, g->mod, "copy16"));
    CU(cu.StreamCreate(&g->st, 0)); /* a blocking stream: ordered with the synchronous copies too */
    for (int i = 0; i < 2; i++) CU(cu.EventCreate(&g->ev[i], 0));
    CU(cu.MemAlloc(&g->sink, 64));
}

static void gpu_close(gpu_t *g) {
    CU(cu.CtxSynchronize());
    CU(cu.MemFree(g->sink));
    for (int i = 0; i < 2; i++) CU(cu.EventDestroy(g->ev[i]));
    CU(cu.StreamDestroy(g->st));
    CU(cu.ModuleUnload(g->mod));
    CU(cu.CtxDestroy(g->ctx));
}

static void launch_mv(gpu_t *g, CUdeviceptr w, CUdeviceptr x, CUdeviceptr y, int64_t rows, int64_t cols) {
    unsigned r = (unsigned)rows, nb = (unsigned)(cols / TR_Q8_0_BLOCK_ELEMS);
    void *args[] = {&w, &x, &y, &r, &nb};
    CU(cu.LaunchKernel(g->f_mv, (r + ROWS_PER_CTA - 1) / ROWS_PER_CTA, 1, 1, 16 * ROWS_PER_CTA, 1, 1, 0, g->st, args,
                       NULL));
}

static void launch_read(gpu_t *g, CUdeviceptr base, size_t bytes) {
    unsigned long long units = bytes / 16;
    void *args[] = {&base, &units, &g->sink};
    CU(cu.LaunchKernel(g->f_read, (unsigned)((units + 1023) / 1024), 1, 1, 256, 1, 1, 0, g->st, args, NULL));
}

static void launch_copy(gpu_t *g, CUdeviceptr src, CUdeviceptr dst, size_t bytes) {
    unsigned units = (unsigned)(bytes / 16);
    void *args[] = {&src, &dst, &units};
    CU(cu.LaunchKernel(g->f_copy, (units + 511) / 512, 1, 1, 512, 1, 1, 0, g->st, args, NULL));
}

static double ev_us(gpu_t *g) {
    float ms = 0.0f;
    CU(cu.EventElapsedTime(&ms, g->ev[0], g->ev[1]));
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

static uint64_t rng(uint64_t *s) { /* splitmix64 */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float unif(uint64_t *s) { /* [-1, 1), 24 bits */
    return (float)((double)(rng(s) >> 40) / 8388608.0 - 1.0);
}

/* The host refuses to leave less than 2 GB or 10% of the RAM free (CLAUDE.md, safety). */
static void ram_check(size_t need) {
    tr_meminfo mi;
    if (tr_mem_info(&mi) != 0) return;
    uint64_t keep = mi.total_bytes / 10 > (2ull << 30) ? mi.total_bytes / 10 : (2ull << 30);
    if (mi.available_bytes < need + keep) {
        fprintf(stderr, "not enough free RAM: %.0f MiB needed, %.0f available\n", (double)need / 1048576.0,
                (double)mi.available_bytes / 1048576.0);
        exit(2);
    }
}

static void vram_check(size_t need) {
    size_t free_b = 0, total_b = 0;
    CU(cu.MemGetInfo(&free_b, &total_b));
    printf("VRAM: %.0f MiB free of %.0f, this run takes %.0f\n", (double)free_b / 1048576.0,
           (double)total_b / 1048576.0, (double)need / 1048576.0);
    if (need > (1ull << 30) || need + (256u << 20) > free_b) {
        fprintf(stderr, "not enough free VRAM, or more than the 1 GiB this bench allows itself\n");
        exit(2);
    }
}

static const tr_gguf_tensor *find_2d(tr_gguf *m, const char *name, tr_type type, int64_t rows, int64_t cols) {
    const tr_gguf_tensor *t = tr_gguf_find_tensor(m, name);
    if (t == NULL || t->type != type || (int64_t)t->ne[0] != cols || (int64_t)t->ne[1] != rows ||
        t->ne[2] != 1 || t->ne[3] != 1) {
        fprintf(stderr, "%s: missing, or not %s %" PRId64 " x %" PRId64 "\n", name, tr_type_get(type)->name, rows,
                cols);
        exit(2);
    }
    return t;
}

static void read_2d(tr_gguf *m, const char *name, tr_type type, int64_t rows, int64_t cols, void *dst) {
    const tr_gguf_tensor *t = find_2d(m, name, type, rows, cols);
    if (tr_gguf_read(m, t, dst) != 0) {
        fprintf(stderr, "%s: read failed\n", name);
        exit(2);
    }
}

static const char *const ROLE[4] = {"attn_q", "attn_k", "attn_v", "attn_output"};

/* ---- check ------------------------------------------------------------------------------------ */

typedef struct {
    const unsigned char *w;
    size_t row_bytes;
    int64_t cols;
    const float *x;
    float *y;
    float (*dot)(const void *row, const float *x, int64_t n);
} ref_ctx;

static void ref_body(void *ctx_, int64_t begin, int64_t end, int worker) {
    const ref_ctx *c = ctx_;
    for (int64_t r = begin; r < end; r++) c->y[r] = c->dot(c->w + (size_t)r * c->row_bytes, c->x, c->cols);
}

/* y = the scalar dot_row of every row: the definition */
static void ref_matvec(tr_pool *pool, const unsigned char *w, int64_t rows, int64_t cols, const float *x, float *y) {
    ref_ctx c = {w, tr_row_bytes(TR_TYPE_Q8_0, cols), cols, x, y, tr_kernels_tier("scalar")->dot_row[TR_TYPE_Q8_0]};
    tr_parallel_for(pool, rows, 64, ref_body, &c);
}

typedef struct {
    int64_t compared, mismatches, active_mismatches, nan_pairs, infs, products, sentinel_bad;
    int64_t sub_scales, zero_scales, q_min, rows_past_end;
} tally;

typedef struct {
    gpu_t *g;
    tr_pool *pool;
    CUdeviceptr dw, dx, dy;
    float *y_gpu, *y_ref, *y_act;
    tally t;
} checker;

/* One product on the GPU, on the CPU twice, every output compared as bits. */
static void check_one(checker *c, const char *what, const unsigned char *w, int64_t rows, int64_t cols,
                      const float *x) {
    CU(cu.MemcpyHtoD(c->dx, x, (size_t)cols * sizeof(float)));
    CU(cu.MemsetD32(c->dy, SENTINEL, (size_t)(rows + TAIL)));
    launch_mv(c->g, c->dw, c->dx, c->dy, rows, cols);
    CU(cu.StreamSynchronize(c->g->st));
    CU(cu.MemcpyDtoH(c->y_gpu, c->dy, (size_t)(rows + TAIL) * sizeof(float)));
    ref_matvec(c->pool, w, rows, cols, x, c->y_ref);
    tr_mat m = {TR_TYPE_Q8_0, rows, cols, w};
    tr_matmul(c->pool, &m, x, 1, c->y_act);
    int shown = 0;
    for (int64_t r = 0; r < rows; r++) {
        float a = c->y_gpu[r], ref = c->y_ref[r], act = c->y_act[r];
        c->t.compared++;
        if (isinf(ref)) c->t.infs++;
        if (isnan(ref) && isnan(a)) c->t.nan_pairs++;
        else if (fbits(a) != fbits(ref)) {
            c->t.mismatches++;
            if (shown++ < 3)
                printf("  MISMATCH %s row %" PRId64 ": gpu %08" PRIX32 " (%.9g), cpu %08" PRIX32 " (%.9g)\n", what, r,
                       fbits(a), (double)a, fbits(ref), (double)ref);
        }
        if (!(isnan(ref) && isnan(act)) && fbits(act) != fbits(ref)) c->t.active_mismatches++;
    }
    for (int i = 0; i < TAIL; i++)
        if (fbits(c->y_gpu[rows + i]) != SENTINEL) c->t.sentinel_bad++;
    c->t.products++;
    c->t.rows_past_end += (ROWS_PER_CTA - rows % ROWS_PER_CTA) % ROWS_PER_CTA;
}

enum { V_UNIFORM, V_OUTLIERS, V_SPECIAL, V_HUGE };

static void make_vector(int kind, uint64_t *s, float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        switch (kind) {
        case V_UNIFORM: x[i] = unif(s); break;
        case V_OUTLIERS: { /* ~N(0, 1) (Irwin-Hall), one channel in 64 fifty times larger */
            float v = 0.0f;
            for (int j = 0; j < 12; j++) v += 0.5f * (unif(s) + 1.0f);
            x[i] = (v - 6.0f) * (rng(s) % 64 == 0 ? 50.0f : 1.0f);
            break;
        }
        case V_SPECIAL: { /* +-0, subnormals, tiny normals (subnormal products), ordinary values */
            uint64_t r = rng(s);
            uint32_t sign = (uint32_t)(r >> 63) << 31;
            switch (r % 8) {
            case 0: x[i] = 0.0f; break;
            case 1: x[i] = -0.0f; break;
            case 2: x[i] = from_bits(sign | (uint32_t)(1 + (r >> 8) % 0x7FFFFF)); break;
            case 3: x[i] = from_bits(sign | 0x00800000u | (uint32_t)((r >> 8) & 0x7FFFFF)) * 3.0f; break;
            default: x[i] = unif(s); break;
            }
            break;
        }
        default: { /* V_HUGE: values near FLT_MAX in one element of 32: lanes overflow to +-inf */
            uint64_t r = rng(s);
            x[i] = r % 32 == 0 ? ((r >> 63) ? -1.0f : 1.0f) * 3.0e38f : unif(s);
            break;
        }
        }
    }
}

/* A synthetic Q8_0 matrix: random finite f16 scales (zero and subnormal ones included), random quants. */
static void make_matrix(uint64_t *s, unsigned char *w, int64_t rows, int64_t cols, tally *t) {
    int64_t nb = rows * (cols / TR_Q8_0_BLOCK_ELEMS);
    for (int64_t b = 0; b < nb; b++) {
        unsigned char *blk = w + (size_t)b * TR_Q8_0_BLOCK_BYTES;
        uint64_t r = rng(s);
        uint16_t h = (uint16_t)((r & 0x83FF) | (((r >> 16) % 31) << 10)); /* exponent 0..30: finite */
        if (b % 97 == 5) h &= 0x8000;                                     /* +-0 */
        if (b % 89 == 7) h &= 0x83FF;                                     /* subnormal */
        memcpy(blk, &h, 2);
        if ((h & 0x7C00) == 0 && (h & 0x3FF) != 0) t->sub_scales++;
        if ((h & 0x7FFF) == 0) t->zero_scales++;
        for (int j = 0; j < TR_Q8_0_BLOCK_ELEMS; j++) {
            blk[2 + j] = (unsigned char)(rng(s) >> 56);
            if (blk[2 + j] == 0x80) t->q_min++;
        }
    }
}

static void report(const char *group, const tally *before, const tally *now) {
    printf("  %-44s %8" PRId64 " floats, %" PRId64 " differ from dot_row (scalar), %" PRId64
           " between scalar and the active tier\n",
           group, now->compared - before->compared, now->mismatches - before->mismatches,
           now->active_mismatches - before->active_mismatches);
    fflush(stdout);
}

static int run_check(gpu_t *g, const char *model, tr_pool *pool) {
    const size_t head_bytes = tr_row_bytes(TR_TYPE_Q8_0, N_EMBD) * N_VOCAB;
    const size_t attn_bytes = tr_row_bytes(TR_TYPE_Q8_0, N_EMBD) * N_EMBD;
    ram_check(head_bytes + 4 * attn_bytes + 4 * (size_t)N_VOCAB * sizeof(float));
    checker c;
    memset(&c, 0, sizeof c);
    c.g = g;
    c.pool = pool;
    vram_check(head_bytes + (N_EMBD + N_VOCAB + TAIL) * sizeof(float));
    CU(cu.MemAlloc(&c.dw, head_bytes));
    CU(cu.MemAlloc(&c.dx, N_EMBD * sizeof(float)));
    CU(cu.MemAlloc(&c.dy, (N_VOCAB + TAIL) * sizeof(float)));
    c.y_gpu = xmalloc((N_VOCAB + TAIL) * sizeof(float));
    c.y_ref = xmalloc(N_VOCAB * sizeof(float));
    c.y_act = xmalloc(N_VOCAB * sizeof(float));
    float *x = xmalloc(N_EMBD * sizeof(float));
    uint64_t seed = 0x5EED0001;
    printf("reference: dot_row of the scalar tier (the definition) and tr_matmul on %s\n", tr_kernels_get()->tier);

    /* synthetic: the edges of the arithmetic and of the grid */
    static const int64_t shapes[][2] = {{37, 2048}, {33, 1024}, {16, 256}};
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
        int64_t rows = shapes[i][0], cols = shapes[i][1];
        size_t bytes = tr_row_bytes(TR_TYPE_Q8_0, cols) * (size_t)rows;
        unsigned char *w = xmalloc(bytes);
        tally before = c.t;
        for (int rep = 0; rep < 8; rep++) {
            make_matrix(&seed, w, rows, cols, &c.t);
            CU(cu.MemcpyHtoD(c.dw, w, bytes));
            for (int kind = V_UNIFORM; kind <= V_HUGE; kind++) {
                char what[64];
                snprintf(what, sizeof what, "synthetic %" PRId64 "x%" PRId64 " v%d", rows, cols, kind);
                make_vector(kind, &seed, x, cols);
                check_one(&c, what, w, rows, cols, x);
            }
        }
        char group[64];
        snprintf(group, sizeof group, "synthetic %" PRId64 " x %" PRId64 " (8 matrices x 4 vectors)", rows, cols);
        report(group, &before, &c.t);
        free(w);
    }

    /* the model */
    char err[256];
    tr_gguf *m = tr_gguf_open(model, err, sizeof err);
    if (m == NULL) {
        fprintf(stderr, "%s: %s\n", model, err);
        return 2;
    }
    float eps = 0.0f;
    if (tr_gguf_get_f32(m, "olmoe.attention.layer_norm_rms_epsilon", &eps) != 0) {
        fprintf(stderr, "%s: no olmoe.attention.layer_norm_rms_epsilon\n", model);
        return 2;
    }
    enum { N_TOK = 3 };
    static const int64_t tokens[N_TOK] = {1, 510, 12345};
    const size_t row_bytes = tr_row_bytes(TR_TYPE_Q8_0, N_EMBD);
    float *emb = xmalloc((size_t)N_TOK * N_EMBD * sizeof(float)), *norm = xmalloc(N_EMBD * sizeof(float));
    float *xr = xmalloc((size_t)N_TOK * N_EMBD * sizeof(float)), *xo = xmalloc((size_t)N_TOK * N_EMBD * sizeof(float));
    {
        const tr_gguf_tensor *te = find_2d(m, "token_embd.weight", TR_TYPE_Q8_0, N_VOCAB, N_EMBD);
        unsigned char *row = xmalloc(row_bytes);
        for (int i = 0; i < N_TOK; i++) {
            if (tr_gguf_read_range(m, te, (uint64_t)tokens[i] * row_bytes, row, row_bytes) != 0) return 2;
            tr_kernels_tier("scalar")->dequant_row[TR_TYPE_Q8_0](row, emb + (size_t)i * N_EMBD, N_EMBD);
        }
        free(row);
    }
    unsigned char *w = xmalloc(4 * attn_bytes);
    tally before = c.t;
    for (int layer = 0; layer < N_LAYER; layer++) {
        char name[64];
        for (int k = 0; k < 4; k++) {
            snprintf(name, sizeof name, "blk.%d.%s.weight", layer, ROLE[k]);
            read_2d(m, name, TR_TYPE_Q8_0, N_EMBD, N_EMBD, w + (size_t)k * attn_bytes);
        }
        snprintf(name, sizeof name, "blk.%d.attn_norm.weight", layer);
        const tr_gguf_tensor *tn = tr_gguf_find_tensor(m, name);
        if (tn == NULL || tn->type != TR_TYPE_F32 || tn->n_elems != N_EMBD || tr_gguf_read(m, tn, norm) != 0) {
            fprintf(stderr, "%s: missing or not f32 [%d]\n", name, N_EMBD);
            return 2;
        }
        /* the layer's input for these tokens (exact at layer 0), and W_v of it: the attention output of
         * position 0 */
        for (int i = 0; i < N_TOK; i++) {
            memcpy(xr + (size_t)i * N_EMBD, emb + (size_t)i * N_EMBD, N_EMBD * sizeof(float));
            tr_rmsnorm(xr + (size_t)i * N_EMBD, norm, N_EMBD, eps);
            ref_matvec(pool, w + 2 * attn_bytes, N_EMBD, N_EMBD, xr + (size_t)i * N_EMBD, xo + (size_t)i * N_EMBD);
        }
        for (int k = 0; k < 4; k++) {
            const unsigned char *wk = w + (size_t)k * attn_bytes;
            CU(cu.MemcpyHtoD(c.dw, wk, attn_bytes));
            snprintf(name, sizeof name, "blk.%d.%s", layer, ROLE[k]);
            for (int kind = V_UNIFORM; kind <= V_SPECIAL; kind++) {
                make_vector(kind, &seed, x, N_EMBD);
                check_one(&c, name, wk, N_EMBD, N_EMBD, x);
            }
            for (int i = 0; i < N_TOK; i++)
                check_one(&c, name, wk, N_EMBD, N_EMBD, (k == 3 ? xo : xr) + (size_t)i * N_EMBD);
        }
    }
    report("q, k, v, output projection of 16 layers (x 6)", &before, &c.t);

    before = c.t;
    {
        unsigned char *wh = xmalloc(head_bytes);
        free(w);
        w = NULL;
        read_2d(m, "output.weight", TR_TYPE_Q8_0, N_VOCAB, N_EMBD, wh);
        const tr_gguf_tensor *tn = tr_gguf_find_tensor(m, "output_norm.weight");
        if (tn == NULL || tn->type != TR_TYPE_F32 || tn->n_elems != N_EMBD || tr_gguf_read(m, tn, norm) != 0) {
            fprintf(stderr, "output_norm.weight: missing or not f32 [%d]\n", N_EMBD);
            return 2;
        }
        CU(cu.MemcpyHtoD(c.dw, wh, head_bytes));
        for (int kind = V_UNIFORM; kind <= V_SPECIAL; kind++) {
            make_vector(kind, &seed, x, N_EMBD);
            check_one(&c, "output", wh, N_VOCAB, N_EMBD, x);
        }
        for (int i = 0; i < N_TOK; i++) {
            memcpy(x, emb + (size_t)i * N_EMBD, N_EMBD * sizeof(float));
            tr_rmsnorm(x, norm, N_EMBD, eps);
            check_one(&c, "output", wh, N_VOCAB, N_EMBD, x);
        }
        free(wh);
    }
    report("output head 50304 x 2048 (x 6)", &before, &c.t);
    tr_gguf_close(m);

    printf("total: %" PRId64 " floats in %" PRId64 " products, %" PRId64 " differ from dot_row, %" PRId64
           " differ between the CPU tiers; %" PRId64 " NaN pairs, %" PRId64 " infinite results\n",
           c.t.compared, c.t.products, c.t.mismatches, c.t.active_mismatches, c.t.nan_pairs, c.t.infs);
    printf("cases seen: %" PRId64 " subnormal and %" PRId64 " zero scales, %" PRId64 " quants of -128, %" PRId64
           " thread rows past the end; %" PRId64 " sentinel floats overwritten\n",
           c.t.sub_scales, c.t.zero_scales, c.t.q_min, c.t.rows_past_end, c.t.sentinel_bad);
    int rc = 0;
    if (c.t.mismatches || c.t.active_mismatches || c.t.sentinel_bad) rc = 1;
    if (!c.t.compared || !c.t.sub_scales || !c.t.zero_scales || !c.t.q_min || !c.t.rows_past_end || !c.t.infs ||
        !c.t.nan_pairs) {
        printf("a covered case never occurred: the check proves less than it says\n");
        rc = 1;
    }
    free(x);
    free(xr);
    free(xo);
    free(emb);
    free(norm);
    free(c.y_gpu);
    free(c.y_ref);
    free(c.y_act);
    CU(cu.MemFree(c.dw));
    CU(cu.MemFree(c.dx));
    CU(cu.MemFree(c.dy));
    return rc;
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

static void line(const char *what, stats s, double bytes) {
    printf("  %-34s median %8.1f us  (min %8.1f, p90 %8.1f, max %8.1f)", what, s.med, s.lo, s.p90, s.hi);
    if (bytes > 0.0) printf("  %6.1f GB/s", bytes / (s.med * 1e3));
    printf("\n");
    fflush(stdout);
}

typedef struct {
    const char *name;
    int64_t rows;
    CUdeviceptr base;
    size_t stride; /* bytes between two copies */
    int copies;
} shape;

static int run_time(gpu_t *g, const char *model, int runs, double gap_us) {
    const size_t row_bytes = tr_row_bytes(TR_TYPE_Q8_0, N_EMBD);
    const size_t attn_bytes = row_bytes * N_EMBD, head_bytes = row_bytes * N_VOCAB;
    const size_t need = N_ATTN * attn_bytes + 2 * head_bytes;
    ram_check(head_bytes);
    vram_check(need + (N_EMBD + N_VOCAB) * sizeof(float));
    char err[256];
    tr_gguf *m = tr_gguf_open(model, err, sizeof err);
    if (m == NULL) {
        fprintf(stderr, "%s: %s\n", model, err);
        return 2;
    }
    /* the 64 attention matrices in one buffer, layer after layer in the order q, k, v, output: the q, k,
     * v of a layer are one 6144 x 2048 matrix; the output head and the embedding, same shape */
    CUdeviceptr d_attn, d_head, dx, dy;
    CU(cu.MemAlloc(&d_attn, N_ATTN * attn_bytes));
    CU(cu.MemAlloc(&d_head, 2 * head_bytes));
    CU(cu.MemAlloc(&dx, N_EMBD * sizeof(float)));
    CU(cu.MemAlloc(&dy, N_VOCAB * sizeof(float)));
    unsigned char *w = xmalloc(head_bytes);
    double t0 = tr_time_sec();
    for (int layer = 0; layer < N_LAYER; layer++)
        for (int k = 0; k < 4; k++) {
            char name[64];
            snprintf(name, sizeof name, "blk.%d.%s.weight", layer, ROLE[k]);
            read_2d(m, name, TR_TYPE_Q8_0, N_EMBD, N_EMBD, w);
            CU(cu.MemcpyHtoD(d_attn + (size_t)(4 * layer + k) * attn_bytes, w, attn_bytes));
        }
    read_2d(m, "output.weight", TR_TYPE_Q8_0, N_VOCAB, N_EMBD, w);
    CU(cu.MemcpyHtoD(d_head, w, head_bytes));
    read_2d(m, "token_embd.weight", TR_TYPE_Q8_0, N_VOCAB, N_EMBD, w);
    CU(cu.MemcpyHtoD(d_head + head_bytes, w, head_bytes));
    free(w);
    tr_gguf_close(m);
    printf("loaded %.0f MiB of real weights in %.1f s\n", (double)need / 1048576.0, tr_time_sec() - t0);

    void *p;
    CU(cu.MemAllocHost(&p, N_EMBD * sizeof(float)));
    float *hx = p;
    CU(cu.MemAllocHost(&p, N_VOCAB * sizeof(float)));
    float *hy = p;
    /* mapped pinned memory: the GPU reads x and writes y there itself, over PCIe */
    CUdeviceptr dxm, dym;
    CU(cu.MemHostAlloc(&p, N_EMBD * sizeof(float), CU_MEMHOSTALLOC_DEVICEMAP));
    float *hxm = p;
    CU(cu.MemHostGetDevicePointer(&dxm, hxm, 0));
    CU(cu.MemHostAlloc(&p, N_VOCAB * sizeof(float), CU_MEMHOSTALLOC_DEVICEMAP));
    float *hym = p;
    CU(cu.MemHostGetDevicePointer(&dym, hym, 0));
    uint64_t seed = 42;
    for (int i = 0; i < N_EMBD; i++) hx[i] = unif(&seed);
    CU(cu.MemcpyHtoD(dx, hx, N_EMBD * sizeof(float)));

    /* warm-up: half a second of products, so the clocks leave their idle state */
    t0 = tr_time_sec();
    for (int i = 0; tr_time_sec() - t0 < 0.5; i++) {
        launch_mv(g, d_head + (size_t)(i & 1) * head_bytes, dx, dy, N_VOCAB, N_EMBD);
        CU(cu.StreamSynchronize(g->st));
    }

    const shape shapes[] = {
        {"2048 x 2048 (q, k, v or output)", N_EMBD, d_attn, attn_bytes, N_ATTN},
        {"6144 x 2048 (q, k, v of a layer)", 3 * N_EMBD, d_attn, 4 * attn_bytes, N_LAYER},
        {"50304 x 2048 (output head)", N_VOCAB, d_head, head_bytes, 2},
    };
    double *x = xmalloc(MAX_RUNS * sizeof(double));
    const int warm = 16;
    for (size_t si = 0; si < sizeof shapes / sizeof shapes[0]; si++) {
        const shape *s = &shapes[si];
        double bytes = (double)row_bytes * (double)s->rows;
        printf("\n%s: %.2f MiB of weights a product\n", s->name, bytes / 1048576.0);
        int n = 0;
        for (int i = 0; i < warm + runs; i++) {
            CU(cu.EventRecord(g->ev[0], g->st));
            launch_mv(g, s->base + (size_t)(i % s->copies) * s->stride, dx, dy, s->rows, N_EMBD);
            CU(cu.EventRecord(g->ev[1], g->st));
            CU(cu.StreamSynchronize(g->st));
            if (i >= warm) x[n++] = ev_us(g);
        }
        line("(a) kernel", stat_of(x, n), bytes);
        n = 0;
        for (int i = 0; i < warm + runs; i++) {
            double a = tr_time_sec();
            CU(cu.MemcpyHtoDAsync(dx, hx, N_EMBD * sizeof(float), g->st));
            launch_mv(g, s->base + (size_t)(i % s->copies) * s->stride, dx, dy, s->rows, N_EMBD);
            CU(cu.MemcpyDtoHAsync(hy, dy, (size_t)s->rows * sizeof(float), g->st));
            CU(cu.StreamSynchronize(g->st));
            double b = tr_time_sec();
            if (i >= warm) x[n++] = 1e6 * (b - a);
        }
        line("(b) round trip (x in, y out, sync)", stat_of(x, n), bytes);
        /* where the round trip goes: the kernel and the wait without the copies, the copies without the kernel */
        n = 0;
        for (int i = 0; i < warm + runs; i++) {
            double a = tr_time_sec();
            launch_mv(g, s->base + (size_t)(i % s->copies) * s->stride, dx, dy, s->rows, N_EMBD);
            CU(cu.StreamSynchronize(g->st));
            double b = tr_time_sec();
            if (i >= warm) x[n++] = 1e6 * (b - a);
        }
        line("(b1) kernel + sync, no copies", stat_of(x, n), bytes);
        n = 0;
        for (int i = 0; i < warm + runs; i++) {
            double a = tr_time_sec();
            CU(cu.MemcpyHtoDAsync(dx, hx, N_EMBD * sizeof(float), g->st));
            CU(cu.MemcpyDtoHAsync(hy, dy, (size_t)s->rows * sizeof(float), g->st));
            CU(cu.StreamSynchronize(g->st));
            double b = tr_time_sec();
            if (i >= warm) x[n++] = 1e6 * (b - a);
        }
        line("(b2) copies + sync, no kernel", stat_of(x, n), 0.0);
        /* the round trip without the copy engine: the CPU writes x to mapped memory, copy16 pulls it into
         * VRAM, q8_mv stores y straight into mapped memory, one wait */
        n = 0;
        for (int i = 0; i < warm + runs; i++) {
            double a = tr_time_sec();
            memcpy(hxm, hx, N_EMBD * sizeof(float));
            launch_copy(g, dxm, dx, N_EMBD * sizeof(float));
            launch_mv(g, s->base + (size_t)(i % s->copies) * s->stride, dx, dym, s->rows, N_EMBD);
            CU(cu.StreamSynchronize(g->st));
            double b = tr_time_sec();
            if (i >= warm) x[n++] = 1e6 * (b - a);
        }
        line("(b3) round trip, compute engine only", stat_of(x, n), bytes);
        {
            size_t last = (size_t)((warm + runs - 1) % s->copies);
            CU(cu.MemcpyHtoD(dx, hx, N_EMBD * sizeof(float)));
            launch_mv(g, s->base + last * s->stride, dx, dy, s->rows, N_EMBD);
            CU(cu.StreamSynchronize(g->st));
            CU(cu.MemcpyDtoH(hy, dy, (size_t)s->rows * sizeof(float)));
            int64_t diff = 0;
            for (int64_t r = 0; r < s->rows; r++) diff += fbits(hym[r]) != fbits(hy[r]);
            printf("       (b3)'s last y against the copy engine's: %" PRId64 " of %" PRId64 " floats differ\n", diff,
                   s->rows);
            if (diff != 0) return 1;
        }
        /* the same with the CPU away between two calls, as the decode leaves it (MoE, attention) */
        n = 0;
        for (int i = 0; i < warm + runs; i++) {
            double a = tr_time_sec();
            memcpy(hxm, hx, N_EMBD * sizeof(float));
            launch_copy(g, dxm, dx, N_EMBD * sizeof(float));
            launch_mv(g, s->base + (size_t)(i % s->copies) * s->stride, dx, dym, s->rows, N_EMBD);
            CU(cu.StreamSynchronize(g->st));
            double b = tr_time_sec();
            if (i >= warm) x[n++] = 1e6 * (b - a);
            while (tr_time_sec() < b + 1e-6 * gap_us) {
            }
        }
        char what_d[64];
        snprintf(what_d, sizeof what_d, "(d) (b3) after %.0f us of CPU gap", gap_us);
        line(what_d, stat_of(x, n), bytes);
        n = 0;
        for (int i = 0; i < warm + runs; i++) {
            CU(cu.EventRecord(g->ev[0], g->st));
            launch_read(g, s->base + (size_t)(i % s->copies) * s->stride, (size_t)bytes);
            CU(cu.EventRecord(g->ev[1], g->st));
            CU(cu.StreamSynchronize(g->st));
            if (i >= warm) x[n++] = ev_us(g);
        }
        line("(c) plain read, same bytes", stat_of(x, n), bytes);
        if (s->copies > 2) { /* a token's worth: every layer's matrices, one after the other */
            n = 0;
            for (int i = 0; i < warm / 4 + runs; i++) {
                CU(cu.EventRecord(g->ev[0], g->st));
                for (int k = 0; k < s->copies; k++)
                    launch_mv(g, s->base + (size_t)k * s->stride, dx, dy, s->rows, N_EMBD);
                CU(cu.EventRecord(g->ev[1], g->st));
                CU(cu.StreamSynchronize(g->st));
                if (i >= warm / 4) x[n++] = ev_us(g) / s->copies;
            }
            char what[64];
            snprintf(what, sizeof what, "(e) %d back to back, per product", s->copies);
            line(what, stat_of(x, n), bytes);
        }
    }
    free(x);
    CU(cu.MemFreeHost(hx));
    CU(cu.MemFreeHost(hy));
    CU(cu.MemFreeHost(hxm));
    CU(cu.MemFreeHost(hym));
    CU(cu.MemFree(d_attn));
    CU(cu.MemFree(d_head));
    CU(cu.MemFree(dx));
    CU(cu.MemFree(dy));
    return 0;
}

/* ---- main ------------------------------------------------------------------------------------- */

static void run_info(gpu_t *g) {
    int v, clock = 0, mem_clock = 0, bus = 0, l2 = 0;
    size_t total = 0;
    CU(cu.DeviceTotalMem(&total, g->dev));
    if (cu.DeviceGetAttribute(&v, ATTR_CLOCK, g->dev) == 0) clock = v;
    if (cu.DeviceGetAttribute(&v, ATTR_MEM_CLOCK, g->dev) == 0) mem_clock = v;
    if (cu.DeviceGetAttribute(&v, ATTR_BUS, g->dev) == 0) bus = v;
    if (cu.DeviceGetAttribute(&v, ATTR_L2, g->dev) == 0) l2 = v;
    printf("%s, sm_%d%d, %d SMs, %.0f MiB, L2 %d KiB, bus %d bit, clock %d MHz, memory %d MHz (peak %.0f GB/s), "
           "driver %d.%d\n",
           g->name, g->cc_major, g->cc_minor, g->sms, (double)total / 1048576.0, l2 / 1024, bus, clock / 1000,
           mem_clock / 1000, 2.0 * mem_clock * 1e3 * (bus / 8) / 1e9, g->driver / 1000, (g->driver % 1000) / 10);
    const struct {
        const char *name;
        CUfunction f;
    } fs[] = {{"q8_mv", g->f_mv}, {"stream_read", g->f_read}, {"copy16", g->f_copy}};
    for (size_t i = 0; i < sizeof fs / sizeof fs[0]; i++) {
        int regs = 0, local = 0, shared = 0;
        CU(cu.FuncGetAttribute(&regs, FATTR_REGS, fs[i].f));
        CU(cu.FuncGetAttribute(&local, FATTR_LOCAL, fs[i].f));
        CU(cu.FuncGetAttribute(&shared, FATTR_SHARED, fs[i].f));
        printf("  %-12s %3d registers, %5d bytes shared, %d bytes local (spills)\n", fs[i].name, regs, shared, local);
    }
    fflush(stdout);
}

static int usage(void) {
    fprintf(stderr, "usage: bench_gpu_q8 info | check | time [--runs R] [--gap-us G]  [--model <file.gguf>]\n"
                    "       [--mutate norn|tree|order]\n");
    return 2;
}

int main(int argc, char **argv) {
    if (argc < 2) return usage();
    const char *mode = argv[1], *model = DEFAULT_MODEL;
    int runs = 200;
    double gap_us = 1000.0;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (strcmp(a, "--runs") == 0 && more) runs = atoi(argv[++i]);
        else if (strcmp(a, "--gap-us") == 0 && more) gap_us = atof(argv[++i]);
        else if (strcmp(a, "--model") == 0 && more) model = argv[++i];
        else if (strcmp(a, "--mutate") == 0 && more) {
            const char *mu = argv[++i];
            g_mut = strcmp(mu, "norn") == 0 ? MUT_NORN : strcmp(mu, "tree") == 0 ? MUT_TREE
                                                    : strcmp(mu, "order") == 0  ? MUT_ORDER
                                                                                : -1;
            if (g_mut < 0) return usage();
        } else return usage();
    }
    if (runs < 50 || runs > MAX_RUNS) {
        fprintf(stderr, "--runs 50..%d\n", MAX_RUNS);
        return 2;
    }
    tr_kernels_init();
    gpu_t g;
    gpu_open(&g);
    if (g_mut) printf("MUTATED kernel (--mutate): check must fail\n");
    int rc = 0;
    if (strcmp(mode, "info") == 0) {
        run_info(&g);
    } else if (strcmp(mode, "check") == 0) {
        run_info(&g);
        tr_pool *pool = tr_pool_create(0);
        if (pool == NULL) return 2;
        double t0 = tr_time_sec();
        rc = run_check(&g, model, pool);
        printf("check took %.1f s\n", tr_time_sec() - t0);
        tr_pool_destroy(pool);
    } else if (strcmp(mode, "time") == 0) {
        run_info(&g);
        rc = run_time(&g, model, runs, gap_us);
    } else {
        rc = usage();
    }
    gpu_close(&g);
    printf("%s\n", rc == 0 ? "ok" : rc == 1 ? "FAILED" : "error");
    return rc;
}
