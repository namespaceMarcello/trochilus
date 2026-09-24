/* gpu_attn.h — a decode token's attention on an NVIDIA GPU, with the CPU's bits
 * (docs/MEASUREMENTS.md question 51, "The decode's attention on the GPU").
 *
 * The driver is opened at run time (tr_lib_open: nvcuda.dll, libcuda.so.1): the core links
 * nothing, and a machine without it runs as before. The kernels are PTX that gpu_attn.c writes
 * and the driver compiles. Every float operation carries its rounding (.rn: never fused into an
 * fma), the exponential is tr_expf's own arithmetic, the sums keep the 16-lane contract and the
 * order of the positions: one decode token's attention gives, head by head, the bits
 * tr_attention_group gives it (tests/test_gpu_attn.c, and on the real model tools/gpu_exact.sh).
 *
 * A session keeps a copy of its keys and values in VRAM, written as it writes its own cache
 * (src/kv/kv.h), so a decode token sends only its query, key and value and reads back its output,
 * through pinned memory the kernels read and write themselves (no copy engine: ~57 µs a layer
 * less). A laptop GPU drops its clocks in the CPU's gap between two layers (~1.6 ms): every
 * decode call leaves one warp spinning on a second stream for TR_GPU_WARM_NS, so the next layer
 * finds the GPU awake, and the warp stops by itself when the decode stops. Measured on the RTX
 * 4070 Laptop: 185 µs a layer at 2048 positions this way, ~390 without the warp, 704 on the CPU.
 *
 * Supported: head_dim 128, one KV head per query head, compute capability 8.0 or later. Anything
 * else is refused when the session's cache is created, and the session stays on the CPU. */
#ifndef TR_GPU_ATTN_H
#define TR_GPU_ATTN_H

#include <stddef.h>
#include <stdint.h>

#define TR_GPU_WARM_NS 3000000 /* how long a call keeps the GPU awake after it: > the CPU's gap */
/* the longest tr_gpu_attn_warm: a kernel running near 2 s trips Windows' timeout (TDR) */
#define TR_GPU_WARM_MAX_NS 200000000
#define TR_GPU_VRAM_RESERVE ((uint64_t)512 << 20) /* VRAM a cache must leave free */

typedef struct tr_gpu tr_gpu;             /* the device, its context, the compiled kernels: one per model */
typedef struct tr_gpu_attn tr_gpu_attn;   /* one session's cache in VRAM and its pinned buffers */

/* The first CUDA device of compute capability 8.0 or later, the kernels compiled for it. NULL
 * with a message in err when there is no driver, no such device, or the compilation fails. */
tr_gpu *tr_gpu_open(char *err, size_t err_len);
void tr_gpu_close(tr_gpu *g); /* NULL: nothing */
/* The device's name, for the command line's report. */
const char *tr_gpu_name(const tr_gpu *g);

/* A cache of n_ctx positions for every layer and head, and the buffers of one decode call. NULL
 * with a message when the shape is not supported or VRAM would keep less than
 * TR_GPU_VRAM_RESERVE free. One session, one thread at a time (the calls make the context
 * current on the calling thread). */
tr_gpu_attn *tr_gpu_attn_create(tr_gpu *g, int64_t n_layers, int64_t n_head, int64_t head_dim, int64_t n_ctx,
                                char *err, size_t err_len);
void tr_gpu_attn_free(tr_gpu_attn *a); /* NULL: nothing */
/* Bytes of VRAM tr_gpu_attn_create takes for that shape. */
uint64_t tr_gpu_attn_bytes(int64_t n_layers, int64_t n_head, int64_t head_dim, int64_t n_ctx);

/* hot: begin */

/* Positions pos0 .. pos0 + n_tok - 1 of one layer, k and v as tr_kv_write takes them
 * ([n_tok][n_head * head_dim] each). Returns when the GPU holds them. 0, or -1 on a driver error
 * or a range outside the cache. */
int tr_gpu_attn_write(tr_gpu_attn *a, int64_t layer, int64_t pos0, int64_t n_tok, const float *k, const float *v);

/* One decode token of one layer at position pos: q, k, v its rows ([n_head * head_dim] each,
 * after the norms and RoPE, as the CPU's attention reads them). Stores k and v at pos in the VRAM
 * cache, attends to positions 0 .. pos and writes out [n_head * head_dim]: for every head h the
 * bits tr_attention_group(q + h*head_dim, ..., n_q = 1, first_n_pos = pos + 1, head_dim, scale, ...)
 * writes. 0, or -1 on a driver error or pos outside the cache: the caller then stays on the CPU,
 * whose own cache is always whole. */
int tr_gpu_attn_decode(tr_gpu_attn *a, int64_t layer, int64_t pos, const float *q, const float *k, const float *v,
                       float scale, float *out);

/* Keeps the GPU awake for ns nanoseconds (at most TR_GPU_WARM_MAX_NS) or until the next decode
 * call starts. After a prompt the GPU has idled for seconds and its clocks take hundreds of ms to
 * climb back: the first decode tokens paid it (a layer at ~3000 positions: 436-694 µs over the first
 * 40 tokens, 315 over 400; docs/MEASUREMENTS.md "The decode's attention on the GPU, in the engine").
 * The last pass of a prompt calls it layer by layer, so decode starts at full clocks. 0, or -1 on a
 * driver error. */
int tr_gpu_attn_warm(tr_gpu_attn *a, uint64_t ns);

/* hot: end */

#endif
