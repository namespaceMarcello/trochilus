/* cpu.h — what this processor can do, detected once at runtime.
 *
 * Kernels are chosen from these flags, never from compile-time -march, so one
 * binary uses AVX-512 VNNI where it exists and falls back elsewhere. An x86
 * AVX / AVX-512 flag is set only if the OS also saves that register state
 * (XGETBV), otherwise the instructions would fault. */
#ifndef TR_CPU_H
#define TR_CPU_H

typedef enum {
    TR_ARCH_OTHER = 0,
    TR_ARCH_X86_64,
    TR_ARCH_ARM64,
} tr_arch;

/* At most this many logical processors are placed; beyond it the extra ones are
 * left to the scheduler. 256 covers every machine the CPU path targets. */
#define TR_CPU_MAX_SLOTS 256

/* SMT siblings of one physical core that are kept; beyond this they are ignored (x86 has 2,
 * POWER up to 8, and nothing measured here needs more than 4). */
#define TR_CPU_MAX_SMT 4

/* One logical processor. `group` is the Windows processor group (always 0 elsewhere), `lcpu`
 * the processor index inside that group (the global index elsewhere), and `core[0..n_core)`
 * every processor of the physical core it belongs to, `core[0] == lcpu`: pinning to the one
 * processor or to the whole core are both one call away. */
typedef struct {
    unsigned short group;
    unsigned short lcpu;
    unsigned short core[TR_CPU_MAX_SMT];
    unsigned char n_core;
} tr_cpu_slot;

typedef struct {
    tr_arch arch;
    char vendor[16];            /* "AuthenticAMD", "GenuineIntel", "Apple", ... or "" */
    char brand[64];             /* marketing name, or "" if unknown */
    int logical_cores;
    int physical_cores;         /* SMT siblings counted once; >= 1 */

    /* Where to put the n-th thread of a pool, best first: one logical processor per
     * physical core, cores taken round robin over last-level caches, SMT siblings only
     * after every core already has a thread. Restricted to the processors this process
     * is allowed to run on, so an outer `taskset` / `start /affinity` still decides.
     * n_slots is 0 when the topology is unknown or the platform has no affinity: then
     * nothing is pinned and the scheduler places the threads (docs/MISURE.md
     * "Dove vanno i thread"). */
    int n_slots;
    tr_cpu_slot slot[TR_CPU_MAX_SLOTS];

    /* x86-64 */
    unsigned sse42 : 1, avx : 1, avx2 : 1, fma : 1, f16c : 1;
    unsigned avx512f : 1, avx512bw : 1, avx512vl : 1, avx512dq : 1;
    unsigned avx512vnni : 1, avx512bf16 : 1, avxvnni : 1;

    /* arm64 */
    unsigned neon : 1, dotprod : 1, i8mm : 1, sve : 1;
} tr_cpu_info;

/* Detected on first call, then cached; thread-safe after the first call returns.
 * The environment variable TR_CPU_MAX (read once, here) caps the kernel tier for
 * testing and measurement: "scalar" clears every SIMD flag, "avx2" clears the
 * AVX-512 ones, "neon" clears dotprod/i8mm/sve. Any other value is reported on the
 * log and ignored (tools/tier_check.sh proves in the gate that a cap took effect). */
const tr_cpu_info *tr_cpu(void);

/* One line such as "x86-64 AMD Ryzen 9 7940HX, 16 cores / 32 threads, avx2 fma avx512f ... avx512vnni". */
void tr_cpu_describe(const tr_cpu_info *c, char *buf, int buf_len);

#endif
