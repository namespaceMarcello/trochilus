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

typedef struct {
    tr_arch arch;
    char vendor[16];            /* "AuthenticAMD", "GenuineIntel", "Apple", ... or "" */
    char brand[64];             /* marketing name, or "" if unknown */
    int logical_cores;
    int physical_cores;         /* SMT siblings counted once; >= 1 */

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
 * AVX-512 ones, "neon" clears dotprod/i8mm/sve. */
const tr_cpu_info *tr_cpu(void);

/* One line such as "x86-64 AMD Ryzen 9 7940HX, 16 cores / 32 threads, avx2 fma avx512f ... avx512vnni". */
void tr_cpu_describe(const tr_cpu_info *c, char *buf, int buf_len);

#endif
