/* cpu.c — runtime CPU feature and topology detection.
 *
 * x86-64 uses CPUID (via <cpuid.h>, available on gcc and clang alike, so the
 * same code compiles on MinGW-w64, Linux gcc and Linux clang) plus XGETBV to
 * confirm the OS actually saves AVX / AVX-512 register state before trusting
 * those CPUID bits. arm64 has no single instruction for this and instead asks
 * the OS (getauxval on Linux, sysctlbyname on macOS, IsProcessorFeaturePresent
 * on Windows). Detection runs once and is cached in a file-scope struct behind
 * a plain "already done" flag, not a mutex: this is race-free only because the
 * whole engine calls tr_cpu() for the first time from the main thread, before
 * the worker pool (and therefore any other thread) exists — tr_pool_create()
 * itself calls tr_cpu() to resolve n_threads <= 0, which establishes this. */

#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE
#endif

#include "cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#  define TR_CPU_X86_64 1
#  include <cpuid.h>
#  include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define TR_CPU_ARM64 1
#endif

#if defined(_WIN32)
#  include <windows.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#else
#  include <dirent.h>
#  include <unistd.h>
#  if defined(TR_CPU_ARM64)
#    include <sys/auxv.h>
#  endif
#endif

/* Kernel-stable arm64 hwcap bit positions (uapi/asm/hwcap.h). Defined here as
 * a fallback so detection still compiles against an older libc/kernel header
 * that has not caught up; the numeric values are part of the Linux ABI. */
#if defined(TR_CPU_ARM64) && !defined(_WIN32) && !defined(__APPLE__)
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#endif

#if defined(_WIN32) && defined(TR_CPU_ARM64)
#ifndef PF_ARM_V8_INSTRUCTIONS_AVAILABLE
#define PF_ARM_V8_INSTRUCTIONS_AVAILABLE 29
#endif
#endif

/* ---- x86-64 --------------------------------------------------------- */

#if defined(TR_CPU_X86_64)

/* Isolated in its own target-attributed function per cpu.h's requirement:
 * XGETBV must not execute unless CPUID already confirmed OSXSAVE, so this is
 * never called on a CPU/OS combination where the instruction would fault. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("xsave")))
#endif
static uint64_t read_xcr0(void) {
    return _xgetbv(0);
}

static void detect_x86(tr_cpu_info *c) {
    unsigned eax, ebx, ecx, edx;

    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        unsigned max_leaf = eax;
        memcpy(c->vendor + 0, &ebx, 4);
        memcpy(c->vendor + 4, &edx, 4);
        memcpy(c->vendor + 8, &ecx, 4);
        c->vendor[12] = '\0';

        int have_osxsave = 0, have_avx_os = 0, have_avx512_os = 0;

        if (max_leaf >= 1 && __get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            c->sse42 = (ecx >> 20) & 1u;
            c->fma = (ecx >> 12) & 1u;
            c->f16c = (ecx >> 29) & 1u;
            have_osxsave = (ecx >> 27) & 1u;
            int avx_bit = (ecx >> 28) & 1u;

            if (have_osxsave && avx_bit) {
                uint64_t xcr0 = read_xcr0();
                have_avx_os = (xcr0 & 0x6) == 0x6;                 /* SSE + AVX state (bits 1,2) */
                have_avx512_os = have_avx_os && (xcr0 & 0xE0) == 0xE0; /* + opmask/ZMM state (bits 5,6,7) */
            }
            c->avx = have_avx_os ? 1u : 0u;
        }

        if (max_leaf >= 7) {
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                c->avx2 = have_avx_os && ((ebx >> 5) & 1u);
                c->avx512f = have_avx512_os && ((ebx >> 16) & 1u);
                c->avx512dq = have_avx512_os && ((ebx >> 17) & 1u);
                c->avx512bw = have_avx512_os && ((ebx >> 30) & 1u);
                c->avx512vl = have_avx512_os && ((ebx >> 31) & 1u);
                c->avx512vnni = have_avx512_os && ((ecx >> 11) & 1u);
            }
            if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
                c->avxvnni = have_avx_os && ((eax >> 4) & 1u);
                c->avx512bf16 = have_avx512_os && ((eax >> 5) & 1u);
            }
        }

        if (__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx) && eax >= 0x80000004u) {
            unsigned brand[12];
            __get_cpuid(0x80000002u, &brand[0], &brand[1], &brand[2], &brand[3]);
            __get_cpuid(0x80000003u, &brand[4], &brand[5], &brand[6], &brand[7]);
            __get_cpuid(0x80000004u, &brand[8], &brand[9], &brand[10], &brand[11]);
            char raw[49];
            memcpy(raw, brand, 48);
            raw[48] = '\0';

            const char *start = raw;
            while (*start == ' ') start++;
            size_t len = strlen(start);
            while (len > 0 && start[len - 1] == ' ') len--;
            if (len >= sizeof c->brand) len = sizeof c->brand - 1;
            memcpy(c->brand, start, len);
            c->brand[len] = '\0';
        }
    }
}

#endif /* TR_CPU_X86_64 */

/* ---- arm64 ------------------------------------------------------------ */

#if defined(TR_CPU_ARM64)

static void detect_arm64(tr_cpu_info *c) {
    c->neon = 1; /* mandatory baseline on arm64 */

#if defined(_WIN32)
    /* Windows exposes no documented PF_* flag for dot-product / I8MM / SVE
     * detection; left at 0 (conservative: falls back to plain NEON kernels).
     * Not exercised by this project's build/test matrix (x86-64 only). */
    (void)IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE);
    c->dotprod = 0;
    c->i8mm = 0;
    c->sve = 0;
#elif defined(__APPLE__)
    int val = 0;
    size_t len = sizeof val;
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &val, &len, NULL, 0) == 0)
        c->dotprod = val ? 1u : 0u;
    val = 0;
    len = sizeof val;
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &val, &len, NULL, 0) == 0)
        c->i8mm = val ? 1u : 0u;
    c->sve = 0; /* no SVE on Apple silicon */
#else
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    c->dotprod = (hwcap & HWCAP_ASIMDDP) ? 1u : 0u;
    c->sve = (hwcap & HWCAP_SVE) ? 1u : 0u;
    c->i8mm = (hwcap2 & HWCAP2_I8MM) ? 1u : 0u;
#endif
}

#endif /* TR_CPU_ARM64 */

/* ---- vendor/brand fallback for non-x86 ------------------------------- */

#if defined(TR_CPU_ARM64)
static void fill_vendor_brand(tr_cpu_info *c) {
#if defined(__APPLE__)
    strncpy(c->vendor, "Apple", sizeof c->vendor - 1);
    size_t len = sizeof c->brand;
    if (sysctlbyname("machdep.cpu.brand_string", c->brand, &len, NULL, 0) != 0)
        c->brand[0] = '\0';
#else
    c->vendor[0] = '\0';
    c->brand[0] = '\0';
#endif
}
#endif

/* ---- core topology ----------------------------------------------------- */

#if defined(_WIN32)

static void detect_cores(tr_cpu_info *c) {
    c->logical_cores = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (c->logical_cores < 1) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        c->logical_cores = (int)si.dwNumberOfProcessors;
    }

    int physical = 0;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (len > 0) {
        char *buf = malloc(len);
        if (buf != NULL) {
            if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                    (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
                DWORD off = 0;
                while (off < len) {
                    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX rec =
                        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
                    if (rec->Relationship == RelationProcessorCore) physical++;
                    off += rec->Size;
                }
            }
            free(buf);
        }
    }

    c->physical_cores = (physical > 0) ? physical : c->logical_cores;
    if (c->physical_cores < 1) c->physical_cores = 1;
    if (c->logical_cores < 1) c->logical_cores = 1;
}

#elif defined(__APPLE__)

static void detect_cores(tr_cpu_info *c) {
    int v = 0;
    size_t len = sizeof v;
    c->logical_cores = (sysctlbyname("hw.logicalcpu", &v, &len, NULL, 0) == 0 && v > 0) ? v : 1;
    v = 0;
    len = sizeof v;
    c->physical_cores = (sysctlbyname("hw.physicalcpu", &v, &len, NULL, 0) == 0 && v > 0) ? v : c->logical_cores;
}

#else /* Linux */

typedef struct { long pkg, core; } pkg_core_t;

static void detect_cores(tr_cpu_info *c) {
    long logical = sysconf(_SC_NPROCESSORS_ONLN);
    c->logical_cores = (logical > 0) ? (int)logical : 1;

    pkg_core_t seen[1024];
    int n_seen = 0;

    DIR *dir = opendir("/sys/devices/system/cpu");
    if (dir != NULL) {
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            const char *name = ent->d_name;
            if (strncmp(name, "cpu", 3) != 0) continue;
            const char *digits = name + 3;
            if (*digits == '\0') continue;
            int all_digits = 1;
            for (const char *q = digits; *q; q++) {
                if (*q < '0' || *q > '9') { all_digits = 0; break; }
            }
            if (!all_digits) continue;

            char path[512];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/%s/topology/physical_package_id", name);
            FILE *fp = fopen(path, "r");
            long pkg = -1;
            if (fp != NULL) {
                if (fscanf(fp, "%ld", &pkg) != 1) pkg = -1;
                fclose(fp);
            }

            snprintf(path, sizeof path, "/sys/devices/system/cpu/%s/topology/core_id", name);
            fp = fopen(path, "r");
            long core = -1;
            if (fp != NULL) {
                if (fscanf(fp, "%ld", &core) != 1) core = -1;
                fclose(fp);
            }

            if (pkg < 0 || core < 0) continue;

            int dup = 0;
            for (int i = 0; i < n_seen; i++) {
                if (seen[i].pkg == pkg && seen[i].core == core) { dup = 1; break; }
            }
            if (!dup && n_seen < (int)(sizeof seen / sizeof seen[0])) {
                seen[n_seen].pkg = pkg;
                seen[n_seen].core = core;
                n_seen++;
            }
        }
        closedir(dir);
    }

    c->physical_cores = (n_seen > 0) ? n_seen : c->logical_cores;
    if (c->physical_cores < 1) c->physical_cores = 1;
}

#endif

/* ---- TR_CPU_MAX cap ------------------------------------------------- */

static void apply_cpu_max(tr_cpu_info *c) {
    const char *v = getenv("TR_CPU_MAX");
    if (v == NULL) return;

    if (strcmp(v, "scalar") == 0) {
        c->sse42 = c->avx = c->avx2 = c->fma = c->f16c = 0;
        c->avx512f = c->avx512bw = c->avx512vl = c->avx512dq = 0;
        c->avx512vnni = c->avx512bf16 = c->avxvnni = 0;
        c->neon = c->dotprod = c->i8mm = c->sve = 0;
    } else if (strcmp(v, "avx2") == 0) {
        c->avx512f = c->avx512bw = c->avx512vl = c->avx512dq = 0;
        c->avx512vnni = c->avx512bf16 = 0;
    } else if (strcmp(v, "neon") == 0) {
        c->dotprod = c->i8mm = c->sve = 0;
    }
}

/* ---- cache and public API -------------------------------------------- */

static tr_cpu_info g_cpu;
static int g_detected = 0; /* see the file header comment for the race-free assumption */

const tr_cpu_info *tr_cpu(void) {
    if (!g_detected) {
        memset(&g_cpu, 0, sizeof g_cpu);
#if defined(TR_CPU_X86_64)
        g_cpu.arch = TR_ARCH_X86_64;
        detect_x86(&g_cpu);
#elif defined(TR_CPU_ARM64)
        g_cpu.arch = TR_ARCH_ARM64;
        fill_vendor_brand(&g_cpu);
        detect_arm64(&g_cpu);
#else
        g_cpu.arch = TR_ARCH_OTHER;
#endif
        detect_cores(&g_cpu);
        apply_cpu_max(&g_cpu);
        g_detected = 1;
    }
    return &g_cpu;
}

void tr_cpu_describe(const tr_cpu_info *c, char *buf, int buf_len) {
    if (buf == NULL || buf_len <= 0) return;

    const char *arch_name = (c->arch == TR_ARCH_X86_64) ? "x86-64"
                            : (c->arch == TR_ARCH_ARM64) ? "arm64"
                            : "unknown";
    const char *name = (c->brand[0] != '\0') ? c->brand
                       : (c->vendor[0] != '\0') ? c->vendor
                       : "unknown CPU";

    int n = snprintf(buf, (size_t)buf_len, "%s %s, %d cores / %d threads,", arch_name, name,
                      c->physical_cores, c->logical_cores);
    if (n < 0) n = 0;
    if (n >= buf_len) return;

    char *p = buf + n;
    int remaining = buf_len - n;

#define APPEND_FLAG(flag, text) \
    do { \
        if ((flag) && remaining > 1) { \
            int w = snprintf(p, (size_t)remaining, " %s", (text)); \
            if (w > 0) { \
                if (w >= remaining) w = remaining - 1; \
                p += w; \
                remaining -= w; \
            } \
        } \
    } while (0)

    if (c->arch == TR_ARCH_X86_64) {
        APPEND_FLAG(c->sse42, "sse4.2");
        APPEND_FLAG(c->avx, "avx");
        APPEND_FLAG(c->avx2, "avx2");
        APPEND_FLAG(c->fma, "fma");
        APPEND_FLAG(c->f16c, "f16c");
        APPEND_FLAG(c->avxvnni, "avxvnni");
        APPEND_FLAG(c->avx512f, "avx512f");
        APPEND_FLAG(c->avx512dq, "avx512dq");
        APPEND_FLAG(c->avx512bw, "avx512bw");
        APPEND_FLAG(c->avx512vl, "avx512vl");
        APPEND_FLAG(c->avx512vnni, "avx512vnni");
        APPEND_FLAG(c->avx512bf16, "avx512bf16");
    } else if (c->arch == TR_ARCH_ARM64) {
        APPEND_FLAG(c->neon, "neon");
        APPEND_FLAG(c->dotprod, "dotprod");
        APPEND_FLAG(c->i8mm, "i8mm");
        APPEND_FLAG(c->sve, "sve");
    }

#undef APPEND_FLAG
}
