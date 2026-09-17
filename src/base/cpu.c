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

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* sched_getaffinity and the CPU_* macros */
#endif
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
#  include <sched.h>
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

/* One physical core: its logical processors, and which last-level cache it shares.
 * `domain` only has to separate the caches from one another, its value means nothing. */
typedef struct {
    int domain;
    int n_cpu;
    tr_cpu_slot cpu[TR_CPU_MAX_SMT];
} core_desc;

static void core_add_cpu(core_desc *co, unsigned group, unsigned lcpu) {
    if (co->n_cpu >= TR_CPU_MAX_SMT) return;
    co->cpu[co->n_cpu].group = (unsigned short)group;
    co->cpu[co->n_cpu].lcpu = (unsigned short)lcpu;
    co->n_cpu++;
    /* every sibling carries the whole core: slot.core[0] stays the sibling itself, so
     * pinning to one processor and pinning to the core are the same call with a different n */
    for (int i = 0; i < co->n_cpu; i++) {
        tr_cpu_slot *s = &co->cpu[i];
        s->core[0] = s->lcpu;
        s->n_core = 1;
        for (int j = 0; j < co->n_cpu; j++)
            if (co->cpu[j].lcpu != s->lcpu && s->n_core < TR_CPU_MAX_SMT)
                s->core[s->n_core++] = co->cpu[j].lcpu;
    }
}

/* Fills c->slot from the cores, best placement first: one logical processor per physical
 * core before any second sibling (two threads on one core measured 30% slower than one
 * per core), and the cores taken round robin over the last-level caches (4+4 over the two
 * chiplets of a 7940HX beat 8 on one by 7-9%). Both numbers: docs/MISURE.md. */
static void order_slots(tr_cpu_info *c, const core_desc *cores, int n_cores) {
    c->n_slots = 0;
    if (n_cores <= 0) return;

    int domains[TR_CPU_MAX_SLOTS], n_domains = 0;
    for (int i = 0; i < n_cores; i++) {
        int seen = 0;
        for (int d = 0; d < n_domains; d++)
            if (domains[d] == cores[i].domain) { seen = 1; break; }
        if (!seen && n_domains < TR_CPU_MAX_SLOTS) domains[n_domains++] = cores[i].domain;
    }

    /* Cores in round-robin order over the domains, keeping each domain's own order. */
    int order[TR_CPU_MAX_SLOTS], n_order = 0;
    for (int rank = 0; n_order < n_cores; rank++) {
        int added = 0;
        for (int d = 0; d < n_domains; d++) {
            int seen = 0;
            for (int i = 0; i < n_cores; i++) {
                if (cores[i].domain != domains[d]) continue;
                if (seen++ != rank) continue;
                order[n_order++] = i;
                added = 1;
                break;
            }
        }
        if (!added) break; /* every domain is exhausted */
    }

    for (int s = 0; s < TR_CPU_MAX_SMT; s++)
        for (int k = 0; k < n_order && c->n_slots < TR_CPU_MAX_SLOTS; k++)
            if (s < cores[order[k]].n_cpu) c->slot[c->n_slots++] = cores[order[k]].cpu[s];
}

#if defined(_WIN32)

/* The processors this process may run on, so an outer `start /affinity` still decides.
 * Only asked for a single-group machine: GetProcessAffinityMask cannot describe more. */
static int win_process_mask(KAFFINITY *out) {
    DWORD_PTR proc = 0, sys = 0;
    if (GetActiveProcessorGroupCount() != 1) return 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) || proc == 0) return 0;
    *out = (KAFFINITY)proc;
    return 1;
}

/* Reads one relationship into a malloc'd buffer; *len is its size. NULL on failure. */
static char *win_proc_info(LOGICAL_PROCESSOR_RELATIONSHIP rel, DWORD *len) {
    *len = 0;
    GetLogicalProcessorInformationEx(rel, NULL, len);
    if (*len == 0) return NULL;
    char *buf = malloc(*len);
    if (buf == NULL) return NULL;
    if (!GetLogicalProcessorInformationEx(rel, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, len)) {
        free(buf);
        return NULL;
    }
    return buf;
}

/* Index of the last-level cache that holds (group, lcpu), or -1. */
static int win_cache_domain(const char *caches, DWORD len, unsigned group, unsigned lcpu) {
    int best = -1, idx = 0;
    BYTE best_level = 0;
    for (DWORD off = 0; off + sizeof(DWORD) * 2 <= len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX rec =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(caches + off);
        if (rec->Size == 0 || off + rec->Size > len) break;
        if (rec->Relationship == RelationCache && rec->Cache.Level >= 2) {
            const GROUP_AFFINITY *ga = &rec->Cache.GroupMask;
            if (ga->Group == group && (ga->Mask & ((KAFFINITY)1 << lcpu)) != 0 &&
                rec->Cache.Level >= best_level) {
                best_level = rec->Cache.Level;
                best = idx;
            }
        }
        if (rec->Relationship == RelationCache) idx++;
        off += rec->Size;
    }
    return best;
}

static void detect_cores(tr_cpu_info *c) {
    c->logical_cores = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (c->logical_cores < 1) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        c->logical_cores = (int)si.dwNumberOfProcessors;
    }

    KAFFINITY allowed = 0;
    int have_allowed = win_process_mask(&allowed);

    DWORD cores_len = 0, caches_len = 0;
    char *cores_buf = win_proc_info(RelationProcessorCore, &cores_len);
    char *caches_buf = win_proc_info(RelationCache, &caches_len);

    core_desc *cores = calloc(TR_CPU_MAX_SLOTS, sizeof *cores);
    int n_cores = 0, physical = 0;

    if (cores_buf != NULL) {
        for (DWORD off = 0; off + sizeof(DWORD) * 2 <= cores_len;) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX rec =
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(cores_buf + off);
            if (rec->Size == 0 || off + rec->Size > cores_len) break; /* truncated record */
            if (rec->Relationship == RelationProcessorCore) {
                physical++;
                if (cores != NULL && n_cores < TR_CPU_MAX_SLOTS) {
                    core_desc *co = &cores[n_cores];
                    co->domain = -1;
                    co->n_cpu = 0;
                    for (WORD g = 0; g < rec->Processor.GroupCount; g++) {
                        const GROUP_AFFINITY *ga = &rec->Processor.GroupMask[g];
                        for (unsigned bit = 0; bit < sizeof(KAFFINITY) * 8; bit++) {
                            if ((ga->Mask & ((KAFFINITY)1 << bit)) == 0) continue;
                            if (have_allowed && ga->Group == 0 &&
                                (allowed & ((KAFFINITY)1 << bit)) == 0) continue;
                            if (co->n_cpu == 0 && caches_buf != NULL)
                                co->domain = win_cache_domain(caches_buf, caches_len, ga->Group, bit);
                            core_add_cpu(co, ga->Group, bit);
                        }
                    }
                    if (co->n_cpu > 0) n_cores++; /* a core with no allowed processor is dropped */
                }
            }
            off += rec->Size;
        }
    }

    c->physical_cores = (physical > 0) ? physical : c->logical_cores;
    if (cores != NULL && n_cores > 0) {
        /* Counted on the processors this process may use: a restricted process must not
         * size its pool on cores it will never run on. */
        int usable = 0;
        for (int i = 0; i < n_cores; i++) usable += cores[i].n_cpu;
        c->physical_cores = n_cores;
        c->logical_cores = usable;
    }
    if (c->physical_cores < 1) c->physical_cores = 1;
    if (c->logical_cores < 1) c->logical_cores = 1;

    if (cores != NULL) order_slots(c, cores, n_cores);
    free(cores);
    free(cores_buf);
    free(caches_buf);
}

#elif defined(__APPLE__)

static void detect_cores(tr_cpu_info *c) {
    int v = 0;
    size_t len = sizeof v;
    c->logical_cores = (sysctlbyname("hw.logicalcpu", &v, &len, NULL, 0) == 0 && v > 0) ? v : 1;
    v = 0;
    len = sizeof v;
    c->physical_cores = (sysctlbyname("hw.physicalcpu", &v, &len, NULL, 0) == 0 && v > 0) ? v : c->logical_cores;
    c->n_slots = 0; /* macOS has no way to pin a thread to a core */
}

#else /* Linux */

/* One number out of a sysfs file, or `def` when it is missing or unreadable. */
static long sysfs_long(const char *fmt, int cpu, const char *leaf, long def) {
    char path[256];
    snprintf(path, sizeof path, fmt, cpu, leaf);
    FILE *fp = fopen(path, "r");
    if (fp == NULL) return def;
    long v = def;
    if (fscanf(fp, "%ld", &v) != 1) v = def;
    fclose(fp);
    return v;
}

static void detect_cores(tr_cpu_info *c) {
    long logical = sysconf(_SC_NPROCESSORS_ONLN);
    c->logical_cores = (logical > 0) ? (int)logical : 1;

    cpu_set_t allowed;
    int have_allowed = sched_getaffinity(0, sizeof allowed, &allowed) == 0;

    core_desc *cores = calloc(TR_CPU_MAX_SLOTS, sizeof *cores);
    long pkg_of[TR_CPU_MAX_SLOTS], core_of[TR_CPU_MAX_SLOTS];
    int n_cores = 0;

    /* cpu ids can be sparse: walk the range instead of readdir, so siblings of one core
     * always come in cpu order and the placement is the same from one run to the next. */
    for (int cpu = 0; cpu < TR_CPU_MAX_SLOTS * TR_CPU_MAX_SMT && cores != NULL; cpu++) {
        long pkg = sysfs_long("/sys/devices/system/cpu/cpu%d/topology/%s", cpu, "physical_package_id", -1);
        long core = sysfs_long("/sys/devices/system/cpu/cpu%d/topology/%s", cpu, "core_id", -1);
        if (pkg < 0 || core < 0) continue;
        if (have_allowed && !CPU_ISSET(cpu, &allowed)) continue;

        int idx = -1;
        for (int i = 0; i < n_cores; i++)
            if (pkg_of[i] == pkg && core_of[i] == core) { idx = i; break; }
        if (idx < 0) {
            if (n_cores >= TR_CPU_MAX_SLOTS) continue;
            idx = n_cores++;
            pkg_of[idx] = pkg;
            core_of[idx] = core;
            cores[idx].n_cpu = 0;
            /* index3 is the L3 on every machine that has one; without it the package
             * separates the chiplets no better and no worse than nothing. */
            long dom = sysfs_long("/sys/devices/system/cpu/cpu%d/cache/index3/%s", cpu, "id", -1);
            cores[idx].domain = (dom >= 0) ? (int)dom : (int)pkg;
        }
        core_add_cpu(&cores[idx], 0, (unsigned)cpu);
    }

    c->physical_cores = (n_cores > 0) ? n_cores : c->logical_cores;
    if (cores != NULL && n_cores > 0) {
        /* Counted on the processors this process may use (an outer taskset restricts them),
         * so a restricted process does not size its pool on cores it will never run on. */
        int usable = 0;
        for (int i = 0; i < n_cores; i++) usable += cores[i].n_cpu;
        c->logical_cores = usable;
    }
    if (c->physical_cores < 1) c->physical_cores = 1;
    if (c->logical_cores < 1) c->logical_cores = 1;

    if (cores != NULL) order_slots(c, cores, n_cores);
    free(cores);
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

static tr_cpu_info g_cpu;     /* global-ok: the CPU is a fact of the process */
static int g_detected = 0;    /* global-ok: CPU detected once; see the file header comment for the race-free assumption */

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
