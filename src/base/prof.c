/* prof.c — tick source, reports. See prof.h. */
#include "prof.h"
#include "platform.h"

#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <x86intrin.h>
#include <cpuid.h>
#define TR_HAVE_RDTSC 1
#else
#define TR_HAVE_RDTSC 0
#endif

static const char *const zone_names[TR_PROF_ZONE_COUNT] = {
    [TR_PROF_TOKEN] = "token",
    [TR_PROF_EMBED] = "embed",
    [TR_PROF_ATTN_NORM] = "attn_norm",
    [TR_PROF_QKV_PROJ] = "qkv_proj",
    [TR_PROF_QK_NORM] = "qk_norm",
    [TR_PROF_ROPE] = "rope",
    [TR_PROF_KV_WRITE] = "kv_write",
    [TR_PROF_ATTENTION] = "attention",
    [TR_PROF_ATTN_OUT_PROJ] = "attn_out_proj",
    [TR_PROF_FFN_NORM] = "ffn_norm",
    [TR_PROF_ROUTER] = "router",
    [TR_PROF_EXPERT_GATE_UP] = "expert_gate_up",
    [TR_PROF_EXPERT_ACT] = "expert_act",
    [TR_PROF_EXPERT_DOWN] = "expert_down",
    [TR_PROF_EXPERT_MIX] = "expert_mix",
    [TR_PROF_OUTPUT_NORM] = "output_norm",
    [TR_PROF_LM_HEAD] = "lm_head",
    [TR_PROF_SAMPLE] = "sample",
    [TR_PROF_WEIGHT_READ] = "weight_read",
    [TR_PROF_POOL_WAIT] = "pool_wait",
};

static const char *const phase_names[TR_PHASE_COUNT] = {"prefill", "decode"};

const char *tr_prof_zone_name(tr_prof_zone z) {
    return (unsigned)z < TR_PROF_ZONE_COUNT ? zone_names[z] : "?";
}

void tr_prof_reset(tr_prof *p) {
    memset(p->acc, 0, sizeof p->acc);
    memset(p->tokens, 0, sizeof p->tokens);
    memset(p->weight_bytes_touched, 0, sizeof p->weight_bytes_touched);
    memset(p->io_bytes, 0, sizeof p->io_bytes);
}

/* ------------------------------------------------------------------ ticks */

static int tick_mode = -1;          /* global-ok: clock of the machine; -1 unknown, 0 OS clock (ns), 1 RDTSC */
static double tick_rate = 1e9;      /* global-ok: calibrated once for the machine */

static void tick_init(void) {
    if (tick_mode >= 0) return;
#if TR_HAVE_RDTSC
    unsigned a, b, c, d;
    if (__get_cpuid(0x80000000u, &a, &b, &c, &d) && a >= 0x80000007u &&
        __get_cpuid(0x80000007u, &a, &b, &c, &d) && (d & (1u << 8))) {
        /* invariant TSC: calibrate against the OS monotonic clock over ~20 ms */
        double t0 = tr_time_sec();
        uint64_t r0 = __rdtsc();
        double t1;
        do { t1 = tr_time_sec(); } while (t1 - t0 < 0.02);
        uint64_t r1 = __rdtsc();
        tick_rate = (double)(r1 - r0) / (t1 - t0);
        tick_mode = 1;
        return;
    }
#endif
    tick_rate = 1e9;
    tick_mode = 0;
}

uint64_t tr_prof_ticks(void) {
    if (tick_mode < 0) tick_init();
#if TR_HAVE_RDTSC
    if (tick_mode == 1) return __rdtsc();
#endif
    return (uint64_t)(tr_time_sec() * 1e9);
}

double tr_prof_ticks_per_sec(void) {
    if (tick_mode < 0) tick_init();
    return tick_rate;
}

/* ------------------------------------------------------------------ reports */

void tr_prof_print(const tr_prof *p, FILE *out) {
    double rate = tr_prof_ticks_per_sec();
    for (int ph = 0; ph < TR_PHASE_COUNT; ph++) {
        const tr_prof_acc *acc = p->acc[ph];
        double token_s = (double)acc[TR_PROF_TOKEN].ticks / rate;
        if (acc[TR_PROF_TOKEN].calls == 0 && p->tokens[ph] == 0) continue;

        fprintf(out, "\n== %s: %llu tokens in %.3f s", phase_names[ph],
                (unsigned long long)p->tokens[ph], token_s);
        if (token_s > 0) fprintf(out, " = %.2f tokens/s", (double)p->tokens[ph] / token_s);
        fprintf(out, "\n");
        if (p->tokens[ph] > 0 && p->weight_bytes_touched[ph] > 0) {
            double per_tok = (double)p->weight_bytes_touched[ph] / (double)p->tokens[ph];
            fprintf(out, "   weights touched: %.1f MiB per token, %.2f GB/s of memory traffic\n",
                    per_tok / (1024.0 * 1024.0), token_s > 0 ? (double)p->weight_bytes_touched[ph] / token_s / 1e9 : 0.0);
        }
        if (p->io_bytes[ph] > 0)
            fprintf(out, "   read from disk: %.1f MiB\n", (double)p->io_bytes[ph] / (1024.0 * 1024.0));

        int order[TR_PROF_ZONE_COUNT], n = 0;
        for (int z = 0; z < TR_PROF_ZONE_COUNT; z++)
            if (acc[z].calls > 0) order[n++] = z;
        for (int i = 1; i < n; i++)             /* insertion sort, most time first (n <= 20) */
            for (int j = i; j > 0 && acc[order[j]].ticks > acc[order[j - 1]].ticks; j--) {
                int t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
            }

        fprintf(out, "   %-16s %10s %11s %7s %11s\n", "zone", "calls", "total ms", "%", "us/call");
        for (int i = 0; i < n; i++) {
            const tr_prof_acc *a = &acc[order[i]];
            double s = (double)a->ticks / rate;
            fprintf(out, "   %-16s %10llu %11.3f %6.1f%% %11.3f\n", zone_names[order[i]],
                    (unsigned long long)a->calls, s * 1e3,
                    token_s > 0 ? 100.0 * s / token_s : 0.0, s * 1e6 / (double)a->calls);
        }
    }
}

void tr_prof_write_json(const tr_prof *p, FILE *out) {
    double rate = tr_prof_ticks_per_sec();
    fprintf(out, "{\"tick_source\":\"%s\",\"phases\":{", tick_mode == 1 ? "rdtsc" : "os_clock");
    for (int ph = 0; ph < TR_PHASE_COUNT; ph++) {
        const tr_prof_acc *acc = p->acc[ph];
        double token_s = (double)acc[TR_PROF_TOKEN].ticks / rate;
        fprintf(out, "%s\"%s\":{\"tokens\":%llu,\"seconds\":%.9f,\"tokens_per_sec\":%.6f,"
                     "\"weight_bytes\":%llu,\"io_bytes\":%llu,\"zones\":{",
                ph ? "," : "", phase_names[ph], (unsigned long long)p->tokens[ph], token_s,
                token_s > 0 ? (double)p->tokens[ph] / token_s : 0.0,
                (unsigned long long)p->weight_bytes_touched[ph], (unsigned long long)p->io_bytes[ph]);
        int first = 1;
        for (int z = 0; z < TR_PROF_ZONE_COUNT; z++) {
            if (acc[z].calls == 0) continue;
            fprintf(out, "%s\"%s\":{\"calls\":%llu,\"seconds\":%.9f}", first ? "" : ",", zone_names[z],
                    (unsigned long long)acc[z].calls, (double)acc[z].ticks / rate);
            first = 0;
        }
        fprintf(out, "}}");
    }
    fprintf(out, "}}\n");
}
