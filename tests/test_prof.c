/* test_prof.c — the profiler: tick rate, accounting, disabled cost, reports.
 *
 * test_reports_known pins the printed table and the JSON, byte for byte, on four profiles with
 * known numbers, one per branch of tr_prof_print: a phase skipped (no token timed, none counted),
 * a phase with tokens and no time (no tokens/s, 0.0%), time and no tokens (no per-token lines),
 * zones with bytes and no time, equal times (the sort keeps zone order), a zone that sorts to the
 * front, and all 21 zones at once (the sort's and the loops' bounds, under ASan in
 * tools/mutate_auto.py). Seen red: every printed mutant of tr_prof_print (docs/MEASUREMENTS.md
 * §Generated mutations). */
#include <string.h>
#include <stdlib.h>

#include "test.h"
#include "../src/base/prof.h"
#include "../src/base/platform.h"

static volatile double sink;
static char tmp_path[512]; /* scratch file next to the test binary */

static void busy(double seconds) {
    double t0 = tr_time_sec();
    while (tr_time_sec() - t0 < seconds) sink += 1.0;
}

/* Fake clocks for tr_prof_calibrate: every read takes 1 us of simulated time, the tick counter
 * runs at 3 GHz, and before read number `stall_at` the thread loses 2 ms, as a preemption does. */
typedef struct {
    double now;         /* simulated seconds */
    long calls, stall_at;
} fake_clock;

static void fake_step(fake_clock *f) {
    if (f->calls++ == f->stall_at) f->now += 0.002;
    f->now += 1e-6;
}
static uint64_t fake_ticks(void *ctx) {
    fake_clock *f = (fake_clock *)ctx;
    fake_step(f);
    return (uint64_t)(f->now * 3e9);
}
static double fake_sec(void *ctx) {
    fake_clock *f = (fake_clock *)ctx;
    fake_step(f);
    return f->now;
}

/* Calibration under a preemption (LESSONS #108): 2 ms lost at any one read -- among the first
 * reads, where the window opens, or the last ones, where it closes -- moves the rate by less than
 * 0.1%. A single unbracketed pair was 10% off when the 2 ms fell between its two reads. */
static void test_calibrate(void) {
    fake_clock f = {0, 0, -1};
    tr_prof_clocks c = {fake_ticks, fake_sec, &f};
    double rate = tr_prof_calibrate(&c, 0.02);
    TR_CHECK(rate > 3e9 * 0.999 && rate < 3e9 * 1.001);
    long total = f.calls, stalled = 0;
    double worst = 0;
    for (long at = 0; at < total; at++) {
        if (at == 40) at = total - 40; /* the loop between the two ends only waits */
        fake_clock g = {0, 0, at};
        tr_prof_clocks cg = {fake_ticks, fake_sec, &g};
        double r = tr_prof_calibrate(&cg, 0.02) / 3e9;
        double err = r > 1 ? r - 1 : 1 - r;
        if (err > worst) worst = err;
        stalled++;
    }
    printf("calibration: %ld reads, a 2 ms stall at each of %ld of them, worst error %.5f%%\n", total, stalled,
           worst * 100);
    TR_CHECK(stalled >= 80);
    TR_CHECK(worst < 0.001);
}

/* ---- the reports, on profiles with known numbers ---- */

/* Times are whole seconds: rounded to a tick, 20 ms were 199.999999 tokens/s in the JSON (half a
 * tick in 4.8e7); half a tick in 1e10 stays below every printed digit. */
static void zone(tr_prof *p, tr_prof_phase ph, tr_prof_zone z, uint64_t calls, double sec, uint64_t bytes) {
    p->acc[ph][z].calls = calls;
    p->acc[ph][z].ticks = (uint64_t)(sec * tr_prof_ticks_per_sec() + 0.5);
    p->acc[ph][z].bytes = bytes;
}

/* what `report` writes for p, into buf */
static void capture(void (*report)(const tr_prof *, FILE *), const tr_prof *p, char *buf, size_t cap) {
    buf[0] = 0;
    FILE *f = fopen(tmp_path, "w+b");
    TR_CHECK(f != NULL);
    if (f == NULL) return;
    report(p, f);
    rewind(f);
    size_t n = fread(buf, 1, cap - 1, f);
    buf[n] = 0;
    fclose(f);
    remove(tmp_path);
}

static void check_text(const char *what, const char *got, const char *want) {
    if (strcmp(got, want) == 0) return;
    size_t i = 0;
    while (got[i] && got[i] == want[i]) i++;
    fprintf(stderr, "%s differs at byte %zu:\n--- got\n%s--- want\n%s---\n", what, i, got, want);
    TR_CHECK(strcmp(got, want) == 0);
}

#define ZONES_HEADER "   zone                  calls    total ms       %     us/call  MiB/token     GB/s\n"

static void test_reports_known(void) {
    static char buf[16384];
    tr_prof p;
    char want[4096];

    /* A: a prefill with every line (time, tokens, weights, KV, disk); equal times (rope and
     * attention, expert_mix and sample) keep zone order; lm_head sorts up past two zones; bytes
     * and no time (expert_mix). A decode with tokens and no time: no tokens/s, no per-token lines
     * (weights, KV, disk all 0), 0.0% on every row; attention sorts to the front. */
    memset(&p, 0, sizeof p);
    p.tokens[TR_PHASE_PREFILL] = 4;
    p.weight_bytes_touched[TR_PHASE_PREFILL] = (uint64_t)40 << 30;
    p.kv_bytes_read[TR_PHASE_PREFILL] = (uint64_t)4 << 30;
    p.io_bytes[TR_PHASE_PREFILL] = 3u << 20;
    zone(&p, TR_PHASE_PREFILL, TR_PROF_TOKEN, 4, 20, 0);
    zone(&p, TR_PHASE_PREFILL, TR_PROF_ROPE, 4, 5, 0);
    zone(&p, TR_PHASE_PREFILL, TR_PROF_ATTENTION, 4, 5, (uint64_t)4 << 30);
    zone(&p, TR_PHASE_PREFILL, TR_PROF_EXPERT_MIX, 1, 0, 2u << 20);
    zone(&p, TR_PHASE_PREFILL, TR_PROF_LM_HEAD, 1, 8, (uint64_t)40 << 30);
    zone(&p, TR_PHASE_PREFILL, TR_PROF_SAMPLE, 1, 0, 0);
    p.tokens[TR_PHASE_DECODE] = 2;
    zone(&p, TR_PHASE_DECODE, TR_PROF_TOKEN, 2, 0, 0);
    zone(&p, TR_PHASE_DECODE, TR_PROF_ATTENTION, 2, 1, 0);
    zone(&p, TR_PHASE_DECODE, TR_PROF_LM_HEAD, 2, 3, 0);
    capture(tr_prof_print, &p, buf, sizeof buf);
    check_text("table A", buf,
               "\n"
               "== prefill: 4 tokens in 20.000 s = 0.20 tokens/s\n"
               "   weights touched: 10240.0 MiB per token, 2.15 GB/s of memory traffic\n"
               "   KV cache read: 1024.0 MiB per token; weights + KV: 2.36 GB/s of memory traffic\n"
               "   read from disk: 3.0 MiB\n" ZONES_HEADER
               "   token                     4   20000.000  100.0% 5000000.000\n"
               "   lm_head                   1    8000.000   40.0% 8000000.000    10240.0     5.37\n"
               "   rope                      4    5000.000   25.0% 1250000.000\n"
               "   attention                 4    5000.000   25.0% 1250000.000     1024.0     0.86\n"
               "   expert_mix                1       0.000    0.0%       0.000        0.5     0.00\n"
               "   sample                    1       0.000    0.0%       0.000\n"
               "\n"
               "== decode: 2 tokens in 0.000 s\n" ZONES_HEADER
               "   lm_head                   2    3000.000    0.0% 1500000.000\n"
               "   attention                 2    1000.000    0.0%  500000.000\n"
               "   token                     2       0.000    0.0%       0.000\n");

    /* the same profile as JSON: zones in their own order, seconds not ticks, 0 tokens/s where no
     * time was taken */
    capture(tr_prof_write_json, &p, buf, sizeof buf);
    snprintf(want, sizeof want, "{\"tick_source\":\"%s\",\"phases\":{"
             "\"prefill\":{\"tokens\":4,\"seconds\":20.000000000,\"tokens_per_sec\":0.200000,"
             "\"weight_bytes\":42949672960,\"kv_bytes\":4294967296,\"io_bytes\":3145728,\"zones\":{"
             "\"token\":{\"calls\":4,\"seconds\":20.000000000,\"bytes\":0},"
             "\"rope\":{\"calls\":4,\"seconds\":5.000000000,\"bytes\":0},"
             "\"attention\":{\"calls\":4,\"seconds\":5.000000000,\"bytes\":4294967296},"
             "\"expert_mix\":{\"calls\":1,\"seconds\":0.000000000,\"bytes\":2097152},"
             "\"lm_head\":{\"calls\":1,\"seconds\":8.000000000,\"bytes\":42949672960},"
             "\"sample\":{\"calls\":1,\"seconds\":0.000000000,\"bytes\":0}}},"
             "\"decode\":{\"tokens\":2,\"seconds\":0.000000000,\"tokens_per_sec\":0.000000,"
             "\"weight_bytes\":0,\"kv_bytes\":0,\"io_bytes\":0,\"zones\":{"
             "\"token\":{\"calls\":2,\"seconds\":0.000000000,\"bytes\":0},"
             "\"attention\":{\"calls\":2,\"seconds\":1.000000000,\"bytes\":0},"
             "\"lm_head\":{\"calls\":2,\"seconds\":3.000000000,\"bytes\":0}}}}}\n",
             tr_prof_ticks_per_sec() == 1e9 ? "os_clock" : "rdtsc");
    check_text("JSON A", buf, want);

    /* B: tokens and bytes but no time (weights and KV at 0.00 GB/s), and an empty decode, which
     * is not printed */
    memset(&p, 0, sizeof p);
    p.tokens[TR_PHASE_PREFILL] = 2;
    p.weight_bytes_touched[TR_PHASE_PREFILL] = (uint64_t)2 << 30;
    p.kv_bytes_read[TR_PHASE_PREFILL] = (uint64_t)2 << 30;
    zone(&p, TR_PHASE_PREFILL, TR_PROF_LM_HEAD, 1, 1, (uint64_t)2 << 30);
    capture(tr_prof_print, &p, buf, sizeof buf);
    check_text("table B", buf,
               "\n"
               "== prefill: 2 tokens in 0.000 s\n"
               "   weights touched: 1024.0 MiB per token, 0.00 GB/s of memory traffic\n"
               "   KV cache read: 1024.0 MiB per token; weights + KV: 0.00 GB/s of memory traffic\n" ZONES_HEADER
               "   lm_head                   1    1000.000    0.0% 1000000.000     1024.0     2.15\n");

    /* C: time and no tokens (a load): no per-token lines, no per-token columns, disk still */
    memset(&p, 0, sizeof p);
    p.weight_bytes_touched[TR_PHASE_PREFILL] = 1u << 20;
    p.kv_bytes_read[TR_PHASE_PREFILL] = 1u << 20;
    p.io_bytes[TR_PHASE_PREFILL] = 1u << 20;
    zone(&p, TR_PHASE_PREFILL, TR_PROF_TOKEN, 2, 10, 0);
    zone(&p, TR_PHASE_PREFILL, TR_PROF_WEIGHT_READ, 1, 4, 1u << 20);
    capture(tr_prof_print, &p, buf, sizeof buf);
    check_text("table C", buf,
               "\n"
               "== prefill: 0 tokens in 10.000 s = 0.00 tokens/s\n"
               "   read from disk: 1.0 MiB\n" ZONES_HEADER
               "   token                     2   10000.000  100.0% 5000000.000\n"
               "   weight_read               1    4000.000   40.0% 4000000.000\n");

    /* D: all 21 zones, time rising with the zone's number: every one printed, in reverse */
    memset(&p, 0, sizeof p);
    p.tokens[TR_PHASE_DECODE] = 1;
    for (int z = 0; z < TR_PROF_ZONE_COUNT; z++) zone(&p, TR_PHASE_DECODE, (tr_prof_zone)z, 1, (z + 1) * 1e-3, 0);
    capture(tr_prof_print, &p, buf, sizeof buf);
    const char *line = strstr(buf, ZONES_HEADER);
    int rows = 0;
    TR_CHECK(line != NULL);
    for (line = line ? line + strlen(ZONES_HEADER) : ""; *line != 0 && rows <= TR_PROF_ZONE_COUNT; rows++) {
        char name[32];
        snprintf(name, sizeof name, "   %s ", tr_prof_zone_name((tr_prof_zone)(TR_PROF_ZONE_COUNT - 1 - rows)));
        TR_CHECK(strncmp(line, name, strlen(name)) == 0);
        const char *nl = strchr(line, '\n');
        line = nl != NULL ? nl + 1 : "";
    }
    TR_CHECK_EQ_INT(rows, TR_PROF_ZONE_COUNT);
}

int main(int argc, char **argv) {
    /* scratch file next to the test binary (tmpfile() needs write access to the drive root on Windows) */
    const char *self = argc > 0 ? argv[0] : "";
    const char *sl = strrchr(self, '/'), *bs = strrchr(self, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    snprintf(tmp_path, sizeof tmp_path, "%.*s/test_prof_tmp.json", sl ? (int)(sl - self) : 1, sl ? self : ".");

    test_calibrate();
    test_reports_known();

    /* the tick rate agrees with the OS clock within 3% over 100 ms, on the best of three windows:
     * a preemption between the two reads of one window is the machine, not the rate, and a wrong
     * rate is wrong in all three (the gate saw one such window under ASan and a loaded machine) */
    double ratio = 0, best = 1e9;
    for (int w = 0; w < 3; w++) {
        uint64_t k0 = tr_prof_ticks();
        double t0 = tr_time_sec();
        busy(0.1);
        uint64_t k1 = tr_prof_ticks();
        double t1 = tr_time_sec();
        TR_CHECK(k1 > k0);
        double r = ((double)(k1 - k0) / tr_prof_ticks_per_sec()) / (t1 - t0);
        if ((r > 1 ? r - 1 : 1 - r) < best) { best = r > 1 ? r - 1 : 1 - r; ratio = r; }
    }
    printf("tick rate %.0f/s, ratio to OS clock %.4f (best of 3)\n", tr_prof_ticks_per_sec(), ratio);
    TR_CHECK(ratio > 0.97 && ratio < 1.03);

    tr_prof p;
    memset(&p, 0, sizeof p);

    /* disabled: nothing is recorded */
    uint64_t s = tr_prof_begin(&p);
    tr_prof_end(&p, TR_PROF_ROPE, s);
    tr_prof_count(&p, TR_PROF_ROPE, 100, 10);
    tr_prof_count_kv(&p, TR_PROF_ROPE, 100);
    TR_CHECK_EQ_INT(p.acc[TR_PHASE_PREFILL][TR_PROF_ROPE].calls, 0);
    TR_CHECK_EQ_INT(p.acc[TR_PHASE_PREFILL][TR_PROF_ROPE].bytes, 0);
    TR_CHECK_EQ_INT(p.weight_bytes_touched[TR_PHASE_PREFILL], 0);
    TR_CHECK_EQ_INT(p.kv_bytes_read[TR_PHASE_PREFILL], 0);
    TR_CHECK(tr_prof_begin(NULL) == 0);
    tr_prof_end(NULL, TR_PROF_ROPE, 0);

    /* enabled: calls, time and counters land in the current phase */
    p.enabled = 1;
    p.phase = TR_PHASE_DECODE;
    for (int i = 0; i < 3; i++) {
        uint64_t tok = tr_prof_begin(&p);
        uint64_t z = tr_prof_begin(&p);
        busy(0.01);
        tr_prof_end(&p, TR_PROF_ATTENTION, z);
        tr_prof_count(&p, TR_PROF_LM_HEAD, 1u << 20, 0);
        tr_prof_count_kv(&p, TR_PROF_ATTENTION, 1u << 10);
        tr_prof_end(&p, TR_PROF_TOKEN, tok);
        p.tokens[p.phase]++;
    }
    const tr_prof_acc *att = &p.acc[TR_PHASE_DECODE][TR_PROF_ATTENTION];
    const tr_prof_acc *tok = &p.acc[TR_PHASE_DECODE][TR_PROF_TOKEN];
    TR_CHECK_EQ_INT(att->calls, 3);
    TR_CHECK(tok->ticks >= att->ticks);
    double att_s = (double)att->ticks / tr_prof_ticks_per_sec();
    TR_CHECK(att_s > 0.029 && att_s < 0.2);
    TR_CHECK_EQ_INT(p.acc[TR_PHASE_PREFILL][TR_PROF_ATTENTION].calls, 0);
    TR_CHECK_EQ_INT(p.weight_bytes_touched[TR_PHASE_DECODE], 3u << 20);
    /* bytes land in the zone that read them, weights and KV cache apart in the phase totals */
    TR_CHECK_EQ_INT(p.kv_bytes_read[TR_PHASE_DECODE], 3u << 10);
    TR_CHECK_EQ_INT(att->bytes, 3u << 10);
    TR_CHECK_EQ_INT(p.acc[TR_PHASE_DECODE][TR_PROF_LM_HEAD].bytes, 3u << 20);
    TR_CHECK_EQ_INT(p.acc[TR_PHASE_PREFILL][TR_PROF_LM_HEAD].bytes, 0);

    TR_CHECK(strcmp(tr_prof_zone_name(TR_PROF_LM_HEAD), "lm_head") == 0);
    TR_CHECK(strcmp(tr_prof_zone_name((tr_prof_zone)999), "?") == 0);
    TR_CHECK(strcmp(tr_prof_zone_name(TR_PROF_ZONE_COUNT), "?") == 0); /* one past the table */
    for (int z = 0; z < TR_PROF_ZONE_COUNT; z++) TR_CHECK(tr_prof_zone_name((tr_prof_zone)z) != NULL);

    /* reports: the table and a JSON object with the expected keys */
    tr_prof_print(&p, stdout);
    char buf[8192];
    FILE *f = fopen(tmp_path, "w+b");
    TR_CHECK(f != NULL);
    if (f) {
        tr_prof_write_json(&p, f);
        rewind(f);
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        fclose(f);
        remove(tmp_path);
        TR_CHECK(strstr(buf, "\"decode\":{\"tokens\":3,") != NULL);
        TR_CHECK(strstr(buf, "\"attention\":{\"calls\":3,") != NULL);
        TR_CHECK(strstr(buf, "\"weight_bytes\":3145728,\"kv_bytes\":3072,") != NULL);
        TR_CHECK(strstr(buf, ",\"bytes\":3072}") != NULL);
        TR_CHECK(buf[0] == '{' && buf[n - 2] == '}');
    }

    tr_prof_reset(&p);
    TR_CHECK_EQ_INT(p.acc[TR_PHASE_DECODE][TR_PROF_ATTENTION].calls, 0);
    TR_CHECK_EQ_INT(p.tokens[TR_PHASE_DECODE], 0);
    TR_CHECK(p.enabled == 1);
    TR_TEST_EXIT();
}
