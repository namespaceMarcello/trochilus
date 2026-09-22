/* test_prof.c — the profiler: tick rate, accounting, disabled cost, reports. */
#include <string.h>
#include <stdlib.h>

#include "test.h"
#include "../src/base/prof.h"
#include "../src/base/platform.h"

static volatile double sink;

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

int main(int argc, char **argv) {
    /* scratch file next to the test binary (tmpfile() needs write access to the drive root on Windows) */
    char path[512];
    const char *self = argc > 0 ? argv[0] : "";
    const char *sl = strrchr(self, '/'), *bs = strrchr(self, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    snprintf(path, sizeof path, "%.*s/test_prof_tmp.json", sl ? (int)(sl - self) : 1, sl ? self : ".");

    test_calibrate();

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
    for (int z = 0; z < TR_PROF_ZONE_COUNT; z++) TR_CHECK(tr_prof_zone_name((tr_prof_zone)z) != NULL);

    /* reports: the table and a JSON object with the expected keys */
    tr_prof_print(&p, stdout);
    char buf[8192];
    FILE *f = fopen(path, "w+b");
    TR_CHECK(f != NULL);
    if (f) {
        tr_prof_write_json(&p, f);
        rewind(f);
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        fclose(f);
        remove(path);
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
