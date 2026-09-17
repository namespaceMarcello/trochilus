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

int main(int argc, char **argv) {
    /* scratch file next to the test binary (tmpfile() needs write access to the drive root on Windows) */
    char path[512];
    const char *self = argc > 0 ? argv[0] : "";
    const char *sl = strrchr(self, '/'), *bs = strrchr(self, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    snprintf(path, sizeof path, "%.*s/test_prof_tmp.json", sl ? (int)(sl - self) : 1, sl ? self : ".");

    /* the tick rate agrees with the OS clock within 3% over 100 ms */
    uint64_t k0 = tr_prof_ticks();
    double t0 = tr_time_sec();
    busy(0.1);
    uint64_t k1 = tr_prof_ticks();
    double t1 = tr_time_sec();
    double ratio = ((double)(k1 - k0) / tr_prof_ticks_per_sec()) / (t1 - t0);
    printf("tick rate %.0f/s, ratio to OS clock %.4f\n", tr_prof_ticks_per_sec(), ratio);
    TR_CHECK(k1 > k0);
    TR_CHECK(ratio > 0.97 && ratio < 1.03);

    tr_prof p;
    memset(&p, 0, sizeof p);

    /* disabled: nothing is recorded */
    uint64_t s = tr_prof_begin(&p);
    tr_prof_end(&p, TR_PROF_ROPE, s);
    tr_prof_count(&p, 100, 10);
    TR_CHECK_EQ_INT(p.acc[TR_PHASE_PREFILL][TR_PROF_ROPE].calls, 0);
    TR_CHECK_EQ_INT(p.weight_bytes_touched[TR_PHASE_PREFILL], 0);
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
        tr_prof_count(&p, 1u << 20, 0);
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
        TR_CHECK(buf[0] == '{' && buf[n - 2] == '}');
    }

    tr_prof_reset(&p);
    TR_CHECK_EQ_INT(p.acc[TR_PHASE_DECODE][TR_PROF_ATTENTION].calls, 0);
    TR_CHECK_EQ_INT(p.tokens[TR_PHASE_DECODE], 0);
    TR_CHECK(p.enabled == 1);
    TR_TEST_EXIT();
}
