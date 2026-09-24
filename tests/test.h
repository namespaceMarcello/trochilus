/* test.h — tiny header-only test helpers, one static counter per .c file that
 * includes it (each test lives in its own translation unit). */
#ifndef TR_TEST_H
#define TR_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <process.h>
static inline long tr_test_pid(void) { return (long)_getpid(); }
#else
#include <unistd.h>
static inline long tr_test_pid(void) { return (long)getpid(); }
#endif

static int tr_test_failures = 0;

/* The directory a test writes its temporary files in: TR_TEST_TMPDIR when set, else the test
 * binary's own (tmpfile() needs write access to the drive root on Windows). The gate sets it to
 * the container's own disk: next to the binary is the Windows bind mount, where writing and
 * reading the synthetic models took 100 s of each build's 120 (docs/LESSONS.md #142). */
static inline void tr_test_tmpdir(const char *argv0, char *dir, size_t len) {
    const char *env = getenv("TR_TEST_TMPDIR");
    if (env != NULL && env[0] != '\0') {
        snprintf(dir, len, "%s", env);
        return;
    }
    const char *sl = strrchr(argv0, '/'), *bs = strrchr(argv0, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    snprintf(dir, len, "%.*s", sl ? (int)(sl - argv0) : 1, sl ? argv0 : ".");
}

/* TR_TEST_FAILFAST=1 in the environment: the first failed check ends the test, exit 1.
 * tools/mutate_auto.py sets it: a mutant is killed by its first failure, not at the end of a
 * test that runs for a minute. */
static inline int tr_test_failfast(void) {
    static int ff = -1;
    if (ff < 0) {
        const char *e = getenv("TR_TEST_FAILFAST");
        ff = e != NULL && e[0] == '1';
    }
    return ff;
}

#define TR_TEST_FAILED() \
    do { \
        tr_test_failures++; \
        if (tr_test_failfast()) exit(1); \
    } while (0)

#define TR_CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            TR_TEST_FAILED(); \
        } \
    } while (0)

#define TR_CHECK_EQ_INT(a, b) \
    do { \
        long long tr_check_a_ = (long long)(a); \
        long long tr_check_b_ = (long long)(b); \
        if (tr_check_a_ != tr_check_b_) { \
            fprintf(stderr, "%s:%d: check failed: %s == %s (%lld != %lld)\n", \
                    __FILE__, __LINE__, #a, #b, tr_check_a_, tr_check_b_); \
            TR_TEST_FAILED(); \
        } \
    } while (0)

#define TR_TEST_EXIT() \
    do { \
        if (tr_test_failures != 0) return 1; \
        printf("ok\n"); \
        return 0; \
    } while (0)

#endif
