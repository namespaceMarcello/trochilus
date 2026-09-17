/* test.h — tiny header-only test helpers, one static counter per .c file that
 * includes it (each test lives in its own translation unit). */
#ifndef TR_TEST_H
#define TR_TEST_H

#include <stdio.h>

static int tr_test_failures = 0;

#define TR_CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            tr_test_failures++; \
        } \
    } while (0)

#define TR_CHECK_EQ_INT(a, b) \
    do { \
        long long tr_check_a_ = (long long)(a); \
        long long tr_check_b_ = (long long)(b); \
        if (tr_check_a_ != tr_check_b_) { \
            fprintf(stderr, "%s:%d: check failed: %s == %s (%lld != %lld)\n", \
                    __FILE__, __LINE__, #a, #b, tr_check_a_, tr_check_b_); \
            tr_test_failures++; \
        } \
    } while (0)

#define TR_TEST_EXIT() \
    do { \
        if (tr_test_failures != 0) return 1; \
        printf("ok\n"); \
        return 0; \
    } while (0)

#endif
