/* test_cli.c — the command line as a user runs it: the trochilus binary built beside this test
 * (BUILD/trochilus), on a synthetic model that carries a tokenizer and a chat template
 * (tests/synth_olmoe.h), its exit code and its output checked.
 *
 * Branches pinned, each seen red before its fix (docs/LESSONS.md #111):
 *  - an id in --tokens beyond int32 is refused (exit 2), not truncated into another id;
 *  - -p and -n far beyond any context: the buffers of generate, generate --spec, run and chat are
 *    sized from the context, not from n * sizeof(int32_t), which wrapped to a few bytes and let
 *    the tokens run past them (under ASan an abort, without it a crash or silent corruption);
 *  - --profile-json names the model in valid JSON when its path holds a control character
 *    (POSIX only: Windows forbids such names);
 *  - a chat template one byte off the known one, at the same length, is refused (src/tokenizer/
 *    chat.c recognizes templates by length and hash: both must match).
 * Every case checks the output it expects too (how many tokens, the reply's length), so a run that
 * stops before its branch -- a control token drawn at once -- fails instead of passing. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "test.h"
#include "synth_olmoe.h"

static char cli[600];      /* the trochilus binary */
static char out_path[600]; /* what a run printed, stdout and stderr */
static char in_path[600];  /* what a run reads on stdin */
static char out[65536];

/* `cli args [< in_path] > out_path 2>&1`; out gets what it printed. The exit code, or 128 + the
 * signal that ended it. */
static int run_cli(const char *args, int with_stdin) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "%s %s%s%s > %s 2>&1", cli, args, with_stdin ? " < " : "", with_stdin ? in_path : "",
             out_path);
    int st = system(cmd);
    out[0] = 0;
    FILE *f = fopen(out_path, "rb");
    if (f != NULL) {
        size_t n = fread(out, 1, sizeof out - 1, f);
        out[n] = 0;
        fclose(f);
    }
#ifdef _WIN32
    return st;
#else
    if (st == -1) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
#endif
}

/* ids on the "tokens:" line of generate */
static int count_tokens(void) {
    const char *p = strstr(out, "tokens: ");
    if (p == NULL) return -1;
    int n = 1;
    for (p += 8; *p != '\n' && *p != 0; p++) n += *p == ',';
    return n;
}

static void expect(const char *what, int rc, int want_rc) {
    if (rc == want_rc) return;
    fprintf(stderr, "%s: exit %d, expected %d; it printed:\n%s\n", what, rc, want_rc, out);
    TR_CHECK_EQ_INT(rc, want_rc);
}

/* the same file name for a POSIX shell and for cmd.exe */
static void native_path(char *p) {
#ifdef _WIN32
    for (; *p != 0; p++)
        if (*p == '/') *p = '\\';
#else
    (void)p;
#endif
}

int main(int argc, char **argv) {
    const char *self = argc > 0 ? argv[0] : "";
    const char *sl = strrchr(self, '/'), *bs = strrchr(self, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    int dir_len = sl ? (int)(sl - self) : 1;
    const char *dir = sl ? self : ".";
#ifdef _WIN32
    snprintf(cli, sizeof cli, "%.*s/../trochilus.exe", dir_len, dir);
#else
    snprintf(cli, sizeof cli, "%.*s/../trochilus", dir_len, dir);
#endif
    snprintf(out_path, sizeof out_path, "%.*s/test_cli_out.txt", dir_len, dir);
    snprintf(in_path, sizeof in_path, "%.*s/test_cli_in.txt", dir_len, dir);
    native_path(cli);
    native_path(out_path);
    native_path(in_path);

    if (run_cli("cpu", 0) != 0) {
#ifdef _WIN32
        /* a freshly linked exe that Smart App Control blocks (docs/LESSONS.md #12) */
        printf("test_cli: SKIPPED, %s does not run here\n", cli);
        return 126;
#else
        fprintf(stderr, "test_cli: %s does not run:\n%s\n", cli, out);
        return 1;
#endif
    }

    synth_params P = {2, 32, 2, 2, 32, 4, 2, SYNTH_TOK_MIN_VOCAB, 64, TR_TYPE_F32};
    synth_with_tokenizer = 1;
    char model[600], args[2048];
    if (synth_write(&P, self, "test_cli.gguf", model, sizeof model) != 0) {
        fprintf(stderr, "test_cli: could not write the model\n");
        return 1;
    }
    native_path(model);

    /* --tokens: an id past int32 is refused, not wrapped into id 1 (or into a negative one) */
    snprintf(args, sizeof args, "generate -m %s --tokens 4294967297 -n 1", model);
    expect("--tokens 2^32+1", run_cli(args, 0), 2);
    snprintf(args, sizeof args, "generate -m %s --tokens 2147483648 -n 1", model);
    expect("--tokens 2^31", run_cli(args, 0), 2);
    snprintf(args, sizeof args, "generate -m %s --tokens 5,7 -n 1", model);
    expect("--tokens 5,7", run_cli(args, 0), 0);
    TR_CHECK_EQ_INT(count_tokens(), 1);

    /* -p past the context: refused before any buffer is sized from it */
    snprintf(args, sizeof args, "generate -m %s -p 4611686018427387905 -n 1 -c 16", model);
    expect("-p 2^62+1", run_cli(args, 0), 1);

    /* -n past the context: every token the context holds, then exit 3; 4 of 16 positions are the
     * prompt, the first token comes from its logits: 13 tokens (12 with --spec, where every token
     * is evaluated) */
    snprintf(args, sizeof args, "generate -m %s -p 4 -n 4611686018427387905 -c 16", model);
    expect("generate -n 2^62+1", run_cli(args, 0), 3);
    TR_CHECK_EQ_INT(count_tokens(), 13);
    snprintf(args, sizeof args, "generate -m %s -p 4 -n 4611686018427387905 -c 16 --spec 4", model);
    expect("generate --spec -n 2^62+1", run_cli(args, 0), 3);
    TR_CHECK_EQ_INT(count_tokens(), 12);

    /* run: the history the draft is read from is sized from the context. This model answers
     * "hello" with 4 tokens and then EOS (256): 5 + 5 entries, where the wrapped size gave 6 */
    snprintf(args, sizeof args, "run -m %s -p hello -n 4611686018427387904 -c 16", model);
    expect("run -n 2^62", run_cli(args, 0), 0);
    TR_CHECK(strstr(out, "generate: 5 tokens") != NULL);

    /* chat: the same; the reply runs until the context is full: 11 prompt tokens, 21 positions
     * left, and the 22nd token, drawn from the last logits, finds no room */
    FILE *in = fopen(in_path, "wb");
    TR_CHECK(in != NULL);
    if (in != NULL) {
        fputs("hello\n", in);
        fclose(in);
    }
    snprintf(args, sizeof args, "chat -m %s -c 32 -n 4611686018427387904", model);
    expect("chat -n 2^62", run_cli(args, 1), 0);
    TR_CHECK(strstr(out, "[22 tokens") != NULL);
    TR_CHECK(strstr(out, "(context full") != NULL);

    /* a template is recognized by its exact bytes: the same length with one byte changed is
     * refused, not rendered as the one it resembles */
    char near[sizeof SYNTH_OLMOE_TEMPLATE], near_model[600];
    memcpy(near, SYNTH_OLMOE_TEMPLATE, sizeof near);
    near[3] = 'B';
    synth_chat_template = near;
    TR_CHECK(synth_write(&P, self, "test_cli_near.gguf", near_model, sizeof near_model) == 0);
    synth_chat_template = SYNTH_OLMOE_TEMPLATE;
    native_path(near_model);
    snprintf(args, sizeof args, "chat -m %s -c 32 -n 4", near_model);
    expect("chat, template one byte off", run_cli(args, 1), 1);
    TR_CHECK(strstr(out, "chat template is not supported") != NULL);
    remove(near_model);
    remove(in_path);

#ifndef _WIN32
    /* --profile-json: the model's path in valid JSON, a tab in it written as \u0009 */
    char odd[600], json_path[600];
    if (synth_write(&P, self, "test_cli\tmodel.gguf", odd, sizeof odd) != 0) {
        fprintf(stderr, "test_cli: could not write the second model\n");
        return 1;
    }
    snprintf(json_path, sizeof json_path, "%.*s/test_cli_profile.json", dir_len, dir);
    snprintf(args, sizeof args, "generate -m '%s' -p 2 -n 1 --profile-json %s", odd, json_path);
    expect("--profile-json", run_cli(args, 0), 0);
    FILE *jf = fopen(json_path, "rb");
    TR_CHECK(jf != NULL);
    if (jf != NULL) {
        size_t n = fread(out, 1, sizeof out - 1, jf);
        out[n] = 0;
        fclose(jf);
        TR_CHECK(strstr(out, "test_cli\\u0009model.gguf\"") != NULL);
        TR_CHECK(strchr(out, '\t') == NULL);
    }
    remove(json_path);
    remove(odd);
#endif

    remove(model);
    remove(out_path);
    TR_TEST_EXIT();
}
