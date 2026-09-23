/* test_cli.c — the command line as a user runs it: the trochilus binary built beside this test
 * (BUILD/trochilus), on synthetic models that carry a tokenizer and a chat template
 * (tests/synth_olmoe.h), its exit code, its stdout and its stderr checked.
 *
 * Branches pinned, each seen red before its fix (docs/LESSONS.md #111) or under a mutant of
 * src/app/main.c (tools/mutate_auto.py, docs/LESSONS.md #120):
 *  - an id in --tokens beyond int32 is refused (exit 2), not truncated into another id;
 *  - -p and -n far beyond any context: the buffers of generate, generate --spec, run and chat are
 *    sized from the context, not from n * sizeof(int32_t), which wrapped to a few bytes and let
 *    the tokens run past them (under ASan an abort, without it a crash or silent corruption);
 *  - --profile-json names the model in valid JSON when its path holds a control character, a
 *    space or a quote (POSIX only: Windows forbids such names);
 *  - a chat template one byte off the known one, at the same length, is refused (src/tokenizer/
 *    chat.c recognizes templates by length and hash: both must match);
 *  - every number on the command line is parsed strictly, at both ends of its range ("-n abc"
 *    was 0), every option given last without its value is refused, every unknown one named;
 *  - every command's usage errors, each condition alone; the dispatch of every command;
 *  - inspect's listing (arrays, a string of exactly 80 bytes, a tensor of exactly 1 KiB, the
 *    totals by type), cpu's RAM line;
 *  - generate: the first token of each step is the argmax of the logits command's row; --spec
 *    gives the same tokens and stops at -n; the prompt that exactly fills the context; the speed
 *    and speculation lines at -n 0; the threads line forced, not yet measured, and measured;
 *    --profile and --profile-json filled; --expert-budget min, 1, and its limits;
 *  - --expert-mask: accepted with duplicates, refused at each bound, on garbage, on a layer left
 *    with fewer experts than a token uses, on a missing file;
 *  - tokenize: ids, --pieces with an added token, --decode, -f, an empty file, --batch and each
 *    of its malformed records;
 *  - run: a control token that is not EOS stops it; -n; --spec prints the same text; -f;
 *    --route-trace's file, and its refusal to write where it cannot;
 *  - chat: -n per reply, -s, /reset, a prompt too long for the context dropped from the
 *    conversation, the cache reused across turns; chat-template's records and their refusals.
 * Every case checks the output it expects too (how many tokens, which line), so a run that
 * stops before its branch fails instead of passing. */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "test.h"
#include "synth_olmoe.h"

static const char *self_arg;       /* argv[0]: synth_write puts its files beside it */
static char bin[600];              /* the trochilus binary */
static char dir[512];              /* where this test lives, and the files it writes */
static char out_path[600], err_path[600], in_path[600];
static char out[65536], err[65536]; /* what the last run printed on stdout, on stderr */
static size_t out_len;

#define OUT_HAS(s) TR_CHECK(strstr(out, s) != NULL)
#define ERR_HAS(s) TR_CHECK(strstr(err, s) != NULL)
#define ERR_LACKS(s) TR_CHECK(strstr(err, s) == NULL)

/* the same file name for a POSIX shell and for cmd.exe */
static void native_path(char *p) {
#ifdef _WIN32
    for (; *p != 0; p++)
        if (*p == '/') *p = '\\';
#else
    (void)p;
#endif
}

static void in_dir(char *path, size_t len, const char *name) {
    snprintf(path, len, "%s/%s", dir, name);
    native_path(path);
}

static size_t slurp(const char *path, char *buf, size_t cap) {
    buf[0] = 0;
    FILE *f = fopen(path, "rb");
    if (f == NULL) return 0;
    size_t n = fread(buf, 1, cap - 1, f);
    buf[n] = 0;
    fclose(f);
    return n;
}

/* "\r\n" made "\n": on Windows the commands that print text (cpu, inspect, every message on
 * stderr) end their lines the Windows way, those that print bytes (run, chat, tokenize) do not;
 * both sides of every comparison go through here. Returns the new length. */
static size_t unix_lines(char *buf, size_t len) {
    size_t w = 0;
    for (size_t r = 0; r < len; r++)
        if (!(buf[r] == '\r' && r + 1 < len && buf[r + 1] == '\n')) buf[w++] = buf[r];
    buf[w] = 0;
    return w;
}

static void write_bytes(const char *path, const char *data, size_t len) {
    FILE *f = fopen(path, "wb");
    TR_CHECK(f != NULL);
    if (f == NULL) return;
    TR_CHECK(fwrite(data, 1, len, f) == len);
    fclose(f);
}

/* one --batch record of chat-template: "<length>\n" then body */
static void write_record(const char *path, const char *body) {
    char rec[1024];
    int n = snprintf(rec, sizeof rec, "%zu\n%s", strlen(body), body);
    write_bytes(path, rec, (size_t)n);
}

/* `cli <args> [< in_path] > out_path 2> err_path`, args formatted like printf. The exit code, or
 * 128 + the signal that ended it. */
static int vcli(int with_stdin, const char *fmt, va_list ap) {
    char args[4096], cmd[8192];
    vsnprintf(args, sizeof args, fmt, ap);
    snprintf(cmd, sizeof cmd, "%s %s%s%s > %s 2> %s", bin, args, with_stdin ? " < " : "", with_stdin ? in_path : "",
             out_path, err_path);
    int st = system(cmd);
    out_len = unix_lines(out, slurp(out_path, out, sizeof out));
    unix_lines(err, slurp(err_path, err, sizeof err));
#ifdef _WIN32
    return st;
#else
    if (st == -1) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
#endif
}

static int cli(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int rc = vcli(0, fmt, ap);
    va_end(ap);
    return rc;
}

/* the same with `input` on stdin (the chat) */
static int cli_stdin(const char *input, const char *fmt, ...) {
    write_bytes(in_path, input, strlen(input));
    va_list ap;
    va_start(ap, fmt);
    int rc = vcli(1, fmt, ap);
    va_end(ap);
    return rc;
}

static void expect(const char *what, int rc, int want_rc) {
    if (rc == want_rc) return;
    fprintf(stderr, "%s: exit %d, expected %d; stdout:\n%s\nstderr:\n%s\n", what, rc, want_rc, out, err);
    TR_CHECK_EQ_INT(rc, want_rc);
}

/* ids on the "tokens:" line of generate */
static int count_tokens(void) {
    const char *p = strstr(out, "tokens:");
    if (p == NULL) return -1;
    if (p[7] == '\n') return 0;
    int n = 1;
    for (p += 7; *p != '\n' && *p != 0; p++) n += *p == ',';
    return n;
}

/* the ids after "tokens: " into ids (at most cap), their count */
static int parse_ids(int32_t *ids, int cap) {
    const char *p = strstr(out, "tokens: ");
    int n = 0;
    if (p == NULL) return 0;
    p += 8;
    while (n < cap && *p >= '0' && *p <= '9') {
        char *end;
        ids[n++] = (int32_t)strtol(p, &end, 10);
        p = end + (*end == ',');
    }
    return n;
}

/* how many times s occurs in err */
static int err_count(const char *s) {
    int n = 0;
    for (const char *p = strstr(err, s); p != NULL; p = strstr(p + 1, s)) n++;
    return n;
}

/* ---- dispatch and cpu --------------------------------------------------------------------- */

static void test_dispatch(void) {
    expect("no command", cli(""), 2);
    ERR_HAS("usage:\n  trochilus cpu");
    expect("unknown command", cli("bogus"), 2);
    ERR_HAS("usage:\n  trochilus cpu");
    expect("cpu", cli("cpu"), 0);
    OUT_HAS("cpu: ");
    OUT_HAS("\nram: ");
    OUT_HAS(" total, ");
    OUT_HAS(" available\n");
    expect("cpu with an argument", cli("cpu x"), 2);
    /* each command reached by its own name: its own usage, not the general one */
    static const char *const names[] = {"generate", "logits", "tokenize", "run", "chat", "chat-template"};
    for (size_t k = 0; k < sizeof names / sizeof names[0]; k++) {
        char want[64];
        snprintf(want, sizeof want, "usage: trochilus %s ", names[k]);
        expect(names[k], cli("%s", names[k]), 2);
        TR_CHECK(strstr(err, want) != NULL);
    }
}

/* ---- inspect ------------------------------------------------------------------------------ */

static void test_inspect(const char *model, const char *model_e8) {
    expect("inspect", cli("inspect %s", model), 0);
    OUT_HAS("gguf v3, 19 metadata entries, 27 tensors\n");
    OUT_HAS(" u32  2\n");
    OUT_HAS("[str x 260] first \"\xC4\x80\"\n"); /* token 0 is byte 0, written U+0100 */
    OUT_HAS("[i32 x 260]\n");                   /* not a string array: no "first" */
    OUT_HAS("[str x 0]\n");                     /* the merges: empty, no "first" */
    OUT_HAS("message['role'] == 'system' %}...\"\n"); /* the template, cut at 80 bytes */
    OUT_HAS(" 32 x 260  ");                     /* two dimensions, no third */
    OUT_HAS(" 32.50 KiB\n");
    OUT_HAS(" 128 B\n");
    const char *total = strstr(out, "tensor data: ");
    TR_CHECK(total != NULL);
    if (total != NULL) {
        TR_CHECK(strstr(total, ", f32 ") != NULL);
        TR_CHECK(strstr(total, ", f16 ") == NULL);
    }
    /* 8 experts: the router's matrix is 32 x 8 floats, exactly 1024 bytes; its template exactly
     * 80 bytes, printed whole */
    expect("inspect, 8 experts", cli("inspect %s", model_e8), 0);
    OUT_HAS(" 32 x 8  ");
    OUT_HAS(" 1.00 KiB\n");
    OUT_HAS("\"0123456789012345678901234567890123456789012345678901234567890123456789012345678X\"\n");
    char missing[700];
    in_dir(missing, sizeof missing, "test_cli_missing.gguf");
    expect("inspect, missing file", cli("inspect %s", missing), 1);
    ERR_HAS("error: ");
    expect("inspect, no file", cli("inspect"), 2);
    expect("inspect, two files", cli("inspect %s %s", model, model), 2);
}

/* ---- the option parser -------------------------------------------------------------------- */

static void test_options(const char *m) {
    /* strict numbers: -n of generate takes 0 .. INT64_MAX */
    static const char *const bad[] = {"abc", "5x", "\"\"", "\" 5\"", "+5", "-", "-1", "99999999999999999999"};
    for (size_t k = 0; k < sizeof bad / sizeof bad[0]; k++) {
        expect(bad[k], cli("generate -m %s -p 4 -n %s -c 16", m, bad[k]), 2);
        ERR_HAS("generate: -n takes a whole number from 0 to 9223372036854775807, not '");
        ERR_HAS("usage: trochilus generate ");
    }
    expect("-n 0", cli("generate -m %s -p 4 -n 0 -c 16", m), 0);
    TR_CHECK_EQ_INT(count_tokens(), 0);
    ERR_HAS("generate: 0 tokens, 0 evaluations in ");
    expect("-n 9", cli("generate -m %s -p 4 -n 9 -c 16", m), 0);
    TR_CHECK_EQ_INT(count_tokens(), 9);
    /* both ends of a range: --spec 0 .. 15 */
    expect("--spec 15", cli("generate -m %s -p 4 -n 2 -c 16 --spec 15", m), 0);
    expect("--spec 16", cli("generate -m %s -p 4 -n 2 -c 16 --spec 16", m), 2);
    ERR_HAS("--spec takes a whole number from 0 to 15, not '16'");
    expect("-p 0", cli("generate -m %s -p 0 -n 2 -c 16", m), 2);
    /* an option given last, with no value: refused, whatever its kind */
    static const char *const last[] = {"-n", "--tokens", "--expert-budget", "--profile-json"};
    for (size_t k = 0; k < sizeof last / sizeof last[0]; k++) {
        char want[64];
        snprintf(want, sizeof want, "generate: %s needs a value", last[k]);
        expect(last[k], cli("generate -m %s -p 4 -n 1 %s", m, last[k]), 2);
        TR_CHECK(strstr(err, want) != NULL);
    }
    expect("unknown option", cli("generate -m %s -p 4 -n 1 --bogus", m), 2);
    ERR_HAS("generate: unknown argument '--bogus'");
    /* generate's usage errors, each alone */
    expect("no -m", cli("generate -p 4 -n 1"), 2);
    expect("no -n", cli("generate -m %s -p 4", m), 2);
    expect("--tokens and -p", cli("generate -m %s -p 4 --tokens 5 -n 1", m), 2);
    expect("neither --tokens nor -p", cli("generate -m %s -n 1", m), 2);
    /* --expert-budget: 'min', or MiB from 1 to what 64 bits of bytes hold */
    expect("--expert-budget min", cli("generate -m %s -p 4 -n 3 -c 16 --expert-budget min", m), 0);
    ERR_HAS(" of 8 units in RAM (");
    ERR_HAS(" hits, ");
    ERR_LACKS("experts: 8 of 8");
    expect("--expert-budget 1", cli("generate -m %s -p 4 -n 3 -c 16 --expert-budget 1", m), 0);
    ERR_HAS("experts: 8 of 8 units in RAM (0 MiB, ");
    expect("--expert-budget 2^43-1", cli("generate -m %s -p 4 -n 3 -c 16 --expert-budget 8796093022207", m), 0);
    static const char *const bad_budget[] = {"0", "5x", "8796093022208", "mini"};
    for (size_t k = 0; k < sizeof bad_budget / sizeof bad_budget[0]; k++) {
        expect(bad_budget[k], cli("generate -m %s -p 4 -n 3 -c 16 --expert-budget %s", m, bad_budget[k]), 2);
        ERR_HAS("--expert-budget takes 'min' or a number of MiB from 1 to 8796093022207, not '");
    }
    /* the other commands' ranges */
    expect("logits -b 0", cli("logits -m %s --tokens 5 --out x -b 0", m), 2);
    expect("run -n -1", cli("run -m %s -p hello -n -1", m), 2);
    expect("chat -n 0", cli("chat -m %s -n 0", m), 2);
    expect("-t -1", cli("run -m %s -p hello -t -1", m), 2);
    expect("--decode-threads -1", cli("run -m %s -p hello --decode-threads -1", m), 2);
    expect("tokenize, unknown flag", cli("tokenize -m %s -p hello --bogus", m), 2);
    ERR_HAS("tokenize: unknown argument '--bogus'");
}

/* ---- generate ----------------------------------------------------------------------------- */

static void test_generate(const char *m) {
    /* --tokens: an id past int32 is refused, not wrapped into id 1 (or into a negative one) */
    expect("--tokens 2^32+1", cli("generate -m %s --tokens 4294967297 -n 1", m), 2);
    expect("--tokens 2^31", cli("generate -m %s --tokens 2147483648 -n 1", m), 2);
    expect("--tokens 5,7", cli("generate -m %s --tokens 5,7 -n 1", m), 0);
    TR_CHECK_EQ_INT(count_tokens(), 1);
    expect("--tokens 0", cli("generate -m %s --tokens 0 -n 1", m), 0);
    /* INT32_MAX parses; the model then refuses it as out of its vocabulary */
    expect("--tokens 2^31-1", cli("generate -m %s --tokens 2147483647 -n 1", m), 1);
    ERR_HAS("prompt evaluation failed");
    static const char *const bad[] = {",5", "-1", "5x", "\"\"", "5,,7"};
    for (size_t k = 0; k < sizeof bad / sizeof bad[0]; k++) {
        expect(bad[k], cli("generate -m %s --tokens %s -n 1", m, bad[k]), 2);
        ERR_HAS("generate: invalid --tokens");
    }
    /* the list grows past its first 16 ids; 40 of them, where a buffer doubled at every id
     * would ask for 2^47 bytes */
    char many[256];
    int k_many = snprintf(many, sizeof many, "1");
    for (int i = 2; i <= 40; i++) k_many += snprintf(many + k_many, sizeof many - (size_t)k_many, ",%d", i);
    expect("40 ids", cli("generate -m %s --tokens %s -n 2", m, many), 0);
    TR_CHECK_EQ_INT(count_tokens(), 2);
    ERR_HAS("prompt: 40 tokens in ");

    /* -p past the context: refused before any buffer is sized from it; -p that fills it exactly
     * leaves room for the one token its logits give */
    expect("-p 2^62+1", cli("generate -m %s -p 4611686018427387905 -n 1 -c 16", m), 1);
    ERR_HAS("does not fit the context of 16 tokens");
    expect("-p 17 -c 16", cli("generate -m %s -p 17 -n 1 -c 16", m), 1);
    ERR_HAS("does not fit the context of 16 tokens");
    expect("-p 16 -c 16", cli("generate -m %s -p 16 -n 1 -c 16", m), 0);
    TR_CHECK_EQ_INT(count_tokens(), 1);

    /* 4 of 16 positions are the prompt, the first token comes from its logits: 13 tokens fit,
     * 12 evaluations, and no evaluation after the last one */
    expect("-n 13 -c 16", cli("generate -m %s -p 4 -n 13 -c 16", m), 0);
    TR_CHECK_EQ_INT(count_tokens(), 13);
    ERR_HAS("generate: 13 tokens, 12 evaluations in ");
    ERR_LACKS("speculation:");
    /* -n past the context: every token the context holds, then exit 3 (12 with --spec, where
     * every token is evaluated) */
    expect("generate -n 2^62+1", cli("generate -m %s -p 4 -n 4611686018427387905 -c 16", m), 3);
    TR_CHECK_EQ_INT(count_tokens(), 13);
    expect("generate --spec -n 2^62+1", cli("generate -m %s -p 4 -n 4611686018427387905 -c 16 --spec 4", m), 3);
    TR_CHECK_EQ_INT(count_tokens(), 12);

    /* each token is the argmax of the logits after the tokens before it (logits -b 1: one row
     * per position), and --spec gives the same tokens and stops at -n */
    int32_t gen[12], spec[12];
    expect("12 tokens", cli("generate -m %s --tokens 5,7,9,11 -n 12 -c 16", m), 0);
    int n_gen = parse_ids(gen, 12);
    TR_CHECK_EQ_INT(n_gen, 12);
    expect("12 tokens, --spec 4", cli("generate -m %s --tokens 5,7,9,11 -n 12 -c 16 --spec 4", m), 0);
    TR_CHECK_EQ_INT(parse_ids(spec, 12), 12);
    TR_CHECK(memcmp(gen, spec, sizeof gen) == 0);
    expect("--spec 4 -n 3", cli("generate -m %s --tokens 5,7,9,11 -n 3 -c 16 --spec 4", m), 0);
    TR_CHECK_EQ_INT(count_tokens(), 3);
    if (n_gen == 12) {
        char ids[256], logits_path[700];
        int k = snprintf(ids, sizeof ids, "5,7,9,11");
        for (int i = 0; i < 11; i++) k += snprintf(ids + k, sizeof ids - (size_t)k, ",%d", gen[i]);
        in_dir(logits_path, sizeof logits_path, "test_cli_logits.bin");
        expect("logits", cli("logits -m %s --tokens %s -b 1 --out %s", m, ids, logits_path), 0);
        FILE *f = fopen(logits_path, "rb");
        TR_CHECK(f != NULL);
        static float rows[15][SYNTH_TOK_MIN_VOCAB];
        if (f != NULL) {
            TR_CHECK(fread(rows, sizeof rows, 1, f) == 1);
            fclose(f);
            for (int i = 0; i < 12; i++) {
                int best = 0;
                for (int v = 1; v < (int)SYNTH_TOK_MIN_VOCAB; v++)
                    if (rows[3 + i][v] > rows[3 + i][best]) best = v;
                TR_CHECK_EQ_INT(gen[i], best);
            }
        }
        remove(logits_path);
    }

    /* drafts accepted: from its 14th token this model repeats a cycle of 10, so --spec 4 verifies
     * whole drafts there, a step gives up to 5 tokens, and every -n from 30 to 40 ends at some
     * point of a step: never a token more, and the tokens of plain decoding */
    int32_t plain[40], drafted[40];
    expect("40 tokens", cli("generate -m %s -p 4 -n 40 -c 64", m), 0);
    TR_CHECK_EQ_INT(parse_ids(plain, 40), 40);
    for (int n = 30; n <= 40; n++) {
        expect("--spec 4, a cycle", cli("generate -m %s -p 4 -n %d -c 64 --spec 4", m, n), 0);
        TR_CHECK_EQ_INT(parse_ids(drafted, 40), n);
        TR_CHECK(memcmp(drafted, plain, (size_t)n * sizeof(int32_t)) == 0);
    }
    unsigned long long accepted = 0, n_drafted = 0;
    const char *sl = strstr(err, "passes, ");
    TR_CHECK(sl != NULL && sscanf(sl, "passes, %llu/%llu", &accepted, &n_drafted) == 2);
    TR_CHECK(accepted > 0);

    /* -n 0: no token, no evaluation, and with --spec no pass: every rate 0, never 0/0 */
    expect("--spec 4 -n 0", cli("generate -m %s -p 4 -n 0 -c 16 --spec 4", m), 0);
    ERR_HAS("speculation: 0 passes, 0/0 drafted tokens accepted (0.0%), 0.00 tokens per pass, mean draft 0.00\n");
    ERR_LACKS("nan");

    /* threads: forced; a pool of 8 not measured after 2 short passes; measured after 39 */
    expect("--decode-threads 1", cli("generate -m %s -p 4 -n 3 -c 16 --decode-threads 1", m), 0);
    TR_CHECK(strstr(err, " 1 decode (forced)\n") != NULL);
    expect("-t 8 -n 3", cli("generate -m %s -p 4 -n 3 -c 16 -t 8", m), 0);
    ERR_HAS("threads: 8 prompt, decode not measured yet\n");
    expect("-t 8 -n 40", cli("generate -m %s -p 4 -n 40 -c 64 -t 8", m), 0);
    const char *th = strstr(err, "threads: 8 prompt, ");
    TR_CHECK(th != NULL);
    if (th != NULL) {
        /* "<w> decode (measured), choices <pos>:<width>[(<picked>)] ...": a choice shows its pick
         * only when the debounce held it back, and every choice is a real one */
        int w = 0, n_choices = 0;
        TR_CHECK(sscanf(th, "threads: 8 prompt, %d decode", &w) == 1);
        TR_CHECK(w == 4 || w == 8);
        const char *p = strstr(th, " decode (measured), choices ");
        const char *eol = strchr(th, '\n');
        TR_CHECK(p != NULL && eol != NULL && p < eol);
        if (p != NULL) p = strstr(p, "choices");
        for (p = p != NULL ? p + 7 : eol; p != NULL && eol != NULL && p < eol;) {
            long long pos;
            int width, picked, used = 0;
            if (sscanf(p, " %lld:%d%n", &pos, &width, &used) != 2) break;
            n_choices++;
            TR_CHECK(pos > 4 && (width == 4 || width == 8));
            p += used;
            if (*p == '(') {
                TR_CHECK(sscanf(p, "(%d)%n", &picked, &used) == 1);
                TR_CHECK(picked != width);
                p += used;
            }
        }
        TR_CHECK(n_choices > 0);
        TR_CHECK(p == eol);
    }

    /* --profile prints the zones, --profile-json writes them */
    expect("--profile", cli("generate -m %s -p 4 -n 2 -c 16 --profile", m), 0);
    ERR_HAS("\n   lm_head ");
    char json_path[700];
    in_dir(json_path, sizeof json_path, "test_cli_profile.json");
    expect("--profile-json", cli("generate -m %s -p 4 -n 2 -c 16 --profile-json %s", m, json_path), 0);
    static char json[65536];
    slurp(json_path, json, sizeof json);
    TR_CHECK(strstr(json, "\"lm_head\":{\"calls\":") != NULL);
    remove(json_path);
    in_dir(json_path, sizeof json_path, "test_cli_no_such_dir/profile.json");
    expect("--profile-json, no such directory", cli("generate -m %s -p 4 -n 2 -c 16 --profile-json %s", m, json_path),
           1);
    ERR_HAS("generate: could not write --profile-json '");
}

/* ---- --expert-mask ------------------------------------------------------------------------ */

static void test_mask(const char *m) {
    char mask[700], lout[700], missing[700];
    in_dir(mask, sizeof mask, "test_cli_mask.txt");
    in_dir(lout, sizeof lout, "test_cli_mask_logits.bin");
    in_dir(missing, sizeof missing, "test_cli_missing_mask.txt");
    /* 2 layers of 4 experts, 2 used per token: layer 0 keeps 2, a line repeated counts once */
    static const char good[] = "0 0\n0 1\n0 1\n1 3\n";
    write_bytes(mask, good, sizeof good - 1);
    expect("mask", cli("logits -m %s --tokens 5,7 --out %s --expert-mask %s", m, lout, mask), 0);
    ERR_HAS("expert mask: 3 of 8 units off (measurement only");
    ERR_LACKS("error");
    expect("mask, run", cli("run -m %s -p hello -n 2 --expert-mask %s", m, mask), 0);
    ERR_HAS("expert mask: 3 of 8 units off");
    static const char *const refused[] = {"2 0\n", "0 4\n", "-1 0\n", "0 -1\n", "0 x\n"};
    for (size_t k = 0; k < sizeof refused / sizeof refused[0]; k++) {
        write_bytes(mask, refused[k], strlen(refused[k]));
        expect(refused[k], cli("logits -m %s --tokens 5,7 --out %s --expert-mask %s", m, lout, mask), 1);
        ERR_HAS("is not a list of 'layer expert' in range");
        ERR_LACKS("expert mask:");
    }
    static const char too_many[] = "0 0\n0 1\n0 2\n";
    write_bytes(mask, too_many, sizeof too_many - 1);
    expect("mask, 3 of 4 off", cli("logits -m %s --tokens 5,7 --out %s --expert-mask %s", m, lout, mask), 1);
    ERR_HAS("a layer would keep fewer experts than a token uses");
    ERR_LACKS("is not a list");
    expect("mask, missing", cli("logits -m %s --tokens 5,7 --out %s --expert-mask %s", m, lout, missing), 1);
    ERR_HAS("--expert-mask: cannot read '");
    remove(mask);
    remove(lout);

    /* logits' usage errors, each alone */
    expect("logits, no -m", cli("logits --tokens 5 --out x"), 2);
    expect("logits, no --tokens", cli("logits -m %s --out x", m), 2);
    expect("logits, no --out", cli("logits -m %s --tokens 5", m), 2);
}

/* ---- tokenize ----------------------------------------------------------------------------- */

static void test_tokenize(const char *m) {
    expect("tokenize -p", cli("tokenize -m %s -p hello", m), 0);
    TR_CHECK(strcmp(out, "104,101,108,108,111\n") == 0);
    TR_CHECK(err[0] == 0);
    expect("--pieces", cli("tokenize -m %s -p \"h<|user|>i\" --pieces", m), 0);
    TR_CHECK(strcmp(out, "68 #258 69\n") == 0);
    expect("--decode", cli("tokenize -m %s -p \"h<|user|>i\" --decode", m), 0);
    TR_CHECK(strcmp(out, "683c7c757365727c3e69\n") == 0);
    expect("--no-parse-special", cli("tokenize -m %s -p \"h<|user|>i\" --no-parse-special", m), 0);
    TR_CHECK(strcmp(out, "104,60,124,117,115,101,114,124,62,105\n") == 0);
    expect("--no-add-special", cli("tokenize -m %s -p hello --no-add-special", m), 0);
    TR_CHECK(strcmp(out, "104,101,108,108,111\n") == 0);
    expect("--pieces --decode", cli("tokenize -m %s -p hello --pieces --decode", m), 2);

    char file[700], missing[700];
    in_dir(file, sizeof file, "test_cli_text.txt");
    in_dir(missing, sizeof missing, "test_cli_missing.txt");
    write_bytes(file, "hello", 5);
    expect("tokenize -f", cli("tokenize -m %s -f %s", m, file), 0);
    TR_CHECK(strcmp(out, "104,101,108,108,111\n") == 0);
    write_bytes(file, "", 0);
    expect("tokenize -f, empty", cli("tokenize -m %s -f %s", m, file), 0);
    TR_CHECK(strcmp(out, "\n") == 0);
    expect("tokenize -f, missing", cli("tokenize -m %s -f %s", m, missing), 1);
    static const char batch[] = "5\nhello2\nhi0\n";
    write_bytes(file, batch, sizeof batch - 1);
    expect("--batch", cli("tokenize -m %s --batch %s", m, file), 0);
    TR_CHECK(strcmp(out, "104,101,108,108,111\n104,105\n\n") == 0);
    /* malformed, each alone: no length; a length not followed by a newline; one byte more
     * than the file holds; a 19th digit */
    static const char *const bad[] = {"\nab", "3xabc", "3\nab", "0000000000000000001\na"};
    for (size_t k = 0; k < sizeof bad / sizeof bad[0]; k++) {
        write_bytes(file, bad[k], strlen(bad[k]));
        expect(bad[k], cli("tokenize -m %s --batch %s", m, file), 1);
        ERR_HAS("error: malformed batch record at byte ");
        TR_CHECK(out[0] == 0);
    }
    remove(file);
    expect("tokenize, no -m", cli("tokenize -p hello"), 2);
    expect("tokenize, -p and -f", cli("tokenize -m %s -p hello -f x", m), 2);
    expect("tokenize, no text", cli("tokenize -m %s", m), 2);
}

/* ---- run ---------------------------------------------------------------------------------- */

static void test_run(const char *m, const char *m257) {
    /* this model answers "hello" with 4 tokens and then token 256, a control token: EOS here,
     * and in m257, whose EOS is 257, a control token that is not EOS -- it stops the reply
     * all the same */
    expect("run", cli("run -m %s -p hello -c 16", m), 0);
    ERR_HAS("generate: 5 tokens in ");
    ERR_LACKS("speculation:");
    char reply[256];
    size_t reply_len = out_len < sizeof reply ? out_len : sizeof reply;
    memcpy(reply, out, reply_len);
    expect("run, EOS 257", cli("run -m %s -p hello -c 16", m257), 0);
    ERR_HAS("generate: 5 tokens in ");
    expect("run --spec 4", cli("run -m %s -p hello -c 16 --spec 4", m), 0);
    TR_CHECK(out_len == reply_len && memcmp(out, reply, reply_len) == 0);
    expect("run -n 2", cli("run -m %s -p hello -n 2", m), 0);
    ERR_HAS("generate: 2 tokens in ");
    expect("run -n 0", cli("run -m %s -p hello -n 0", m), 0);
    ERR_HAS("generate: 0 tokens in ");
    expect("run --spec 4 -n 0", cli("run -m %s -p hello -n 0 --spec 4", m), 0);
    ERR_HAS("speculation: 0 passes, 0/0 drafted tokens accepted (0.0%), mean draft 0.00\n");
    ERR_LACKS("nan");

    char file[700], missing[700], trace[700], no_dir[700];
    in_dir(file, sizeof file, "test_cli_prompt.txt");
    in_dir(missing, sizeof missing, "test_cli_missing.txt");
    write_bytes(file, "hello", 5);
    expect("run -f", cli("run -m %s -f %s -c 16", m, file), 0);
    TR_CHECK(out_len == reply_len && memcmp(out, reply, reply_len) == 0);
    remove(file);
    expect("run -f, missing", cli("run -m %s -f %s", m, missing), 1);

    /* the route trace: "TRROUTE2", 8 int64 (tokens, prompt, layers, experts, used, pred, ...),
     * then chosen, pred_in, pred_out (uint16), tokens (int32), margins (float pairs) */
    in_dir(trace, sizeof trace, "test_cli_trace.bin");
    in_dir(no_dir, sizeof no_dir, "test_cli_no_such_dir/trace.bin");
    expect("--route-trace", cli("run -m %s -p hello -c 16 --route-trace %s", m, trace), 0);
    static char tb[65536];
    size_t tn = slurp(trace, tb, sizeof tb);
    TR_CHECK(tn >= 72 && memcmp(tb, "TRROUTE2", 8) == 0);
    if (tn >= 72) {
        int64_t h[8];
        memcpy(h, tb + 8, sizeof h);
        TR_CHECK_EQ_INT(h[0], 10); /* 5 prompt tokens, 5 generated */
        TR_CHECK_EQ_INT(h[1], 5);
        TR_CHECK_EQ_INT(h[2], 2);
        TR_CHECK_EQ_INT(h[3], 4);
        TR_CHECK_EQ_INT(h[4], 2);
        int64_t want = 72 + 2 * (h[0] * h[2] * h[4] + 2 * h[0] * h[2] * h[5]) + 4 * h[0] + 4 * h[0] * h[2] * 2;
        TR_CHECK_EQ_INT((int64_t)tn, want);
    }
    remove(trace);
    expect("--route-trace, no such directory", cli("run -m %s -p hello -c 16 --route-trace %s", m, no_dir), 1);
    ERR_HAS("run: could not write the route trace to '");

    /* run's usage errors, each alone */
    expect("run, no -m", cli("run -p hello"), 2);
    expect("run, -p and -f", cli("run -m %s -p hello -f x", m), 2);
    expect("run, no text", cli("run -m %s", m), 2);

    /* the history the draft is read from is sized from the context: 5 + 5 entries, where the
     * size from -n wrapped to 6 */
    expect("run -n 2^62", cli("run -m %s -p hello -n 4611686018427387904 -c 16", m), 0);
    ERR_HAS("generate: 5 tokens in ");
}

/* ---- chat and chat-template --------------------------------------------------------------- */

static void test_chat(const char *m) {
    /* the reply runs until the context is full: 11 prompt tokens, 21 positions left, and the
     * 22nd token, drawn from the last logits, finds no room; the size of the cache comes from
     * the context, where one from -n wrapped for n near 2^62 */
    expect("chat -n 2^62", cli_stdin("hello\n", "chat -m %s -c 32 -n 4611686018427387904", m), 0);
    ERR_HAS("[22 tokens");
    ERR_HAS("(context full");
    ERR_LACKS("not enough free RAM"); /* -c given: no halving, no message */
    ERR_HAS("\nexperts: 8 of 8 units in RAM");
    expect("chat -n 2", cli_stdin("hello\n", "chat -m %s -n 2", m), 0);
    ERR_HAS("[2 tokens");
    ERR_LACKS("not enough free RAM"); /* the training context fits */
    /* a system prompt: 17 prompt tokens; its reply fills the 64 positions */
    expect("chat -s", cli_stdin("hello\n", "chat -m %s -c 64 -s sys", m), 0);
    ERR_HAS("[48 tokens");
    ERR_HAS("; 17 new prompt tokens in ");
    ERR_LACKS("out of memory");
    /* the chat's reply is run's on the text the template renders (chat-template, hex): chat
     * prints "\n> " <reply> "\n" and the next "\n> ", run <reply> "\n" */
    expect("chat -s -n 4", cli_stdin("hello\n", "chat -m %s -c 64 -s sys -n 4", m), 0);
    static char chat_out[4096];
    size_t chat_len = out_len;
    memcpy(chat_out, out, out_len + 1);
    char rec_path[700], text_path[700];
    in_dir(rec_path, sizeof rec_path, "test_cli_render.bin");
    in_dir(text_path, sizeof text_path, "test_cli_render.txt");
    write_record(rec_path, "1system\x1Fsys\x1Euser\x1Fhello");
    expect("chat-template, the conversation", cli("chat-template -m %s --batch %s", m, rec_path), 0);
    char text[512];
    size_t n_text = 0;
    for (const char *h = out; h[0] != '\n' && h[0] != 0 && h[1] != 0 && n_text < sizeof text; h += 2) {
        unsigned byte;
        if (sscanf(h, "%2x", &byte) != 1) break;
        text[n_text++] = (char)byte;
    }
    TR_CHECK(n_text > 20);
    write_bytes(text_path, text, n_text);
    expect("run on the rendered text", cli("run -m %s -f %s -c 64 -n 4", m, text_path), 0);
    TR_CHECK(chat_len > 6 && strncmp(chat_out, "\n> ", 3) == 0 && out_len == chat_len - 6 &&
             memcmp(out, chat_out + 3, out_len) == 0);
    /* that reply is a cycle, "j" "\n" (a byte each): run --spec 4 accepts whole drafts in it, and
     * still stops at -n, inside a step or at its end */
    expect("run, 12 tokens", cli("run -m %s -f %s -c 64 -n 12", m, text_path), 0);
    char plain[64];
    size_t plain_len = out_len < sizeof plain ? out_len : sizeof plain;
    memcpy(plain, out, plain_len);
    TR_CHECK_EQ_INT(plain_len, 13);
    for (int n = 5; n <= 12; n++) {
        char want[48];
        snprintf(want, sizeof want, "generate: %d tokens in ", n);
        expect("run --spec 4, a cycle", cli("run -m %s -f %s -c 64 -n %d --spec 4", m, text_path, n), 0);
        TR_CHECK(strstr(err, want) != NULL);
        TR_CHECK(out_len == (size_t)n + 1 && memcmp(out, plain, (size_t)n) == 0);
    }
    unsigned long long accepted = 0, n_drafted = 0;
    const char *sl = strstr(err, "passes, ");
    TR_CHECK(sl != NULL && sscanf(sl, "passes, %llu/%llu", &accepted, &n_drafted) == 2);
    TR_CHECK(accepted > 0);
    remove(rec_path);
    remove(text_path);
    /* the second turn evaluates only what the cache does not hold: the reply's last token (never
     * evaluated), then EOS, "\n<|user|>\nhi\n<|assistant|>\n": 1 + 9 */
    expect("chat, two turns", cli_stdin("hello\nhi\n", "chat -m %s -c 64 -n 4", m), 0);
    ERR_HAS("; 11 new prompt tokens in ");
    ERR_HAS("; 10 new prompt tokens in ");
    /* /reset: the second turn starts from nothing again */
    expect("chat /reset", cli_stdin("hello\n/reset\nhello\n", "chat -m %s -c 64 -n 4", m), 0);
    OUT_HAS("(new conversation)");
    TR_CHECK_EQ_INT(err_count("; 11 new prompt tokens in "), 2);
    /* a message longer than the context is refused and dropped: the next one is a conversation
     * of its own */
    expect("chat, message too long",
           cli_stdin("0123456789012345678901234567890123456789\nhi\n", "chat -m %s -c 32 -n 4", m), 0);
    TR_CHECK_EQ_INT(err_count("(context full: the conversation is 46 tokens"), 1);
    ERR_HAS("; 8 new prompt tokens in ");
    expect("chat, no -m", cli_stdin("", "chat -n 4"), 2);

    /* a template is recognized by its exact bytes: the same length with one byte changed is
     * refused, not rendered as the one it resembles */
    char near[sizeof SYNTH_OLMOE_TEMPLATE], near_model[600];
    synth_params P = {2, 32, 2, 2, 32, 4, 2, SYNTH_TOK_MIN_VOCAB, 64, TR_TYPE_F32};
    memcpy(near, SYNTH_OLMOE_TEMPLATE, sizeof near);
    near[3] = 'B';
    synth_chat_template = near;
    TR_CHECK(synth_write(&P, self_arg, "test_cli_near.gguf", near_model, sizeof near_model) == 0);
    synth_chat_template = SYNTH_OLMOE_TEMPLATE;
    native_path(near_model);
    expect("chat, template one byte off", cli_stdin("hello\n", "chat -m %s -c 32 -n 4", near_model), 1);
    ERR_HAS("chat template is not supported");
    remove(near_model);
}

/* records "<length>\n<flag><role 0x1F content>[0x1E ...]" (src/app/main.c cmd_chat_template);
 * a role other than system, user and assistant renders nothing (src/tokenizer/chat.c) */
static void test_chat_template(const char *m) {
    char file[700];
    in_dir(file, sizeof file, "test_cli_records.bin");
    /* the flag alone: no message, the generation prompt only */
    write_record(file, "1");
    expect("chat-template, no message", cli("chat-template -m %s --batch %s", m, file), 0);
    TR_CHECK(strchr(out, '\n') != NULL && out[0] != '\n');
    expect("chat-template --ids", cli("chat-template -m %s --batch %s --ids", m, file), 0);
    TR_CHECK(out[0] >= '0' && out[0] <= '9');
    /* a role of 31 bytes fits its buffer, one of 32 does not */
    write_record(file, "1abcdefghijklmnopqrstuvwxyz01234\x1Fhi");
    expect("role of 31 bytes", cli("chat-template -m %s --batch %s", m, file), 0);
    write_record(file, "1abcdefghijklmnopqrstuvwxyz012345\x1Fhi");
    expect("role of 32 bytes", cli("chat-template -m %s --batch %s", m, file), 1);
    ERR_HAS("malformed message in batch record");
    write_record(file, "1uhi");
    expect("message without a role", cli("chat-template -m %s --batch %s", m, file), 1);
    ERR_HAS("malformed message in batch record");
    /* 64 messages fit, 65 do not */
    for (int count = 64; count <= 65; count++) {
        char body[900];
        int k = snprintf(body, sizeof body, "0");
        for (int i = 0; i < count; i++)
            k += snprintf(body + k, sizeof body - (size_t)k, "%suser\x1Fx", i ? "\x1E" : "");
        write_record(file, body);
        expect(count == 64 ? "64 messages" : "65 messages", cli("chat-template -m %s --batch %s", m, file),
               count == 64 ? 0 : 1);
    }
    write_bytes(file, "0\n", 2);
    expect("empty record", cli("chat-template -m %s --batch %s", m, file), 1);
    ERR_HAS("malformed batch record");
    remove(file);
    expect("chat-template, no --batch", cli("chat-template -m %s", m), 2);
    expect("chat-template, no -m", cli("chat-template --batch x"), 2);
}

int main(int argc, char **argv) {
    const char *self = self_arg = argc > 0 ? argv[0] : "";
    const char *sl = strrchr(self, '/'), *bs = strrchr(self, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    snprintf(dir, sizeof dir, "%.*s", sl ? (int)(sl - self) : 1, sl ? self : ".");
#ifdef _WIN32
    snprintf(bin, sizeof bin, "%s/../trochilus.exe", dir);
#else
    snprintf(bin, sizeof bin, "%s/../trochilus", dir);
#endif
    native_path(bin);
    in_dir(out_path, sizeof out_path, "test_cli_out.txt");
    in_dir(err_path, sizeof err_path, "test_cli_err.txt");
    in_dir(in_path, sizeof in_path, "test_cli_in.txt");

    if (cli("cpu") != 0) {
#ifdef _WIN32
        /* a freshly linked exe that Smart App Control blocks (docs/LESSONS.md #12) */
        printf("test_cli: SKIPPED, %s does not run here\n", bin);
        return 126;
#else
        fprintf(stderr, "test_cli: %s does not run:\n%s%s\n", bin, out, err);
        return 1;
#endif
    }

    /* the model of every case; the same with EOS 257 (token 256 still a control token); one with
     * 8 experts and a template of exactly 80 bytes, for inspect */
    synth_params P = {2, 32, 2, 2, 32, 4, 2, SYNTH_TOK_MIN_VOCAB, 64, TR_TYPE_F32};
    synth_with_tokenizer = 1;
    char model[600], model257[600], model_e8[600];
    int ok = synth_write(&P, self, "test_cli.gguf", model, sizeof model) == 0;
    synth_set_key[0] = "tokenizer.ggml.eos_token_id";
    synth_set_value[0] = 257;
    ok = ok && synth_write(&P, self, "test_cli_eos257.gguf", model257, sizeof model257) == 0;
    synth_set_key[0] = NULL;
    synth_params P8 = P;
    P8.n_expert = 8;
    synth_chat_template = "0123456789012345678901234567890123456789012345678901234567890123456789012345678X";
    ok = ok && synth_write(&P8, self, "test_cli_e8.gguf", model_e8, sizeof model_e8) == 0;
    synth_chat_template = SYNTH_OLMOE_TEMPLATE;
    if (!ok) {
        fprintf(stderr, "test_cli: could not write the models\n");
        return 1;
    }
    native_path(model);
    native_path(model257);
    native_path(model_e8);

    test_dispatch();
    test_inspect(model, model_e8);
    test_options(model);
    test_generate(model);
    test_mask(model);
    test_tokenize(model);
    test_run(model, model257);
    test_chat(model);
    test_chat_template(model);

#ifndef _WIN32
    /* --profile-json: the model's path in valid JSON, a tab in it written as \u0009, a quote and
     * a backslash escaped, a space as itself */
    char odd[600], json_path[700];
    if (synth_write(&P, self, "test_cli\tmo\"d\\el x.gguf", odd, sizeof odd) != 0) {
        fprintf(stderr, "test_cli: could not write the second model\n");
        return 1;
    }
    in_dir(json_path, sizeof json_path, "test_cli_profile.json");
    expect("--profile-json", cli("generate -m '%s' -p 2 -n 1 --profile-json %s", odd, json_path), 0);
    static char json[65536];
    slurp(json_path, json, sizeof json);
    TR_CHECK(strstr(json, "test_cli\\u0009mo\\\"d\\\\el x.gguf\"") != NULL);
    TR_CHECK(strchr(json, '\t') == NULL);
    remove(json_path);
    remove(odd);
#endif

    remove(model);
    remove(model257);
    remove(model_e8);
    remove(out_path);
    remove(err_path);
    remove(in_path);
    TR_TEST_EXIT();
}
