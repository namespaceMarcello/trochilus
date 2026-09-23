/* serve.h — `trochilus serve`: one process keeps the model, its pool and its expert store alive
 * between commands (docs/MEASUREMENTS.md question 49). See serve.c. */
#ifndef TR_SERVE_H
#define TR_SERVE_H

#include <stdint.h>
#include "../base/threads.h"
#include "../memory/experts.h"
#include "../models/model.h"

/* A command of the command line: the arguments after its name. */
typedef int (*serve_command_fn)(int argc, char **argv);

typedef struct {
    const char *model_path;    /* loaded at start when not NULL */
    int64_t n_threads;
    uint64_t expert_budget;
    int64_t idle_minutes;      /* exit after this long without a request; 0: never */
} serve_config;

/* The server: listens, runs each request's command through `dispatch` (argv[0] is the command's
 * name), until `serve --stop` or the idle time. 0 on a clean stop, 1 if it could not listen
 * (another server holds the endpoint) or failed to load cfg->model_path. */
int serve_run(const serve_config *cfg, serve_command_fn dispatch);

/* The client side of a command. argv[0] is the command's name. When a server answers, it runs the
 * command on this process's streams and *rc receives its exit code: returns 1. No server (or
 * TR_SERVER=0): returns 0 and the caller runs the command itself. `reads_stdin`: the command
 * reads the standard input (chat). */
int serve_forward(int argc, char **argv, int reads_stdin, int *rc);

/* `serve --stop` and `serve --status`: 0 done, 1 no server answered. --status prints one line. */
int serve_stop(void);
int serve_status(void);

/* What the commands load through, instead of tr_pool_create / tr_model_load_budget /
 * tr_model_free + tr_pool_destroy. In an ordinary process these are exactly those calls, the load
 * showing its progress bar on stderr when stderr is a terminal (src/app/bar.c tr_bar_load); in a
 * server the pool and the model are kept for the next request, which gets them back when it asks
 * for the same threads, file and budget. app_release takes either one NULL. */
tr_pool *app_pool(int64_t n_threads);
tr_model *app_model(const char *path, tr_pool *pool, uint64_t expert_budget, char *err, size_t err_len);
void app_release(tr_model *model, tr_pool *pool);
/* tr_model_expert_stats counted from this request's app_model: a kept store's counters would
 * otherwise carry every request before. */
int app_expert_stats(const tr_model *model, tr_experts_stats *out);

#endif
