# Code origins

Record of every file ported from colibri or ds4. It serves two purposes: respect licenses
(Apache-2.0 asks to say what changed) and know where to report improvements later.
Updated in the same commit that brings the code.

## Fixed references

| Project | License | Commit | Local worktree | Origin clone |
|---|---|---|---|---|
| colibri (`JustVugg/colibri`, branch `dev`) | Apache-2.0 | `a90bed9` (2026-09-17) | `ref/colibri` | `Desktop\colibri` |
| ds4 (`antirez/ds4`, branch `main`) | MIT (contains ggml code, MIT) | `8db1d1d` (2026-09-16) | `ref/ds4` | `Desktop\ds4` |
| llama.cpp (`ggml-org/llama.cpp`, `master`) — reference for comparison and source of ideas, no code ported | MIT | `b49650a` (2026-09-17) | `ref/llama.cpp` (shallow clone; build in `build-trochilus/` with `tools/build_llamacpp.sh`) | — |

To move a reference: `git -C <clone> fetch`, then `git -C ref/<project> checkout --detach <commit>`,
and update the table. Files already ported remain tied to the commit written in their row.

## Mandatory header of a derived file

```c
/* <filename> — <what it does>.
 * Derived from <colibri|ds4> <commit> <path> (<license>), modified: <what changed, one line>. */
```

## Ported files

| Trochilus file | From | Commit | Origin path | What changed |
|---|---|---|---|---|
| `src/format/gguf.c` (type table only) | ds4 | `8db1d1d` | `ds4.c` `gguf_types[]` | added tq1_0/tq2_0, iq4_nl fixed to 32 × 18 byte blocks; parser is new |
| `tools/make_tiny_olmoe.py` | colibri | `a90bed9` | `c/tools/make_olmoe_tiny.py` | GQA 4/2, 16 tokens, logit reference per position |

## Ideas taken without code

Do not require attribution, recorded to know where to look when that piece improves.

| Idea | From | Where in origin | In Trochilus |
|---|---|---|---|
| persistent thread pool, no OpenMP | ds4 | `ds4.c` `ds4_parallel_for` | `src/base/threads.c` |
| threads on physical cores, not logical | colibri | `c/omp_tune.h` | `tr_pool_create(0)` |
| each thread of the pool pinned to its own logical processor | llama.cpp (the two system calls, and the idea of a per-thread mask for `--cpu-strict`) | `ggml-cpu.c` `ggml_thread_apply_affinity`, `ggml_thread_cpumask_next` | `src/base/platform.c` `tr_thread_pin`, slot order in `src/base/cpu.c`, pinning in pool in `src/base/threads.c` |
| weights with `pread` instead of `mmap` | colibri | `c/st.h` (comment on RSS bug) | `src/base/platform.h` |
| oracle on tiny generated models | colibri | `c/tools/make_*_tiny.py`, CI job | `tools/`, `make oracle` |
| tokenizer read from GGUF metadata, "a b" merge as rank key | ds4 | `ds4.c` `vocab_load`, `bpe_rank` | `src/tokenizer/tokenizer.c` (merge becomes id pair → rank and result) |
| pretokenizer replays regex in C on codepoints; Unicode classes from generated tables | colibri | `c/tok.h` `pretok_chunk`, `c/tok_unicode.h` | `src/tokenizer/tokenizer.c` `split_gpt2`, `tools/gen_unicode_tables.py` (tables probed on HF, not Python) |
| prompt in passes of at most 512 tokens | llama.cpp | `common/common.h` `n_ubatch` | `olmoe.c` `OLMOE_DEFAULT_BATCH`, `-b` |
| (token, expert) pairs ordered by expert with a counting sort, one job per expert | ds4 (llama.cpp with an index map) | `ds4.c` `layer_routed_moe_batch`; `ggml-cpu.c` `ggml_compute_forward_mul_mat_id` | `olmoe.c` `forward_pass`, `tr_matmul_grouped` |
| one weight row against a token block, instead of whole matrix per token | llama.cpp (16 × 16 blocks), ds4 and colibri (row × all tokens) | `ggml-cpu.c` `..._mul_mat_id_one_chunk`; `ds4.c` `matmul_q8_0_batch_worker`; `c/olmoe.c` `matmul` | `kernels.c` `matmul_tiled` (blocks of `TR_MATMUL_TILE` tokens) |
| one weight row against many tokens in registers (2 in ds4, 4 in Trochilus): one load and one conversion only for all | ds4 | `ds4.c` `dot_q8_0_row_2` | `kernels.c` `k_dot_row_x4_q8_0`, `kernels_x86.c` `avx512_dot_row_x4_q8_0` |
| K and V of batch written all, then attention in parallel for (head, token) | colibri (ds4 optional) | `c/olmoe.c` `attention`; `ds4.c` `layer_attention_prefix_batch` | `olmoe.c` `attn_body` |
| logit only of the last token of the prompt | all three | `c/olmoe.c` `step`; `ds4.c` `output_logits_one` | `olmoe_eval` |
| draft from text already in prompt: search backward for the last occurrence of the tail n-gram and propose what followed | colibri (exact scan, no cache), llama.cpp (n-gram cache with counts) | `c/deepseek_v4.c` `v4_ngram_draft`; `common/ngram-cache.cpp` `common_ngram_cache_draft` | `src/gen/lookup.c` `tr_lookup_draft` (new code) |
| verify 1 + k positions in one pass, accept while the proposed token is what the model would choose | all three | `examples/lookup/lookup.cpp`; `ds4.c` `metal_graph_verify_decode2_exact`; `c/deepseek_v4.c` (acceptance loop) | `src/gen/greedy.c` `tr_greedy_step` |
| undo rejected draft by truncating cache index | llama.cpp (ds4 and colibri cannot: recurrent attention, must restore snapshot) | `llama_memory_seq_rm` | `tr_session_rewind` (`s->pos = n`) |

### Prefill in blocks: three sources (2026-09-17)

| | llama.cpp `b49650a` | ds4 `8db1d1d` | colibri `a90bed9` |
|---|---|---|---|
| prompt split | `n_ubatch` 512 | none on CPU: whole prompt; FFN in sub-batches of 128 (`DS4_PREFILL_BATCH`) | whole prompt in `step()` |
| projections | sgemm, Q8_0 activations | weight row × tokens pairwise, Q8_0 activations | weight row × tokens, float |
| MoE | rows per expert with index map, 16 × 16 blocks, dot Q8 × Q8 | counting sort per expert; gate and up with one quantization only; down and sum per row, experts in id order | no grouping: token by token |
| attention | flash attention or matrix with mask | serial KV write, then attention per token (optional per head and token) | all K and V, then (head, token) in parallel |
| exactness against one token per pass | no (int8 activations) | no, no test | not declared or proved |

Not taken: int8 activations (not exact: lever 2, separate option) and ds4's per-row sum over all
experts, which rereads intermediate activations of the whole batch at every weight row (with 512
tokens ~16 MB per row): Trochilus keeps expert order in token blocks, and sums per token.

### Thread placement: three sources (2026-09-17)

| | llama.cpp | colibri | ds4 |
|---|---|---|---|
| how many threads | `logical / 2`, or physical cores where Windows count is compiled (excluded on MinGW-w64, UPSTREAM #4) | physical cores counted truly (`GetLogicalProcessorInformationEx`, `thread_siblings_list`, `hw.perflevel0`), never guessed | `omp_get_max_threads()` |
| where they go | nowhere: without `--cpu-mask` does not call affinity | `OMP_PROC_BIND=close` passed to libgomp, Linux only, with process re-exec | nothing |
| who decides the mask | the user, by hand (`--cpu-mask`, `--cpu-strict` = one processor per thread) | libgomp | — |
| beyond 64 processors | does not reach (`SetThreadAffinityMask`, UPSTREAM #3) | depends on libgomp | — |

Taken: the two system calls (Windows and Linux) and the idea of `--cpu-strict`, that is, a mask
of one processor only per thread instead of a common mask. Not taken: the mask written by hand
by the user (here the topology is counted by the engine), `OMP_PROC_BIND` with re-exec (no
OpenMP, and environment variables read inside the engine are anti-pattern from colibri), and
`SetThreadAffinityMask` (replaced by `SetThreadGroupAffinity`, which reaches every group).
New here: the **order** of slots (one thread per physical core before any SMT sibling, and cores
taken in rotation across last-level caches, because 4+4 on two chiplets beats 8 on one), respect
of a mask already set on the process (`taskset`, `start /affinity`), restore of caller's affinity
when the pool dies, and a test that **asks the operating system** which processor ran each chunk
(`tests/test_base.c`): none of the three sources verify this.

### Speculation from prompt: three sources (2026-09-17)

| | llama.cpp | ds4 | colibri |
|---|---|---|---|
| where the draft comes from | three n-gram caches (context, previous session, corpus), n from 4 to 1, vote `count × corpus count` | no prompt lookup: DSpark and MTP, that is, trained heads (`docs/SPECULATIVE_DECODING.md`) | `v4_ngram_draft`: 3-gram then 2-gram, backward scan, most recent occurrence |
| verify | one `llama_decode` on draft batch, logits at each position | 2–3 rows per pass, `row0_top == draft1` | loop `predictions[i] == drafts[i]` |
| undo a rejection | cache truncation by index | buffer swap with snapshot (recurrent state) | `spec_attention_restore` and prefix re-execution (measured so costly it was disabled for MTP) |
| tokens identical to generation without speculation | algebraically yes, bit by bit **not declared** (batch may sum in another order) | declared **not** identical to bit on long continuations | not declared |

Taken: the idea of draft from context (colibri, simplest: no state) and verify in one pass with
logits at each position (all three). Not taken: llama.cpp's caches with counts (allocations and
state, to evaluate only if measurements ask), trained heads (DSpark, MTP, EAGLE3: other models),
ds4's "opportunistic" sampling. Restore by snapshot is not needed: our cache is indexed per
position, we go back with `s->pos = n`. Plus, here tokens are **byte-identical**, because each
row of the batch is the same computation as the one-token pass (`tests/test_prefill.c`,
`tests/test_spec.c`): none of the three sources prove this.


### Expert store: what was taken from colibri and what was not (2026-09-20)

`src/memory/experts.{h,c}` is new code (no line ported), written after reading
`ref/colibri/c/olmoe.c` (`LCache`, `Slot`, `expert_get`, `pilot_worker`) and with measurements
of questions 13–16 in front (`docs/MEASUREMENTS.md` §M1). Three bugs found in that reading are in
`docs/UPSTREAM.md` #7–#9.

| Choice | colibri | Trochilus | Why |
|---|---|---|---|
| find an expert in cache | array `slot_by_expert[eid]`, O(1) (`olmoe.c:158`) | `slot_of[unit]`, O(1), unit = (layer, expert) | taken: the right thing |
| one expert in memory | one block for three matrices (`load_expert_merged`, `olmoe.c:636`) | one slot per unit, three parts aligned to 64 inside the slot | taken halfway: in our GGUF the three tensors are separate, so three parts are still read with three `pread` (single read impossible without our format, excluded 2026-09-17) |
| choose who to evict | linear scan of all slots on every miss, O(capacity) (`olmoe.c:696`) | doubly-linked list, O(1) | rejected: worsens as budget grows, the very case we care about |
| keep in RAM the most-used experts (pin) | `pin_hot_experts` after 5 tokens, with escape if all slots blocked (`olmoe.c:745`) | no pin: LRU only | rejected **by numbers**: on routing trace pin from use reads more than LRU at every capacity (at 75%: 94 vs 32 MiB per token, question 14) |
| preload experts of the next layer | one thread, prediction with mini-matmul of router plus moving average (`PILOT`, `olmoe.c:1110`) | none, for now | rejected **by numbers**: prediction is good (92–95%, question 13) but on a 1.5 GB/s disk every wrong candidate is a read and total read regresses (question 43) |
| I/O thread | one, preload only | none: read on calling thread | disk gives same bandwidth to 1 and 8 readers (question 16): a thread would serve only to overlap, and without preloading there is nothing to overlap |
| a read error | `exit(1)` inside `st_pread_full` (`st.h:268`), even for optional preload | eval returns -1 and session stays as it was | an engine in a library cannot close the process |
| where to read from | own format converted (`model-*.safetensors`, `st.h`) | standard GGUF, at tensor positions | decision 2026-09-17: Hugging Face files open without conversion |
