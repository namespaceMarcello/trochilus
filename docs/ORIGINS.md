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
| ik_llama.cpp (`ikawrakow/ik_llama.cpp`, `main`) — source of ideas for the K-quant kernels, no code ported | MIT | `f3d6e6e` (2026-09-23) | `ref/ik_llama.cpp` (shallow clone, 150 MB) | — |

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
| preload experts of the next layer | one thread, prediction with mini-matmul of router plus moving average (`PILOT`, `olmoe.c:1110`) | in the layer-major prompt only, with no prediction: layer L+1's missing units while L computes (the prompt uses nearly all of them); in decode none | prediction rejected **by numbers**: it is good (92–95%, question 13) but on a 1.5 GB/s disk every wrong candidate is a read and total read regresses (question 43). The prompt's read-ahead: 1.22× at 2048 tokens, half budget (MEASUREMENTS §Reading the next layer) |
| I/O thread | one, preload only | one, the prompt's read-ahead only (`tr_experts_prefetch`); demand reads stay on the calling thread | disk gives same bandwidth to 1 and 8 readers (question 16): a thread serves only to overlap, and there is something to overlap only in the layer-major prompt |
| a read error | `exit(1)` inside `st_pread_full` (`st.h:268`), even for optional preload | eval returns -1 and session stays as it was | an engine in a library cannot close the process |
| where to read from | own format converted (`model-*.safetensors`, `st.h`) | standard GGUF, at tensor positions | decision 2026-09-17: Hugging Face files open without conversion |

### Q4_K on the CPU: what was taken from llama.cpp and ik_llama.cpp (2026-09-24)

`src/kernels` Q4_K (scalar in `kernels.c`, AVX2 and AVX-512 in `kernels_x86.c`) is new code, written
after reading llama.cpp `b49650a` (`ggml-common.h` `block_q4_K`, `ggml-quants.c`
`dequantize_row_q4_K` and `get_scale_min_k4`, `ggml-cpu/arch/x86/quants.c`
`ggml_vec_dot_q4_K_q8_K`, `ggml-cpu/repack.cpp` `block_q4_Kx8`, `gguf-py/gguf/quants.py` `Q4_K`) and
ik_llama.cpp `f3d6e6e` (`ggml/src/iqk/iqk_gemm_kquants.cpp` `DequantizerQ4K`, `Scales8K`;
`iqk_common.h` `make_q4_scales`; `iqk_quantize.cpp` `repack_q4_k`).

| Choice | llama.cpp | ik_llama.cpp | Trochilus | Why |
|---|---|---|---|---|
| the block and its weights | `block_q4_K`: f16 d and dmin, 12 bytes of 6-bit scales and mins, 128 of nibbles; weight `d*sc*q - dmin*m` | the same block | read as it is in the file; the weight `(d*sc)*q - (dmin*m)`, each product rounded to float | taken: the file format, and gguf-py's order of operations, so a dequantized row is gguf-py's bit for bit (`tools/check_dequant.py` in the gate) |
| the activations | quantized to Q8_K (8 bits, blocks of 256 with their sums), integer dot | Q8_K, Q8_K32, Q8_1_X4, integer `madd` | float, never quantized | rejected as for Q8_0 (question 21): 8-bit activations are another number, not exact |
| the min term | `dmin*m * sum(y)` per sub-block, from the sums stored with the Q8_K activations | the same (`accum_mins`) | inside each weight, element by element | the dot of the dequantized weight is the definition: bit for bit dequantize-then-dot, the closest to transformers' float matmul; the factored form rounds otherwise |
| decoding the weights in SIMD | nibble masks, then int8 products | the same, 8 scales unpacked at once with 32-bit masks (`make_q4_scales`) | AVX2: nibbles to float, `scale*q - min`; **AVX-512: the 16 values a sub-block can take computed once, `vpermps` picks each weight by its nibble**: no conversion and no mask per element | ours: float activations make a weight a lookup among 16 floats, and one AVX-512 register holds exactly 16. The 32-bit unpack of the scales was not needed: it runs once per 256 elements |
| rows repacked at load | `block_q4_Kx8`: 8 rows interleaved, `gemv`/`gemm` over them | `-rtr`, `Q4_K_R4`: 4 rows interleaved | not taken | the expert store reads the file's bytes into its slots (M1): a repack would be a transform on every read. What it buys, one activation load for several rows, matters little in decode (the activations sit in L1); a question for later |
| large prompts | 8-bit GEMM | Q4_K converted to 8-bit rows (`iqk_convert_q4_k_q8_1_r8`), then 8-bit GEMM | not taken (8 bits) | the float form of the idea, a row dequantized once per tile of tokens and the F32 kernel on it, is to be measured |
| IQ_K types | — | its own types (IQ2_K … IQ6_K), better quality per bit | not now | not among the GGUF types of llama.cpp that our reader follows; such files come only from ik's quantizer |

### Q6_K on the CPU: what was taken from llama.cpp and ik_llama.cpp (2026-09-24)

`src/kernels` Q6_K (scalar in `kernels.c`, AVX2 and AVX-512 in `kernels_x86.c`, the block in
`kernels_internal.h`) is new code, written after reading llama.cpp `b49650a` (`ggml-common.h`
`block_q6_K`, `ggml-quants.c` `dequantize_row_q6_K` and `quantize_row_q6_K_ref`,
`gguf-py/gguf/quants.py` `Q6_K`, `src/llama-quant.cpp` for which tensors Q4_K_M makes Q6_K) and
ik_llama.cpp `f3d6e6e` (`ggml/src/iqk/iqk_gemm_kquants.cpp` `DequantizerQ6K`, AVX2 and AVX-512).

| Choice | llama.cpp | ik_llama.cpp | Trochilus | Why |
|---|---|---|---|---|
| the block and its weights | `block_q6_K`: 128 bytes of low nibbles, 64 of high 2 bits, 16 int8 scales, f16 d last; weight `d*sc*(q-32)` | the same block | read as it is in the file; the weight `(d*sc)*(q-32)`, each product rounded to float | taken: the file format and gguf-py's order (`tools/check_dequant.py`: 1M floats bit for bit, scale 0 times a negative quant is -0 in both) |
| the activations | Q8_K, integer dot | Q8_K, the `-32` folded into a min term (`-32*d` times the activations' sums) | float, never quantized; the `-32` inside each weight | as for Q4_K: the dot of the dequantized weight is the definition |
| decoding the weights in SIMD | masks and shifts on bytes, int8 products | **the 6-bit values assembled on bytes, 32 at a time: low nibbles OR (high bits shifted into 0x30)** (`DequantizerQ6K::prepare`) | taken: a block's 256 quants assembled that way once, minus 32, into int8 on the stack; then Q8_0's path per element (widen, convert, multiply by the sub-block's scale) | the byte form makes 32 quants in ~5 instructions; a lookup like Q4_K's AVX-512 one does not pay here (64 values per 16 weights) |
| several weight rows per activation load | `block_q4_Kx8`, `block_q8_0x8`: 8 rows interleaved at load, then a gemm over them | `Q4_K_R4` and friends: 4 rows interleaved | **our own, without repacking: `dot_row2_x4`** takes two rows of the file as they are against four input rows (eight float accumulators, AVX-512) | the same idea, register blocking over rows, where it pays for us: the prefill's matmul is bound by loading its inputs (MEASUREMENTS §Two weight rows, prefill 1.20–1.26× on Q8_0); no repack, so the expert store still reads the file's bytes |

## Sources not yet studied (2026-09-24)

Proposed by Marcello to improve and evolve Trochilus; my first read of each, from what they declare
and what I know of them, not from reading their code. Their numbers are their authors' claims: each
counts only once measured here, with its prediction written first (MEASUREMENTS §Open questions).
When one is studied, its row moves to the tables above (ported, or ideas taken) with its commit.

| Project | What it is | What we would take | When | First read |
|---|---|---|---|---|
| ktransformers (kvcache-ai) | MoE inference on consumer hardware, CPU and GPU together: hot experts on the GPU, the rest on the CPU with its own kernels | the CPU/GPU split of the experts and how it decides what goes where; its CPU MoE kernels | **M3** (CUDA module, hot experts in VRAM): it is M3 almost to the letter | yes, at M3; little to take before |
| Adaptive-K routing | fewer than top-k experts for a token when the router is confident (its entropy); declared 24-33% less compute, perplexity within 0.5% (Mixtral, Qwen-MoE) | the idea, as our own experiment: a declared, non-exact mode | M1/M2, as a question (MEASUREMENTS question 50) | interesting here more than for them: under a partial budget every expert skipped is a unit not read from disk, and the disk is our bottleneck. Our tools measure it (expert mask, route trace, KL). Caution from our data: switching off the least-used experts overall changed the token in 6-19% of positions (§Route trace); dropping the lowest-weight ones per token should cost far less |
| ArcLight | an architecture for many-core CPUs and NUMA: memory placement and thread scheduling against cross-NUMA access; declared up to +46% throughput | pool and memory placement per NUMA node | when Trochilus targets servers | later: the reference laptop has one NUMA node, nothing to gain here today. Not known to me first-hand: verify it exists as described before counting on it |
| PCoMoE | fine-grained paths inside the experts instead of whole experts; declared up to 1.31× end to end with better accuracy | — | not now | no, for now: from what is declared it seems to need modified or retrained models, and Trochilus runs models as they are. Verify before ruling it out for good |
| bitnet.cpp (Microsoft) | an engine for ternary (1.58-bit) models, lookup-table kernels | its lookup-table kernels, for the 1-2 bit formats of M2 (TQ1_0, TQ2_0, IQ2), which our GGUF reader already knows | M2, the extreme formats | later: it runs only models trained ternary, and none exists for OLMoE or DeepSeek; the kernels are what could serve |
