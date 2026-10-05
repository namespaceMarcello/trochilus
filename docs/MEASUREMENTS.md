# Measurements

Every optimization attempt, successful or rejected, with its number. Rule
(`docs/ARCHITECTURE.md` §C and assembly): a gain counts only if it shifts the median by more
than the spread measured on the same row. Base tables are replaced when they change; attempts
are added at the end.

**Never a number from a single run** (Marcello, 2026-09-17): every measurement is the median
of N runs with min, max, and spread, and the generated tokens must be identical across all
runs and with every number of threads (`tools/profile_suite.py` verifies it and exits with an
error if not). Rows marked below as «one run» predate the rule and need to be remeasured.

## Open questions: what to measure

Tests to drill down and find where to focus. Added when a doubt arises, moved to the right
section with its number when measured.

| # | Question | How to measure | Why it matters |
|---|---|---|---|
| 1 | ~~How much does waking threads on native Windows cost?~~ Measured: §Thread pool | — | — |
| 2 | ~~Do 8 threads win because they're in one core group (CCD, 32 MB L3)?~~ No: 4+4 on two CCDs go **better** than 8 on one. Measured: §Where threads go | — | — |
| 3 | ~~Do SMT copies help or hurt?~~ **Hurt**: 32 pinned threads (two siblings of each core) vs 16 (one per core) give prefill 110 vs 164 and decode 2.2 vs 22.9. Measured: §SMT | — | — |
| 4 | ~~What is real RAM bandwidth on this machine?~~ **~57 GB/s** reading with 4-6 threads, then drops
(8: 55.7, 12: 54.4, 16: 49.5; one thread 26, two 52): remeasured clean machine 2026-09-19
(§Clean machine remeasure; earlier «~54, one thread 22-25, two 43-49» had four cores taken,
LESSONS #84), same sequential and 2 MiB or 256 KiB sparse blocks; 4 KiB page sparse 7 GB/s
one thread and 34-41 from 6-16. Engine matmul reads 46-58. Earlier «41 GB/s» was 17/09 engine
speed, not limit. `memcpy` not measured: decode reads, doesn't write. Measured: §Decode at
long context | — | — |
| 5 | ~~How many TLB misses per token, and value of 2 MB pages?~~ **Closed as "no" on Windows
2026-09-19 (Marcello)**: large pages need privilege (`SeLockMemoryPrivilege`) normal user
doesn't have, target is PC without options. On Linux optional measurement, not a step | Linux:
weights with `madvise(MADV_HUGEPAGE)` vs without, same median tok/s | 300,000 4 KB pages
touched per token |
| 6 | Does laptop slow when hot? | 60 s continuous decode, tok/s every 5 s | short measure may not hold for real use |
| 7 | ~~How much does attention cost as context grows (128, 1024, 4096 tokens)?~~ **In decode**
(§Decode at long context): 0.5 ms at context 32, 2.9 at 512, 11.5 at 2048, 22.6 at 4000 on
25-48 ms token, i.e., KV bytes read at 47 GB/s. **In prefill** (§Prefill on long prompts): zone
was 5%, 18%, and 31-35% at 512, 2048, 4000, mostly softmax (51-72%: C lib `expf` at 30 ns),
then products and weighted sum (27-37%), and repeated key/value read only at 4000 (0-39% per
run). With grouped attention zone is 1.09×, 1.12×, 1.13×; what remains is question 37 | — | — |
| 8 | What does profiler cost on real model? | same scenarios with and without `--profile` | to trust zone percentages |
| 9 | Native Windows and Linux on same machine same speed? | same scenario native and Docker/WSL | in Docker waking thread cost ~20 µs |
| 10 | Do more lanes (32 or 64) unlock sum chain on one thread? | experimental kernel, `make bench` | AVX-512 goes like AVX2 for that limit; would change numbers (stated) |
| 11 | Cost of division per row and indirect call in `matmul_body`? | kernel takes whole matrix vs one per row | 1024 calls and 2 64-bit divisions per matrix |
| 12 | ~~Do experts chosen in consecutive tokens repeat?~~ **Partly yes** (§Adversarial review): one more
row in same pass reads 2.3-4.7 new experts per layer of 8, i.e., 40-70% of its experts already
read by other rows. By time (native zones, idle machine): one more row costs **17.6 ms** if only
one and **13.7 ms** each if eight, vs 31.3 per pass; experts have **91%** and **61%**, time
follows MiB of new experts (30 MiB/ms at both points); dense multiplies free for first row and
cost 4.1 ms per row at nine. Draft cost attaches on weights side | — | — |
| 13 | ~~Does next-layer router applied to current-layer state guess experts that will be chosen?~~
**Yes** (2026-09-20, §M1 before writing code): predicting 12 finds 92.4% of chosen ones already
with state entering FFN (before experts run), 94.9% with layer output; first layer exception
(75-80%) | — | prediction exists, but **on 1.5 GB/s disk prefetch doesn't pay** (§M1 point 6:
each wrong candidate is a read, read is bottleneck): first M1 reads on demand; prefetch only
on fast disks (question 43) |
| 14 | ~~How many bytes per token remain to read from disk with expert cache 25, 50, 75% of model?~~
**Measured on real engine** (2026-09-20 night, §M1 measured): **3.7 / 1.8 / 0.5 MiB per token**
right after prompt and **0.9 / 0.2 / — far from prompt** (0.6 / 0.3 / 0.1 and 0.14 / 0.03
misses). Trace simulation said 348 / 143 / 32 MiB (54.6 / 22.4 / 5.0 units; model superseded by measurement):
answered different question, engine enters decode with LRU filled by prompt, reads
whole model anyway. Strategy comparison holds: static pin-by-use loses to LRU at all capacity |
second model and prose (mask for use invalid outside domain, question 44) | M1 default: LRU,
on-demand reads; cost in prompt, not token |
| 15 | ~~Stream from disk whole layers (DeepSpeed) or experts: bytes per token?~~ **Per experts**:
whole layers 5106 / 3404 / 1702 MiB per token, 15-53× more (§M1 before writing code) | — | — |
| 16 | ~~What does NVMe really read in expert-sized blocks at random positions, no cache?~~
**~1.5 GB/s** (1.4-1.5 at 2.125 MiB blocks, 1.6-1.75 at 6.375 MiB), same with 1, 4, or 8
readers; with question 14: 7-10 tok/s cache 50%, 17+ at 75%, ~4 at 25% (§M1 before writing
code). Criterion «≥ 5 tok/s at 50%»: passed. Old KV on SSD: not measured, expect no | `make
bench-disk` on other machines | M1 for experts; 1-2 I/O threads enough |
| 17 | Reloading KV from disk for already-read file cheaper than prefill? KV in q8 changes token? |
checkpoint: logits identical to bit after reload, time ≥ 10× below prefill; q8: greedy tokens
same ≥ 99% on 1000 code tokens | coding levers (files already read, long contexts), M5 |
| 18 | ~~Why does decode lose 11-13% from 32 to 512 token context, llama.cpp only 5-7%?~~ Because
KV read **slower than weights**: 32-36 GB/s vs 47-54, one head reading 512 bytes every 8 KiB.
With one head's positions in a row KV reads at 47 GB/s and 32-512 decode loses 8.8% (38.9 →
35.5), cost of extra bytes. Measured: §Decode at long context. Comparison with llama.cpp
alternating runs stays question 19. At 2048 (2026-09-24, §Decode at context 2048): the whole gap
(1.16–1.26×) is our F32 KV against their F16, 518 against 259 MiB per token; the lever is point 6 | — | — |
| 19 | Decode 4-16 threads: is llama.cpp really ahead 8-12%? | `tools/speed_compare.py` two engines
alternating run per run in same session (between sessions llama.cpp median moved 3-7%) | says
if leverage in multi-thread decode or just noise |
| 20 | ~~Prefill yields 1.16× 8 to 16 threads: laptop power limit?~~ No, not CCD either: scheduler
puts two threads on same physical core. Measured: §Where threads go | — | — |
| 22 | Pin threads to physical cores worth **on decode** and **with machine busy**? Measured
(§Thread pinning): on 16-thread decode **not distinguishable** from no pin (A/A remeasure:
−1.8%, threshold 2.2%; old 0.82× processor pin doesn't reproduce) and **yes at 4-8** (1.14-1.32×,
processor pin, 3 rounds); machine busy prefill holds (1.19×) and decode loses (0.67×, 3 rounds,
not remeasured). Only **Linux** part remains: not measurable on this machine (no native Linux,
WSL2 topology synthetic, LESSONS #47); Linux code runs and tested in `make check`, numbers no |
real Linux machine, same scenarios | pin is product decision: other rows already decided (on) |
| 26 | ~~How many threads does **each phase** want?~~ **Remeasured clean machine 2026-09-19**
(§Clean machine remeasure; 18/09 numbers had four cores taken, LESSONS #84). Prefill all (16
vs 8: 1.49× at 512, 1.57× at 2048). Decode **few**: 4 short context (512 auto picks 4, beats 8
by 4-6%, above threshold), 8 long context, 16 wins nowhere but **not distinguishable from 8**
(1.00-1.02× at 512, 0.99-1.09× at 2048, threshold 4.6%): «1.10× at 8 threads» 18/09 was
taken-cores effect. Why: RAM bandwidth hits ceiling 4-6 readers then drops (question 4). Engine
doesn't carry written number: each session measures its width and `--decode-threads` forces;
but estimator needs rewrite (question 31) | — | — |
| 31 | **Does decode width estimator rewritten 2026-09-19 choose on real model what forced truth
indicates?** (LESSONS #88, §Clean machine remeasure point 3). Old one guessed: widest within fixed
1% of fastest on three one-token passes, which have 2-5% noise (different choice every 8 quiet
machine, 4×4 1×8 3×16 with noise, 3-5 changes on 8 at 2048 with `yes`; median didn't show). New
one (`src/models/model.h`): center = fastest pass, noise = second's gap; **narrowest** within
pair's noise (it and fastest, never third width's); if margin decides other passes on both, up to
6; remeasure each context doubling from 32; change only with two agreeing measures (second
after ≤ 128 passes); no probes below 4 threads. **No real-model numbers yet.** Then: holds on
other machines (4-8 core, more channels, Apple Silicon no pin, P/E cores)? | done: three pure
functions and session with fake clock in `tests/test_phase.c`, ten red mutations
(`tools/mutate_tune.sh`), choice history in `threads:` row. To do **quiet machine, night**:
`decode_context.sh widths` (forced truth: 16 vs 8, 4 vs 8), `decode_context.sh measure 16`,
`decode_context.sh long` (1500 tokens after 1000-token prompt: two remeasures, changes from
history), `threads_phase.sh widths`. Criterion: auto within A/A of best forced each context,
zero changes between runs, at most 4→8 change in long run | first next step: rule picks each
user's each session default |
| 32 | `--spec`: can decide **before trying** if draft is worth it, instead of discovering wrong
(Marcello idea, 2026-09-19)? Today just last 2 words appearing in text (`TR_LOOKUP_NGRAM_MIN` 2)
tries it, new code pair returns by chance | count without stopwatch, accepted and proposed
drafts **by match length** (2, 3, 4) on `code.txt` and `code-edit.txt`; then exact replay with
measured costs (17.6 and 13.7 ms per row) of rule asking longer matches when drafts failing | worst
case 0.95× and pause constants not enough (max 0.97-0.98×): if short matches fail, −5% drops
without touching +17% |
| 33 | Between float (exact) and int8 (1.8-2.3× kernel, not exact) middle path for prefill (Marcello
question, 2026-09-19)? Candidate: **16-bit** activations with `vpdpwssd` (VNNI), rounding error
256× smaller than int8 | in bench like int8 (`tests/bench_kernels.c`, same 4-token kernel
structure, alternating runs), more than logits move on 2-layer model | speed expectations low
(half int8's lanes plus 16-bit weight widening) but cheap to know; other paths: prefill serial
part (question 30, exact) and 4-bit models accelerating decode, exact for own weights |
| 27 | ~~Why does 16-thread pinned decode go 17% slower than no pin?~~ **Premise doesn't hold**:
with 8 rounds and A/A check processor pin at −3.4% from no pin and −1.6% from core pin (not
distinguishable), not −17%. **Core** pin stays default for prefill (1.27× on no pin, +9% on
processor pin). Measured: §Thread pinning, §Adversarial review | — | — |
| 23 | ~~How many drafts accepted on real code work?~~ Measured: §Speculation from prompt. 64%
rewriting file already in prompt (1.42×), 13% writing new code (0.62×) | — | — |
| 24 | ~~How much does wrong draft cost?~~ **Much more than container measure said** (5.2 ms):
native with pin, one more row costs **13.7-17.6 ms** vs 31.3 per pass (profiler zones,
§Adversarial review; tok/s gave 15-27), because draft token picks different experts and pass
reads their weights too. Break-even not at 15% draft acceptance but **44-56%** (§Adaptive draft,
LESSONS #59) | — | — |
| 25 | ~~Draft shortening after reject, lengthening after accept removes worst case?~~ **Reduces,
doesn't remove.** Shortening not enough (0.89×): need **stop** after full-wrong draft, pause
doubles. With it, A/A remeasured 8 rounds: **0.953×** when model invents (earlier 3 rounds said
1.01×) and **1.175×** when copies (§Adaptive draft, §Adversarial review) | — | — |
| 21 | ~~How much of prefill gap vs llama.cpp is int8 VNNI dot vs our float?~~ **Almost all**
(§Adversarial review, LESSONS #65; first answer «nothing» compared one row vs one token):
with our 4-token kernel structure int8 VNNI 512-bit dot is **1.8-2.3×** our x4 float, llama.cpp
ahead 1.57× one thread. Int8 not exact: whether to write it and how as explicit mode, Marcello
decides | — | — |
| 28 | ~~Where is the prefill gap that remains with llama.cpp, if not in int8?~~ It was in int8 (question 21). The exact "more tokens in registers" path is measured and does not pay off: one row against 8 tokens gives 1.13× at n=1024, **0.79×** at 2048 and 0.93× at 4096 (§Adversarial review) | — | — |

| 29 | ~~How much is a SIMD variant for **F16** worth?~~ **49×** on the kernel (`make bench`, n=2048: 466 M el/s the scalar, 22.8 G el/s AVX2 with F16C, 22.7 AVX-512; before, every tier ran at the scalar's 466-824). The half→float conversion is exact, so it stays bit-identical (`tests/test_kernels.c`, also with subnormals, infinities and NaN). Written 2026-09-19 because the new check, `tests/test_tier_used.c`, requires that **every type** of weight pass through the tier's kernels (LESSONS #78) | — | — |
| 30 | ~~Can the prefill's serial part (per-expert copy, KV write, norms, RoPE) be split by token?~~ **Yes** (§Prefill on long prompts): it cost 8% of the prefill at 512 and 5-6% at 4000, counting the router's choice (57 ms over 512 tokens) that sat inside the `router` zone. Split on the pool per token (at least 8 per piece) it goes from 121 to 48 ms at 512 and from 880 to 353 at 4000: norms and RoPE divide by 6, the memory copies (rows for the experts, KV write: 27 → 19 ms) only by 2 and by 1.1-1.4, because there the limit is writing to RAM. With int8 (question 21) the weight of what remains doubles | — | — |
| 34 | **Closed as "no" on 2026-09-19 (Marcello)**: 2.1% estimated at context 4000, below the measurement threshold; it reopens if a model with contexts past 4000 pushes it above. ~~Width per zone~~: the decode's attention over the whole pool and the rest over the decode's width (exact: the pool's contract). Potential measured per zone (§Decode at long context and RAM bandwidth): at context 4000 attention takes 21.6 ms on 16 threads and 22.6 on 8, while experts and projections on 16 lose 1.3 ms; keeping the best of the two removes **1.0 ms of 48.1 (2.1%)**, at 2048 0.24 ms of 36.4 (0.7%) | `tr_pool_set_active` around the `attention` zone in short passes; needs a threshold lower than tonight's 3.8% (more runs, or 200 tokens generated instead of 48) | grows with context: past 4000 attention exceeds half the token. Below the threshold today, so not written |
| 35 | What is left under the ceiling in the decode, on the exact side? `attn_out_proj` reads at 43-52 GB/s where the other multiplications sit at 52-55 (0.2-0.3 ms per token); the zones with no bytes (norms, RoPE, per-expert copy, activation, sum, token choice) cost **1.1 ms per token**, 4.4% at short context; attention reads at 47 GB/s where reading the same bytes alone gives 52-56 (the dot product, softmax and weighted sum cost 12% of the zone) | profile per zone with the small zones split; `bench_mem kv` with the kernel in pieces | in all the 6-11% that separates the decode's 48-51 GB/s from RAM's 54 |
| 36 | **16-bit KV** (not exact, Marcello decides): by how much do the logits move? Gain estimated from the bytes (§Decode at long context and RAM bandwidth): +5% at context 512, +16-19% at 2048, +26-31% at 4000, and half the KV's memory | a declared mode, off by default; against the exact mode on the real model: average KL and first token equal on ≥ 1000 code positions up to context 4000 (`trochilus logits` in both modes, `tools/compare_llamacpp.py` already computes the KL: llama.cpp sits at 9e-3), greedy tokens equal on 1000 generated (question 17's threshold: ≥ 99%), and no K or V value beyond 16 bits' maximum (65504) | it is the only big lever left to the decode at long context: past the KV per head, attention already reads at RAM's bandwidth |
| 37 | ~~**Our own `expf`**: the softmax (attention, router) and the experts' activation called the C library's `expf`, 30 ns a call with MinGW and different bits with glibc~~ **Scalar written on 2026-09-19** (`tr_expf`, §`tr_expf`): correctly rounded on all 4 278 190 082 floats by exhaustive proof in `make check` (gcc and clang), 3.5 ns a call, no library call inside; on Windows the same bits as before (logits identical to the byte), on Linux KL 3.9e-13 and 1000 of 1000 tokens equal, and the same bytes as the emulation build; **Windows and Linux now give the same logits to the byte**. Prefill **1.03-1.08× at 512, 1.20-1.23× at 2048, 1.29-1.31× at 4000**, decode 1.02-1.10× (estimated 1.12 / 1.23 / 1.27× and +5-10%). SIMD remains: question 39 | — | — |
| 38 | Grouped attention with **GQA** past 4096 tokens (Qwen3-Coder: 32 query heads over 4 key heads): a group today is 16 tokens of **one** query head; the 8 heads that share keys and values could sit in the same group and read them only once. And at 16-32 thousand tokens a head's keys and values (16-32 MB) fit in no cache: there the repeated read, which at 4000 ranges from 1% to 21-39% of the zone depending on the run, becomes the bulk | `make bench-attn` with the new model's shapes and `--heads`/`--group`; prefill at 8, 16, 32 thousand tokens once the model exists | coding uses long contexts; at 4000 a group of 4, 16 or 64 performs the same, beyond that is not known |
| 39 | **Closed as "no" on 2026-09-19 (Marcello)**: 1.01-1.04× estimated sits at the noise threshold, for about 200 lines in the project's most delicate piece; the scalar has already taken almost everything. ~~`tr_expf` in SIMD, yes or no~~ (numbers in §`tr_expf` point 7). After the scalar, in the prefill at 512 / 2048 / 4000, the attention's softmax remains at about 9 / 130 / 610 ms and `expert_act` at 25 / 108 / 218 ms. Estimate with an AVX-512 exponential at 1.0-1.3 ns per element: prefill **1.01× / 1.03× / 1.04×**, decode +1-2%; cost about 200 lines, the delicate piece (table gather, mask for lanes that fail the test, bit identity with the scalar proven on every float for every tier) | a sketch in the bench (`tests/bench_expf.c`) would give the real per-element cost before deciding; then an `exp` entry in the kernel table, `make bench-expf` per tier, `prefill_context.sh change` | the scalar has taken almost everything: at 4000 tokens the estimated gain is close to a good session's threshold (3%) |
| 40 | ~~Are Windows's and Linux's **RoPE tables** the same entry by entry?~~ **Yes**, measured on 2026-09-19 (`sh tools/platform_bits.sh`, §`tr_expf` point 5): over 4096 positions × 64 pairs the two libraries' `cos` and `sin` **in double** differ in 0.83-0.85% of entries (2175 and 2216 of 262 144: glibc and MinGW round the double's last digit differently), `pow` never (64 of 64 correctly rounded on both), and **none** of the 524 288 entries in float differ: rounding to float absorbs the difference. With `tr_expf` the engine therefore gives **the same bytes on both platforms** up to OLMoE's training context | — | — |
| 41 | **Why does the disk give 1.5 GB/s, a third of its card's spec (Micron 2400, 4.5 GB/s)?** **Answered 2026-10-02 (§The disk at its limit)**: the request's size: 1.9 GB/s at 2.125 MiB, **3.5 at 64 MiB** (78% of the spec), flat past it; the order, the requests in flight and the handles do not move it; one request scattered into the slots runs level; the decryption costs the System process +0.8 logical processors at 2 GB/s, +3.8 at 3.5. The engine's lever: a layer's experts in three requests (the GGUF's tensors are already in a row). Three hypotheses: Windows's volume encryption (read with `manage-bde -status C:` as administrator), QLC with no DRAM on sparse reads, the synchronous 2 MiB request (try larger, overlapped requests). Worth three times M1's tokens per second with the cache at 50%. **2026-09-20: the volume is encrypted** (`manage-bde`: BitLocker ON, XTS-AES 128, so in software). It is not proven to be the cause, nor the only one: during reads the System process goes from 0.79 to 0.94 processors (`tools/background_load.ps1`, `build/disk/run3.txt`), no core is saturated, so if the limit is there it is latency, not compute. The real test needs an unencrypted volume on the same disk, and **encryption is not removed for a measurement**: Windows 11 PCs ship encrypted from the factory, so ~1.5 GB/s **is** the target, and M1's automatic plan measures the disk it finds | `bench_disk --block` at 16 and 64 MiB; more requests in flight per reader; the same bench on an external disk or an unencrypted partition, if one turns up | after the first M1 run: the exact path first, then the bandwidth |
| 43 | **From what disk speed up does prefetching pay off, and with how many candidates?** The time model says: at 1.5 GB/s no (k=8 neutral, k=12 −40%), at 4.3 GB/s k=8 gives 11-15% (§M1 point 6) — model, not measured: one disk, one read at a time — and the decode by now has little to hide: far from the prompt it misses by 0.03-0.14 units per token (§M1 measured: what experts really cost from disk), so if prefetching helps it is in the **prompt** | with M1 running: `--expert-budget` at 50%, prefetching off and with k=8, on this disk and on a fast one (or on the file in the system cache, which gives 12-26 GB/s); threshold in the automatic plan | after the first M1 |
| 42 | **How is the first layer predicted?** Prediction with the next layer's router takes 92-95% everywhere except at layer 0 (75-80%, question 13) | from the trace: layer 0's experts chosen equal per token (do they depend almost only on the embedding?); alternatively layer 0 always stays in RAM (64 experts = 408 MiB, 6% of the model) | M1's project |
| 44 | ~~Is the model's behavior on code a small, deterministic graph?~~ **No, in all four senses** (Marcello, 2026-09-20; closed on the night of 2026-09-20 with the five texts of the functional proof). From the trace at hand (OLMoE-1B-7B, `code-1000`, 1204 tokens; one-off computation on the trace, not yet in the report): **small, no**: 1012 of 1024 units used, 89% already after 100 tokens; the top 25% used covers 68% of activations, the top 50% 88%, the top 75% 97%; usage entropy 5.1 of 6 bits per layer. **Static, no**: a cross-layer co-occurrence graph learned on the first 900 tokens guesses 53.7% of the next layer's experts on the following 300 (frequency alone: 40.3%; the router on the live state: 82-86%); no whole path repeats (0 of 1204), the single set of 8 does (32%); 3.6 of 8 experts shared with the previous token (chance: 1.0), which is what the LRU exploits. Consistent with the usage pin that loses against the LRU (question 14) and with the balancing loss MoEs are trained with. Limits: one general-purpose model, one prompt, one language | missing, with the thresholds written **beforehand**: (1) token ids in the trace → same token, same experts? (the "table" version of the hypothesis, plausible at layer 0); (2) 4-6 traces (different files and languages, and control prose) → does the overlap of hot units code-to-code exceed code-to-prose?; (3) the margin between the 8th and 9th expert (probability in the trace); (4) the proof that decides, functional and not about routing: **masking** the experts outside the top X% used and measuring equal tokens and KL against the whole model on code never seen (measurement-only mode) **Thresholds written before measuring (2026-09-20, Marcello's yes)**: *small* = the top 25% of units covers ≥ 99% of activations on code; *code graph* = the overlap (Jaccard) of the hottest 25% between two code traces exceeds that between code and prose by ≥ 0.20; *table* = same token id → same set of experts at layer 0 in ≥ 95% of repeats (the other layers are reported); *functional* = with 50% of the units masked (the least used, on **another** code file) the greedy token matches in ≥ 99% of positions and the average KL is ≤ 1e-2 on code never seen (the project's yardstick for a non-exact mode: llama.cpp sits at 9e-3). A missed threshold falsifies that part of the hypothesis for OLMoE-1B-7B; to say it "of models" needs at least a second model | **Answer (§Is code behavior a small, deterministic graph?)**: small no, table no, static no, functional no (already on the mask's own text: 93.9% of tokens with 50% off); one **region of code** yes (Jaccard 0.68-0.77 between C, Python and shell, 0.07-0.09 with English prose), and usage orders the experts 13-50 times better than chance **inside code** and not at all outside it (on English prose the usage mask performs like a random one). Functional proof on five texts out of five: with 50% off the token matches in 93.9 / 92.8 / 86.0 / 81.4% (mask's own text, other C, Python, shell). A second model remains |
| 45 | **Can the first prompt's cost be removed, or only hidden?** Under budget, every start rereads the whole model: 4.7 s measured at 2048 tokens, and that is the gap that remains against the resident model (12.41 vs 7.47 s). **Watch the mechanism**: reloading at startup a list of the units that were in RAM removes nothing — rereading them from disk *is* that time, and with direct reads the system has no copies to give us for free. It is removed only one way, **keeping the bytes in RAM between one start and the next** (a process that stays up, question 49); otherwise it is **hidden** behind the human's own time (question 48) | the second start with the live store against a cold one, same prompt: prefill and the first 100 tokens' misses | it is the gap that remains between a partial budget and the resident model |
| 46 | ~~Why does the decode under budget improve with the length of generation?~~ **It is the store, not a warm-up** (2026-09-21, §M1 measured: what experts really cost from disk): with the resident model the time per token does not change from 200 to 1000 (34.64 → 33.48, within the spread), under budget it does (26.53 → 29.84 at 25%). A fixed cost of ~0.35 s at 50% and ~0.84 s at 25%, which the first tokens' extra misses explain for only ~115 ms. **Remains**: where the rest goes | zone profile of the first 64 tokens against tokens far from the prompt, same budget | says whether there is a lever in the LRU past the prompt, and what number to promise with half the model in RAM |
| 47 | ~~Does the prefill under budget read the experts once per prompt?~~ **Now yes** (per-layer order, 2026-09-21: 6 273 MiB and 12.41 s at 2048, **1.89×**). Before: once per pass (2026-09-21, §Prefill reads model once per pass). The prompt is processed in blocks of 512 tokens (`OLMOE_DEFAULT_BATCH`) and every pass walks all the layers, so under budget it rereads the whole table: at 2048 tokens **22 880 MiB instead of 6 528**, 3.65×, growing with the prompt. With a single pass (`-b 2048`) the prefill takes 11.27 s instead of 23.51 (**2.09×**) and the last logit row is identical to the byte | the per-layer order's timing once it exists, and the same measurement at 4000 tokens (eight passes) | the prefill's biggest lever under budget, and it costs no precision |
| 48 | **How much of the first prompt hides behind the time the user spends typing?** A declared preparation phase (Marcello's idea, 2026-09-21): as soon as the session opens, and before the prompt arrives, the engine starts reading the experts and says so ("reading the model, 6.5 GiB"). It does not remove the 4.7 s, it puts them where they do not bother anyone; it costs no bit of precision and needs no new process. To decide: what to read first when the prompt is still unknown (the layer order is the right one, layer 0 is needed first) | time between opening and the first token, with and without the early read, on a prompt that arrives after 5, 15 and 30 seconds | it is the cheapest lever: no new structure |
| 49 | ~~Does a process that stays up between sessions pay for what it costs?~~ **At full budget yes, at half budget little** (measured 2026-09-23, §The engine kept between commands). `trochilus serve` (built 2026-09-23, Marcello's go) keeps pool, model and store; `generate`, `logits`, `run` and `chat` run in it when its endpoint answers, byte for byte the same output (`tests/test_serve.c`). First prompt of 2048 tokens: **full budget 12.79 → 7.81 s** (the 4.98 s load gone, prefill unchanged); **half budget 13.00 → 12.04 s** (0.926×, A/A 1.020×), 139 misses fewer of 1 157. Prediction written before measuring: full loses the load (held); half gains little because the sweep evicts what the next sweep needs first, misses warm = cold (held in the mechanism, off by 12% in the misses) | where the 139 fewer misses come from (per-phase misses: prefill and decode apart); a sweep-resistant eviction at half budget (model: up to ~511 units kept, ~2.5 s) | the gap to the resident model is gone where the model fits; below budget the lever is the eviction policy |
| 50 | **How much disk does Adaptive-K take away, and what does it change?** (Marcello's source, docs/ORIGINS.md §Sources not yet studied): a token uses fewer than 8 experts when the router is confident, the lowest-weight ones dropped until the kept ones hold a share p of the router's weight. Prediction, written first: at p = 0.9 about 5-6 experts a token, misses at half budget down 20-35%, KL against the exact mode small but not 0; the token changed in a few % of positions | route trace (`--route-trace`): per p, experts kept per token and their weight (no engine change); then a declared mode (env or flag, never the default) at half budget through `tools/ab_modes.sh` (misses, prefill, decode) and KL against the exact mode (`tools/mask_quality.sh`) | under a partial budget every expert skipped is a unit not read from disk: a lever on M1's bottleneck, if the quality holds |
| 51 | ~~Can the decode's attention run on the GPU with the same bits, and what does it give at long context?~~ **Premise measured 2026-09-24** (§The decode's attention on the GPU: the premise): 0 of 25 344 head outputs differ; 185 µs a layer at 2048 and 319 at 4000 with keep-warm and zero copy (CPU 704 and 1415), 1.29× and 1.57× projected; **in the engine 1.31–1.32× at 2048 and 1.54–1.58× at 4000** with the measured width (§The decode's attention on the GPU, in the engine). (2026-09-24, after the skip family closed, §Skipping cached positions exactly). Prediction, written first: bits identical if every float op carries an explicit `.rn` and `tr_expf` is ported whole (its double arithmetic too); per layer (16 heads) at 2048 the kernels ~150 µs (32 MiB at ~220 GB/s) and the round trip (copies of q, k, v in and the output out, launch, sync, WDDM) 30–60 µs, so 180–210 µs against the CPU's 704: the token 36.8 → ~28.7 ms (**1.28×**); at 4000 the attention 22.6 → ~5.4 ms, the token 48.1 → ~31 ms (**~1.55×**) | `tests/bench_gpu_attn.c`: the driver loaded at run time, the kernels as PTX; every query of the six dumps of `make attn-probe` against the CPU's bits, a mutation seen red; per layer at 2048 and 4000 the kernels alone, the round trip, a plain streaming read (median of ≥ 50) | the one exact lever left of the size of the KV's bytes: the same bytes read 4-5× faster, and more the longer the context |
| 52 | ~~How many tokens does a speculative pass give at long context?~~ **Closed 2026-09-24** (§Speculation at long context): 1.04-1.13 on free prose, 1.57 rewriting code, 3.29 on repetitive code (1.66x net at 4000); a pass of ~4 rows reads ~2.8x the experts. Prediction, written first: on tasks that quote or rework the prompt (rewriting code, summarizing section by section) 1.5–2.5 tokens a pass, on free prose ~1.1; the KV bytes per token divide by it, the union of the experts of a multi-row pass eats part of the gain | `run -f <task of ~2000 and ~4000 tokens> -n 256` with and without `--spec 8`: tokens, passes, tokens identical | the engine already reads the KV once per pass (`tr_attention_group`): what it is worth where the KV is 30-46% of the token |
| 53 | ~~Draft on the GPU, exact verification: how often do the Q4_K and the Q8_0 of the same model pick the same token?~~ **Closed 2026-09-24** (§A Q4 draft against the exact Q8_0): **88.7 / 94.7 / 90.6%** on the Q8_0's greedy trajectory (prose, code, Italian; Q4_K_M 90.4 / 93.6 / 92.8%), 83–91% on real text, KL 3.4–5.8e-2; 5.5 / 7.2 / 6.4 tokens a pass of 8. Below the threshold, but with 54's union a pass of 9 rows reads ~2.7× a token's bytes: **~2.0× fewer bytes a token** on prose, ~1.3–1.9× in time once the draft's own GPU time is paid. (2026-09-24) A Q4 copy of the model on the GPU drafts 8 tokens, the exact engine checks them in one pass (batch = token by token: the accepted tokens have exact logits). Prediction, written first: top-1 agreement 95–98% on prose and 97–99% on code (llama.cpp's int8 activations sit at KL 9e-3 from us; 4.5-bit weights move more), so 6–7.5 tokens accepted a pass of 8 on prose, 7–8 on code | teacher-forced on real text (the session's prose and code prompts, `tools/quantize_q4k.sh` makes the Q4_K): feed the Q8_0's greedy tokens to both, count top-1 agreement per position; then accepted tokens per pass of 8 | at ≥ ~95% agreement, ~600 MB read a token instead of 1200 at 2048 (dense weights and KV once per pass, experts ~5×): **~2×** on top of the GPU attention, exact by construction; the draft needs M3's Q4 model whole on the GPU |
| 54 | ~~How many experts does a pass of k rows read?~~ **Closed 2026-09-24** (§Experts read by a pass of k rows): **1.55× at k = 2, 2.33× at 4, 3.34× at 8, 4.42× at 16** (random subsets: 1.88 / 3.31 / 5.25 / 7.06), below the prediction at every k; code overlaps more than prose. Co-activation grows with depth (17–21% of pairs above lift 2 at layers 8–15). (2026-09-24) Prediction, written first: against one token's experts, 1.7–1.8× at k = 2, 2.8× at 4 (measured), 4.2–5× at 8, 5.5–6.5× at 16 (it saturates at 64 / 8 = 8×) | from `--route-trace` on real text: the union of the experts of k consecutive tokens, per layer, k = 1..16 | measured 2.8× at ~4 rows (§Speculation at long context); it sets the cost of 53's verification and of speculation on any MoE, and fine-grained MoEs (256 experts) may overlap less |
| 55 | ~~How close is the prefill's matmul to the CPU's peak?~~ **Closed 2026-09-24** (§The CPU's peak): **49–53% of the no-FMA peak on one core** (87 of 166.5 GFLOP/s), ~40–45% on 16 (noisy); the kernel's own stream in asm with 8 tokens a weight reaches 119.4, **1.34× today's matmul**: the microkernel has margin. (2026-09-24) Prediction, written first: 35–50% of the no-FMA peak (0.7–1.0 of ~2 TFLOP/s on 16 cores) | FLOP/s of `tr_matmul` on the real shapes (`bench_kernels`' whole-matrix lines) against the no-FMA peak of this Zen 4 (2 FMUL + 2 FADD pipes) | estimated 40-50% (~0.75 of ~2 TFLOP/s): room for a hand-written AVX-512 assembly microkernel using all 32 zmm (M2's assembly workshop; gcc spills, LESSONS #45); near the peak, none |
| 56 | ~~How far does lossless compression shrink the real weights?~~ **Closed as no 2026-09-24** (§The model read like a genome): Q8_0's codes carry **7.59–7.69 bits of 8**, Q4_K's **3.83–3.87 of 4**; a lossless coder gives ~7% of a Q8_0 file (half of it the scales) and ~4% of a Q4_K, nothing beyond order 0 in the experts. (2026-09-24) Prediction, written first: Q8_0's codes carry 6.3–7.0 bits of their 8 (12–20% of the code bytes), Q4_K's 3.5–3.8 of their 4 (5–12%) | entropy of the codes of Q8_0 and Q4_K per block and per tensor (and FP8/BF16 when such a model arrives); a layout that decodes at memory speed (fixed width, or block ANS) | bytes are the wall at every level (RAM, disk, PCIe) while compute idles in decode; guess ~8-12% on Q4, more on FP8/BF16 (~30% on BF16 elsewhere); for a model streamed from disk the gain is proportional |
| 57 | ~~Does the KV packed in 28 bits pay on a still machine?~~ **Closed as no 2026-09-24** (§The KV packed in 28 bits): **0.97–1.05×** the F32 attention on the six dumps; decoding eats the 1.13× in bytes (44–45 GB/s against 49–51). (2026-09-24) Prediction, written first: 1.00–1.08× on the attention zone at 2048 and 4000: the four positions decoded at a time come in the x4 order that cost 10% (§The decode's attention, one position at a time), and may eat most of the 1.13× in bytes | `build/tests/bench_kvpack.exe time --run all`, native, marker held | exact, 1.13× fewer bytes (§The KV packed in 28 bits): ~1.04× at 2048, ~1.06× at 4000 for a CPU-only machine, if decoding hides under the memory time (four positions at a time may inherit the x4 order's penalty) |
| 58 | ~~The float-only exact exp in SIMD: what does it give the prefill?~~ **Closed 2026-09-24** (§An exact exp in float32 only): AVX-512 and AVX2 tiers, **0 of 2^32 differ**, **0.73 / 0.80 ns a value against `tr_expf`'s 3.52 (4.8× / 4.4×)**; the zones ~2.4× (softmax) and ~1.7× (SiLU), **the prefill ~1.02–1.03×**: question 39's range, not wired into the CPU engine; the GPU prefill's reference. (2026-09-24) Prediction, written first: the softmax and SiLU zones 3–8× faster, the prefill 1.03–1.06× at 2048–4000 | an AVX-512 and an AVX2 tier of `tests/bench_expf32.c`'s algorithm, exhaustively proven like `tr_expf`, then the prefill's softmax and SiLU zones | 3-6% of the prefill at 2048-4000; the GPU prefill needs it anyway (FP64 at 1/64 rate) |
| 59 | **FMA in the definition?** (Marcello's decision, 2026-09-24) **Numbers on this Zen 4** (§The CPU's peak): FMA's peak = mul + add's (166 GFLOP/s a core); the prefill kernel's 8-token stream with FMA **1.19×** the one without (141.1 / 118.8), 1.17× on 16 cores | the prefill's matmul with and without FMA on an ARM (Apple) and an Intel machine | about free on this Zen 4 (separate FADD pipes), up to half the peak compute on ARM and some Intel; C99 `fmaf`, FMA3, NEON and CUDA `fma.rn` keep every tier identical; changes today's bytes once |
| 60 | **An exactly rounded dot product (Kulisch-style accumulation) as the definition?** (Marcello's decision, 2026-09-24) | the cost of an exact dot on AVX-512 VNNI with the activations' mantissas sliced in bytes, against the float lane path | a result independent of the order: any SIMD width, the GPU or a sum split across machines gives the same bits by construction, and it is more accurate; changes today's bytes once |
| 62 | ~~The model scanned like a genome: is there exact structure nobody reads once?~~ **Closed 2026-09-24** (§The model read like a genome, §Experts read by a pass of k rows): weights **0.008% of Q8_0 blocks repeated** (220 identical rows of the output head: the tokens never learned; Q4_K 504 rows, 2 140 zero blocks), no zero blocks in Q8_0; routing co-activation strong (lift up to 67, top-3 partners 18–28% of co-firings); **the KV: only layer 0's values are token-determined** (61.6–81.6% of positions repeat), layers 1–15 and every key 0. Tools: `tools/weights_genome.py`, `tools/route_union_report.py`, `tools/kv_repeats_report.py`, to run again on the next model. (Marcello, 2026-09-24: a discovery comes from looking patiently at real data others filtered away) Approximate engines compress the weights with loss and never look for exact repeats; an exact engine can read anything that repeats exactly once, without changing a bit. Prediction, written first: trained weights hold almost no exact repeats (≤ 0.1% of Q8_0 blocks), the routing holds strong co-activation, the KV holds token-determined parts beyond layer 0 in no layer | a hash of every Q8_0 block (34 bytes) over the whole file: duplicates, all-zero blocks, rows shared between experts, per-tensor entropy (with 56); in the routing traces the pairs and groups of experts that fire together (co-activation arrays, for placement and file order); in the probe's dumps, any layer whose keys or values depend on the token alone | cheap, on real data, not done by anyone: most of it may give zero, like the exact skips; what repeats exactly is bytes saved exactly |
| 61 | **Exactness as the asset for a frontier model locally** (after M4, 2026-09-24): a computation that gives the same bytes anywhere can be moved in time (the KV of your files computed while the machine idles, reused byte for byte), in space (several home machines splitting a model: their RAM bandwidth adds up), and checked (work done by an untrusted fast machine, verified by recomputing random spots) | a KV checkpoint to disk and back per 1000 tokens; one layer across two machines on a LAN (~28 KB a hop at 7168 dims); detection probability against spot-check cost | a 671B MoE reads ~20 GB a token at Q4 (0.35 s from RAM, 13 s from this disk): the wall moves only by adding bandwidth or by not paying the prefill at question time; approximate engines cannot do any of the three |
| 63 | ~~Decode a panel of weight rows once, then F32 over every token?~~ **Closed 2026-09-24** (§The prompt's matmul against the four references): the same bytes, but **Q8_0 1.00–1.04×, Q4_K 1.10–1.12×** on a core (predicted 1.13–1.27×): the decode is 29% of the stream in L1, the matmul is bound by its input rows' loads from L2 | — | ik_llama.cpp's convert-then-gemm, kept exact |
| 64 | ~~A tile of more weight rows per input load (4 rows × 6 tokens)?~~ **Closed 2026-09-25 as no** (§More weight rows per input load): the streams of 4 × 6 and 3 × 8 run 1.02× and 1.04× x8's, but in the matmul, exact, they lose (Q8_0 0.88–0.90× and 0.93–0.94×, Q4_K 0.84× and 0.90×; predicted 1.00–1.12×): the matmul is not bound by its input loads | — | the tile stays 2 × 8 |
| 65 | ~~The decode's matmul (one token, bound by memory): do ggml's kernels read the weights faster than ours?~~ **Closed 2026-09-25** (§The decode's matmul against ggml's): **Q8_0 level from 4 threads** (0.97-1.06x, both at the ceiling), ggml 1.4-1.7x at one thread; **Q4_K ours bound by its arithmetic** (9.3 GB/s a core), ggml 1.34-1.40x at 4 threads, level from 8. The decode's gap is attention (row 3). Three premises for a faster exact Q4_K dot closed as no (four rows at once 0.5-0.9x from RAM, vector scales 0.86x, arithmetic weight 0.77x) | — | — |
| 66 | ~~Reach llama.cpp's Q4_K decode at 4 threads, exactly (Marcello, 2026-09-25)~~ **Reached 2026-09-25** (§The Q4_K decode dot sequenced): the kernel read like a genome first (the scalar scale decode was 24% of a row), then the scales in a vector one block ahead and two rows a call, the same bits: the real model's decode at 4 threads **40.3 -> 50.4 tok/s (1.25x), llama.cpp 50.2**; the matmul alone ggml 1.37x -> 1.04x. Premise e (a dictionary of tables) closed: 94-99.6% of the sub-blocks have a pair of their own. Open: SMT (+9-13% from RAM at 4 cores), the prefill at 4 threads (llama.cpp 1.6x) | — | — |
| 67 | ~~SMT in the decode: two threads a core beat llama.cpp on the same 4 cores? (Marcello, 2026-09-25)~~ **Closed 2026-09-25 as no** (§SMT in the decode): natively two threads a core give the decode +1.6-5.7% at 4 cores, nothing at 8 (60.06 against 60.01 tok/s) and a collapse at 16 (8.4 against 58.3); the engine's measured width is 8 cores, one thread each (59.8 tok/s, 8 forced 60.2): no default width gains, not wired. The container cannot answer it (its siblings are not a core's two threads). On the way: the pool's tail (§The pool's tail), tr_parallel_for_balanced | — | — |
| 68 | Why does the decode collapse at 32 threads on 16 cores (8.4 tok/s against 58.3 at 16, native, TR_POOL_PIN=1; 2.2 against 22.9 on 2026-09-18)? A pool trace (TR_POOL_TRACE) at 32 would say whether it is the wait of every call or a few calls | — | — |
| 69 | ~~Pass llama.cpp's decode at long context with the KV exact (Marcello, 2026-09-25)~~ **Closed 2026-09-26 on the CPU** (§The attention against llama.cpp's): level with no context, 5.31 against 3.24 µs a cached position, the bytes of their F16 KV; our attention reads at a plain read's speed, and F32 at their slope needs 81 GB/s (the RAM reads 53-57). On the GPU the same bits pass it (1.31x at 2048) | — | — |
| 70 | The softmax's exponential in a vector (question 58's AVX-512 kernel, 0.73 ns against `tr_expf`'s 3.52, 0 of 2^32 differ): with the tiles it is half of a prompt's (query, position) pair (4.45 of 9.2 ns on a core). Predicted: the prompt's attention 1.5-1.8x more, the prefill at 4000 +5-8%. **Built 2026-09-26** (§The softmax's exp in a vector): the table's `expf_f32`, AVX-512 and AVX2 without FMA, 0 of 2^32 differ from `tr_expf` in each tier; the prompt's attention 12.6 -> 8.9 ns a pair on a core (1.42x, loaded machine, indicative); in the engine on a free machine (8 pairs, background 2.3-2.4): the attention zone 1.34-1.37x, **the prefill 1.05x at 2048 and at 4000** | — | — |
| 71 | A draft inside the exact bits (Q8_0 high nibble, KV high 16 bits) verified k tokens a pass: can it pass llama.cpp's bytes a token? (Marcello, 2026-09-26) **Measured** (§The engine read as entangled pairs): 97-100% where the exact model is sure, 82-98% overall; **0.81-1.04x llama.cpp's bytes on the real text** (model), 0.97-1.09x on the greedy continuation: this form closed on the CPU. **The two levers measured 2026-09-26** (§The draft's two levers): the second choice as a leaf +0.07-0.10x on prose and Italian; a KV of constant bytes (the previous token's top positions and their successors) keeps the sure positions at 512 and on code, not on prose at 2048-4000; **1.15-1.46x llama.cpp only on code** (repetitive), 0.89-1.04x on prose and Italian (model, upper bounds). Open: the skipped part of the softmax handed over by the exact pass; the selection's staleness | `make draft-probe`, `tools/draft_probe_report.py` | a pass's own drafts are 56% of its bytes; passes end where the draft doubts |
| 72 | ~~Gate and up interleaved at load, one call with the activation~~ **Closed 2026-09-26 as no** (§The engine read as entangled pairs): on the free machine 1.006-1.023x of three calls from RAM (predicted 1.03-1.08x): with the experts balanced three calls already read at 97-98% of a plain read | — | — |
| 73 | The head's argmax from its high plane: bounds from the high nibbles, exact dots only for the rows that can still win; greedy needs the argmax, `logits` keeps every row. **Rows left measured 2026-10-05** (§A draft from the bytes the machine holds, `tools/head_bound.py`): Q8_0's high nibble leaves 0.2-3.8% of the rows, the head read at 0.53-0.55, a RAM-bound Q8_0 token 1.039x by bytes; Q4_K's top 3 of 4 bits 0.2-3.7%, the head at 0.78, 1.017x. **Phase 1 done 2026-10-05** (§The head's argmax by a bound): the head reads at the RAM on avg and pc (1.92-2.03 ms Q8_0, 1.02-1.06 Q4_K, 7.2-7.9% of a token); greedy positions leave 0.28-0.36% of the rows, long contexts 1.6-2.5%; the whole argmax by the bound measured in a bench (16 threads: Q8_0 2102 -> 1162-1278 us, Q4_K 1141 -> 928-1048); by the model a token pc and avg Q8_0 1.034-1.037x, weak Q4_K 1.018x, pc Q4_K 1.010x, avg Q4_K level, weak Q8_0 1.002x. Decided: build (phase 2), greedy first. **Phase 2 done 2026-10-05**: built, exact on 13 texts x 2 models at every position, ~10 rows computed a token, raced: the decode Q8_0 1.046x pc, 1.041x avg; Q4_K 1.022x pc, 1.009x avg, 1.015x weak; on by default. Open: the verify passes (a k-row bound kernel) | the rows left after the bounds, on the real text | <= 49 MiB a token (4%) |
| 74 | The draft on the GPU: the Q8_0 high-nibble planes and the draft's KV in VRAM, the CPU verifying from RAM, overlapped (the idea bounce of 2026-09-26, `build/entangled/bounce.md`): ~2.2x overlapped, ~1.7x serial in time (model). For Marcello: does the GPU count in the race (the GPU attention did) | a ~50-line CUDA probe timing one batch-1 draft step (plane 635 MiB + KV halves 256): pass <= 6 ms a draft token, fail > 15 ms | the draft's 891 MiB a draft token off the RAM bus |
| 75 | ~~A draft KV in int4 (or int8) written once per position, and a draft head of the top ~16k rows~~ **Measured 2026-09-26** (§The draft's KV in 8 and 4 bits): the 8-bit copy changes no draft token and lifts every cell 0.02-0.14x; out of sample prose 0.93-1.09x, Italian 0.95-1.12x, code 1.14-1.26x; the 16k head covers 85-93% of the exact tokens: out. Prose does not pass 1.1x: the draft's weight plane is what is left | — | — |
| 76 | Why does a 4-bit draft KV agree with the exact model more often than a 16-bit one (18 flips won, 3 lost, mostly where the exact margin is under 1 nat)? A guess: noise on the old keys inflates their softmax weight (Jensen) against the last 64 kept at 16 bits | kv4 with no positions kept at 16 bits; Gaussian noise on the 16-bit draft's old keys | a draft that is cheaper and closer at once |
| 78 | The verify pass of 2-4 rows at its bytes (§The post-it taken apart): while no group reaches 4 rows the grouped matmul cut by weight rows and balanced (every input row of a group against a weight row while it is in cache), a kernel decoding a weight row once for 2-3 tokens. **Q8_0 done 2026-09-27** (§The short verify pass at its bytes: 3 rows at its bytes, a row 0.31 of a pass; 2 rows 0.43). **Q4_K 2026-09-27** (§The Q4_K short passes: the W16 panel's rows brought ahead, a 2-row group by `q4x_dot2` pairs: 3 rows 28.0 -> 26.5 ms, 2 rows 22.9 -> 22.2 at 8 threads); then **road (c)** (§A Q4_K weight row decoded once for 2-3 tokens: `q4x_dot_xt`, two weight rows decoded once for a group's 2-3 rows: 2 rows 0.94-0.98, 3 rows 0.96-0.985 in three sessions, the prompt unchanged, the decode +0.8-1.7% spread over every zone: open, the code's layout?); left: the rest's +0.30-0.42 ms a row and the dense at 3 rows, +0.39-0.45 (xt(3) reads at 0.88 of the bytes at 8 threads) | `sh tools/row_price.sh` before and after (2, 3, 5, 9 rows), also with a 2048 and 4000-position prefix (the context is modelled today); then `tools/draft_gate_sim.py` | new code's drafts are short: a row costs 0.50-0.72 of a pass today, 0.31-0.36 at its bytes; every draft source gains |
| 79 | A row that costs no bytes. **As a draft, answered by question 94 (2026-10-05, model, not measured; §A draft from the bytes the machine holds)**: it re-reads the dense 37-40% of a token's bytes for each drafted token, 0.74-0.86x with the measured chains; as a filter of the gate's draft, open. Asked: the model restricted to the experts the pass already reads (the union of its exact rows'), as a draft, or as a filter of the gate's draft before its exact rows (two-stage verification) | a `draft_probe` variant: agreement with the exact argmax by margin, restricted to the union of 1 and of 2 exact rows | the model's knowledge of names at ~0.03-0.05 of a pass a row, where the table knows 22% of them |
| 80 | Drop a draft row mid-pass when the main row's intermediate state says its draft is wrong (logit lens on the draft's and the gate's candidates' rows of the head, a few KB) | per layer, the draft's logit-lens rank among the candidates on the replay's positions | a wrong row then costs its new experts only up to the layer where it is dropped |
| 81 | The drafts on code written with a real context: a function of a real file continued mid-file with the rest of the file and its neighbours in the prompt, and a table built from the user's own repository. **Answered 2026-09-27** (§Drafts with a real context): the engine's lookup 1.067x, the gate 1.146x at the measured prices (1.27-1.32x with the context modelled), the repo table adds nothing to the gate; left: the row's price measured at ~3000 positions | the 18-prompt pipeline (`tools/draft_table_gen.sh`, `tools/draft_gate_sim.py`) on prompts cut from this repo's `src/` and `tools/`, the table from the repo minus the file under test | the 18 prompts are file beginnings (~100-200 positions): real use has thousands, and the context sources are the best drafters measured |
| 82 | The short passes' width when every pass drafts (LESSONS #228): the tuner probes one-row passes only, so a session with a draft on every pass runs its 2-4-row passes on the whole pool. **Answered 2026-09-27** (§The verify passes' own width): every size of verify pass measures its own width, the narrowest skipped; on a pool of 16 Q8_0 0.961-0.966 of the pool, Q4_K 0.982-1.005, 8 threads picked everywhere; left: the probes' cost on long sessions (a re-measure every doubling) | the passes of 1, 2 and 3 rows at 4, 8 and 16 threads forced (`--decode-threads`) on a pool of 16, alternated; then the tuner fed the short passes too | after question 78 the short passes are still 1.04-1.09x slower at 16 threads than at 8 |
| 83 | Half a prep a token, or less (Marcello): Q4_K's input rows prepared once a distinct row, in the call that makes them, with fewer pool calls a layer. **P1 + P2 done 2026-09-27** (§The prep once a row: q/k/v from one prep, gate and up from the tokens' rows through a map, no gather: the prompt 1.058, a 3-row pass 1.026, the same bits; the decode with P4 1.025, the arms aperiodic); **P4** (swiglu with the down's prep in one call): the prompt 1.0087, the decode level; left: gate and up in one call, countdown epilogues, the idle workers prefetching their next call, then the prep's bit arithmetic (0.70 us a prep: ~0.1 us to take) | `tools/ab_inproc.sh <bin> <switch>` (prompt: AB_INPROC_PROMPT=8), each piece with its own switch; the traced timeline's no-weight calls | the prompt's preps and gather were 8.6% of a 512-token prompt, 3% of a decode pass |
| 84 | ~~Does the run past a stream's end cost the weights' reads too?~~ **Answered 2026-09-27** (§The idle workers, and what lies past a region's end): the page after each thread's region dropped, a dense call 0.8% faster (the read 1.7%), an expert's call level; a thread's regions of consecutive calls one after the other, a dense call 1.0-1.75% faster. A lever of 0.2-0.4% of a token in the dense calls' layout, none in the experts'; taken with piece 4 (q, k, v in one call lays them out again) | — | — |
| 85 | The serial steps as messages (LESSONS #255): the idle workers' hints gain the next call up to the serial step's length in the bench (0.85-0.96 W) and nothing in the engine, where the step after each hint reads the rows the workers wrote (attn_out, h3) and stores over lines they share (normed, the prepared row). If the steps stop touching the workers' lines (the add and the mix inside the calls that make their rows, normed and the prepared row in buffers the workers have not read since), do the serial steps get faster by themselves, and do the hints then pay? Step 1 raced 2026-10-02, `pretouch` (the calling thread's own next lines asked before each hint): level (1.0027 +- 0.0022), attn_norm's +54 us a pass unchanged, so a step does not pay its own lines' latency but a share of the core or the fabric under the flood (§The idle workers) | the steps' zones with `TR_POOL_HINT=2` against 0 in one process (`idle`), then each step's lines moved | the serial zones are ~260 us a token (1.7%); the hints' gross gain 150-190 us |
| 86 | A team of small models (Marcello, 2026-10-01; placed 2026-10-02 after rung 2, block x-team: then two families run exactly, and a 1-3B model is what a below-average machine holds): can three diverse 1-3B models (different families: Qwen, Llama, Mistral...) match or beat one ~7B, and at fewer bytes read a token? Three forms: a vote on the whole answer (any tokenizer); a token-level vote (shared tokenizer: mean or product of probabilities, or the most confident model speaks); the cascade, where the cheapest model runs first and the others wake only when it is unsure (entropy). Prior art, all outside inference engines: Product of Experts (Hinton 2002), self-consistency (2022), LLM-Blender (2023), FrugalGPT (2023) and RouteLLM (2024) cascades on whole answers, Mixture-of-Agents (2024), Branch-Train-MiX (2024), mergekit-moe, DeePEn and UniTE (2024); not seen: an exact token-level team inside an engine, counted in bytes a token. Three traps found in a review by Claude Opus 5.5 at medium effort (2026-10-01; the top-1 minus top-2 gate came from DeepSeek): (1) a model woken at token t lacks tokens 0..t-1 in its KV, so the output is wrong, not just slow: it must catch up with a prefill of the skipped tokens on waking, and that catch-up cost decides whether the cascade pays; (2) the mean of logits and the product of probabilities pick the same token (logits are defined up to a constant), so the real fusion modes are the mean of probabilities after softmax, the product of probabilities, and the most confident model (top-1 minus top-2 margin, a better gate than entropy alone: a model sure and wrong wakes nobody); (3) a shared tokenizer is checked by a hash of the whole token list, not by vocab size, and n_ctx is the minimum over the models. A fusion defined as a fixed sequence of operations can be exact against a Python oracle | Python first (`tools/.venv`, transformers, no C): GSM8K (math) and MMLU (knowledge) subsets, the team against the single ~7B; accuracy and bytes read a token for each form; the prediction written before the run; the RAM guard (one model loaded at a time if the sum does not fit) | a capability nobody offers (an exact team that wakes only when needed); a zero is written and closed |

| 87 | **The dedicated and the integrated GPU together** (Marcello, 2026-10-03; a machine with both, like this PC, RTX 4070 + Radeon 610M, and many 16 GB laptops): today the engine takes the dedicated one and leaves the integrated idle. Could the integrated take the work the dedicated cannot hold, or that would travel to it (it reads the RAM in place: the experts the VRAM has no room for, a layer's prompt while the dedicated takes the decode), the same bits on both? | after R1 phase 3 has the integrated GPU (Vulkan): the bytes each one reads a token and its share of the time, one GPU against both, on a machine with both; the prediction first | a laptop's two GPUs are two engines, and one of them sits idle today |
| 88 | **What the integrated GPU gives on a real machine** (Marcello, 2026-10-03): **until real PCs are available, every number of the integrated GPU is an estimate**, and is written as one (Iris Xe or Radeon 680M ~2-3.4 TFLOPS against ~0.4 of four AVX2 cores: a prompt 3-5x; the UHD of the cheapest machines ~0.2-0.4, about the CPU). This PC's Radeon 610M (2 compute units) checks the bits, not the speed | `tools/machines.sh` on a real 8 or 16 GB laptop with an integrated GPU: the prompt and the decode with and without it, the same tokens | the emulation gives fewer cores and less memory, never another GPU: the integrated machines' prompt limits stay unmeasured until then |
| 89 | **A MoE larger than RAM: its experts at the disk's limit, the VRAM as their cache** (Marcello, 2026-10-04; rung R3, **not before R2 closes**): Qwen3.8-Flash-Next FP8, 185 GB, 48 layers x 512 experts of 4.9 MB, 10 a token a layer. On this 8 GB-GPU, 31 GB laptop colibri decodes it at 0.75-1.01 tok/s and is disk-bound: it reads the experts buffered at 0.6 GB/s, queue depth ~1.5, against 2.1-3.0 GB/s O_DIRECT on the same drive (LESSONS #297); its VRAM gives more as expert cache than as trunk compute (#296; §Qwen3.8-Flash-Next FP8 on colibri's Vulkan tier). How close to the drive's limit can our store read 4.9 MB experts, and how many misses a token does the VRAM save as a cache tier under the RAM's? | the drive's ceiling for 4.9 MB reads at depth 1-16 (`make bench-disk`); a routing trace of the real model replayed through RAM + VRAM slots (`tools/evict_replay.py`); then the engine on the real model, the same tokens, against colibri on this machine | colibri used 0.6 of the drive's 2-3 GB/s: on a disk-bound decode that is up to 3.5-5x before any compute (an estimate); the VRAM holds ~5% of the experts (1248 of 24576), the RAM ~9% |
| 90 | **A decode layer's experts as their bytes arrive** (R1 phase 3, the 8 GB machine; 2026-10-04): today a layer's misses are read on the calling thread, then its experts computed. Computing the resident ones while an I/O thread reads the misses, then the late ones, hides a miss under its own layer (§The routes as time: the model 1.084 Q4_K, 1.038 Q8_0, the same bytes); the next layer's router reading ahead adds 4% on Q4_K at +11% bytes. Does the engine keep the model's word, and what do the extra pool regions cost? | the expert stages split (resident, then late) with the store's reads on an I/O thread, the same bits (`make check`); counted (`MACHINES_COUNTS=1`), then raced on the 8 GB machine (`sh tools/machines.sh <binary> 5 q4k q8`) | Q4_K 1.06-1.08, Q8_0 1.03-1.04 (the model less two or three region ends a layer with a miss) |
| 92 | ~~**The router's read ahead on top of the arrival order**~~ **Answered 2026-10-05** (§The router's read ahead): the 8 GB machine's decode Q4_K **1.031**, Q8_0 1.000 (k 8) and 1.005 (k 4), the same bits; a guess read in pieces and stopped when the next router does not name it, read a part a request once it does, k the next layer's units in 27 MiB. Open beside it: the race against the old binary (#332), the average machine's Q8_0 | ab_env.sh, `TR_AHEAD_K=0` against the default | Q4_K +2.5..4.5%, in |
| 94 | **A draft from the bytes the machine already holds**. **Phase 1 answered 2026-10-05: not built** (§A draft from the bytes the machine holds): below 1.0x on every machine (model, not measured: on the measured chains and the replayed store) (weak Q8_0 0.87-0.98, weak Q4_K 0.82-0.94, RAM-bound 0.74-0.86): the store already gives the union's saving, a chain's rejected rows are read again, the RAM-bound draft re-reads the dense bytes. Measured: the resident draft agrees 0.89-0.98 on code, 0.50-0.88 on prose. Open only for an engine's measurement, which the model does not ask for. Asked (Marcello, 2026-10-05: the one piece that would change the whole structure, a gain on every machine): the reads for one token are near their ceilings (Q8_0 at 97% of its disk on 8 GB, the RAM-bound machines at 98-99%), and from R2 on every ordinary machine's experts come from disk. A pass of k rows reads 2.33x one row's experts at k 4 (question 54) and yields up to k exact tokens; the draft restricted to the experts already in RAM (disk-bound) or already streamed by the pass (RAM-bound, question 79) costs nothing on the bottleneck, the verification exact. How many exact tokens a pass, on each machine? | phase 1 (build/prompt-draft.md): agreement of each draft (a `draft_probe` variant, chains of k tokens), then a bytes-and-time model per machine checked against the engine's counts; built only at >= 1.2x and never below 1.0 | disk-bound 1.3-1.6x, RAM-bound 1.0-1.2x (question 71's draft inside the bits was 0.81-1.09x of llama.cpp's bytes) |
| 95 | **The free drafts on the 8 GB machine's Q4_K** (question 94's survivor, 2026-10-05): there a token is half disk waits, half the four AVX2 cores' compute, and a verify row may cost far less than a token (the prompt's tiles: weakr's prompt 7.9 ms a row against a 31.3 ms token). A draft that reads nothing (the engine's lookup, question 81's gate) pays only its agreement; with the resident draft's chains the model gives code 1.07-1.40x, prose 1.03-1.23x, by the row's price (0.8-0.3 of a token). What is the weak machine's row price, and what do the lookup and the gate give there? | `tools/row_price.sh` with the weak machine's environment (2, 3, 5, 9 rows), then `tools/draft_gate_sim.py` at those prices on question 81's prompts; then `spec_replay --chain` for the disk | R1 phase 3 on the weakest machine; `--spec` by default there if it pays |
Reference machine: Ryzen 9 7940HX (Zen 4, 16 core / 32 thread, AVX-512 VNNI/BF16), 31 GB
RAM (2×16 GB DDR5-5200), NVMe Micron 1 TB, GPU RTX 4070 Laptop 8 GB and Radeon 610M (not used
until M3/M6), Windows 11, MinGW-w64 gcc 15.2 `-O2`. Laptop on power, other programs open
(Docker containers active): the spread reflects this.

## CPU kernels — scalar baseline (2026-09-17)

`build/bench/bench_kernels.exe --runs 5 --ms 100` (`tests/bench_kernels.c`). Million elements
per second for a dot product between a weight row and an f32 vector, one thread.

| Kernel | n=64 | n=1024 | n=2048 | n=4096 | spread |
|---|---|---|---|---|---|
| dot_f32 | 2779 | 3195 | 3201 | 3230 | 1.5-4% |
| dot_row f16 | 699 | 700 | 632 | 639 | 1-14% |
| dot_row q8_0 | 1728 | 1796 | 1909 | 1952 | 4-15% |

`tr_matmul` q8_0 1024×2048 (shape of an OLMoE expert projection), one token:
1 thread 1.070 ms (spread 1.6%); 16 thread 0.350 ms (spread 74%).

Notes:
- f16 is slowest: converting half → float element-by-element costs more than the product.
- With 16 threads a single matrix is only 3× faster, with huge spread: for a one-millisecond
  task, waking threads costs as much as the calculation. Threads should be used on larger
  blocks (a whole layer, all experts for one token), not per matrix.

## Engine profile — tiny model (2026-09-17)

`make profile` (3 scenarios on tiny OLMoE: hidden 64, 4 layers), scalar kernels, native Windows.
Numbers that only serve to test the suite: with such small sizes the fixed cost of each call
dominates, and the proportions don't hold for a real model.

| Scenario | Prefill tok/s | Decode tok/s | Heaviest zones in decode |
|---|---|---|---|
| f32, prompt 6 → 64 token | 13380 (±3.5%) | 10909 (±0.9%) | attention 28%, rope 17%, expert gate_up 13%, qkv projections 13% |
| f32, prompt 96 → 8 token | 9784 (±4.1%) | 7318 (±3.7%) | attention 51%, rope 12% |
| q8_0, prompt 6 → 64 token | 10704 (±18%) | 8872 (±2.3%) | attention 24%, gate_up 16%, qkv 16%, rope 14% |

Cost of profiler on: +0.25% (within noise). To verify on real model: rope at 17% suggests
sine and cosine recalculated on each call (a precomputed table would be correct).

## Engine profile — OLMoE-1B-7B Q8_0, scalar kernels (2026-09-17)

`build/trochilus generate -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -p 32 -n 16 -c 128
--profile`, native Windows, 16 threads (physical cores), one run: baseline, not yet median.
Free RAM before loading 10.8 GiB (Docker containers active).

| Phase | Token/s | Weights touched per token | Memory traffic |
|---|---|---|---|
| prefill (one token at a time) | 6.92 | 1200 MiB | 8.7 GB/s |
| decode | 6.87 | 1200 MiB | 8.7 GB/s |

Where decode time goes: expert gate+up 44.9%, expert down 22.3%, q/k/v projections 14.3%,
attention output projection 5.0%, lm_head 4.8%, expert activation 3.5%, rope 2.7%, attention
1.3%, router 0.9%, everything else below 0.2%.

Notes:
- **Limited by compute, not memory**: 8.7 GB/s versus ~80 GB/s theoretical RAM. SIMD kernels
  are the right lever now; memory will become the limit only later.
- **92% of time is in q8_0 matrix-vector products** (experts 67%, projections 19%, lm_head 5%):
  the first kernel to vectorize, then move to assembly, is `dot_row q8_0`.
- Prefill runs at the same speed as decode because it processes one token at a time: batched
  prefill (multiple tokens in the same multiplication) is the second lever.
- Rope at 2.7%: not a priority on the real model (on the tiny one it looked like 17%).
- Tokens generated from a synthetic prompt repeat `64,301`: normal for greedy on nonsense
  input, but correctness on the real model isn't yet verified (needs comparison with a
  reference: llama.cpp or transformers).

## CPU kernels — AVX2 and AVX-512, bit-identical (2026-09-17)

`make bench` native, same machine, million elements per second on one thread (n = 2048):

| Kernel | scalar | AVX2 | AVX-512 |
|---|---|---|---|
| `dot_f32` | 3 310 | 26 356 (8.0×) | 31 807 (9.6×) |
| `dot_row q8_0` | 1 957 | 19 532 (10.0×) | 19 837 (10.1×) |

`tr_matmul` q8_0 1024×2048 (one expert matrix, 1 token): 1 thread 0.100 ms, 16 threads 0.146 ms.

Notes:
- AVX-512 runs almost as fast as AVX2: the limit is the **chain of sums**. Each lane sums one
  product after another, and a sum waits on the previous one (~3-4 clock cycles): 16 lanes give
  at most 16 elements every 3-4 cycles, with 256- or 512-bit registers. More lanes would break
  the scalar definition (different numbers): worth evaluating only if a single thread becomes the limit.
- On paper 16 threads at 20 000 M el/s would exceed RAM's bandwidth (~80 GB/s); on the real model:
  traffic stops at 22 GB/s because threads don't pay (section below).
- On an expert matrix 16 threads are **slower** than one: waking the pool costs more than
  the work (0.1 ms). Confirmed on real model (section below).

## Engine profile — OLMoE-1B-7B Q8_0, AVX-512 kernels (2026-09-17)

`trochilus generate -p 32 -n 16 -c 128 --profile`, one run per row (not yet median of 5).
Generated tokens identical across all rows and with `TR_CPU_MAX=scalar`.

| Kernel | Threads | Decode tok/s | Prefill tok/s | Memory traffic |
|---|---|---|---|---|
| scalar | 16 | 6.87 | 6.92 | 8.7 GB/s |
| scalar | 8 | 7.18 | — | — |
| AVX-512 | 16 | 17.72 | 17.83 | 22.3 GB/s |
| AVX-512 | 8 | **21.08** | — | — |
| AVX-512 | 4 | 20.68 | — | — |
| AVX-512 | 1 | 13.85 | — | — |

Decode zones at 8 threads: expert gate+up 34.8%, rope 13.0%, q/k/v 11.3%, expert activation 8.5%.

Notes:
- **3× on real model** (6.87 → 21.08), but 16 threads are slower than 8 and 1 thread already
  does 13.85: threads only yield 1.5×. The pool wakes 464 times per token for 0.1 ms matrices.
  The default (threads = physical cores) is wrong for decode today; the fix isn't a fixed
  number but parallelizing larger jobs (the 8 experts for one token, the whole layer).
- **Rope rose to 13%**: computes `pow`, `cos` and `sin` for every element at every token.
  A table per position gives the same numbers (same operations in double) and removes almost all.
- **Expert activation at 8.5%**: scalar `expf` on 1024 elements per expert.
- Traffic (22 GB/s) is still below RAM bandwidth (~80 GB/s): the limit is still compute
  and thread coordination, not memory.

## Thread pool: busy-wait instead of sleep (2026-09-17)

`bench_kernels --runs 7 --ms 60` on Linux (Docker on WSL2, same machine), median of 7 runs.
Dispatch = an empty `tr_parallel_for` that assigns a piece to each thread; matmul = a q8_0
expert matrix 1024×2048, median of 50 consecutive calls.

| Thread | Dispatch before | Dispatch after | Matmul before | Matmul after |
|---|---|---|---|---|
| 1 | 0.002 µs | 0.002 µs | 0.099 ms | 0.098 ms |
| 2 | 41.0 µs | 0.10 µs | 0.100 ms | 0.049 ms |
| 4 | 90.0 µs | 0.37 µs | 0.126 ms | 0.025 ms |
| 8 | 181.8 µs | 1.62 µs | 0.211 ms | 0.015 ms |
| 16 | 361.9 µs | 2.80 µs | 0.376 ms | 0.014 ms |

Notes:
- Before each system-woken thread cost ~20 µs in Docker: more threads, slower.
- Now threads busy-wait up to 2 ms (instruction `pause`), then sleep: the matrix scales 2× at
  2 threads, 4× at 4, 6.5× at 8; at 16 it barely gains more (0.015 → 0.014 ms): unclear if
  it's RAM bandwidth or two core groups (questions 2 and 4).

Native Windows, same benchmark (old pool compiled separately), median of 7 runs:

| Thread | Dispatch before | Dispatch after | Matmul before | Matmul after |
|---|---|---|---|---|
| 1 | 0.024 µs | 0.017 µs | 0.123 ms | 0.099 ms |
| 2 | 0.48 µs | 0.05 µs | 0.106 ms | 0.092 ms |
| 4 | 2.2 µs | 0.19 µs | 0.051 ms | 0.047 ms |
| 8 | 5.7 µs | 0.89 µs | 0.052 ms | 0.024 ms |
| 16 | 53.5 µs | 1.48 µs | 0.079 ms | 0.014 ms |

- Windows wakes threads much faster than Docker (53 µs versus 362 at 16 threads), but gains
  remain large from 8 threads on: at 16 the matrix is 5.6× faster.
- High spread (up to 1305% on dispatch at 2 threads): a single wake from sleep weighs on
  a 0.05 µs measurement. Matmuls, longer, stay below 36%.

## Engine profile — OLMoE-1B-7B Q8_0, AVX-512 + new pool (2026-09-17)

`make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json`: prompt 32, 32 generated tokens,
context 128, native Windows, **median of 5 runs** (plus one warmup). Tokens identical across
all 20 runs and with every number of threads.

| Thread | Decode tok/s (min–max) | Before (old pool, one run) | Memory traffic |
|---|---|---|---|
| 16 | **26.90** (26.54–27.53) | 17.72 | 33.9 GB/s |
| 8 | **27.02** (26.42–27.06) | 21.08 | 34.0 GB/s |
| 4 | 24.48 (24.37–25.45) | 20.68 | 30.8 GB/s |
| 1 | 13.99 (12.47–14.39) | 13.85 | 17.6 GB/s |

Decode zones at 16 threads: expert gate+up 32.9%, expert down 16.7%, q/k/v 12.3%,
**expert activation 10.7%**, **rope 9.2%**, lm_head 5.9%, **attention 5.4%**, attention
output 5.3%.

Notes:
- 16 threads aren't slower than 8, but also not faster: 27 tok/s for both.
- **A quarter of the token runs on one thread only**: expert activation, rope and attention
  (bold above) account for 25%. With 16 threads that part doesn't speed up (Amdahl): it's
  the next exact work (table for rope, experts in parallel, vectorized attention).
- **Experts are now memory-limited**: an expert multiplication on the real model takes ~48 µs
  at 16 threads, versus 14 µs in the benchmark where the matrix is already cached. Each
  matrix is 2.2 MB to read from RAM: 2.2 MB in 48 µs is ~46 GB/s, near the real bandwidth
  of two DDR5 modules. Measure that bandwidth (question 4) before looking elsewhere on experts.

## Engine profile — OLMoE-1B-7B Q8_0, single-thread part parallelized (2026-09-17)

Same command as profile above (`make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json`,
native Windows, median of 5), before and after in the same hour: rope from table per position,
all expert activations as one parallel job, attention split per head with AVX-512 weighted sum
(`axpy_f32`). Tokens identical before and after, across runs and threads; real model logits
identical to the bit at baseline (200 tokens with 16, 3, and 1 threads, and with
`TR_CPU_MAX=scalar`).

| Thread | Decode before (min–max) | Decode after (min–max) | Gain | Memory traffic after |
|---|---|---|---|---|
| 16 | 26.34 (26.15–26.51) | **32.78** (32.35–33.27) | +24% | 41.3 GB/s |
| 8 | 26.02 (25.68–26.40) | **33.88** (30.39–34.91) | +30% | 42.6 GB/s |
| 4 | 23.69 (23.60–24.54) | **32.93** (32.06–33.60) | +39% | 41.5 GB/s |
| 1 | 13.11 (12.04–13.32) | 14.47 (14.36–14.61) | +10% | 18.2 GB/s |

Decode zones at 16 threads, ms over 32 tokens:

| Zone | Before | After |
|---|---|---|
| expert activation | 124.2 (10.6%) | 10.0 (1.1%) |
| rope | 107.6 (9.1%) | 0.8 (0.1%) |
| attention | 62.2 (5.2%) | 23.6 (2.5%) |
| expert gate+up | 389.8 (33.1%) | 397.5 (41.9%) |
| expert down | 196.3 (16.7%) | 199.2 (20.9%) |
| q/k/v | 143.6 (12.2%) | 144.9 (15.3%) |

Notes:
- The single-thread part went from 25% to 4% of the token: all the expected gain is there.
- **4 threads perform like 16** (32.9 versus 32.8 tok/s): what remains are matrix multiplications
  that read 1.2 GB of weights per token, and at 41 GB/s the limit is memory, not cores. Question 4
  (real RAM bandwidth: 2×16 GB DDR5-5200, ~83 GB/s theoretical) is now the first priority.
- At one thread +10%: rope and vectorized weighted sum, no parallelism.
- 13% spread at 8 threads (one run at 30.4): machine noise, other rows are below 5%.

## Correctness on real model — Trochilus vs llama.cpp (2026-09-17)

`tools/compare_llamacpp.py` in container, OLMoE-1B-7B-0125-Instruct Q8_0, prompt
`bench/prompts/dante.txt` (chat template), 128 greedy tokens, llama.cpp `b49650a` on CPU.

| What | Result |
|---|---|
| prompt tokens (27) | identical to `llama-tokenize` |
| greedy generation | identical for first 19 tokens, then paths diverge |
| same best word, 155 positions (prompt + generated, teacher forcing) | 149/155; in 6 different Trochilus' margin between top two is 0.04–0.18 |
| KL(llama.cpp ‖ Trochilus) | mean 8.9e-3, max 0.30 |
| maximum logit difference | 3.63 |

Inconclusive: llama.cpp multiplies Q8_0 weights with 8-bit quantized activations and keeps KV
cache in f16, Trochilus uses exact activations. The decisive test is transformers on the same
weights (section below). Both engines invent Dante verses: model's limit.

## Correctness on real model — Trochilus vs transformers, 2 layers (2026-09-17)

`make oracle-real`: first 2 layers of real GGUF Q8_0 copied byte-for-byte
(`tools/make_olmoe_2layer_gguf.py`); transformers 5.14.1 float32 with same weights dequantized
and config read from GGUF (`tools/make_olmoe_2layer_ref.py`). 32 greedy tokens and logits at
every position (teacher forcing). Linux container, gcc.

| Prompt | Token | Greedy | Best word | Maximum logit difference |
|---|---|---|---|---|
| `bench/prompts/dante.txt` | 27 + 32 | identical | 59/59 | 1.3e-5 (position 8) |
| first 1024 tokens of `src/models/olmoe.c` | 1024 + 32 | identical | 1056/1056 | 2.4e-4 (position 813) |

Check threshold: 1e-3. Residual difference comes from arithmetic, not the model: Trochilus
multiplies Q8_0 blocks with exact activations, torch multiplies already-dequantized matrices;
grows with position (longer attention sums). Cost: reference 29 s and 6.0 GB peak (only when
missing), comparison 33 s and 1.1 GB at every `make check`.

## Speed — Trochilus vs llama.cpp and colibri, OLMoE-1B-7B (2026-09-17)

`tools/speed_compare.py` in Linux container (`trochilus-dev`, gcc; Docker VM sees 32 CPUs and
15 GB), models in Docker volume `trochilus-models` (not Windows disk: LESSONS #41), one engine
at a time, **median of 5 runs** (min–max) plus one warmup; other projects' containers on but
idle. Trochilus and llama.cpp on same GGUF Q8_0; colibri (`a90bed9`, `ARCH=native`) on its
int8 conversion from Hugging Face, cache of 64 experts per layer (all), new process each run.
llama.cpp `b49650a` with `GGML_NATIVE`: `llama-bench`, prefill with empty context and decode
after context as large as the prompt (`-d`). Decode = evaluations of one token per second
(LESSONS #40). Prompt tokens: real code (`src/models/olmoe.c`) for Trochilus and colibri,
random for llama-bench.

**Prompt 32, 32 generated** (tok/s)

| Thread | Trochilus prefill | Trochilus decode | llama.cpp prefill | llama.cpp decode | colibri prefill | colibri decode |
|---|---|---|---|---|---|---|
| 16 | 30.4 (28.5–33.0) | 31.1 (28.0–31.7) | 232.9 (230.3–245.4) | 34.8 (34.0–35.3) | 10.1 (3.9–10.8) | 12.4 (11.5–15.1) |
| 8 | 34.0 (33.1–34.1) | 34.0 (31.3–34.2) | 200.9 (198.0–212.6) | 36.9 (36.3–38.2) | 12.1 (11.2–13.1) | 13.3 (11.1–14.0) |
| 4 | 33.7 (32.9–34.1) | 33.3 (30.3–34.5) | 126.0 (124.7–129.5) | 36.1 (35.7–36.4) | 12.3 (12.2–12.8) | 12.5 (10.8–13.6) |
| 1 | 15.4 (13.8–15.6) | 15.4 (15.0–15.5) | 38.3 (37.9–38.9) | 22.5 (22.0–22.7) | 8.3 (7.9–8.6) | 6.2 (5.4–6.5) |

**Prompt 512, 128 generated** (tok/s; colibri not measured)

| Thread | Trochilus prefill | Trochilus decode | llama.cpp prefill | llama.cpp decode |
|---|---|---|---|---|
| 16 | 29.6 (28.7–29.9) | 27.7 (27.5–29.0) | 376.6 (361.3–382.5) | 33.1 (32.5–34.2) |
| 8 | 31.8 (31.1–32.4) | 29.7 (28.2–31.0) | 270.2 (262.7–275.3) | 34.2 (33.9–34.5) |

**Tokenizer** on 2.2 MB of code and text (llama.cpp sources, colibri README in Italian,
English, and Chinese), time excluding vocabulary loading:

| Tokenizer | Seconds (min–max) | MB/s | Tokens |
|---|---|---|---|
| Trochilus | 0.100 (0.097–0.104) | 22.0 | 815 460 |
| HF `tokenizers` 0.22.2 (in-process) | 1.31 (1.24–1.45) | 1.7 | 815 460 |
| llama.cpp `llama-tokenize` | 3.60 (3.52–3.64) | 0.6 | 814 194 (different from HF, not investigated) |

Notes:
- **Prefill: llama.cpp 8× faster at 16 threads with prompt 32, 13× with prompt 512** (2.5× at 1
  thread). Multiplies all prompt tokens together; in Trochilus prefill = decode. It's the first
  job.
- **Decode at 4-16 threads: llama.cpp ahead by 8-12%** at short context, 15-19% at context 512.
  With prompt 512 Trochilus loses 11-13%, llama.cpp 5-7%: the cost of long context is ours
  (question 18). The first llama.cpp series (decode with empty context) gave 33.2 / 34.4 / 33.6:
  between two sessions the median shifts by 3-7%, precise number needs alternating runs
  (question 19).
- **1 thread: llama.cpp 1.46× in decode**: int8 activations with VNNI (lever 2, not exact).
- 8 threads beat 16 in both engines: the limit is memory bandwidth.
- colibri at 12-13 tok/s restarts each run with empty expert cache (90% hits): measures its
  cold start, not its steady state.
- **Tokenizer: Trochilus 13× HF and 36× llama.cpp**, same tokens as HF.
- Trochilus in container is ~5% below native Windows (31.1 versus 32.8 tok/s at 16 threads).

## Batched prefill in exact C + 4-token kernel — OLMoE-1B-7B Q8_0 (2026-09-17)

Container, volume `trochilus-models`, prompt 512 synthetic tokens, 16 generated, **alternating
runs** old binary / new binary each round (`tools/ab_speed.sh`; LESSONS #46: measuring all A
then all B lets machine warmup distort comparison), median of 5 rounds plus one warmup. «Before»
= binary at `38d0771` (one token per pass). «After» = passes of 512 tokens, (token, expert)
pairs sorted by expert, matrices in blocks of 16 tokens with one weight row against 4 tokens
in registers, logits of last token only.

| Thread | Prefill before | Prefill after | Gain | Decode before | Decode after |
|---|---|---|---|---|---|
| 16 | 30.1 (26.9–31.1) | **196.9** (158.3–211.2) | 6.5× | 27.9 (24.9–30.9) | 29.4 (24.8–30.9) |
| 8 | 31.0 (29.5–31.9) | **170.3** (147.9–185.1) | 5.5× | 29.6 (26.9–31.6) | 29.7 (28.3–29.8) |
| 4 | 31.0 (30.7–32.0) | **100.0** (94.4–101.2) | 3.2× | 29.4 (26.0–31.1) | 27.0 (26.6–30.1) |
| 1 | 14.4 (13.4–14.6) | **28.4** (26.8–29.0) | 2.0× | 13.5 (12.8–13.7) | 13.4 (12.7–13.8) |

Against llama.cpp, also alternating runs (same GGUF, `llama-bench -p 512 -n 0 -r 1`):

| Thread | Trochilus | llama.cpp | Gap |
|---|---|---|---|
| 16 | 162.8 (121.8–173.8) | 321.0 (184.9–349.7) | 1.97× |
| 8 | 156.7 (154.2–163.2) | 216.8 (180.0–251.6) | 1.38× |
| 4 | 93.6 (90.4–96.8) | 146.3 (136.9–148.3) | 1.56× |
| 1 | 26.1 (24.3–27.0) | 41.0 (40.9–41.2) | 1.57× |

Correctness on entire model (16 layers), 200 tokens: `logits` from before and after binary,
one token per pass, identical to the bit (`cmp`), and identical between 16 and 3 threads; passes
of 16, 64, and 200 tokens identical to the bit at same positions. On 2-layer model (`make
oracle-real`) passes of 3, 64, and 1056 tokens identical to the bit; `generate` on 1024-token
prompt gives transformers' greedy tokens.

Notes:
- **Prefill 6.5× at 16 threads** (30 → 197 tok/s), 2.0× also at 1 thread. Gap with llama.cpp
  drops from 11.8× to **1.97×** (1.4-1.6× with fewer threads).
- **Decode unchanged** (differences within spread): same code with one-token pass, logits
  identical to the bit.
- **Still scales poorly**: 4 to 16 threads yields 2.0×, 8 to 16 only 1.16×; llama.cpp 2.2× from
  4 to 16. The two Trochilus measurements at 16 threads this session (197 and 163) have wide
  spread: at 16 threads the machine is at power limit and variability rises (question 20).
- **At 1 thread llama.cpp ahead by 1.57×**: same work with int8 VNNI dot (lever 2, question 21).

## Prefill profile — where time goes (2026-09-17)

`bench/scenarios-olmoe-1b-7b.json` (scenarios `prefill512-t*`, `tools/profile_suite.py` now
reports zones in prefill too), prompt 512, median of 5, before 4-token kernel.

| Thread | Prefill tok/s | matmul | attention | serial (norms, RoPE, KV, router, gather, mix) |
|---|---|---|---|---|
| 16 | 144.3 | 89.8% | 2.4% | 7.6% |
| 8 | 114.9 | 91.1% | 2.3% | 6.4% |
| 4 | 66.2 | 93.6% | 2.2% | 4.0% |
| 1 | 18.2 | 95.5% | 2.1% | 2.3% |

Zone by zone at 16 threads: `expert_gate_up` 43.9%, `expert_down` 23.2%, `qkv_proj` 17.2%,
`attn_out_proj` 5.5%, `attention` 2.4%, `kv_write` 1.6%, `router` 1.5%, `expert_gather` 1.5%.
Distinct weight bytes 12.5 MiB/token (1.9 GB/s): prefill isn't bandwidth-limited like decode
(1.2 GiB/token, 41 GB/s), time is in multiplications.

## Where threads go — 16-thread prefill (2026-09-17)

Native Windows (container topology isn't real, LESSONS #47), binary at `5db8117`, OLMoE-1B-7B
Q8_0 from disk, synthetic 2048-token prompt, `-n 0`. Process affinity set at start, before pool
runs; cases alternate each round (LESSONS #46), median of 3 rounds plus one warmup. Clock from
Windows counter «Processor Information(_Total)\% Processor Performance» (nominal 2400 MHz),
sampled while prefill runs.

| Placement | Threads | Cores used | Prefill tok/s (min-max) | Median clock |
|---|---|---|---|---|
| default, no affinity | 16 | Windows picks from 32 logical | 134.1 (134.1-138.2) | 4.4-4.6 GHz |
| one per physical core (`0x55555555`) | 16 | 16 physical, both CCDs | **173.9** (173.3-183.1) | 4.2-4.5 GHz |
| one CCD only (`0x0000FFFF`) | 8 | cores 0-7, one 32 MB L3 | 78.6 (77.7-80.8) | 4.4-4.6 GHz |
| other CCD (`0xFFFF0000`) | 8 | cores 8-15 | 80.8 (79.1-82.8) | 4.6-4.7 GHz |
| 4+4 across CCDs (`0x00FF00FF`) | 8 | cores 0-3 and 8-11 | **86.1** (84.2-89.0) | 4.7-4.8 GHz |

Clock during prefill without affinity, medians of 3 rounds: 1 thread 4.66-4.80 GHz, 4 threads
4.64-4.78, 8 threads 4.53-4.69, 16 threads 4.46-4.54.

Notes:
- **Not power limit** (question 20): 1 to 16 threads clock drops 6-8%, not 40% needed to explain
  1.16×. On power the machine holds ~4.5 GHz on all cores.
- **Not the CCD** (question 2): 8 threads split 4+4 across chiplets go **7-9% better** than 8 on
  one chiplet. Two 32 MB L3s and double L2 beat locality, and clock rises.
- **It's where threads end up**: with 16 threads on 32 logical processors Windows puts two on
  same physical core and leaves cores free. One thread per physical core gives **+30%** (134 →
  174 tok/s). From 8 cores (86.1) to 16 (173.9) prefill scales **2.02×**: parallelism scales,
  placement didn't (LESSONS #48).
- At 2048-token prompt prefill does 174 tok/s versus 197 measured at 512 without pin: attention
  per token grows with context, measure separately (question 7).

Next step from this: pin pool threads to physical cores, and remeasure speed rows with that base.

## Thread pinning to physical cores (2026-09-18)

Native Windows (LESSONS #47), idle machine: other projects' containers stopped and Docker VM
cache returned, 15.9 GB free (measuring while an agent compiles in Docker gives false numbers,
LESSONS #57). One binary and three modes, chosen with `TR_POOL_PIN`: **0** no affinity, **1**
each thread on its own logical processor, **2** (the default) each thread on its whole
**physical core**, i.e., free among the two SMT siblings of that core but only that core.
**Alternating runs** (`tools/ab_speed.sh`, LESSONS #46), median of rounds after one warmup,
OLMoE-1B-7B Q8_0 from Windows disk, prompt 512, 24 generated tokens.

**Default vs no pin** (2 vs 0), 6 rounds:

| threads | prefill core | prefill none | | decode core | decode none | |
|---|---|---|---|---|---|---|
| 16 | **217.8** (184.9-220.3) | 175.1 (148.4-183.4) | **1.24×** | 28.7 (26.4-29.8) | 29.9 (27.2-30.3) | 0.96× |
| 8 | **147.3** (141.9-151.4) | 113.4 (105.6-117.3) | **1.30×** | 31.0 (28.4-31.5) | 30.4 (26.4-30.9) | 1.02× |

**Two pin modes** (2 vs 1), 3 rounds: at 16 threads whole core gives prefill 212.5 vs 178.6
and decode 27.2 vs 21.5; at 8 threads 145.9 vs 141.4 and 29.4 vs 31.1. *16-thread numbers
superseded by A/A remeasure below.*

**Logical processor pin vs no pin** (1 vs 0), 3 rounds, the abandoned path:

| threads | prefill | none | | decode | none | |
|---|---|---|---|---|---|---|
| 16 | 231.9 | 191.3 | 1.21× | 25.1 | **30.5** | **0.82×** |
| 8 | 171.0 | 125.9 | 1.36× | 32.7 | 28.6 | 1.14× |
| 4 | 90.5 | 60.4 | 1.50× | 32.5 | 24.6 | 1.32× |
| 1 | 24.0 | 24.0 | 1.00× | 12.2 | 10.7 | 1.14× |

Same mode (1 vs 0) at 2048-token prompt, 16 tokens: 16 threads prefill 159.2 vs 147.2 (1.08×)
and decode 19.5 vs 23.4 (0.83×); 8 threads prefill 130.1 vs 86.7 (1.50×) and decode 21.4 vs 22.0.
Machine busy (4 processes running idle, prompt 512, 16 threads): prefill 206.5 vs 173.6
(**1.19×**), decode 20.3 vs 30.4 (0.67×).

**16-thread A/A remeasure** (8 rounds, round-robin order, `tools/ab_modes.sh`; table and
thresholds in §Adversarial review). Prefill: to core **242.9**, to processor 223.3 (0.92×,
unstable: 185-240), no pin 190.8 (0.79×); all differences above threshold (1.2%). Decode: 31.26,
30.74, and 31.82; between core pin and the other two **not distinguishable** (−1.6% and +1.8%,
threshold 2.2%). 16-thread decode of processor pin (25.1 and 21.5 in tables above, 0.82×)
**does not reproduce**: vs no pin is 0.97×.

Notes:
- **Prefill always gains**, more with fewer threads: at 4 threads worth 1.50×, because without
  affinity Windows puts those 4 threads on 2 physical cores. With default (core pin) at 16
  threads is 1.24× (**1.27×** in A/A remeasure). Even with machine busy gain remains (1.19×):
  pin isn't fragile under contention (question 22).
- **Pin mode shows in prefill, not decode** (corrected by A/A remeasure). First measure (3
  rounds, fixed order) gave **logical processor** pin 17-18% cost on 16-thread decode: with 8
  rounds and A/A control doesn't reproduce (−3.4% vs no pin; −1.6% vs core pin, not
  distinguishable). What **core** pin adds more is prefill: **+9%** over processor pin, 8 rounds
  out of 8, stable (spread 5% vs 25%). Keeps the default: prevents two threads landing on same
  core without pinning them tight. Busy-machine data (0.67×) not remeasured. Question 27:
  premise doesn't hold.
- Not wakeup latency: lengthening spin from 2 to 20 ms made 16-thread decode pinned to
  processor 21.0 instead of 23.6, i.e., worse.
- **Decode prefers 8 threads to 16**: with default 34.7 vs 31.0-31.7 in A/A remeasure
  (**1.09-1.12×**, disjoint intervals, threshold 2.2%; before 31.0 vs 28.7 with overlapping).
  Prefill wants all cores (242.5 vs 170.9 at 8, 1.42×), decode doesn't. Today thread count is
  one for both phases: it's an open lever (question 26).
- At 1 thread pin doesn't change prefill (nothing to place) and gives +14% on decode.

## Int8 activations with VNNI: what lever 2 would give (2026-09-18)

Question 21. Candidate written **in benchmark**, not in kernels (`tests/bench_kernels.c`):
quantizes activations to int8 in blocks of 32 like Q8_0 and uses `_mm256_dpbusd_epi32`, with
same ggml structure (int32 accumulate per block, convert and scale in float accumulator, block
weight sums precomputed for +128 correction). Not ours and won't be: 8-bit activations are
another number, so can't be bit-identical. `make bench`, 9 runs of 40 ms, median.

| kernel | activations | n=1024 | n=2048 | n=4096 |
|---|---|---|---|---|
| `dot_row q8_0` (AVX-512) | float | 11 083 | 11 278 | 11 379 |
| `dot q8_0 x q8_0` VNNI (candidate) | int8 | 11 184 | 11 398 | 12 310 |
| `dot_row q8_0 x4` (already ours) | float, 4 tokens | **28 644** | **29 969** | **30 565** |

Million elements per second per activation row. Quantizing activations costs 467 M elem/s
(scalar), which one matmul pays once every 1024 rows: negligible.

Notes (**corrected by 2026-09-18 review**, §Adversarial review and LESSONS #65: table is
right, the conclusion drawn from it isn't):
- This table compares one row against **one** token, int8 vs float: there int8 is worth +1-8%.
  But the candidate has half a row scalar work per block (half→float, correction chain), and
  prefill doesn't run with that kernel: runs with 4-token kernel.
- Our 4-token kernel is 2.6-2.7× faster than both because it reuses weight row. The right
  question was what int8 gives **with that same structure**: measured later, gives **1.8-2.3×**
  more, while «wider blocks» in float (8 tokens) doesn't pay. Gap with llama.cpp is in 8-bit
  activations, not wider blocks.
- At these n the dot isn't memory-limited (12 GB/s vs ~41 measured): it's latency and
  instructions per block. One instruction doing 4 products doesn't change the total only while
  instructions per block stay the candidate's; remove them, it changes.

## SMT: 32 threads vs 16 (2026-09-18)

Same method, 2048-token prompt, 16 tokens, both cases **pinned**: A = 16 threads, one per
physical core; B = 32 threads, one per logical processor (two siblings of each core).

| | 16 thread (un core ciascuno) | 32 thread (fratelli SMT) |
|---|---|---|
| prefill tok/s | **163.6** | 110.2 |
| decode tok/s | **22.9** | 2.2 |

Prefill loses a third, decode crashes **tenfold**: two threads on same core fight over cache and,
with 464 dispatches per token, each waits for its sibling. Closes question 3: SMT copies don't
help, and the default (threads = physical cores) is right.

## Speculation from prompt — how much it helps and hurts (2026-09-17)

Container, volume `trochilus-models`, whole OLMoE-1B-7B Q8_0, `run -f <prompt> -n 200`,
`--spec 0` and `--spec k` **alternated run by run** (`tools/ab_spec.sh`, LESSONS #46), median of 3
runs after one warm-up. The generated text is identical on every run: the script stops if it is
not. No thread pinning and measurements in a container: redone native and with pinning in §Adaptive
draft: what one more row really costs, where the cost of one more row comes out about 3 times what
is written below (13.7-17.6 ms measured per zone against 5.2; LESSONS #59).

| Prompt | Draft | Decode tok/s (min-max) | Drafts accepted | Against `--spec 0` |
|---|---|---|---|---|
| `bench/prompts/code-edit.txt` (the file is already in the prompt, to be rewritten with a change) | 0 | 29.1 (27.1-30.7) | — | — |
| | 2 | 32.5 (31.4-33.5) | 120/160 = 75% | **1.11×** |
| | 8 | **41.3** (41.0-43.2) | 170/264 = 64% | **1.42×** |
| `bench/prompts/code.txt` (new code, the model copies nothing) | 0 | 29.6 (28.5-31.8) | — | — |
| | 8 | 18.4 (18.4-19.5) | 57/435 = 13% | **0.62×** |
| same prompt, 4 threads | 0 | 29.1 (28.7-29.4) | — | — |
| | 8 | 16.3 (16.2-16.4) | 57/435 = 13% | **0.56×** |

Notes:
- **On real code work the lever is there**: when the text to produce is already in the prompt
  (rewriting a file with a change, the case the lever was chosen for), the 8-row draft gives
  **1.42×** with 64% of tokens accepted, and the tokens are the same.
- **When the model invents, it loses**: 13% accepted and 0.62×. A pass of 1 + k positions costs
  ~34 ms of weights plus ~5.2 ms per extra row (34 ms at one row, 76 ms at nine): break-even is
  around **15% of drafts accepted**, and below that threshold it costs more.
- The cost per extra row (~5.2 ms) is the same per-element work as the prefill, so pinning threads
  to physical cores lowers it: after that step the break-even threshold falls and the long draft
  pays off more often. To be remeasured there.
- Consequence: `--spec` stays **off by default** until the draft shortens itself after a rejection
  (llama.cpp does this). With the adaptive draft, the worst case becomes "like without".

## Adaptive draft: what one more row really costs (2026-09-18)

Windows **native**, still machine, pinned to core (the default), 16 threads, `run -f <prompt> -n
200`, `--spec 0` and `--spec 8` **alternated run by run** (`tools/ab_spec.sh`), median of 3 runs
after one warm-up, identical text on every run. `--spec-fixed` picks the fixed draft. The two
"with pause" rows are the **remeasure at 8 runs with an A/A control** (`tools/ab_modes.sh`,
§Adversarial review): the earlier 3 runs gave 1.15× (28.63 → 32.98) and 1.01× (27.49 → 27.76), and
the second one was wrong.

| Prompt | Draft policy | `--spec 0` | `--spec 8` | | Accepted | Average draft |
|---|---|---|---|---|---|---|
| `code-edit.txt` (the file is already in the prompt) | fixed 8 | 27.99 | **37.61** | **1.34×** | 64.4% | 8.00 |
| | adaptive, shortens only | 26.96 | 32.81 | 1.22× | 70.9% | 2.86 |
| | adaptive with pause (today), 8 runs with A/A | 31.87 | 37.45 | **1.175×** | 69.4% | 2.34 |
| `code.txt` (new code) | fixed 8 | 29.65 | 17.67 | **0.60×** | 13.1% | 3.04 |
| | adaptive, shortens only | 29.61 | 26.22 | 0.89× | 41.3% | 0.70 |
| | adaptive with pause (today), 8 runs with A/A | 32.67 | 31.15 | **0.953×** | 42.4% | 0.38 |

The cost of one more row, derived from the printed passes (ms per pass = 1000 · tokens / tok/s
/ passes, minus the ~35 ms of a one-token pass, divided by the average draft): **15 ms** with a
fixed draft of 8, **20 ms** with an average draft of 2.3, **27 ms** with an average draft of 0.4.
Measured then per zone (§Adversarial review): **17.6 ms** if it is the only extra row, **13.7 ms**
each if there are eight, against a pass's 31.3.

Notes:
- **On a MoE one more row is not nearly free, the way it would be on a dense model.** The decode
  costs because it reads the weights of the 8 experts chosen in each layer; the draft token
  chooses others, and the pass reads those too. One more row thus costs half a pass or more, and
  break-even sits at **44-56% of drafts accepted** (row cost / pass cost, measured per zone in
  §Adversarial review; the tok/s estimate said 60-75%), not the 15% the container measurement said
  (LESSONS #59). This is also why the long draft only pays off where the text is already in the
  prompt: there the experts chosen are the same, because the tokens are the same.
- **Shortening the draft was not enough**: at 41% accepted, 11% was still lost. Stopping entirely
  after a fully wrong draft (pause 1, 3, 7, 15 steps, capped at 16) brings the worst case to
  **0.95×** (A/A remeasure: −4.7%, above the threshold, 8 of 8 runs; the earlier 3 runs said
  1.01×) and leaves the good case at **1.175×**. "Like without" is **not reached**: turning
  `--spec` on by default means accepting −5% where the model invents for +17.5% where it copies.
  Question 25 stays closed (the pause is measured), the default's choice is Marcello's.
- The fixed draft stays the fastest where the model copies (1.34×): whoever knows they are
  rewriting a file can still ask for `--spec 8 --spec-fixed`.

## Kernel: one weight row vs 4 tokens (2026-09-17)

`make bench` (`tests/bench_kernels.c`, container, median of 5 × 100 ms, one thread), million
elements per second **per input row**:

| Kernel | n=1024 | n=2048 | n=4096 |
|---|---|---|---|
| dot_row q8_0 (one row, one token) | 20 525 | 21 154 | 21 517 |
| dot_row q8_0 x4 (one row, 4 tokens) | 41 467 | 44 169 | 36 898 |
| dot_f32 (float reference) | 31 421 | 31 653 | 29 153 |

`tr_matmul` on a 1024×2048 q8_0 matrix (shape of an expert projection), ms per token:

| Thread | 1 token | 64 token |
|---|---|---|
| 1 | 0.100 | 0.059 |
| 4 | 0.027 | 0.016 |
| 8 | 0.014 | 0.008 |
| 16 | 0.014 | 0.009 |

Notes:
- The 4-token kernel is worth **2.1×** on the dot product and even beats `dot_f32`: the q8_0
  weight is read and converted once for four products, and occupies a quarter of a float's bytes.
- The whole matrix gains less than the kernel (1.7× at 1 thread): there the bytes read weigh in too.
- From 1 to 16 threads the same matrix gives 6.5×, not 16: at 16 threads the machine is at its
  power limit (question 20).

## Speed levers: what research says (2026-09-17)

Numbers from sources, on other machines: point the direction, don't promise. Exact = same
numbers; declared = different numbers, difference to measure on real model before adopting.

| # | Lever | Type | Reported gain | For us |
|---|---|---|---|---|
| 1 | SIMD on the row × vector product | exact | — | done: 10× on the kernel |
| 2 | int8 + VNNI activations (`VPDPBUSD`), as llama.cpp does for Q8_0 | declared | llama.cpp uses this path for Q8_0 | high: fewer bytes and less work per element |
| 3 | Batched prefill (more tokens in the same multiplication) | exact | prefill linear in the batch ([discussion #18030](https://github.com/ggml-org/llama.cpp/discussions/18030)) | done with 4: prefill 4.3× at 16 threads, decode unchanged |
| 4 | Tokens grouped by expert in the prefill | exact | no isolated number | done with 3 |
| 5 | Weights reordered in row blocks (8×8 repack) | exact | +53% prefill, ~0% decode on Zen 5 ([PR #9532](https://github.com/ggml-org/llama.cpp/pull/9532)) | after 3-4 |
| 6 | Cache blocking (Goto/BLIS) | exact | base of 3-5 ([Goto 2008](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)) | only with more tokens together |
| 7 | llamafile's tinyBLAS | exact | prefill 1.3-4×, decode little ([PR #6414](https://github.com/ggml-org/llama.cpp/pull/6414)) | ideas for 3-5 |
| 8 | Speculative decoding | exact (greedy) | 2-3× ([arXiv 2211.17192](https://arxiv.org/abs/2211.17192)) | needs a draft model; for coding the prompt's own text can be used |
| 9 | Skipping the weak expert | declared | 1.2-1.3×, -3 quality points ([arXiv 2402.14800](https://arxiv.org/abs/2402.14800)) | optional only, with a quality measurement |
| 10 | Activation sparsity (PowerInfer) | declared | 2.9× ([arXiv 2312.12456](https://arxiv.org/abs/2312.12456)) | needs a predictor trained per model |
| — | Tables (T-MAC, bitnet.cpp), AMX, huge pages, NUMA | — | — | not for us: 1-4 bit weights, Intel only, I/O-bound, a single socket |

7940HX cache: L1d 32 KB and L2 1 MB per core, L3 32 MB for each of the two 8-core groups.
An expert row (2 KB) and the vector (8 KB) already fit in L1: in the decode a weight is read once
per token, and there is nothing to keep in cache. Blocking matters when several tokens reuse the
same rows (levers 3-6).

## Adversarial review (2026-09-18)

Re-read of entire repository by another model (Fable 5.1), instructed to bring evidence not
opinions. New errors and checks: LESSONS #60-#68. Here are the numbers. Container
`trochilus-dev`, other containers on: so comparisons are only those with effects well above
spread (kernel on one core, alternating runs) or **no stopwatch** (counters, replay). None of
what follows is a product speed measurement except the last section (**Native A/A remeasure**),
which is native and machine idle.

**Invariants: they hold.** `trochilus logits` on 40-70 tokens, 3 tiers (`TR_CPU_MAX` scalar, avx2,
avx512) × 3 thread counts (1, 5, 16) × 3 passes (`-b` 1, 7, 512): 27 files per model, identical to
the byte on the comparable rows, for the three tiny models (f32, f16, q8_0) and for the real
OLMoE cut to 2 layers (201 216 bytes per row). The model tests pass under `scalar` and `avx2` too.
It is now a check: `make tier-check`. Tokenizer: 90 000 strings built to break it (storms of
combining marks, jamo, every Unicode space, contractions, added tokens glued to marks and
spaces) × 2 modes against HF `tokenizers`: 0 differences, and 12 000 equal decodings.

**int8 activations, second measurement** (question 21, LESSONS #65). `make bench`, "one row, by"
section: the four kernels run alternated on the same row, 9 runs of 40 ms, one core. Nanoseconds
per input row and ratio to our kernel; float x8 is verified bit-identical to x4, the int8 ones
against their plain-C dot.

| kernel | n=1024 | n=2048 | n=4096 |
|---|---|---|---|
| `dot_row q8_0 x4` float, AVX-512 (ours) | 25.6 ns | 47.6 ns | 114.1 ns |
| int8 VNNI x4, 256 bit, ggml's sign trick | 1.64× | 1.62× | 1.96× |
| int8 VNNI x4, 512 bit, two blocks per instruction | **1.83×** | **1.79×** | **2.26×** |
| float x8, exact (the "wider blocks" path) | 1.13× | **0.79×** | 0.93× |

Spread 3-8% per row; with clang 1.78× / 1.84× / 2.13× and 1.15× / 0.79× / 0.91×. Notes:
- The prefill is 90% multiplications and llama.cpp is ahead 1.57× at one thread: it is int8,
  almost entirely. Float x8 at n=2048 (OLMoE's width) **loses**: 8 rows of float activations are
  64 KB and L1 holds 32; in int8 they are 16 KB.
- Limit of the measurement: it is a kernel on one core with everything in cache; on the whole
  prefill the gain will be lower (the 10% non-matmul remains, and the activations' quantization).
  And int8 **is not exact**: the prefill's logits would no longer be the token-by-token ones. It
  can only exist as a declared mode; the decision is Marcello's.

**How much a draft row reads** (question 12, LESSONS #67). The profiler's `weight_bytes` counter
divided by the passes: deterministic, real model, 200 tokens, `generate --spec k`.

| prompt | draft | rows/pass | MiB of weights/pass | MiB per extra row | new experts per layer (of 8) |
|---|---|---|---|---|---|
| all | 0 | 1.00 | 1200.4 | — | — |
| `code-edit` | fixed 1 | 2.00 | 1675.8 | 475 | 4.7 |
| `code-edit` | fixed 8 | 9.00 | 3221.3 | 253 | 2.5 |
| `code-edit` | fixed 15 | 16.00 | 3891.0 | 179 | 1.8 |
| `code-edit` | adaptive | 3.34 | 2001.0 | 342 | 3.4 |
| `code` | fixed 1 | 1.46 | 1404.9 | 445 | 4.4 |
| `code` | fixed 8 | 4.04 | 1916.7 | 236 | 2.3 |
| `code` | adaptive | 1.38 | 1337.6 | 361 | 3.5 |

One expert is 6.375 MiB (three 2048×1024 Q8_0 matrices). At 36 GB/s, 253 MiB is 7 ms of the 15 of
one more row: reading the new experts explains **half** the cost, not all of it. In the same
profile (times in a container, indicative only) one more row costs 4-5.6 ms even in the dense
zones (`qkv_proj`, `attn_out_proj`, `lm_head`), where there is no new weight to read. The timed
part is measured native further below (**How much a draft row costs, by zone**): 61-91% of the
cost sits in the experts, and the dense zones only weigh in with long drafts.

**The pause's constants, without a stopwatch** (LESSONS #60, #67). Whether a draft gets accepted
depends only on the tokens, and the tokens are the same under every policy: every policy can be
replayed on the recorded continuation. The replay reproduces the engine's counters exactly (77
passes, 180 drafts, 125 accepted on `code-edit`; 172, 65, 28 on `code`). Time = passes × 35 ms +
draft rows × C; gain over `--spec 0` for C = 15 and 20 ms:

| policy (draft 8) | `code-edit` passes / drafts | C=15 | C=20 | `code` passes / drafts | C=15 | C=20 |
|---|---|---|---|---|---|---|
| fixed | 33 / 264 | 1.37× | 1.09× | 143 / 435 | 0.61× | 0.51× |
| adaptive, shortens only | 66 / 189 | 1.36× | 1.15× | 155 / 109 | 0.99× | 0.92× |
| pause 1, 3, 7, 15, 16 (today) | 77 / 180 | 1.30× | 1.11× | 172 / 66 | 1.00× | 0.95× |
| pause 1, 3, 7, 15, 31, 16 (error #60) | 77 / 180 | 1.30× | 1.11× | 172 / 65 | 1.00× | 0.96× |
| pause 1, 2, 4, 8, 16 | 71 / 183 | 1.34× | 1.14× | 165 / 83 | 1.00× | 0.94× |
| pause 2, 5, 11, 16 | 78 / 169 | 1.33× | 1.15× | 168 / 65 | 1.02× | 0.97× |
| pause always 4 | 77 / 168 | 1.34× | 1.16× | 173 / 68 | 0.99× | 0.94× |

Notes: the constants shift 2-3%, less than a timed measurement on this machine can see; they stay
as they are. The worst case "1.01×" measured native over 3 runs corresponds to C ≈ 15 ms; with
C = 20-22 (the value the same section derives for short drafts) the replay gives 0.94-0.96×. The
question of "`--spec` on by default" must be decided on a measurement with more runs and an A/A
control (`tools/ab_modes.sh`, LESSONS #66): done, it is the **Native remeasure** block further
below, and it gives **0.953×**.

**The pool with more than one pool** (LESSONS #61-#63). Test program, Linux: `H1` caller first
`[0-31]`, then two pools created and destroyed in birth order `[0,1]`; `H2` process restricted to
`[0,2]`, pool of 4: workers 2 and 3 on `[0]`; `H3` worker 1 of two live pools both on `[2,3]`; `H4`
inner pool of 2, worker id reaching the body: 3. H1 and H2 correct and under test; H3 and H4
remain a declared debt in `threads.h`.

**Native remeasure with an A/A control** (LESSONS #66 and #67; `sh tools/remeasure.sh`, each run
in `build/remeasure/`). Native Windows, still machine: the script stops the 9 other projects'
containers and restarts them at the end. Binary of commit 87297b4, OLMoE-1B-7B Q8_0, 16 threads
unless stated otherwise. `tools/ab_modes.sh`: 8 runs after one warm-up, the round's first mode
rotates, a mode given twice acts as an A/A control. Median (min-max), ratio to the first mode.

| measurement | mode | prefill tok/s | | decode tok/s | |
|---|---|---|---|---|---|
| 1. `--spec`, worst case: `run -f code.txt -n 200` | `--spec 0` | 215.1 (210.1-221.9) | — | 32.67 (31.93-33.00) | — |
| | `--spec 0`, A/A copy | 217.6 (212.2-227.4) | 1.012× | 32.39 (31.72-33.26) | 0.991× |
| | `--spec 8` | 219.2 (211.6-227.6) | 1.019× | 31.15 (30.68-31.44) | **0.953×** |
| 2. `--spec`, good case: `run -f code-edit.txt -n 200` | `--spec 0` | 229.2 (221.2-232.5) | — | 31.87 (31.06-32.22) | — |
| | `--spec 8` | 229.4 (222.8-232.8) | 1.001× | 37.45 (36.98-38.09) | **1.175×** |
| 3. pin: `generate -p 512 -n 24` | to core (2, the default) | 242.9 (234.7-246.7) | — | 31.26 (30.54-32.02) | — |
| | to core, A/A copy | 241.0 (235.3-246.1) | 0.992× | 31.11 (30.24-31.58) | 0.995× |
| | to processor (1) | 223.3 (185.3-240.4) | **0.919×** | 30.74 (29.63-31.35) | 0.984× |
| | none (0) | 190.8 (187.7-193.1) | **0.785×** | 31.82 (30.98-32.38) | 1.018× |
| 4. threads: `generate -p 512 -n 48` | 16 | 242.5 (236.3-247.0) | — | 31.70 (30.37-31.89) | — |
| | 16, A/A copy | 240.9 (236.9-243.2) | 0.994× | 31.00 (30.39-32.10) | 0.978× |
| | 8 | 170.9 (165.5-172.6) | **0.705×** | 34.67 (33.94-35.04) | **1.094×** |

**The session's A/A noise** (two identical modes, difference between medians): decode 0.9%, 0.5%,
2.2%; prefill 1.2%, 0.8%, 0.6%. The same decode at 16 threads gives 0.5% in block 3 and 2.2% in
block 4: a single A/A pair is itself a noisy measurement. The threshold used here is the
**session's worst A/A for that phase** (decode 2.2%, prefill 1.2%), and a difference only counts
if it exceeds this against **both** copies of the control. Below that: "not distinguishable".

Notes:
- **`--spec`'s worst case: 0.95×, not 1.01×.** −4.7% against `--spec 0` and −3.8% against its own
  copy (threshold 2.2%), disjoint intervals, `--spec 8` below both in 8 of 8 runs. Counters: 172
  passes, 28/66 drafts accepted (42.4%), average draft 0.38. The pause takes the loss from 0.60×
  to 0.95×, it does not remove it: the condition "worst case like without" is **not met**. On the
  prefill +1.9% and +0.8% against the two copies: not distinguishable.
- **Good case: 1.175×** (it was 1.15× over 3 runs), disjoint intervals; 77 passes, 125/180 drafts
  accepted (69.4%), average draft 2.34. Prefill 1.001×: not distinguishable.
- **Pin to core against no pin: prefill 1.27×** (1.26× from the copy), disjoint intervals. Decode
  −1.8% and −2.2% from the two copies, straddling the threshold: **not distinguishable**.
- **Pin to core against pin to processor: the difference is in the prefill, not in the decode.**
  The processor-pinned prefill sits at −8.1% and −7.3% (threshold 1.2%), below the core-pinned one
  in 8 of 8 runs, and it is unstable: spread 24.7% (185-240) against 5.0%. Decode −1.6% and −1.2%:
  **not distinguishable**. The processor pin's "−17-18% on decode" (3 runs, §Thread pinning to
  physical cores) **does not reproduce**: against no pin it is −3.4% (above the threshold, 7 of 8
  runs, overlapping intervals). The path to one pool and 16 threads is the same in both binaries
  (87297b4's diff to `threads.c` touches the caller with several pools and workers with no slot):
  the difference does not come from the code. The cause was not sought; that measurement had 3
  runs, a fixed order and no A/A.
- **The decode wants 8 threads: 1.09× against 16 and 1.12× against its own copy** (threshold
  2.2%), disjoint intervals (33.9-35.0 against 30.4-32.1), 8 of 8 runs. The prefill wants 16:
  1.42× against 8. Confirms question 26's premise, which before sat inside the spread.

**How much a draft row costs, by zone** (question 12, the timed half; LESSONS #67). Same session:
`generate --tokens <code-edit> -n 200 --profile-json`, three modes alternated for 3 runs
(`--spec 0`, `--spec 1 --spec-fixed`, `--spec 8 --spec-fixed`). Milliseconds per pass, median
(min-max). "Dense" are `qkv_proj`, `attn_out_proj`, `lm_head`; "experts" are `expert_gate_up` and
`expert_down`. The cost of one more row is the difference with the one-row pass of the same run,
divided by the extra rows. Without A/A, but every mode's three runs stay within 0.9%.

| rows per pass | ms per pass | dense | experts | MiB of weights |
|---|---|---|---|---|
| 1 | 31.27 (31.20-31.34) | 9.23 | 16.37 | 1200.4 |
| 2 | 48.84 (48.55-48.89) | 9.16 | 32.35 | 1675.8 |
| 9 | 141.06 (140.10-141.33) | 42.25 | 83.42 | 3221.3 |

| one more row costs | total | experts | dense | attention | MiB of new experts |
|---|---|---|---|---|---|
| if it is the only one (2 rows) | **17.6 ms** (17.4-17.6) | 16.0 (91%) | −0.1 | 1.1 | 475 |
| if there are eight (9 rows) | **13.7 ms** (13.6-13.8) | 8.4 (61%) | 4.1 (30%) | 0.5 | 253 |

Notes:
- A draft row costs **more than half a pass** when it is the only one (17.6 ms of 31.3) and 13.7
  ms each when there are eight. Before it was derived from tok/s (15-27 ms, §Adaptive draft: what
  one more row really costs); now it is measured per zone. Break-even is row cost / pass cost:
  **56% of drafts accepted** for short drafts, **44%** for eight-row ones. This matches the two
  prompts: 42.4% accepted loses (0.953×), 69.4% gains (1.175×).
- **The cost sits in the experts**: 91% with one more row, 61% with eight. At the two measured
  points the experts' time per row follows the MiB of new experts counted above (475 MiB in 16.0
  ms, 253 in 8.4: 29.8 and 30.2 MiB/ms). A draft's cost attaches on the weights' side (which
  experts are read), not the compute's: this closes question 12. Limit: per MiB the new experts
  cost 1.65 times those of the one-row pass (816 MiB in 16.4 ms, 49.8 MiB/ms), and the reason is
  not measured.
- **The dense multiplications are free for the first extra row** (−0.1 ms: the 4-token kernel
  reads the weight row only once) and cost 4.1 ms per row at nine rows: the 4-5.6 ms seen in the
  container hold for long drafts, not short ones.
- **The replay holds up to the stopwatch test**: passes × 31.3 ms + draft rows × 17.6 ms gives
  0.95-0.96× for today's policy on `code` (172 passes, 66 rows, against 199-200 passes with no
  draft); the timed measurement of block 1 gives 0.953× and 0.962× against the two copies. With
  the same costs, none of the replay table's policies reaches 1.00× on `code` (the best, pause 2,
  5, 11, 16, gives 0.97-0.98×): the pause's constants are not enough, the lever is the row's cost.

## Threads per phase (2026-09-18)

Question 26. Native Windows, machine idle (`tools/threads_phase.sh` stops 9 containers of
other projects and restarts at end), OLMoE-1B-7B Q8_0, pinned to cores, `tools/ab_modes.sh`:
8 rounds after one warmup, first mode of round rotates, A/A control. Median (min-max), ratio
to first mode. Each run in `build/threads_phase/`.

**1. How many threads each phase wants** (`sh tools/threads_phase.sh sweep`: a single `-t` for
both phases, binary of commit b231b28, `generate -p <context> -n 48`).

| context | threads | prefill tok/s | | decode tok/s | |
|---|---|---|---|---|---|
| 512 | 16 | 231.2 (228.7-237.4) | — | 30.77 (29.31-31.67) | — |
| | 16, A/A copy | 233.4 (228.5-236.3) | 1.009× | 30.70 (30.39-31.12) | 0.998× |
| | 12 | 210.5 (204.5-212.2) | 0.910× | 31.50 (30.34-31.79) | 1.024× |
| | 8 | 168.4 (167.5-170.6) | 0.728× | 33.79 (33.36-34.67) | **1.098×** |
| | 4 | 92.1 (91.5-92.8) | 0.398× | 34.39 (33.34-34.97) | **1.118×** |
| 2048 | 16 | 186.5 (183.2-188.0) | — | 24.06 (22.19-24.45) | — |
| | 16, A/A copy | 187.2 (184.9-189.8) | 1.004× | 23.91 (23.42-24.51) | 0.994× |
| | 12 | 166.4 (162.3-170.2) | 0.892× | 23.20 (21.72-24.05) | 0.964× |
| | 8 | 134.9 (133.2-137.2) | 0.723× | 24.65 (24.17-25.12) | **1.025×** |
| | 4 | 74.3 (73.8-74.4) | 0.398× | 23.77 (23.32-24.27) | 0.988× |

Threshold, the session's worst A/A: decode 0.6%, prefill 0.9%.

- **The prefill wants all the threads**, at every context: 16 against 8 gives 1.37× at 512 and
  1.38× at 2048.
- **The decode at context 512**: 8 and 4 threads beat 16 by 9.8% and 11.8% (10.1% and 12.0%
  against the copy), with intervals disjoint from that of 16. Between them 4 is ahead by 1.8%,
  with the intervals nested inside each other (33.3-35.0 and 33.4-34.7). 12 gives 2.4%.
- **At context 2048** 8 is the only number above the threshold against both copies (+2.5%, +3.1%).
  4 falls back (−1.2% and −0.6%: not distinguishable from 16, and −3.6% from 8) and 12 is the
  worst (−3.6%, −3.0%).
- **Choice: n = 8**, the only number above the threshold at both contexts. 4 wins by a little
  where the context is short and loses where it is long, and in coding the context is long
  (LESSONS #70).
- Why: the decode reads 1.2 GB of weights per token and 4 threads already fill the bus (34 tok/s
  is 41 GB/s); past that point every extra thread only adds waiting at each `parallel_for`'s
  barrier. As the context grows, attention grows too, and it is compute that splits per head:
  there the extra threads start to help again, and the optimum shifts from 4-6 toward 8.
  (Corrected on 2026-09-19, §Decode at long context and RAM bandwidth: the bus fills at ~54 GB/s,
  not 41, and attention was not compute but the KV read in jumps; with the KV per head at context
  4000 the session chooses 16 threads in 13 of 16 runs.)

**2. The decode at forced width** (new engine, `--decode-threads n`: the prompt always on 16
threads, short passes on the first n slots; context 512; session with the 2% rule, which only
counts for the last row). The prefill sits between 233.6 and 236.7 tok/s in every mode (A/A 1.3%):
the decode's width does not touch it.

| decode on | decode tok/s | |
|---|---|---|
| 16 | 30.88 (29.75-31.60) | — |
| 16, A/A copy | 31.13 (30.56-31.75) | 1.008× |
| 12 | 31.63 (30.16-32.63) | 1.025× |
| 8 | 34.07 (33.54-34.81) | **1.103×** |
| 6 | 34.63 (31.41-35.19) | **1.122×** |
| 4 | 33.84 (33.29-34.67) | **1.096×** |
| measured by the session (8 in 7 runs, 4 in one) | 33.82 (32.04-34.15) | **1.095×** |

Between 4 and 8 threads the curve is flat (33.8-34.6, differences below the session's threshold,
1.4%) and drops past 8. `-t 16 --decode-threads 8` performs like `-t 8` (34.07 against the sweep's
33.79): the workers left out sleep and cost nothing.

**3. The default's rule is a measurement, not a number.** How many threads fill the bus is a fact
of the machine (memory channels, bandwidth per core), and from a single machine no formula can be
derived that holds on the others: `min(core, 8)` and `core/2` both give 8 here, and they err in
opposite directions on a 4-core laptop and on a machine with many channels. So every session
measures its own: the first 9 one-token passes run in turn on the whole pool, on half and on a
quarter (16, 8, 4), three times each; it keeps the best time of each width and picks the widest
within 1% of the fastest; it remeasures every 1024 tokens, because the optimum shifts with the
context. A pass is "short" up to 4 rows (those the 4-token kernel covers with a single read of
the weight row); the others use the whole pool. `--decode-threads n` forces the width and measures
nothing. Tokens identical to the bit by construction (the pool's contract) and by proof:
`tests/test_phase.c`, `tests/test_base.c`, `make tier-check`.

The margin was measured three times, counting which width each run picks (`ab_modes.sh`'s `width`
column, 16 runs per context):

| margin | picks at 512 | picks at 2048 | decode after/before at 512 | at 2048 |
|---|---|---|---|---|
| 3% | 8 in 15, 4 in 1 | **16 in 10**, 8 in 6 | 1.096× and 1.091× | 0.996× and 1.012× |
| 2% | 8 in 15, 4 in 1 | 16 in 5, 8 in 11 | 1.081× and 1.092× | 1.014× and 1.026× |
| **1%** (adopted) | 8 in 11, 4 in 5 | 16 in 2, **8 in 14** | 1.085× and 1.088× | **1.027× and 1.019×** |

At 2048 the runs that pick 8 give 24.0-25.1 tok/s and those that pick 16 give 23.4-23.8: every
wrong pick costs 4%, and it happens because the best time of three passes has about 2% noise, the
same as the distance between 16 and 8 at that context (LESSONS #71). With 1% at 512 the pick falls
on 4 one time in three: there 4 and 8 perform the same, and the remeasure every 1024 tokens
brings it back to 8 as the context grows. A more robust estimator is question 31.

**4. `--spec 8` with the new engine** (`run -f <prompt> -n 200 --spec 8 -t 16`; "16 rows" is
`TR_DECODE_ROWS=16`: every verify pass at the decode's width, not just those up to 4 rows). 2%
rule.

| prompt | mode | decode tok/s | |
|---|---|---|---|
| `code-edit` (the draft is accepted) | before | 37.11 (36.63-37.51) | — |
| | before, A/A copy | 37.03 (36.73-37.79) | 0.998× |
| | after | 37.91 (36.60-38.29) | 1.021× |
| | after, 16 rows | 37.75 (36.54-38.49) | 1.017× |
| `code` (the model invents) | before | 31.01 (30.55-31.58) | — |
| | before, A/A copy | 30.98 (30.13-31.41) | 0.999× |
| | after | 33.00 (32.55-33.41) | **1.064×** |
| | after, 16 rows | 33.05 (32.45-33.51) | **1.066×** |

- Where the draft is accepted the passes almost all have more than 4 rows and run as before:
  +2.1%, straddling the threshold. Where the model invents the passes are short and take the
  decode's narrow-width gain: **+6.4%**, disjoint intervals. In that case the session picks 4
  threads in 8 of 8 runs.
- **The boundary at 4 or 16 rows cannot be distinguished** (−0.4% and +0.2%): it stays at 4, which
  leaves the long verify passes exactly as they were (no risk where compute matters more).
- The ratio between `--spec 8` and `--spec 0` on the worst case was not remeasured: both now take
  the narrow decode's gain, the first 6.4% and the second about 9%.

**5. Before and after** (`sh tools/threads_phase.sh change build/trochilus-before.exe`: b231b28's
binary against the new engine with the 1% rule, both at `-t 16`, each with its own A/A copy, plus
the new engine with `--decode-threads 8`; `generate -p <context> -n 48`).

| context | binary | prefill tok/s | | decode tok/s | |
|---|---|---|---|---|---|
| 512 | before | 235.3 (227.3-238.6) | — | 30.82 (30.03-31.29) | — |
| | before, A/A copy | 233.8 (229.5-236.7) | 0.994× | 30.71 (29.76-31.64) | 0.996× |
| | after | 233.6 (227.0-236.1) | 0.993× | 33.45 (32.46-34.19) | **1.085×** |
| | after, A/A copy | 235.7 (230.7-239.1) | 1.002× | 33.53 (32.85-34.13) | **1.088×** |
| | after, `--decode-threads 8` | 236.3 (233.4-237.7) | 1.004× | 33.50 (31.91-34.41) | **1.087×** |
| 2048 | before | 193.4 (189.7-194.0) | — | 23.79 (23.45-24.43) | — |
| | before, A/A copy | 192.5 (191.3-195.8) | 0.995× | 23.55 (23.29-24.47) | 0.990× |
| | after | 192.6 (189.3-193.7) | 0.996× | 24.43 (24.09-24.79) | **1.027×** |
| | after, A/A copy | 192.8 (190.8-194.2) | 0.997× | 24.24 (23.34-25.04) | **1.019×** |
| | after, `--decode-threads 8` | 193.3 (192.2-194.1) | 0.999× | 24.98 (24.14-25.21) | **1.050×** |

Session threshold: decode 1.0%, prefill 0.6%.

- **Decode at context 512: 1.085× and 1.088×** (1.089× and 1.092× against the copy), disjoint
  intervals (32.5-34.2 against 29.8-31.6). The measured default performs like the forced width.
- **Decode at context 2048: 1.027× and 1.019×** (1.037× and 1.029× against the copy), above the
  threshold against both copies; with 8 forced **1.050×**. The half that the default is missing
  sits in the 2 of 16 runs that pick 16 and in the 9 measurement passes (3 on 16 threads and 3 on
  4, both slower than 8 at this context) out of 47.
- **Prefill: not distinguishable** at either context (from −0.7% to +0.4%).
- On a 48-token reply the measurement weighs in; on a 200-token one it is 9 passes out of 200, and
  then 9 every 1024.

## Decode at long context and RAM bandwidth (2026-09-19)

Questions 4 and 18, point 6 of next steps. Native Windows, machine idle, OLMoE-1B-7B Q8_0,
pinned to cores, all in **one window** (`sh tools/decode_context.sh change
build/trochilus-before.exe`, 105 minutes, each run in `build/decode_context/`): `tools/ab_modes.sh`,
8 rounds after one warmup, `generate -p <context> -n 48 -t 16`. Four contexts in **same session**,
16 modes in order putting each context after each other once (de Bruijn sequence): one run on
machine warmed by 4000-token prompt isn't always same context, and one mode and its A/A copy never
follow same context. «Before» = binary at commit 6734910; «After» = KV with one head's positions
in a row (below). Tables from `tools/decode_context_report.py speed | model | zones`.

**1. RAM's bandwidth** (`make bench-mem`, `tests/bench_mem.c`: 2 GiB read by the pool's threads,
pinned as in the engine; median of 7; two runs on the same night, two hours apart, agreeing within
5-10%). GB/s:

| read | 1 thread | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| sequential | 24.8 | 47.8 | 57.6 | 52.7 | 53.0 | 52.5 | 53.0 |
| sparse, 2 MiB blocks (one expert's matrix) | 24.6 | 49.1 | 57.0 | 56.2 | 54.7 | 53.2 | 52.0 |
| sparse, 256 KiB blocks | 24.0 | 37.3 | 54.4 | 55.3 | 53.7 | 52.0 | 49.9 |
| sparse, 4 KiB pages | 7.2 | 14.0 | 29.1 | 33.5 | 39.1 | 36.6 | 34.2 |
| the engine's matmul, Q8_0, 8 random expert matrices per call | 19.1 | 32.9 | 58.1 | 53.0 | 46.3 | 50.4 | 50.0 |

- **The ceiling is ~54 GB/s** (52-58) and is reached with 4 threads; one thread pulls 22-25. This
  is not the two-channel DDR5-5200's theoretical 83 GB/s, and it is not the "41 GB/s" written so
  far: that was the engine's speed on 09/17 (no pinning, no threads per phase), not a RAM limit.
- **Reading sparse costs nothing**, as long as the pieces are large: 2 MiB or 256 KiB blocks
  perform like the sequential read. Randomly chosen experts pay nothing for being sparse. It costs
  to read at sparse pages (7 GB/s at one thread, 34-41 from 6 up): that is the earlier KV's case,
  below.
- The engine's matmul at one thread is compute-bound (19 GB/s); from 4 threads up it pulls the
  full bandwidth (46-58, the noisiest row: spread up to 30%).

**2. One token of attention on the two ways of keeping the KV** (`bench_mem kv <positions>`: a
cache shaped like OLMoE's, 16 layers × 16 heads × 128, f32, K and V, 1 GiB; between one layer and
the next 64 MiB of other memory pass through the caches, like the weights in the engine; real
kernel `tr_attention_head`). "Rows" is the earlier way, `[position][head]`: a head reads 512 bytes
every 8 KiB, a new page at every position. "Heads" is `[head][position]`: a head reads its
positions in a row. 8 threads, ms per token (KV's GB/s):

| positions | rows | heads | | heads, 16 threads | reading the same bytes alone: rows / heads |
|---|---|---|---|---|---|
| 512 | 4.17 (32.1) | 2.90 (46.3) | **1.44×** | 2.69 (49.9) | 21.5 / 52.9 GB/s |
| 2048 | 15.93 (33.7) | 11.52 (46.6) | **1.38×** | 10.61 (50.6) | 20.0 / 54.1 GB/s |
| 4000 | 29.54 (35.5) | 22.01 (47.6) | **1.34×** | 20.97 (50.0) | 19.6 / 53.7 GB/s |

The same bytes, read in jumps, pass at 20-22 GB/s; in a row at 53-54, that is, at the ceiling. With
the real kernel the difference is 32-35 against 46-48 GB/s (in the night's first run at 4000 rows
gave 28.0).

**3. The decode at the four contexts, before and after** (tok/s, median (min-max) of 8, then
after/before against both copies).

With `--decode-threads 8` (`speed-d8`; session threshold, worst A/A: decode 3.8%, prefill 2.8%):

| context | before | before, A/A copy | after | after, A/A copy | after / before |
|---|---|---|---|---|---|
| 32 | 38.00 (37.31-38.69) | 37.95 (35.65-39.01) | 38.30 (35.64-39.58) | 39.55 (37.20-40.09) | 1.008-1.042×: not distinguishable (A/A 3.3%) |
| 512 | 32.30 (31.15-34.16) | 33.52 (32.58-34.14) | 35.48 (34.25-36.49) | 35.47 (33.15-36.33) | **1.058-1.098×** |
| 2048 | 24.18 (23.68-24.61) | 24.30 (22.92-25.03) | 27.32 (25.34-27.88) | 27.62 (26.73-27.95) | **1.124-1.142×** |
| 4000 | 17.91 (15.66-18.50) | 18.16 (16.87-18.42) | 20.84 (19.83-21.33) | 21.04 (20.46-21.32) | **1.147-1.175×** |

With the default, the width measured by the session (`speed-auto`; decode threshold 3.6%, prefill
4.8%; a noisier session than the other, spread 9-29% against 4-16%: another Claude window was
working, with no container; the medians agree with the table above):

| context | before | before, A/A copy | after | after, A/A copy | after / before | widths chosen (after, 16 runs) |
|---|---|---|---|---|---|---|
| 32 | 38.05 | 37.55 | 38.83 | 38.63 | 1.015-1.034×: not distinguishable | 4 in 13, 8 in 3 |
| 512 | 32.16 | 32.77 | 33.13 | 32.47 | 0.991-1.030×: not distinguishable | 4 in 8, 8 in 4, 16 in 4 |
| 2048 | 22.27 | 22.75 | 25.93 | 26.20 | **1.140-1.176×** | 8 in 10, 16 in 6 |
| 4000 | 17.09 | 16.86 | 19.44 | 20.13 | **1.137-1.195×** | 16 in 13, 8 in 3 |

The prefill, in the same runs (tok/s, `speed-d8`): at 512 **not distinguishable** (234.9 and 234.9
before, 232.9 and 235.5 after); at 2048 **1.096-1.109×** (185.7 → 204.8); at 4000 **1.392-1.430×**
(119.4 → 166.1). The prompt's attention also reads one head at a time: its zone goes from 17.8 to
8.5 s over 4000 tokens (from 52% to 34% of the prefill) and from 3.15 to 1.86 s at 2048. The KV
write, now head by head, costs 25.7 ms on a 512-token prompt instead of 15.7 (question 30).

**4. STATUS's hypothesis: the shape holds, the numbers don't.** The hypothesis was
`tok/s = bandwidth / (weights + KV per token × context)` with 41 GB/s, 1.2 GB of weights and 262 KB
of KV per context token. The decode really is bytes divided by bandwidth, but there were **two**
bandwidths, and neither was 41:

| context (mid-reply) | measured before | formula at 41 GB/s | bytes per second, before | measured after | formula at 48.2 GB/s | bytes per second, after |
|---|---|---|---|---|---|---|
| 56 | 37.97 | 33.11 (−13%) | 47.0 GB/s | 38.91 | 38.91 | 48.2 GB/s |
| 536 | 33.00 | 30.06 (−9%) | 45.0 | 35.48 | 35.32 (−0.5%) | 48.4 |
| 2072 | 24.19 | 23.21 (−4%) | 42.7 | 27.50 | 27.27 (−0.8%) | 48.6 |
| 4024 | 18.04 | 17.99 (−0.3%) | 41.1 | 20.91 | 21.15 (+1.2%) | 47.7 |

(`--decode-threads 8`, one mode and its copy together: 16 runs per point.)

- **Before**: the line through the four points says 26.2 ms at zero context and 7.29 ms every 1000
  context tokens, that is, the weights read at **47 GB/s** and the KV at **36 GB/s** (31.6 with the
  default). The KV was read slower than the weights, the opposite of STATUS's first hypothesis
  ("the KV reads at higher bandwidth"). The 41 GB/s formula came out right at 4000 **by chance**:
  too low on the weights, too high on the KV, the two errors cancel only there.
- **After**: 25.2 ms and 5.57 ms every 1000 tokens, weights at 48.6 and KV at **47.1 GB/s**: a
  single bandwidth. `tok/s = 48.2 / (1.2236 + 0.000262 × context)` (GB) is off by at most 1.2% at
  every context.
- On a model with GQA the KV per token is smaller and the falloff with context shortens
  proportionally; the formula stays the same.

**5. Where the token goes, by zone** (`tools/profile_suite.py` on
`bench/scenarios-decode-context.json`, median of 5, `--decode-threads 8`; "before" is the twin of
the earlier binary compiled with per-zone bytes, same layout; the two profiles are not alternated,
so between before and after only the large differences count). ms per token, and GB/s read inside
the zone:

| zone | MiB per token | before, at 32 / 512 / 2048 / 4000 | after, at 32 / 512 / 2048 / 4000 |
|---|---|---|---|
| `attention` | 14 / 134 / 518 / 1006 | 0.64 / 4.36 / 15.79 / **30.53** ms (22.9 / 32.3 / 34.4 / 34.6 GB/s) | 0.48 / 2.90 / 11.52 / **22.64** ms (30.5 / 48.4 / 47.1 / 46.6 GB/s) |
| `expert_gate_up` | 544 | 10.6-10.9 ms (52.2-53.6) | 10.5-10.8 ms (52.6-54.5) |
| `expert_down` | 272 | 5.4-5.6 ms (51.2-52.8) | 5.3-5.6 ms (51.3-53.3) |
| `qkv_proj` | 204 | 4.1-4.2 ms (50.6-52.3) | 4.0-4.2 ms (50.4-53.4) |
| `lm_head` | 104 | 2.0-2.1 ms (52.7-53.7) | 2.0-2.1 ms (53.2-54.8) |
| `attn_out_proj` | 68 | 2.0-2.2 ms (32.5-36.2) | 1.4-1.6 ms (43.2-52.0) |
| `router` | 8 | 0.24-0.26 ms | 0.23-0.24 ms |
| zones with no bytes (norms, RoPE, KV write, per-expert copy, activation, sum, choice) | — | 1.06-1.10 ms | 1.09-1.15 ms |
| **token** | 1214 / 1334 / 1718 / 2206 | 26.8 / 29.6 / 41.4 / **56.3** ms (47.5 / 47.3 / 43.5 / 41.1 GB/s) | 25.0 / 27.5 / 36.4 / **48.1** ms (50.9 / 50.9 / 49.5 / 48.1 GB/s) |

- **The multiplications on the weights were already at the ceiling** (50-55 GB/s of 54): there is
  nothing to gain there without reading fewer bytes. Below the ceiling there was **only
  attention** (32-35 GB/s), and with it `attn_out_proj`, the zone right after it (32-36 GB/s: the
  jumping reads left it TLB and cache misses to redo).
- **After, attention reads at 47-48 GB/s**: at 4000 tokens −7.9 ms of 56.3.
- **How far from RAM's ceiling**: the decode moves **48-51 GB/s of ~54**, 6-11%. What is left on
  the exact side is small and scattered (question 35): 1.1 ms of zones with no bytes, attention at
  47 instead of 52-56 (the compute is worth 12% of the zone), `attn_out_proj`. From here on, going
  faster means reading **fewer bytes**.
- **8 or 16 threads for attention** (after, context 4000): on 16 the zone takes 21.6 ms instead of
  22.6, but experts and projections lose 1.3 ms and the token performs the same (48.4 against
  48.1). A width per zone would take 1.0 ms of 48.1 (2.1%; at 2048, 0.7%): below tonight's
  threshold, not written (question 34).

**6. The exact lever: one head's positions in a row** (`src/kv/kv.h`). The KV was
`[layer][position][head]`: for every token generated, every head read 512 bytes every 8 KiB, a new
page at every position, and the prefetcher had nothing to follow. It is now
`[layer][head][position]`: two streams in a row per head, the keys and then the values. It changes
where a number sits, not the number: the same calls to `dot_f32` and `axpy_f32` on the same floats
in the same order. Proof: `tests/test_kv.c` (the layout and the writes; 5 of 5 mutations seen by
the tests, three from `test_kv`, all from the oracle), `make check`, and on the real model
`tools/decode_context.sh`'s `exact` stage: logits identical to the byte against the earlier binary
over 600 positions at one token per pass (120 MB) and at passes of 64 on 8 threads, tokens
identical after a 4000-token prompt.

**7. The non-exact levers: the numbers to decide with.** Past the KV per head, attention reads at
RAM's bandwidth: to go faster at long context only fewer bytes remain. Estimates from the bytes,
with today's `attention` zone (2.90 / 11.52 / 22.64 ms at 512 / 2048 / 4000) and the 12% of the
zone that is compute and does not halve (for 8 bits, between 12% and double: there is decoding):

| lever | KV bytes | decode at 512 | at 2048 | at 4000 | KV memory at 4096 |
|---|---|---|---|---|---|
| today, f32 (exact) | 262 KB per context token | 36.4 tok/s | 27.5 | 20.8 | 1.07 GB |
| 16-bit KV | half | ~38 (+5%) | ~32 (+16-19%) | ~26-27 (+26-31%) | 0.54 GB |
| 8-bit KV (Q8_0-style blocks) | 27% | ~39 (+6-7%) | ~33-34 (+20-25%) | ~28-30 (+33-43%) | 0.28 GB |

How quality would be measured is written in question 36 (KL and first token against the exact mode
on the real model, greedy tokens equal, no value beyond 16 bits' maximum). Reference: llama.cpp,
which keeps the KV at 16 bits **and** the activations at 8, sits at KL 9e-3 from us. Not
implemented: Marcello decides. The comparison with llama.cpp's decode at long context (question
19) needs redoing now: its advantage there was also this.

## Prefill on long prompts (2026-09-19)

Point 4 of the next steps, questions 7 and 30. Native Windows, still machine, OLMoE-1B-7B Q8_0, 16
threads, pinned to core. Three windows of `tools/prefill_context.sh`: `measure` at 05:27 (the
06f8e30 binary alone: microbenchmark, speed with A/A, profile; 21 minutes), `bench` at 07:55 (only
the microbenchmark, with the new kernel) and `change build/trochilus-before.exe` at 10:13 (before
against after; 40 minutes). Each run sits in `build/prefill_context/`; the tables come from
`tools/prefill_context_report.py attn | zones` and from `tools/decode_context_report.py speed`. The
512, 2048 and 4000-token prompts sit in the **same session** as `tools/ab_modes.sh` (8 rounds after
one warm-up, `generate -p <prompt> -n 48 -t 16`), in an order that puts every length after every
other; a mode and its A/A copy never follow the same length.

**1. A prompt's attention, taken apart** (`make bench-attn`, `tests/bench_attn.c`: one layer, 16
heads of 128, KV `[head][position]`, 512-token passes as in the engine, 64 MiB of other memory read
between one pass and the next; median of 5). "One query at a time" is the engine's kernel
(`tr_attention_head`): for every token the products with all its keys, the row's softmax, the
weighted sum of the values. "Grouped" is the same computation with 16 queries against a block of 64
keys (and then of values) while the block is in cache; "x4" adds one query against 4 keys in
registers. Every grouped variant prints a hash of all the prompt's outputs: **the same bits** as
the engine's kernel, at every length. 16 threads and 16 heads, ms per layer (ns of thread per
query-position pair). Three runs on the same day (05:27, 07:55, 10:43), the last two also with the
kernel written into the engine (`tr_attention_group`): at 512 and 2048 they agree within 4-13%; at
4000 the first says one thing and the other two another (431.9 and 428.2 ms), and the table has the
first and the third:

| | prompt 512 | 2048 | 4000, 05:27 run | 4000, 10:43 run |
|---|---|---|---|---|
| one query at a time, all | 5.95 (45.3) | 104.1 (49.6) | 518.2 (64.8) | 428.2 (53.5) |
| — without the softmax | 2.56 | 44.2 | 342.0 | 184.5 |
| — products only (reads K) | 1.27 (9.7) | 20.2 (9.6) | 79.1 (9.9) | 85.4 (10.7) |
| — softmax only | 4.27 (32.6) | 69.6 (33.2) | 266.7 (33.3) | 278.8 (34.8) |
| — weighted sum only (reads V) | 1.36 (10.4) | 19.3 (9.2) | 140.0 (17.5) | 82.6 (10.3) |
| grouped, all | 6.32 | 109.9 | 382.7 | 396.9 |
| grouped, without the softmax | 2.60 | 35.6 | 141.9 | 156.7 |
| grouped + x4 (bench prototype), all | 5.49 | 95.1 | 355.2 | 370.6 |
| grouped + x4, without the softmax | 1.53 | 22.8 | 101.2 | 92.1 |
| **the engine's kernel, `tr_attention_group`**, against "one query at a time" of the same run (07:55 / 10:43) | 1.06× / 0.95×: not distinguishable | **1.12× / 1.16×** | — | **1.15× / 1.15×** (376.4 and 373.3 ms; 1.38× on the 05:27 run) |

Over 16 layers the earlier kernel takes 95, 1665 and 6850-8291 ms: the profile's `attention` zone
measures 116, 1883 and 7420. A single thread with a single head (all caches to itself) takes
**42.7, 43.0 and 42.6 ns per pair**: at one thread the length does not matter, and the time is 73%
softmax, 14% products, 12% weighted sum; there the engine's kernel is worth 1.05-1.09× (it is x4
alone). A group of 4, 16 or 64 queries and a block of 16, 64 or 256 positions at 4000: 352-387 ms,
all within the spread (2-15%): 16 × 64 is kept.

**2. The hypothesis: half true.** The hypothesis was: "at 4000 tokens the prompt's attention costs
34% and reads 2 TB in 8.5 s, almost all from cache: every token rereads the same keys and the same
values". The 34% is there (31-35% in two profiles) and so are the 2 TB (283 GB/s in the zone:
cache, not RAM). But the repeated read **is not the bulk**:

| of the `attention` zone, 16 threads | prompt 512 | 2048 | 4000, 05:27 run | 4000, 07:55 and 10:43 runs |
|---|---|---|---|---|
| softmax (`expf`, max, sum, division) | 72% | 61-70% | 51% | 63-65% |
| products and weighted sum, with keys and values in cache | 28% | 30-34% | 27% | 36-37% |
| keys and values reread at every token | nothing | 0-8% | **21-39%** | **0-6%** |

(The softmax is its own row; the repeated read is what remains after removing the softmax and
"grouped without softmax", or "one query at a time" minus "grouped" without softmax: the phases
measured alone do not sum exactly, the spread at 16 threads reaches 20%.) Up to 2048 tokens a
head's keys and values (2 MB) fit in the caches and rereading them costs nothing. At 4000 they are
4 MB per head: 64 MB for the 16 threads against L3's 64 MB, **right on the edge**, and how much it
costs depends on what else sits in L3 at that moment: a quarter of the zone in the first run, almost
nothing in the other two (the engine, which lets the weights pass through between one attention and
the next, sits in between: 464 ms per layer). Grouped, the time **no longer depends on this**: 355,
349 and 371 ms in the three runs. Past 4000 tokens the edge is crossed for everyone (question 38).

The bulk is the **softmax**, that is, the C library's `expf`: **30 ns a call** with MinGW
(`tests/bench_expf.c`; x87 instructions), against glibc's **2.3 ns** in the container, on the same
machine. The same `expf` is almost all of the `expert_act` zone (5-7% of the prefill, 39 ns per
element). It is the only function in the hot zone that is not ours: see point 6.

**3. The prefill before, and its serial part** (binary of commit 06f8e30; tok/s, median (min-max)
of 8; zones from the profile, median of 5, `bench/scenarios-prefill-context.json`):

| | prompt 512 | 2048 | 4000 |
|---|---|---|---|
| prefill, tok/s | 231.7 (230.2-235.7) | 203.9 (200.7-206.9) | 169.5 (147.2-173.0) |
| its A/A copy | 233.3 (0.7%) | 202.9 (0.5%) | 168.1 (0.8%) |
| prompt time, ms | 2175 | 10157 | 23639 |
| `attention` | 116 (5.3%) | 1883 (18.5%) | 7420 (31.4%) |
| multiplications (experts, q/k/v, output, `lm_head`) | 1682 (77%) | 6767 (67%) | 13291 (56%) |
| `expert_act` (`expf`) | 161 (7.3%) | 656 (6.5%) | 1295 (5.4%) |
| `router` | 80 (3.6%) | 324 (3.2%) | 627 (2.6%) |
| **work for a single token, on one thread**: norms, RoPE, KV write, rows for the experts, embedding | **121 (5.6%)** | **455 (4.5%)** | **880 (3.7%)** |
| — of which KV write, head by head | 27.4 | 85.7 | 160.5 |
| — of which rows copied for the experts | 38.4 | 142.4 | 281.5 |
| — of which q and k norms / layer norms / RoPE | 21.4 / 20.5 / 12.7 | 87.9 / 83.2 / 52.1 | 172.9 / 156.4 / 102.3 |

The serial part is not only that: inside `router` the token-by-token choice of experts runs on one
thread, and the **one-thread** scenario says so (prompt 512: `router` 415 ms at one thread, 80 at
16: about 57 ms do not split). And the other half of the zone was a missing kernel: the router's
matrix is **F32**, and F32 rows ran on the scalar loop in every tier (0.11 GB/s in the zone;
LESSONS #78). In all, the single-thread work cost **8%** of the prefill at 512 and 5-6% at 4000.

**4. The three exact levers.** Each changes where and when a computation happens, never the
computation: the same kernel calls on the same floats in the same order for every output.

- **Grouped attention** (`tr_attention_group`, `src/kernels/kernels.c`): 16 consecutive tokens of a
  head form a group; a block of 64 positions meets all the group's queries while it sits in cache,
  4 positions per query load (`dot_f32_x4`) and 4 values per output load (`axpy_f32_x4`); then each
  row's softmax; then the values, the same way. Keys and values are read once per group: the
  profiler counts **31 MiB of KV per token at 4000 instead of 500** (at 2048: 16 instead of 256).
  The decode is a group of one query: a single path.
- **The single-token work on the pool**: embedding, norms, q and k norms, RoPE, KV write, residual
  sum, router choice (one scratch per worker), rows for the experts; at least 8 tokens per piece, so
  a short pass stays on the calling thread, as before.
- **F32 rows with the tier's kernel**: the router's matrix no longer goes through the scalar loop.

Proof: `tests/test_kernels.c` (x4 against scalar and against 4 calls, 525 groups against
`tr_attention_head` query by query), `test_prefill` with a 141-token prompt, `test_hot` under
ThreadSanitizer, 20 of 20 mutations seen (`tools/mutate_prefill.sh`), `make check`; and on the real
model the `exact` stage: logits **identical to the byte** against the earlier binary over 600
positions at one token per pass (120 MB, 16 threads), at passes of 64 on 8 threads, on a 4000-token
prompt at passes of 512 and of 100 on 16 threads (the last row equal in both forms), and the same
tokens after a 4000-token prompt.

**5. Before and after** (`sh tools/prefill_context.sh change build/trochilus-before.exe`; tok/s,
median (min-max) of 8; session threshold, worst A/A: **prefill 2.1%, decode 2.4%**; "after" is the
second copy of the new binary, LESSONS #81, hash in `build/prefill_context/binaries.sha256`):

| prefill | before | before, copy | after | after, copy | after / before |
|---|---|---|---|---|---|
| prompt 512 | 228.7 (222.5-236.1) | 230.2 | 240.9 (222.2-247.2) | 246.1 | **1.047-1.076×** |
| 2048 | 197.8 (184.5-201.2) | 197.6 | 211.3 (202.1-219.4) | 213.3 | **1.069-1.079×** |
| 4000 | 160.1 (156.9-164.1) | 161.2 | 181.7 (178.8-182.5) | 179.6 | **1.114-1.135×** |

The decode after those prompts **is not distinguishable**: 33.0 against 33.1 tok/s at 512
(1.000-1.013×), 25.7 against 25.6 at 2048 (0.973-1.005×), 19.9 against 19.9 at 4000
(0.972-1.001×). (In this session the earlier binary gives 160 tok/s at 4000 where in the morning it
gave 169: the machine had just come out of two hours of standby, LESSONS #82. The comparison within
the session is what counts.)

Per zone (morning profile against the new binary's profile, ms of the prompt):

| | 512: before | after | 2048: before | after | 4000: before | after |
|---|---|---|---|---|---|---|
| `attention` | 115.9 | 106.2 | 1883 | 1688 | 7420 | 6568 (**1.13×**) |
| `router` | 79.7 | **6.2** | 323.5 | **21.8** | 626.9 | **45.2** |
| single-token work | 121.2 | 48.2 | 454.6 | 174.2 | 880.1 | 353.1 |
| — q and k norms, layer norms, RoPE | 54.6 | 8.9 | 223.2 | 34.1 | 431.6 | 70.2 |
| — rows for the experts | 38.4 | 19.5 | 142.4 | 67.4 | 281.5 | 133.2 |
| — KV write | 27.4 | 19.3 | 85.7 | 71.8 | 160.5 | 147.4 |
| whole prompt | 2175 | 2071 | 10157 | 9503 | 23639 | 22044 |

The norms and RoPE, which are compute, divide by 6; the rows for the experts and the KV write,
which are memory copies, only by 2 and by 1.1-1.4: there the limit is writing to RAM, not the
thread. `router` runs 13 times faster: 57 ms were the choice on one thread, the rest the scalar
kernel. The multiplications were not touched (in the after profile they cost 2-3% more: a
different session, not an effect).

After, at 4000 tokens attention is **30%** of the prefill and inside it is almost only the softmax
(4.3 s of 6.6): little is left on the exact side in attention. The next piece is `expf` (point 6).

**6. The non-exact levers: the numbers to decide with.** Not implemented.

- **Our own `expf`** (question 37). Today `expf` is the C library's: 30.2 ns a call on Windows
  (MinGW), 2.3 ns in the container (glibc). `tests/bench_expf.c` compares it, over **all 4 278 190 082
  non-NaN floats**, with the double-precision exponential rounded to float, which is the correctly
  rounded value (barring a handful of cases out of 4 billion):

  | library | `expf`'s cost | results differing from correct rounding | in the softmax's range (−104…0) |
  |---|---|---|---|
  | MinGW-w64 (native Windows) | 30.2 ns | **0** | **0** |
  | glibc 2.39 (container) | 2.3 ns | 170 648 (0.004%), always by one unit in the last digit | 97 052 (0.009%) |

  Two consequences. (a) **Today Windows and Linux do not give the same bits**: on a 4000-token
  prompt the softmax calls `expf` two billion times, and glibc rounds about one in ten thousand of
  them wrong. (b) An `expf` we write ourselves that rounds correctly (scalar C as the definition,
  AVX variants bit-identical, like every other kernel) would give **the same bits as today on
  Windows**, by exhaustive proof (the same `bench_expf` with ours in place of the reference: 0
  differences over 2^32), and on Linux it would change 0.004% of calls by one unit in the last
  digit, making the two platforms identical. "It changes the bits" therefore only holds for Linux,
  and there for the better. Estimated gain with the softmax at 2-3 ns per element instead of 32 and
  `expert_act` at a tenth:

  | | prompt 512 | 2048 | 4000 |
  |---|---|---|---|
  | attention's softmax, ms of the prefill | 83 | 1262 | 3821 |
  | `expert_act`, ms | 161 | 656 | 1295 |
  | estimated prefill, on the earlier binary | **1.11×** | **1.21×** | **1.25×** |
  | estimated prefill, on the later binary (2071 / 9503 / 22044 ms) | **1.12×** | **1.23×** | **1.27×** |
  | estimated decode (softmax + `expert_act` per token: 1.4 / 2.8 / 4.5 ms of 27.9 / 36.4 / 47.5) | +5% | +8% | +10% |

  How quality would be measured: `bench_expf` on both platforms (0 differences on Windows is the
  condition for calling it exact); `tools/prefill_context.sh`'s `exact` stage before against after
  on Windows (expected: identical to the byte); on Linux, KL and first token against the earlier
  binary over 1000 positions and unchanged oracles. The risk is in writing it (a correctly rounded
  `expf` needs a slow path for cases near halfway between two floats), not in measuring it.
- **16-bit KV**: no longer needed for the prefill. With grouped attention, keys and values are read
  once every 16 tokens, and the repeated read was the only byte-bound piece of the zone. It remains
  a lever for the **decode** at long context (question 36: +16-31% at 2048-4000).
- **int8/VNNI in the multiplications** (questions 21 and 33): the multiplications are 77% of the
  prefill at 512 and 56% at 4000; with the kernel at 1.8-2.3× (§int8 activations, second
  measurement) the prefill would run **1.5-1.8×** at 512 and **1.3-1.5×** at 4000. It is not exact:
  the decision remains STATUS's point 3.

**7. Our own `expf`: prepared, not written** (question 37; Marcello decides). **Later written, the
same day: the scalar, with the measured numbers, sits in §`tr_expf`.** The three numbers asked for
then:

- **The test over the 2^32 floats is ready**, and it has already run on a prototype. `make
  bench-expf` compares a candidate with the library's `expf` and with the reference over every
  float; the candidate is given at compile time (`EXTRA_CFLAGS=-DTR_EXPF_CANDIDATE=tr_expf` once
  the engine has one). Without it, a sketch is tried that sits **in the bench, not in the engine**:
  25 lines of scalar C, a table of 64 values of 2^(j/64), a degree-5 polynomial in double, and a
  rounding test (if an error of 2^-50 cannot move the float, the result is certain; otherwise a
  slow path).

  | | cost per call | different from the library | different from the reference | slow path |
  |---|---|---|---|---|
  | Windows (MinGW): library / sketch | 30.2 ns / **3.7 ns** | **0 of 4 278 190 082** | 0 | 8 arguments |
  | Linux (glibc): library / sketch | 2.5 ns / 3.8 ns | 170 648 (where glibc is wrong) | 0 | 8 arguments |

  The reference itself is proven: `bench_expf --hard` writes the 369 arguments whose double
  exponential falls within 2^-45 of a rounding boundary (the same on both platforms), and
  `tools/expf_hard_cases.py` recomputes them at 200 bits: **0 errors**. So MinGW's `expf` is
  correctly rounded over every float, and "zero differences from the library" is a proof, not a
  sample. The scalar sketch is already 8 times faster than the library on Windows; the AVX-512
  version (8 doubles per register) is estimated at 0.8-1 ns per element.
- **The KL on Linux against the earlier binary** (`tools/expf_quality.sh` in the container: the
  engine compiled with every `expf` replaced by the correctly rounded value, by emulation, against
  the normal binary; real model, `code-edit.txt` prompt plus 1000 generated tokens, 1411 positions
  at one token per pass): no logit row identical to the byte, maximum difference between two logits
  **2.3e-5** (over a range of 65.8), **average KL 3.9e-13, maximum 1.5e-11**, **0 positions**
  choosing a different token, **1000 of 1000 tokens** equal in greedy generation. (llama.cpp sits at
  9e-3 from us: ten orders of magnitude above.) On Windows there is nothing to measure: the same
  bits.
- **The cost of writing it**: about **450 lines of C and 40 of Python**, one work session.

  | piece | lines | difficulty |
  |---|---|---|
  | scalar `tr_expf`, the definition: special cases, exact reduction (ln2/64 in two pieces), polynomial, scaling, rounding test | 70 | medium: it is the sketch, with the table from constants rather than from `exp2` |
  | table of 64 correctly rounded doubles, generated with mpmath | 64 + 40 of Python | low |
  | slow path: the arguments that fail the test are **8 of 4 billion**, so a table of exceptions verified at 200 bits (or 40 lines of double-double) | 20 | low, but must be reproven exhaustively at every change |
  | AVX2 (4 doubles) and AVX-512 (8 doubles) variants bit-identical to the scalar: table gather, integer scaling, mask for the slow path, tails | 140 | medium-high: this is the delicate piece |
  | `exp` entry in the kernel table, softmax and SwiGLU that use it (one stack scratch for the activation) | 60 | medium |
  | tests: tier against scalar at every length and on special values, a broken variant the test must catch, `bench-expf` with the candidate inside `make check` (15 s: the exhaustive proof at every gate) | 110 | low |

  The risk is not quality (the proof is exhaustive and takes 15 seconds) but the SIMD piece's time.
  What is gained is in point 6: prefill 1.12× / 1.23× / 1.27×, decode +5-10%, and Windows and Linux
  with the same bits.

**8. How to redo it.** Every length is a scenario of `bench/scenarios-prefill-context.json` (512,
2048 and 4000 at 16 threads, 512 at one thread) and every measurement is a command:

| measurement | command | takes |
|---|---|---|
| the attention taken apart (points 1-2) | `make bench-attn`, or on a still machine `sh tools/prefill_context.sh bench` | 3 minutes |
| `expf`: library, candidate, all 2^32 floats (points 6-7) | `make bench-expf`, native and in the container; `build/tests/bench_expf.exe --hard <file>` and `tools/expf_hard_cases.py <file>` for the boundary cases | 15 s |
| what a correct `expf` would change on Linux (point 7) | `tools/expf_quality.sh` in the container, with the models volume | 6 minutes |
| speed with A/A and a binary's zones (point 3) | `sh tools/prefill_context.sh measure` | 21 minutes |
| logits to the byte, before against after with A/A, bench, zones (point 5) | `sh tools/prefill_context.sh change <before binary>` | 40 minutes |
| zones only | `make profile SCENARIOS=bench/scenarios-prefill-context.json` | 6 minutes |
| the tables | `tools/prefill_context_report.py attn` and `zones`, `tools/decode_context_report.py speed` | — |
| do the tests see the errors? | `tools/mutate_prefill.sh` in the container | 45 minutes |


## Clean machine remeasure (2026-09-19)

Four `yes` processes, left over from a test on 09/17, sat at 100% of one core each under **every
native measurement of 09/18 and -19** (LESSONS #84). Remeasured on 09/19 from 16:09 to 17:52,
without them, the very things the engine's default had been chosen from: threads per phase and
decode at long context. Binary with `tr_expf` (hash in `build/prefill_context/binaries.sha256`), 8
rounds, A/A, each run in `build/threads_phase/` and `build/decode_context/`; the sessions with the
`yes` processes stay in `build/threads_phase-with-yes/` and `build/decode_context-with-yes/`. Every
session declares in its log the **background load** (LESSONS #85): the kernel's `System` process
alone holds 0.7-0.9 core, always; it is a fact of this machine and sits under every number in this
document.

| session | time | background load before → after (logical processors busy, of which `System`) | worst A/A |
|---|---|---|---|
| forced widths at 512 (`threads_phase.sh widths`) | 16:09 | 3.82 (0.84; `svchost` 1.23, Windows indexer) → 2.14 (0.72) | decode 2.6% |
| decode at 16 against 8 threads, four contexts (`decode_context.sh widths`) | 16:19 | 2.01 (0.78) → 1.17 (0.71) | decode 4.6% |
| automatic against 8, RAM bandwidth, zones (`decode_context.sh measure`) | 16:51 | 1.25 (0.75) → 0.98 (0.73) | decode 2.5% |
| thread sweep (`threads_phase.sh sweep`) | 17:30 | 0.99 (0.71) → 1.02 (0.74) | decode 2.7%, prefill 4.1% |

**The day's noise is a fact, not a flaw to hide**: with Marcello at the machine, another window
working and the Windows indexer, the earlier sessions have 15-30% spread between minimum and
maximum and cannot distinguish differences below 5%; in the evening, with the machine left alone,
A/A returns to 1-3%. The tight sessions of 09/18 were tight because they ran at night, not thanks
to the `yes` processes. Measurements that decide are made at night, and now they can be left alone
(orphans rejected, a CPU guard before every run, load declared).

**1. RAM's bandwidth** (`bench_mem ram`, sequential read; question 4):

| threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| with `yes` (09/19 morning), GB/s | 22.4 | 42.7 | 54.0 | 54.0 | 52.3 | 51.0 | 51.5 |
| clean machine, GB/s | 26.2 | 52.2 | **57.3** | **57.6** | 55.7 | 54.4 | 49.5 |

The ceiling is **~57 GB/s, not 54**, reached with 4-6 threads and past that it **drops**: the
decode, which is a memory read, gains nothing from more readers. This is why everything below
follows.

**2. The decode at forced width, 16 against 8 threads** (tok/s, median of 8; every mode with its
own copy; ratio of 8 to 16 against both copies):

| context | 16 threads | 8 threads | 8 / 16 | A/A | with `yes` |
|---|---|---|---|---|---|
| 32 | 38.5 / 37.7 | 40.2 / 39.3 | 1.02-1.07× | 1.9% / 2.4% | — |
| 512 | 34.6 / 34.3 | 34.7 / 35.0 | **1.00-1.02×** | 0.9% / 1.0% | **1.10×** |
| 2048 | 27.1 / 25.9 | 28.1 / 26.9 | 0.99-1.09× | 4.6% / 4.3% | 1.03-1.05× |
| 4000 | 21.1 / 21.5 | 21.6 / 22.2 | 1.01-1.06× | 1.9% / 2.9% | — |

With the session's threshold (4.6%) 8 and 16 **are not distinguishable at any context**; 8 is ahead
by a nominal 0-7% and never behind. The 09/18 "+10% at 8 threads" was the effect of the four cores
taken: 16 threads pinned to cores paid for the slowest thread, 8 did not.

**3. The automatic against 8 threads, and what it picks** (the quietest session; threshold 2.5%).
"Picks": how many times out of 8 runs the session picked each width, and in parentheses how many
changes from one run to the next (`tools/decode_context_report.py speed`), for the mode and for its
copy:

| context | automatic | 8 forced | 8 / automatic | picks, clean machine | picks, with `yes` (09/19 morning) |
|---|---|---|---|---|---|
| 32 | 40.7 / 41.1 | 40.6 / 40.6 | 0.99-1.00× | 7×4 1×8 (1) · 7×4 1×8 (1) | 7×4 1×8 (1) · 6×4 2×8 (3) |
| 512 | **36.9** / 36.1 | 34.8 / 34.6 | **0.94-0.96×** | 7×4 1×8 (1) · 6×4 1×8 1×16 (3) | 6×4 2×8 (4) · 2×4 2×8 4×16 (4) |
| 2048 | 28.5 / 27.7 | 28.1 / 28.6 | 0.99-1.03× | 8×8 (0) · 7×8 1×16 (2) | 6×8 2×16 (3) · 4×8 4×16 (5) |
| 4000 | 22.7 / 22.1 | 22.4 / 22.6 | 0.99-1.02× | 7×8 1×16 (2) · 5×8 3×16 (3) | 1×8 7×16 (2) · 2×8 6×16 (2) |

At 512 the automatic, which almost always picks **4 threads**, beats 8 by 4-6%, above the threshold
against both copies: at short context the decode wants 4 threads, at long context 8, and 16 never
wins. The direction of 09/18's conclusion ("few threads") was right; the measurement was inflated.
The forced widths at 512 from the first (noisy) session say the same: 16 threads 33.1 (copy 32.3),
12 31.1, 8 31.1, 6 33.1, 4 33.1, automatic 33.6 (picks 4×4 1×8 3×16, 3 changes); with `yes` they
were 30.9, 31.6, 34.1, 34.6, 33.8 and 33.8.

**The estimator is drawing lots** (LESSONS #88, question 31). The picks change from one run to the
next: little on a quiet machine (one run in eight), a lot as soon as there is noise (4×4 1×8 3×16),
and it already happened with `yes`, where at 2048 it changed 3-5 times out of 8: the median speed
did not show it, because where widths are equivalent an unstable pick costs nothing. The margin
(the widest within **1%** of the fastest, over three one-token passes) is below those passes' noise
(2-5%), and it had been tightened from 3% to 1% **on the machine with the cores taken**, where 16
threads really did lose. The estimator needs rewriting, not the default: a margin from measured
noise, more rounds if it does not decide, hysteresis on remeasures, and on a tie the **narrower**
width, which never loses on a clean machine and is worth 10% under other load.

**4. The thread sweep** (`-t n` for both phases; the decode picks its own width inside the pool,
written in parentheses; tok/s):

| `-t` | 512: prefill | decode | 2048: prefill | decode |
|---|---|---|---|---|
| 16 | 316.8 (copy 303.7) | 37.4 (4) | 308.5 (copy 305.8) | 29.5 (8) |
| 12 | 287.0 | 37.3 (6) | 265.7 | 29.3 (6) |
| 8 | 213.1 | 36.7 (4) | 196.7 | 28.0 (8) |
| 4 | 112.5 | 32.3 | 104.5 | 24.6 |

The prefill wants the whole pool (from 8 to 16 threads 1.49× at 512 and 1.57× at 2048); the decode
performs the same with any pool as long as the picked width is 4-8. With `-t 4` the estimator also
tries 2 and 1 threads: in a 48-token run those six slow passes weigh 14-17%, so that number is the
probing's cost, not the regime (another argument for question 31: do not probe widths below 4).

**5. The decode's model** (`decode_context_report.py model`, automatic): **24.5 ms at zero context
plus 5.0 ms every 1000 context tokens** (the KV read at 52 GB/s); 41.0 / 36.6 / 28.4 / 22.5 tok/s at
context 32 / 512 / 2048 / 4000, that is, 3-8% more than the morning with `yes` (38.9 / 35.5 / 27.5 /
20.9 at 8 threads). The decode moves 50-51 GB/s of 57: 10-12% remains on the exact side.

**6. How to redo it**, with the machine left alone: `sh tools/threads_phase.sh widths`, `sh
tools/decode_context.sh widths` (16 against 8 at the four contexts, 40 minutes), `sh
tools/decode_context.sh measure`, `sh tools/threads_phase.sh sweep`; the tables with
`tools/decode_context_report.py speed <file>` (picks and changes included) and `model`.


## `tr_expf`: our exponential, scalar (2026-09-19)

Question 37, first half: **the scalar only**, which is the definition. SIMD is a separate decision
(point 7). Speeds measured on the clean machine from the section above, from 17:58 to 18:39:
background load 1.10 → 0.98 processors busy (0.68-0.71 from `System`), worst A/A 3.0% on the
prefill and 1.9% on the decode; every run in `build/prefill_context/`, hash in `binaries.sha256`
("before" is the binary of commit ef3cb99). The two earlier sessions of the day, one with `yes`
running (spread 24-57%) and one interrupted, stay in `build/prefill_context-expf-noisy/` and
`-interrupted/`.

**1. The function** (`src/kernels/expf.c`). exp(x) = 2^(k/64) · exp(r), k = round(x · 64/ln2), r = x
− k · ln2/64 with ln2/64 in two pieces (the first exact for every k), all in double: 64 values of
2^(j/64), a degree-5 Taylor polynomial, and a **rounding test**: if y·(1 − 2^-50) and y·(1 + 2^-50)
round to the same float, exp(x) rounds there. The arguments that fail it are **8 of 4 278 190 082**
and sit in an exception table with the exponential computed at 200 bits (and at 400: the same). An
argument that failed the test without being in the table would give a NaN: a rounding nobody has
proven does not leave the function as a number. Inside there is no call to the C library, and no
constant comes from a library: `tools/gen_expf_table.py` writes them with mpmath
(`src/kernels/expf_table.h`, checked against its generator by `tools/lint.py`, LESSONS #83). The 8
exceptions are found to agree by three paths: C on Windows (MinGW gcc 15.2), C on Linux (gcc 13.3),
and a bit-for-bit numpy mirror of the fast path over every float in range
(`gen_expf_table.py --scan`, 4 minutes). Softmax (attention and router) and SwiGLU use it;
`tools/lint.py` rejects `expf(` and `exp(` in the hot zone.

**2. The proof is exhaustive** (`make bench-expf`: all 2^32 floats against the reference, which is
proven at 200 bits on the 369 boundary cases; in `make check` with gcc and with clang):

| | cost per call, library / `tr_expf` | different from the library | different from the reference | unproven | time |
|---|---|---|---|---|---|
| Windows (MinGW-w64) | 29.9 ns / **3.5 ns** | **0 of 4 278 190 082** | **0** | 0 | 18 s |
| Linux (glibc), gcc | 2.4 ns / 3.8 ns | 170 648 (where glibc is wrong) | **0** | 0 | 7 s |
| Linux (glibc), clang | 4.0 ns / 4.1 ns | 170 648 | **0** | 0 | 6 s |

Red first: with an empty exception table, 8 NaNs and `check: FAILED`. `tools/mutate_expf.sh`:
**17 of 17 mutations seen** by at least one check (the quick test `tests/test_expf.c`, the
exhaustive proof, the comparison against the generator, the hot zone's lint). Three things the
mutations teach. (a) Three constants shifted by one unit in the last digit (2^(1/64), 1/6, ln2/64's
low piece) **change no result** over 4 billion floats: only the header-against-generator comparison
catches them. (b) With the test's margin at 2^-60 (which for double means no margin at all) the
results stay **all correct**: the fast path's double already rounds to the right float everywhere.
The test and the table do not correct numbers: they make the proof structural, because a change
that shifts a result leaves unproven arguments or dead table entries (8, in that mutation), which
the check counts, instead of a silently wrong rounding. (c) Softmax or SiLU put back on the
library's `expf`: on Linux `test_expf` catches it (softmax and SiLU are their definition with
`tr_expf`, bit for bit), the lint catches it everywhere; on Windows the bits are the same and the
lint remains.

**3. On Windows the same bits as before.** `tools/prefill_context.sh`'s `exact` stage, against the
previous commit's binary, run three times during the day: logits **identical to the byte** over 600
positions at one token per pass (120 MB, 16 threads), at passes of 64 on 8 threads, on a 4000-token
prompt at passes of 512 and of 100, and the same tokens after a 4000-token prompt.

**4. On Linux it changes as predicted, and only the exponential** (`tools/expf_quality.sh` in the
container, real model, `code-edit.txt` prompt plus 1000 generated tokens, 1411 positions). Against
the earlier binary (`build/linux-before`, on glibc's `expf`): no row identical to the byte, maximum
difference between two logits **2.3e-5** over a range of 65.8, **average KL 3.9e-13, maximum
1.5e-11, 0 positions** choosing a different token, **1000 of 1000 tokens** equal. Against the
emulation build (the earlier commit with every `expf` replaced by `(float)exp((double)x)`,
`tools/cr_expf_emul.h`): **1411 of 1411 rows identical to the byte** and the same tokens, and it is
a check (the script exits 1 if not): in the engine only the exponential changed, nothing else.

**5. Windows and Linux give the same bytes** (`sh tools/platform_bits.sh`): logits identical to the
byte between the native binary and the container's on the tiny F32, F16 and Q8_0 fixtures (120
positions) and on the **real 2-layer model over 1000 positions (201 MB)**. The suspicion about the
RoPE table (the C library's `pow`, `cos` and `sin`, in double, rounded to float: the only place left
where numbers come from a library) is measured entry by entry (`tests/dump_rope.c`,
`tools/rope_table_compare.py`; the tool's second copy, LESSONS #81, because Smart App Control
blocked the first): over 4096 positions × 64 pairs the **doubles** of `cos` and `sin` differ between
the two libraries in 0.8% of entries (2175 cosines, 2216 sines of 262 144), `pow` never, and the
engine table's **floats never**: rounding to float absorbs the double's last digit. Question 40
closed: up to OLMoE's training context the two platforms compute on the same numbers.

**6. Before and after** (tok/s, median (min-max) of 8; decode forced to 8 threads in both binaries,
because the width chosen by the session oscillates, question 31; threshold: prefill 3.0%, decode
1.9%):

| | before | before, copy | after | after, copy | after / before | estimate (§Prefill on long prompts, point 6) |
|---|---|---|---|---|---|---|
| prefill, prompt 512 | 296.3 (271.6-313.7) | 290.8 | 305.5 (298.7-349.7) | 314.6 | 1.03-1.08× | 1.12× |
| 2048 | 249.4 (239.4-261.5) | 252.7 | 303.2 (297.8-321.9) | 307.8 | **1.20-1.23×** | 1.23× |
| 4000 | 210.9 (204.2-218.5) | 214.0 | 276.1 (267.8-281.3) | 276.5 | **1.29-1.31×** | 1.27× |
| decode after 512 | 35.1 (34.8-36.1) | 35.1 | 36.0 (35.2-36.5) | 35.9 | **1.02-1.03×** | +5% |
| after 2048 | 28.0 (27.6-28.2) | 27.5 | 29.3 (28.6-30.0) | 29.4 | **1.05-1.07×** | +8% |
| after 4000 | 21.4 (21.1-21.8) | 21.2 | 23.2 (22.9-23.4) | 23.3 | **1.08-1.10×** | +10% |

At 512 the prefill is above the threshold against three of four copies (1.031× against the fourth):
"between 1.03 and 1.08×". Per zone (profile of both binaries in the same session, ms of the
prompt):

| | 512: before | after | 2048: before | after | 4000: before | after |
|---|---|---|---|---|---|---|
| `attention` | 92.3 | 40.5 | 1537 | 591 | 5960 | 2331 (**2.56×**) |
| `expert_act` | 138.2 | 24.6 | 594.9 | 108.0 | 1201 | 218.4 (**5.5×**) |
| `router` | 5.0 | 4.0 | 19.6 | 16.4 | 41.8 | 34.8 |
| whole prompt | 1802 | 1644 (1.10×) | 8236 | 6825 (1.21×) | 19087 | 14693 (1.30×) |

The softmax goes from 32-35 to **4.4-6.6 ns per element** (`make bench-attn`) and from 61-73% to
22-31% of the attention. At one thread the prefill at 512 runs 1.15× (25.2 → 28.9 tok/s).

**7. SIMD: the numbers to decide with** (not written). After the scalar, in the prefill at 512 /
2048 / 4000 there remains: the attention's softmax at about 9 / 130 / 610 ms (22-27% of a zone
worth 2.5 / 8.7 / 15.9% of the prompt) and `expert_act` at 24.6 / 108 / 218 ms (1.5%). With an
AVX-512 exponential estimated at 1.0-1.3 ns per element (8 doubles per register; on Zen 4 the 512
bits are two 256-bit passes and the gather costs) and the rest of the softmax vectorized, softmax
and activation would drop by 65-70%: estimated prefill **1.01× / 1.03× / 1.04×**, decode +1-2%.
Cost: about 200 lines (AVX2 and AVX-512 variants bit-identical to the scalar, table gather, mask for
lanes that fail the test, and tails; an `exp` entry in the kernel table with a scratch for the
activation; the exhaustive proof for every tier, 7 s each at the gate). The scalar has taken almost
everything: at 4000 tokens the gain left is below a good session's threshold. Marcello decides
(question 39).

**8. How to redo it.**

| measurement | command | takes |
|---|---|---|
| the proof over every float, and the cost per call | `make bench-expf`, native and in the container | 7-18 s |
| the constants from the generator; the exceptions found again from scratch | `tools/gen_expf_table.py [--check]`; `--scan` | 1 s; 4 minutes |
| do the checks see the errors? | `sh tools/mutate_expf.sh` in the container | 8 minutes |
| Linux: KL and tokens against the earlier binary, bytes against the emulation | `sh tools/expf_quality.sh` in the container, with the models volume | 8 minutes |
| Windows and Linux, the same bytes? | `sh tools/platform_bits.sh` | 3 minutes |
| logits to the byte, before and after with A/A, bench, zones | `GEN_EXTRA="--decode-threads 8" PROF_BEFORE=<before> sh tools/prefill_context.sh change <before>` | 45 minutes |


## M1, before writing code: routing and disk (2026-09-19/20)

Questions 13-16. A routing trace on the real model and a disk bench; no stopwatch on the engine, so
no dependency on the width estimator, not yet validated.

**The trace.** `trochilus run --route-trace <file>` records, for every token and every layer, the 8
experts chosen and two predictions of the next layer's top 16, with the next layer's router:
`pred_in` on the state entering the current layer's FFN (known **before** its experts run),
`pred_out` on the layer's output passed through the next layer's `ffn_norm` (known only once the
layer is finished). Off, it costs nothing and changes no byte. `tests/test_route.c` proves it exact
on synthetic models (with the attention's output zeroed, `pred_out` **is** the next layer's choice;
with layer 0's router generated as layer 1's, `pred_in` **is** layer 0's choice), five mutations
seen red (`tools/mutate_route.sh`). Run: OLMoE-1B-7B Q8_0, `bench/prompts/code-1000.txt` (904 tokens
of our own code), 300 tokens generated, in the container (experts are counted, not seconds). Tables
from `tools/route_trace_report.py build/route/code-1000.bin`.

Question 13, the share of layer L+1's chosen experts that were among the top K predicted:

| | `pred_in` 8 | 12 | 16 | `pred_out` 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| all tokens | 82.4% | 92.4% | 95.8% | 86.1% | 94.9% | 97.2% |
| prompt only | 81.4% | 91.7% | 95.3% | 85.5% | 94.5% | 97.0% |
| generated only | 85.2% | 94.5% | 97.1% | 88.0% | 96.0% | 97.8% |
| worst layer (L=0) | 63.2% | 74.6% | 82.0% | 67.7% | 79.7% | 85.7% |

The threshold was "at least 80% predicting at most 12": **passed** (92-96%), and already with the
prediction taken **before** the experts run, which leaves the disk a whole layer's worth of time;
waiting for the layer's output gives 2-3 more points and half the time. The first layer is the
exception (75-80%): what predicts it is the embedding, not a layer.

Question 14, a cache of units (layer, expert) simulated over the whole trace (model superseded by
measurement: §M1 measured: what experts really cost from disk), counted on the generated tokens
only; a layer's expert is 6.375 MiB (three Q8_0 matrices), the model has 1024:

| cache | LRU: misses per token | LRU: MiB per token | usage pin from the prompt: misses | MiB |
|---|---|---|---|---|
| 25% (256) | 54.6 | 347.9 | 58.3 | 371.7 |
| 50% (512) | 22.4 | 143.0 | 35.0 | 222.9 |
| 75% (768) | 5.0 | 32.2 | 14.7 | 93.8 |

A token uses 128 units (8 × 16): with half the model in RAM, 22 are missed. **The static usage pin
loses to the LRU at every capacity** (at 75% it reads three times as much): M1's plan said "usage
pin", and on this trace it is the worse of the two choices.

Question 15, streaming whole layers (every token touches every layer): 5106 / 3404 / 1702 MiB per
token with 25 / 50 / 75% of the layers in RAM, **15-53 times** streaming by experts. Closed: by
experts.

**The direct read, on the real model, native** (2026-09-20, one run per mode, not a measurement:
it only serves to say the chain holds up outside the container and outside the tiny models). Budget
3264 MiB, prompt 8, 2 tokens: `experts: 511 of 1024 units in RAM (3264 MiB, direct), 109 hits, 584
misses, 3723 MiB read in 2.96 s`, and with `TR_EXPERT_DIRECT=0` the same hits, misses and MiB in
3.41 s. Two things: **3723 MiB in 2.96 s is 1258 MiB/s**, that is, the disk's bandwidth as measured
by `make bench-disk` (~1.4 GB/s at 2 MiB blocks) and not RAM's, so it really is reading from disk
even on an encrypted volume; and the slots are **511 against 512**, because in direct mode each one
carries a margin sector on each side (3 × 4 KiB over 6.4 MiB: 0.2%).

**The cost of a read call** (LESSONS #94, measured on 2026-09-20 natively,
`bench_event_overhead` in `tests/bench_disk.c`): on Windows `tr_file_pread` creates and closes an
event on every call. On a 4 KiB block already in cache, 20 000 rounds: **2.54-2.61 µs** against
**2.11-2.20 µs** with an event kept alive, that is, 0.41-0.44 µs per call over three runs. A 2 MiB
read on this disk is 1.4 ms: 0.03%. Not changed.

**The disk** (question 16). `make bench-disk` (`tests/bench_disk.c`): blocks the size of one
expert's matrix (2.125 MiB) and of three (6.375 MiB), at random aligned positions of the model file
(6.85 GiB), opened without the system cache (`FILE_FLAG_NO_BUFFERING`; `O_DIRECT` on Linux), with
1-16 readers; before every number the direct reader and `tr_file_pread` must give the same bytes.
Native, still machine (`tools/machine_still.sh`), two sessions, median of 5 runs each
(`build/disk/run1.txt`, `run2.txt`); 1 TB NVMe Micron 2400 (QLC, no DRAM).

| block | 1 reader | 4 | 8 | 16 | spread |
|---|---|---|---|---|---|
| 2.125 MiB | 1396 / 1454 MB/s | 1489 / 1471 | 1504 / 1514 | 1373 | 8-16% |
| 6.375 MiB | 1641 / 1718 | 1749 / 1696 | 1666 / 1723 | 1614 | 7-20% |

Through the system cache (`tr_file_pread`, 8 readers): the first read 1.66-1.93 GB/s, the second
read of the same blocks **12-26 GB/s** (it is a copy from RAM).

1. **The disk gives ~1.5 GB/s and readers don't matter**: one alone gets as much as eight, sixteen
   lose. M1's "I/O pool" is not for bandwidth: 1-2 threads are enough, and they only serve to keep
   compute from stalling. Three matrices in a row give 15% more than three separate reads.
2. **1.5 GB/s is a third of the disk's spec** (4.5 GB/s sequential read). It is not known why:
   volume encryption, QLC with no DRAM on sparse reads, or the synchronous 2 MiB request. Question
   41.
3. **Expected tokens per second** (decode at 8 threads ~28-36 ms of compute per token; 1 MB = 10^6
   bytes, 1.5 GB/s = 1430 MiB/s): with the cache at 50%, 143 MiB are read = 100 ms per token, that
   is, **7-8 tok/s** if reading and compute add up and **10 tok/s** if they fully overlap (this is
   what question 13's prediction is for); at 75%, 22 ms, **17-20 tok/s** summed and compute alone
   (28-36) if overlapped; at 25%, 243 ms, **3.6-4.1 tok/s**. The criterion was "by experts if at
   least 5 tok/s at 50%": **passed**.
4. The system cache gives 12-26 GB/s on what it has already read, but it cannot be governed (neither
   how much it keeps nor what it drops): M1's RAM budget stays ours, and the read must be done
   without passing through RAM twice (either direct, or ours: to be decided in the project).
5. Limits of the measurement: a single trace (code, 904 + 300 tokens) and a single model, with small
   experts (8 active of 64); the prompt is counted token by token, while a batched prefill touches
   almost every expert of every layer in one pass (for the prompt, streaming by experts costs as
   much as reading the whole model once).
6. **With this disk, prefetching does not pay, and with 12 candidates it hurts.** The report's
   simulation (model, not measured) of the same LRU with prefetching of `pred_in`'s top k (per
   generated token, cache at 50%): stalls drop from 22.4 to 7.2 (k=8) and 5.5 (k=12), but reads
   rise from 22.4 to 29.9 and 49.8, because 5 and 21 are experts that then are not needed. The disk
   is the bottleneck, so what counts is the total read, not the stalls. **Time model** (model, not
   measured: one disk, one read at a time at 4.46 ms per unit, 33 ms of compute per token;
   `route_trace_report.py`, hand-checked with `--check` and with four mutations seen red):

   | cache | without | k=8 | k=12 | k=16 | | disk 3× (4.3 GB/s): without | k=8 | k=12 |
   |---|---|---|---|---|---|---|---|---|
   | 25% | 3.6 tok/s | 3.5 | 2.2 | 1.6 | | 8.8 | 10.0 | 6.4 |
   | 50% | 7.5 | 7.0 | 4.5 | 3.0 | | 15.1 | 17.4 | 12.7 |
   | 75% | 18.0 | 18.1 | 13.0 | 9.0 | | 24.7 | 26.9 | 24.6 |

   Question 13's threshold ("80% with 12 candidates") measured the wrong thing: the prediction is
   good, but every extra candidate is a read, and the read is what is missing. The first M1 is
   **LRU and on-demand reads, with no prefetching**; prefetching (k = the top 8 most likely, never
   more) remains an option the automatic plan turns on only on a fast disk, where it is worth
   11-15%. Question 43.

## M1 measured: what experts really cost from disk (2026-09-20 night)

The real engine, native Windows, still machine (background load 1.0-1.4 processors of 16, almost
all the System process), `sh tools/experts_budget.sh measure | misses | long | direct`. A 512-token
prompt, decode width **forced to 8** (the estimator is not validated, point 0), direct read
verified before every session, rotating order and one A/A copy per mode (`tools/ab_modes.sh`). The
model has 1024 units (layer, expert) of 6.375 MiB: 6528 MiB in all.

**1. Speed at the four budgets** (median of 6, 64 tokens generated; the A/A copy in parentheses):

| budget | decode tok/s | vs 100% | prefill tok/s | run's misses | MiB read |
|---|---|---|---|---|---|
| 100% (resident) | 35.45 (35.66) | 1.00× | 306.9 (319.4) | — | — |
| 75% | 29.56 (29.32) | 0.83× | 80.5 (80.9) | 1009 | 6432 |
| 50% | 24.27 (24.25) | 0.68× | 82.3 (82.1) | 1106 | 7051 |
| 25% | 20.56 (20.64) | 0.58× | 84.0 (83.5) | 1204 | 7676 |

**The cliff is between resident and non-resident, not between the budgets.** The prefill runs at
306.9 tok/s if the model is in RAM and at 80-84 with any partial budget — 75% does not help more
than 25%, because the prompt reads the whole model regardless: 6432-7676 MiB against the minimum
6528 (every unit once). That is ~4.3 s of disk at 1.5 GB/s, paid **once per session**. The budget
decides the decode.

**2. How much a generated token costs** (difference between two generation lengths, so the prompt
drops out of the count; the counters do not move from run to run, 0.0% spread):

| budget | 72 against 8 tokens: misses/token | MiB/token | 1000 against 200: misses/token | MiB/token |
|---|---|---|---|---|
| 75% | 0.1 | 0.5 | — | — |
| 50% | 0.3 | 1.8 | 0.03 | 0.2 |
| 25% | 0.6 | 3.7 | 0.14 | 0.9 |

**Question 14's simulation (model superseded by measurement) said 22.4 misses and 143 MiB per token
at 50%: the engine gives 0.3 right after the prompt and 0.03 far from the prompt**, that is, 70-700
times fewer. It is not a counting error: that simulation (model superseded by measurement) counted
misses over a 1000-token generation starting from an arbitrary cache, while the engine reaches the
decode with the LRU just filled by the prompt — which has touched almost every unit. The last 512
(or 256) units touched are exactly the ones needed, and the further generation goes the **less** it
misses, not more (LESSONS #98).

**3. The decode improves with the length of generation**, and the store is the reason:

| budget | 64 tokens | 200 tokens | 1000 tokens | extra ms/token at 200 | fixed cost |
|---|---|---|---|---|---|
| 100% (resident) | 35.45 | 34.64 | 33.48 | **−1.0** | **none** |
| 50% | 24.27 | 29.21 | 30.79 | +1.75 | ~0.35 s |
| 25% | 20.56 | 26.53 | 29.84 | +4.18 | ~0.84 s |

The deciding session is the one from the night of 2026-09-21, with the **full budget** inside: with
the resident model the time per token does not improve at all going from 200 to 1000 (it actually
worsens by 1 ms, within a 4-7% spread). So the fixed cost is **not** kernel warm-up, nor the first
token, nor our own yardstick: it appears only when reading from disk and grows the tighter the
budget is. It is the store resettling after the prompt, and a startup preparation phase cannot hide
it, because it happens *after* the prompt.

The first tokens' extra misses explain part of it (at 25%: 27 extra units in the first 200 tokens,
172 MiB, ~115 ms of ~840). The rest is to be found with a per-zone profile of the first 64 tokens
against tokens far from the prompt: question 46, narrowed to this.

**4. Direct read against the system cache** (budget 50%, median of 6, same misses and same MiB in
both modes: only where the bytes come from changes):

| | prefill tok/s | decode tok/s |
|---|---|---|
| `direct` (`FILE_FLAG_NO_BUFFERING`) | 81.4 | 24.2 |
| `buffered` (`TR_EXPERT_DIRECT=0`) | 197.0 | 32.6 |
| | **2.42×** | **1.35×** |

With 7 GiB read and 31 GiB of RAM the system cache holds the whole model: "buffered" measures RAM,
not the disk. This is why `experts_budget.sh` refuses to start if the `experts:` line does not say
`direct` — without that guard every M1 number would be inflated 2.4× on the prompt.

## Prefill reads model once per pass (2026-09-21)

Question 47, born from the question of I/O-compute overlap: how much of the prefill is disk wait
and how much is compute. The profile separates the two (the `weight_read` zone), and the count
found something else. `sh tools/prefill_overlap.sh 5` (median of 5 rounds plus one warm-up,
rotating order, still machine at 1.07-1.19 processors of 16, binary `build/native2` because Smart
App Control blocked the first, LESSONS #81):

| mode | prefill s | disk s | compute s | MiB read | GB/s | overlap ceiling |
|---|---|---|---|---|---|---|
| prompt 512, resident | 1.65 | 0.00 | 1.65 | 0 | — | — |
| prompt 512, budget 50% | 6.11 | 4.44 | 1.68 | 6 031 | 1.33 | 1.38× |
| prompt 2048, resident | 7.11 | 0.00 | 7.11 | 0 | — | — |
| prompt 2048, budget 50% | 23.51 | 16.26 | 7.24 | **22 880** | 1.37 | 1.45× |
| prompt 2048, budget 50%, **one pass** (`-b 2048`) | **11.27** | 4.60 | 6.67 | **6 273** | 1.33 | 1.69× |

Spread 0.7-8.7%. The "overlap ceiling" is `total / max(disk, compute)`: **model, not measured**,
it is the limit of a reader that does not exist yet.

**The count checks out to the cent**: at 512, the under-budget prefill minus the resident one gives
`6.11 − 1.68 = 4.43` against the 4.44 s the zone declares, and on the resident modes the disk is
0.00. The profile's accounting is exact, and the disk runs at 1.33-1.37 GB/s as `make bench-disk`
says.

**The defect**: 22 880 MiB is **3.5 times** the experts' table (6 528 MiB). The prompt is processed
in blocks of 512 tokens (`OLMOE_DEFAULT_BATCH` in `src/models/olmoe.c`) and **every pass walks all
the layers**: under budget the LRU cannot keep the model between one pass and the next, so every
pass rereads everything. 2048 / 512 = 4 passes, 3.65× the bytes. At 4000 tokens it is eight.

**Exact, and already provable without touching code**: with `-b 2048` the bytes drop to 6 273 MiB
(once only, plus the sectors' margin) and the prefill goes from 23.51 to 11.27 s. `logits -b 512`
and `logits -b 2048` on the same prompt give the **last row identical to the byte** (201 216 bytes,
`cmp`), which is what the two forms have in common: the pass does not change the numbers, it
changes how many times the disk is read.

**The compute does not worsen with the large pass**: 6.67 s against 7.24 at four passes (and 7.11
resident). So the reason not to simply raise `-b` is **not the kernels, it is memory**: the
activations, the KV and one pass's scratch grow with the number of tokens, and at 4000 or 32000 a
single pass does not fit in RAM.

**The cure is the per-layer order, written on 2026-09-21** (`src/models/olmoe.c`: a layer's body
extracted into `forward_layer`, and `forward_prompt_layer_major`, which for every layer walks all
the prompt's passes; the hidden state of the whole prompt sits in `s->x_all`, allocated when the
session is created and only when the store is partial — nothing is allocated in the hot zone).
Every expert is read once per prompt while keeping the blocks small; at full budget the path is the
earlier one, unchanged.

| prompt 2048, budget 50% | before | after | |
|---|---|---|---|
| MiB read | 22 880 | **6 273** | 3.65× fewer |
| prefill | 23.51 s | **12.41 s** | **1.89×** |
| compute | 7.24 s | 7.73 s | not distinguishable |

Median of 5 rounds plus one warm-up, spread 11.4%, still machine (0.94-1.37 processors of 16). In
the same session `-b 2048` (one pass) gives 12.03 s and the same 6 273 MiB: the two forms agree
within 3%, that is, the per-layer order reaches the experiment's ceiling without growing the
activations. At prompt 512 (already a single pass) nothing changes: 6.27 s against 6.11, within the
spread.

The check that closes lesson #99 is `tests/test_stream.c` §`once_per_prompt`: 36 tokens in passes
of 12 on the minimal store, the units read must stay within `n_units + n_slots`. Seen red before
the fix (33 read, 16 in the table) and green after. **With `--route-trace` active the earlier order
stays in place**: the trace numbers rows from `n_tokens`, which advances only after the last layer,
so in per-layer order the blocks would overwrite each other; it is a measurement-only mode and is
not worth a second way of counting.

**The levers of the under-budget prefill, in order of payoff** (the first two multiply):

| lever | gain | cost |
|---|---|---|
| experts read once per prompt (per-layer order) | **2.09×** at 2048, more on long prompts | exact, none |
| a store that survives the session (question 45) | removes the disk from the second session on | a daemon |
| overlapping disk and compute | 1.38-1.69×, ceiling (model, not measured) | a reader, a thread |
| file reordering by co-activation (mbolt, MIT) | ≤ 1.15× here | a format of our own |

On mbolt (`github.com/doramirdor/mbolt`): it exists, MIT, and its README claims 2.23× **on reads**
and 1.55× end-to-end on Qwen3-Next-80B, 512 experts per layer, with the model larger than RAM. It
also says where it stops paying off: "~4 MB slices cap gains at 1.3×" and "greatest benefit at deep
offload (≳ 2.5× RAM ratio)". Our experts are 6.375 MiB and the model fits in 7 GiB of 31 of RAM: we
are outside its range, and our `bench-disk` (1.4-1.5 GB/s at 2.125 MiB blocks against 1.6-1.75 at
6.375) puts the ceiling at ~1.15×. Not measured by us: it is a reading, not a measurement.

## Is code behavior a small, deterministic graph? (2026-09-20)

Question 44 (Marcello), with the thresholds written before measuring (row 44 of §Open questions:
what to measure). Six routing traces on OLMoE-1B-7B Q8_0, 300 tokens generated each, in the
container (experts are counted, not seconds): four of our own code (`code-1000` C, `trace-c2` C,
`trace-py` Python, `trace-sh` shell; 842-904 prompt tokens) and two control prose ones
(`trace-prose-it`, a piece of this project in Italian, 941 tokens; `trace-prose-en`, an English text
about hummingbirds, 583 tokens and 265 generated). Trace version 2: also every token's id and the
router's margin (probability of the last expert chosen and of the best excluded one), exact by
proof (`tests/test_route.c`: at layer 0 the excluded expert of a model with 2 experts per token
**is** the last one chosen by the same model with 3). Tables from `tools/route_graph_report.py`
(`--check` computed by hand, mutations in `tools/mutate_reports.py`); files in `build/route/`.

| | code-1000 | c2 | py | sh | prose it | prose en |
|---|---|---|---|---|---|---|
| units used, of 1024 | 1012 | 998 | 1006 | 1011 | 1013 | 1018 |
| top 25% used covers | 68.1% | 73.2% | 72.5% | 70.0% | 60.9% | 51.7% |
| top 50% used covers | 87.7% | 92.5% | 91.2% | 87.7% | 83.6% | 79.2% |
| units for 99% of activations | 84.9% | 75.6% | 81.8% | 84.0% | 85.6% | 89.5% |
| static co-occurrence graph, top 8 (the live router: 82-86%) | 53.7% | 64.2% | 64.8% | 54.3% | 59.3% | 51.3% |
| frequency alone, top 8 (chance: 12.5%) | 40.2% | 51.9% | 54.2% | 40.0% | 37.7% | 20.7% |
| same token id → same set, layer 0 / average / last | 14% / 11% / 12% | 16 / 14 / 16 | 13 / 11 / 14 | 9 / 10 / 14 | 6 / 9 / 27 | 3 / 6 / 8 |
| same, average Jaccard, layer 0 / average | 0.51 / 0.60 | 0.51 / 0.64 | 0.46 / 0.62 | 0.42 / 0.62 | 0.40 / 0.56 | 0.33 / 0.51 |
| experts shared with the previous token, of 8 (chance 1.0) | 3.6 | 3.9 | 3.9 | 4.0 | 3.3 | 3.0 |
| whole paths repeated | 0 | 0 | 0 | 0 | 0 | 0 |
| router margin, median (share of the last chosen expert's probability) | 5.1% | | | | | 6.3% |

Overlap of the hottest 25% between two traces (Jaccard; in parentheses the share of the second
trace's activations that fall in the first's hot set):

| | c2 | py | sh | prose it | prose en |
|---|---|---|---|---|---|
| code-1000 | 0.75 (71%) | 0.73 (70%) | 0.68 (68%) | 0.34 (45%) | 0.09 (19%) |
| c2 | | 0.77 (71%) | 0.72 (68%) | 0.33 (45%) | 0.07 (17%) |
| py | | | 0.74 (68%) | 0.37 (48%) | 0.08 (17%) |
| prose it | | | | | 0.16 (24%) |

With the hottest 50%: code-to-code 0.72-0.77 (85-90% of activations inside), code-to-English-prose
0.28 (46%).

1. **Small: no.** Threshold "the top 25% of units covers 99%": it covers 68-73% on code, and for
   99% it needs 76-85% of the model. Every trace touches 97-99% of the units within a thousand
   tokens.
2. **Table: no, not even at layer 0.** Threshold "same id → same set in 95% of cases": 9-16% on
   code (Jaccard 0.42-0.51). The same token picks different experts depending on context already at
   the first layer, where between the embedding and the router there is only one attention.
3. **Static: no.** A stateless co-occurrence graph guesses 54-65% of the next layer, the router on
   the live state 82-86%; no whole path repeats, in any trace. And the choices are tight: half are
   decided by less than 5% of the last chosen expert's probability (median 5.1%), so the discrete
   graph is also fragile to small perturbations.
4. **A region of code: yes, clearly.** Threshold "code-to-code exceeds code-to-prose by at least
   0.20": 0.68-0.77 against 0.07-0.09 with English prose, and 0.31-0.37 with Italian technical prose
   (which talks about code). C, Python and shell heat the same units; the text about hummingbirds
   almost none of them. Code is also more concentrated than prose (25% → 68-73% against 52-61%) and
   more predictable from frequency alone (40-54% against 21%). It is domain specialization, not a
   small graph: one file's hot set covers only 68-71% of another's activations.
5. For the engine: the LRU remains the right choice within a session (3.6-4.0 of 8 experts shared
   with the previous token); the region of code says that a cache **warm across code sessions**
   already starts with 68-71% of what is needed (question 45), and says nothing about how much of
   the model can be dropped: only the functional proof below says that.
6. Limits: a general-purpose 64-expert model, six texts, 300 tokens generated per trace, prompts
   made of our own code; "deterministic" here means predictable without the state, not repeatable
   (it is repeatable by construction).

**The functional proof** (`sh tools/mask_quality.sh`: experts turned off with `--expert-mask`,
measurement-only mode; mask = the units outside the top 75 / 50 / 25% used in `code-1000`, and for
control the same number picked at random; logits at every position against the whole model).
**Five texts out of five** (night of 2026-09-20, `build/mask/quality.txt`): `code-1000` is the
**most favorable** case (it is the one the mask is derived from, 904 positions), `trace-c2` a C
file never seen (860), `trace-py` Python (842), `trace-sh` shell (875), `trace-prose-en` English
prose (583), that is, the out-of-domain control:

| units kept | | by usage: average KL | different token | random: average KL | different token |
|---|---|---|---|---|---|
| 75% (256 off) | code-1000 | 0.0154 | 29 (3.2%) | 0.598 | 178 (19.7%) |
| | trace-c2 | 0.0198 | 30 (3.5%) | 0.702 | 209 (24.3%) |
| | trace-py | 0.0539 | 63 (7.5%) | 0.877 | 246 (29.2%) |
| | trace-sh | 0.084 | 83 (9.5%) | 1.26 | 319 (36.5%) |
| | trace-prose-en | 1.01 | 212 (36.4%) | 0.865 | 226 (38.8%) |
| 50% (512 off) | code-1000 | 0.0902 | 55 (6.1%) | 4.58 | 663 (73.3%) |
| | trace-c2 | 0.103 | 62 (7.2%) | 4.70 | 644 (74.9%) |
| | trace-py | 0.276 | 118 (14.0%) | 5.02 | 713 (84.7%) |
| | trace-sh | 0.384 | 163 (18.6%) | 5.05 | 759 (86.7%) |
| | trace-prose-en | 2.41 | 350 (60.0%) | 4.43 | 486 (83.4%) |
| 25% (755 off) | code-1000 | 0.309 | 121 (13.4%) | 8.27 | 880 (97.3%) |
| | trace-c2 | 0.334 | 117 (13.6%) | 8.39 | 848 (98.6%) |
| | trace-py | 0.691 | 199 (23.6%) | 8.84 | 822 (97.6%) |
| | trace-sh | 0.976 | 264 (30.2%) | 8.47 | 867 (99.1%) |
| | trace-prose-en | 6.37 | 550 (94.3%) | 6.35 | 549 (94.2%) |

7. **Functional: no.** Threshold "with 50% off, same token in 99% of positions and KL ≤ 1e-2":
   93.9% and KL 0.090 on the mask's own text, 92.8% and 0.103 on a C file never seen, and on the
   other two languages it drops (86.0% and 0.276 in Python, 81.4% and 0.384 in shell); not even
   keeping 75% passes (96.8 / 96.5 / 92.5 / 90.5%, KL 0.015-0.084). The small graph is not there
   even in the weak sense: keeping the behavior needs almost every expert.
8. **The mask holds on unseen C, wears out on the other languages.** With 50% off the token matches
   in 93.9% (the mask's own text), 92.8% (other C), 86.0% (Python), 81.4% (shell): the region from
   point 4 exists but has a center, and the further from C, the more it costs. It is not memory of
   the text (between the two C files there is one percentage point), it is distance from the
   language the mask comes from.
9. **But usage says a great deal about which experts are needed**: for the same number of units
   off, the usage mask moves the output 13-50 times less than the random one (KL 0.09-0.38 against
   4.6-5.1 with 50% off, across all three languages). It is not a small graph, it is a steep
   ranking: the model tolerates losing the right experts poorly and losing the wrong ones almost
   not at all.
10. **Outside the domain the ranking is worth nothing anymore**, and this is the control that was
    needed: on English prose the usage mask performs like the random one (with 75% kept, 36.4%
    against 38.8% of different tokens; with 25%, 94.3% against 94.2%; KL 6.37 against 6.35). The
    hot experts on code are not "the best experts", they are code's experts: a cache warmed on one
    domain is a gain for that domain and zero for another (question 45).

## The gate: where its time goes (2026-09-22)

`make check` on this machine, gate in Docker, one run per configuration: these are wall times that
size the steps, not medians that compare code. Background: the OpenEMR stack of another window
running its own tests the whole time (declared, not measured). Logs `build/check-serial.log`,
`build/check-par.log`, `build/check.log`; tables by `tools/gate_times.py`.

| step | before | after | why |
|---|---|---|---|
| `tools/test_cleanup.sh` | 101 s | 5 s | an orphan kept on purpose held the pipe of `$(...)` open (LESSONS #107) |
| C tests: gcc, clang, ASan, TSan | 56 + 46 + 71 + 42 s, in a row | at once, ~2 min for the four | own BUILD dirs, own temp files (`test_base` wrote to a fixed path) |
| `oracle-real` under the smallest store | 308 s | 57–73 s | the 2-layer cut read over the 9p bind mount (LESSONS #109) |
| tokenizer, tiny oracles, tier-check | 39 + 4 + 3 + 14 s, in a row | beside the real model's steps | they share nothing with them but the binary |
| **whole gate** | **~935 s** (839 s measured with the cleanup already fixed, + 96 s) | **335 s** ("check passed in 335 s"; 423 s with the model steps still in a row) | 2.8× |

The four builds share the processors, so each is slower than alone (ASan 71 → ~100 s): the gain is
in the overlap, not in any single step. The real model's steps stay in a row: each loads the model,
and two at once would ask the engine's memory guard for twice the room.

### 2026-09-24: 428 → 238 s

Same method before and after: `make check` with the time on every line, `tools/gate_times.py`,
nothing changed in the code between the run and the one before it (every build up to date). One
run each: sizes, not medians. Background: the OpenEMR stack of another window, idle. Logs
`build/check-before.log`, `build/check-after6.log`; per test, `build/pertest-*.txt`.

| block | before | after | why |
|---|---|---|---|
| scripts' tests (cleanup, marker, ab_modes) + lint | 22 s | 13 s | `test_marker.sh`: poll of 0.2 s, the wait for the other window by its announcement instead of `sleep 3`, the caps at 0 (10 → 2.7 s, 10 branches still reached) |
| C tests of four builds, at once | 170 s | 45 s | the tests wrote their synthetic models on the Windows bind mount: `test_model_load` 66 s → 0.4, `test_stream` 16 → 5, `test_gguf` 13 → 1, the same in every build (gcc 121 → 22 s). `TR_TEST_TMPDIR` on the container's disk (LESSONS #142); the 20 runs of `test_hot` four at a time (34 → ~8 s), each in its own directory (LESSONS #143) |
| model steps | 210 s, two lanes (tiny ‖ real in a row) | 153 s, three lanes | tiny 32 s ‖ whole model: chat, speculation, then the tokenizer oracle ~150 s ‖ the 2-layer cut: `oracle-real` 31 s then under the smallest store 92 s |
| native tests at the end | 22 s | 22 s | — |
| **whole gate** | **428 s** | **238 s** (299 s when the Makefile changed and every build recompiles) | 1.8× |

The longest lane now holds the whole model and then the tokenizer oracle: 34 s alone, 4.0 GB of
peak RSS (measured), so it never runs beside the whole model (7.4 GB) — three lanes keep the peak
near 10 of the Docker VM's 15 GB, where the engine's memory estimate never refuses. Next levers,
not taken: the native side (scripts' tests, lint, the Windows build and its C tests, ~35 s) beside
the container instead of before and after it; `oracle-real` under the smallest store, 92 s for a
2-layer cut, not looked into. The `trochilus-t3` clone (a gate rewrite never verified) was read and
deleted: it had found the same bind mount cost; taken from it, the platform stamp written
atomically (LESSONS #144) and the tokenizer oracle's peak; its diff is in
`build/trochilus-t3-uncommitted.diff`.

## The engine kept between commands: the first prompt (2026-09-23)

Question 49: `trochilus serve` keeps pool, model and expert store; the same `generate` runs in a
new process (cold, `TR_SERVER=0`) or through the server (warm), which finds the store the run
before left. `sh tools/serve_first_prompt.sh 5` on `build/trochilus.exe`: OLMoE-1B-7B Q8_0, prompt
2048, 8 tokens, `-t 16 --decode-threads 8`, median of 5 rounds after one of warm-up, rotating
order, `AB_WALL=1` (`wall_ms`: the whole command, load included). Machine marker taken, other
windows' containers up and paused by it; background load 2.94 logical processors busy before the
first run and 3.34 after the last (0.83 of them the kernel's `System`, chrome 0.5-0.9). Logs in
`build/serve_first_prompt/`.

| budget | mode | wall s | prefill tok/s | misses | MiB read | A/A |
|---|---|---|---|---|---|---|
| half (3 264 MiB, 511 of 1024 units) | cold, new process | 13.00 (12.91–13.57) | 175.5 | 1 157 | 7 376 | coldagain: wall 1.020×, prefill 0.980× |
| | warm, through the server | **12.04** (11.86–12.37) | 187.6 | 1 018 | 6 490 | |
| full (whole table resident) | cold, new process | 12.79 (12.73–12.95) | 279.7 | — | — | coldagain: wall 1.004× |
| | warm, through the server | **7.81** (7.77–8.14) | 275.4 | — | — | warmagain: wall 1.020× |

1. **Full budget: the server removes the load, 4.98 s of 12.79 (0.61×).** The prefill is the same
   (279.7 against 275.4 tok/s, within spread): what goes is the resident table's read at start,
   the 4.7 s the question was about. The half server served 7 requests and the full one 12, each
   with the model loaded once (`serve --status`, `server-*.log`).
2. **Half budget: 0.96 s of 13.00 (0.926×), beyond the A/A (1.020×), the ranges apart.** The
   prediction was "misses warm = misses cold": measured 139 fewer of 1 157 (12%, 886 MiB), and a
   prefill 7% faster by as much (0.76 s at 2048 tokens); the other ~0.2 s is the dense part's
   load. Where the 139 come from is not measured: a guess is the units the previous run's decode
   reread in the first layers, which the next sweep reaches before it evicts them. The mechanism
   holds (a store smaller than one sweep is rewritten by every prompt), its size was off.
3. **At half budget the next lever is what a sweep evicts, not the server.** A 2048-token prompt
   sweeps all 1024 units in layer order; LRU over a loop larger than the store keeps what the next
   loop needs last. A policy that keeps a fixed part of the loop (the first layers, evicting the
   most recent during a sweep) would carry up to ~511 units from one prompt to the next: up to
   ~3 260 MiB and ~2.5 s less at the 1.33 GB/s of §Prefill reads model once per pass, with the
   server. A model, not a measurement.

## Reading the next layer while this one computes (2026-09-23)

M1 work order item 3. In the layer-major prompt an I/O thread reads layer L+1's missing units
while layer L computes (`tr_experts_prefetch`, `TR_PREFETCH=0` turns it off); the logits are the
same bytes (`tests/test_prefetch.c`, TSan clean in `make check`). `SET=prefetch sh
tools/prefill_overlap.sh 5`: OLMoE-1B-7B Q8_0, half budget (3 264 MiB), median of 5 after one of
warm-up, rotating order; machine marker taken, background load 2.08 / 1.71 logical processors
(0.87 / 0.72 of them `System`). "disk s" is now the time the compute waited for the disk, no
longer all the disk's time. Logs in `build/prefill_overlap/`.

| mode | prefill s | disk wait s | compute s | MiB read | spread |
|---|---|---|---|---|---|
| 2048, read ahead | **9.39** | 2.07 | 7.32 | 6 343 | 2.5% |
| 2048, read ahead, A/A | 9.33 | 2.07 | 7.27 | 6 343 | 3.0% |
| 2048, `TR_PREFETCH=0` | 11.45 | 4.33 | 7.11 | 6 273 | 1.3% |
| 512, read ahead | 5.92 | 4.16 | 1.77 | 6 031 | 0.7% |
| 512, `TR_PREFETCH=0` | 5.83 | 4.16 | 1.67 | 6 031 | 1.8% |

1. **At 2048 tokens: 11.45 → 9.39 s, 1.22×** (A/A 0.6%). The ceiling was 1.61×: half of the disk
   wait is still there (2.07 of 4.33 s), and the compute is 3% slower beside the reads (7.32
   against 7.11). Layer 0 cannot be read ahead (~0.3 s); where the other ~1.8 s goes is not
   measured yet (a profile per layer: is a layer's read longer than the layer before's compute?).
2. **At 512 tokens: nothing** (5.92 against 5.83, within twice the spread, no A/A at 512). A
   512-token prompt is one block, so it takes the pass-major path, where nothing reads ahead: as
   designed, and the next lever there (ceiling 1.40× at 512).
3. **70 MiB more read at 2048** (6 343 against 6 273, 11 units, 1.1%): the rule reads all of the
   next layer's missing units once the layer just done used nearly all of its own, and a few are
   not asked for.


`tools/mutate_auto.py` swaps operators, drops `± 1`, swaps `return 0` / `return -1`, drops `(!`,
one at a time, rebuilds, runs the given tests and, with `--cmd`, the oracles; a mutant whose object
file is byte-identical to the original (a branch this platform does not compile, or folded by the
compiler) is set aside. A check that fails is run again on the same mutant, and only a second
failure kills it (LESSONS #115: under memory pressure a load refused had counted as a kill); a
check out of time runs again with three times the budget, and a mutant that times out twice is
listed (#117); a failure that is the machine's memory (the engine's guard, the OOM killer) is no
verdict, and that mutant is judged again alone at the end (#119: a refusal that outlasted the
repeat had killed an equivalent mutant); a check out of time is killed with every process it
started, and a run that leaves one behind says so and fails (#122: the looping children of
`main.c`'s timeouts slowed every file after it); the checks run fastest first, each with ten
times its own time, and a C test stops at its first failure (#124); every verdict is logged with
the check that gave it, in `build/mutate/<name>.progress`, so that a kill can be audited (#125:
five kills of `olmoe.c` did not repeat). Every file of
`tools/mutate_files.sh` ran again on 2026-09-23 with the corrected tool (the rows from `olmoe.c`
down); `gguf.c`, `experts.c`, `threads.c` and `platform.c` are not in it, and their rows predate
#117-#124. Linux
container, `--asan` where a mutant can read out of bounds without crashing; the checks and the jobs
of each file are in `tools/mutate_files.sh` (jobs sized to the VM's 15 GB, not to its processors).
Background: 2026-09-23, OpenEMR's stack up and idle, the ds4 window's single-core builds at nice 19.
The whole rerun took 42 minutes without `olmoe.c` (`main.c` 23 of them) and 31 for `olmoe.c`, where
the run with the orphans of #122 had spent 37 on `main.c` and 111 on `olmoe.c`.

| file | checks | mutants | killed | survived first | survived now | same object |
|---|---|---|---|---|---|---|
| `src/format/gguf.c` (ASan) | `test_gguf` | 221 | 163 (+9 timed out) | 89 | 43 | 6 |
| `src/memory/experts.c` (ASan) | `test_experts`, `test_stream` | 100 | 87 | 27 (`test_experts` alone) | 9 | 4 |
| `src/base/threads.c` | `test_base`, `test_hot` | 75 | 35 (+9 timed out) | 33 | 23 | 7 |
| `src/base/platform.c` | `test_base` | 170 | 9 (+35 timed out) | 77 | 22 | 104 (Windows branches) |
| `src/base/prof.c` (ASan) | `test_prof`, `test_model_prof` | 46 | 36 | 42 | 10 | 0 |
| `src/models/olmoe.c` (ASan) | the 10 model tests, the tiny oracle, the same under the smallest store, the options oracle | 400 | 319 (+1 timed out) | 92 (9 tests, 2 oracles) | 70 (rerun: 65, and 5 kills that did not repeat, #125) | 10 |
| `src/kernels/kernels.c` (ASan) | `test_kernels`, `test_expf`, `test_tier_used` (also under `TR_CPU_MAX=scalar`), `test_prefill`, both tiny oracles | 110 | 76 (+5 timed out) | 37 | 26 | 3 |
| `src/kernels/kernels_x86.c` (ASan) | `test_kernels`, `test_tier_used` (also scalar) | 39 | 22 (1 did not build) | 20 | 16 | 0 |
| `src/kernels/expf.c` | `test_expf` (the proof on every float is `bench_expf --check`, in `make check`) | 7 | 5 (1 did not build) | 1 | 1 | 0 |
| `src/tokenizer/chat.c` (ASan) | `test_tokenizer`, `test_cli`, the tokenizer oracle | 31 | 24 (+1 timed out) | 4 (with kills made by the OOM killer) | 6 | 0 |
| `src/tokenizer/unicode.c` (ASan) | `test_unicode`, `test_tokenizer`, the tokenizer oracle without its sweep | 128 | 96 (+11 timed out) | 27 | 19 (17 named, 2 die under the sweep) | 2 |
| `src/tokenizer/tokenizer.c` (ASan) | `test_tokenizer`, `test_cli`, the tokenizer oracle without its sweep | 327 | 271 (+12 timed out, each a real loop) | 75 | 43 (all read) | 1 |
| `src/app/main.c` (ASan) | `test_cli`, the tiny oracle, the tokenizer oracle without its sweep | 418 (646 before one parser) | 368 (+1 timed out) | 334 | 43 (all read) | 6 |

What the survivors that are left are, read one by one:
- `gguf.c`: `return -1` → `0` after a failed read (parsing goes on over the garbage and the file is
  refused a few bytes later, with another message); boundaries at the limits (a 64 MiB string, 2^28
  array elements, 2^40 per dimension); the arena's and the hash table's rounding (equivalent);
  out-of-memory paths.
- `experts.c`: the out-of-memory paths of `tr_experts_create` (reachable only with an allocator that
  fails on demand, which the tests do not have).
- `threads.c`: the spin budget's arithmetic (a different wait, the same result), the pinning of
  threads beyond the machine's slots, the Win32 branch of thread creation (which the container
  compiles out: those are the "same object").
- `platform.c`: the POSIX error paths (`open` or `pread` failing mid-way) and the affinity calls; its
  Windows half is tested natively since #103, not here.
- `prof.c`: the calibration's constants (8 brackets or 9, the first or the last of two equally
  narrow ones, one more read of the window: the same rate), and the CPUID test and the re-read of
  the tick mode, identical on a machine with an invariant TSC. The printed table is pinned
  (`test_prof`, four profiles with known numbers; all 30 of its mutants killed).
- `olmoe.c`: out of memory, 42 (`track`, the session's and the trace's allocations, `err` NULL
  after one); a read error of a dense tensor, 1 (no fault injection for them); the direct-read
  probe, 1 (a filesystem that accepts `O_DIRECT` and fails its first read); `TR_MEM_AVAILABLE_MIB=0`,
  1 (measurement only); the arithmetic of the "not enough memory" message, 2; `vocab <= 0`, 1 (the
  reader refuses a zero dimension first); the memory guard's scratch estimate, 1; the layer-major
  path taken with a resident store or at exactly `n_batch`, 2 (the same bits by construction,
  `test_prefill`); `trace_begin(0)` allocating zero bytes, 1 (glibc returns a pointer); boundaries
  that change nothing, 18 (a clamp at exactly ±c, an empty group of queries, an extra empty block,
  ties of distinct ids, `n <= 0` already caught by `n_logits`, the offsets' last entry rewritten
  by the shift, the trace's pad slot overwritten by the next layer).
- `kernels.c`: all 26 change speed, not bits: the matmul's tiles and chunks (an empty extra block,
  the x4 kernel skipped for the last whole group, which dot_row gives bit for bit), the redundant
  guards of `tr_matmul_grouped` (an empty range does nothing either way), the attention's first query
  per block (`j0` lower: a query that does not see the block does no work in it) and its block
  bounds, a softmax maximum found at a tie.
- `kernels_x86.c`: 16, each the last whole chunk left to the tail (the scalar lanes, bit for bit)
  or an AVX-512 tail with an empty mask.
- `expf.c`: the search of the exception table one entry too far: never reached, every argument
  that gets there is in the table (proved on every float by `bench_expf --check`).
- `chat.c`: a buffer grown one step early or one doubling more, 2; `err_len > 0` at 0, 2 (snprintf
  of 0 bytes writes nothing); `err` NULL in the "not supported" branch, 1 (both callers pass a
  buffer); `tr_chat_render` out of memory, 1. The first run's 4 were fewer because the OOM killer
  had failed the oracle under three of them.
- `unicode.c`: 2 more (a Hangul syllable's end, the ASCII class table's end; U+0080's encoding length now dies without it) die only under the tokenizer oracle's full Unicode sweep, which `make check` runs and a run per mutant cannot afford (4 GB, `tools/mutate_files.sh unicode-sweep`); 7 died to new boundary cases in `test_unicode` (each UTF-8 length at both edges, U+10FFFF, U+DFFF, the jamo just outside the composing ranges). The 17 left: out of memory and the size overflow checks of `tr_nfc` (10, a text is at most 1 GiB); `TR_CP_INVALID_BASE + 0`, 2 (byte 0 is valid ASCII, never an invalid byte); the equal case of two binary searches, 2 (returned before); the NFC fast path's test, 3 (U+00C0, the threshold, is NFC-stable alone, and slower is not different).
- `tokenizer.c`: 75 → 43. New cases in `test_tokenizer`: every metadata refusal by its own message
  (tokens missing, not strings, empty; types not int32; merges not strings; a merge missing only one
  side, or making an unused token; a merge starting with a space), and the edges accepted (token
  types 0 and 6, id 0 as a merge side and result, bos and eos 0 with `add_eos`, a merge making a
  user-defined token); U+0144, one past the byte alphabet; byte 0 alone and in a piece; `'re`, and
  the `'re` rule at a segment's end over reused buffers; a 64-byte piece filling the symbol array;
  an empty control token against a text of every byte value; and one helper for the added-token
  predicate, which had two copies (LESSONS #118). The 43 left: out of memory, 14 (the `NULL` checks
  and the `return -1` after them; `tr_nfc` fails only past its size limits); the limits themselves,
  6 (2^24 tokens or merges, 4 GiB of token text, a 1 GiB text, `grow` at `SIZE_MAX`); sizes that
  change nothing, 8 (`malloc(0)` or 1 on glibc, 3; a hash table or a buffer one doubling larger, 3;
  the merge buffer's terminator, never written; `cp[n]`, never read); orderings reached only with
  distinct operands, 6 (the added-token sort, 4; the BPE heap's rank and position, 2); the heap's
  bounds, 4 (the slot past the end is `last` itself); the split's guards, 3 (`j = s` where `cls[s]`
  is `k` anyway; the "every piece advances" guard, never taken); `err_len` at 0, 1; the switch's
  default, 1 (one rule set). The 11 timeouts are real loops (a hash probe that does not move, a
  doubling from 0, a match of length 0); the one refused for memory (`sym[n].next = n`) loops
  emitting ids until memory runs out: caught, by its own hunger.
- `main.c`: 334 → 43 (LESSONS #120, #122). The five option loops became one parser with a range per
  option (646 mutants → 418), the two `--batch` readers one, the route trace one chain of writes,
  and the chat's UTF-8 cut went to `unicode.c` (`tr_utf8_whole_prefix`, 13 cases in `test_unicode`).
  `test_cli` now runs every command: every range at both ends and every kind of malformed number,
  each usage error alone, `inspect`'s listing, generate's tokens against the `logits` command's
  argmax, `--spec` with drafts accepted (the model falls into a cycle) stopping exactly at `-n`, the
  threads line forced, unmeasured and measured, every `--expert-mask` refusal, `tokenize`'s modes and
  malformed records, `run` stopped by a control token that is not EOS, the route trace's bytes, the
  chat against `run` on the text its template renders, `/reset`, a message too long for the
  context, `chat-template`'s refusals. `step[]` starts zeroed: `j <= got` read a slot the step had
  not written, and was killed or not by whatever the stack held. The 43 left: limits no file
  reaches, 3 (a size past TiB, a type the reader refuses, the shape text's bound: 4 dimensions of
  20 digits fit); out of memory, 5 (`parse_tokens`, `tokenize_one` also past 1 GiB of text,
  `conv_push`); `argmax` one float past the row (the session's buffer holds more, never larger
  here) and a tie taken last (the synthetic model draws none), 2; a model with no experts, 3 (OLMoE
  is the only architecture); `fclose(NULL)` where `fopen` failed, 1 (glibc declares the argument
  nonnull and gcc drops the call); the threads line past 32 measurements, 2; `-p 0`, which its range
  refuses first, 1; `malloc(0)`, not NULL on glibc, 1; a step that returns no token, which never
  happens, 2; the speed lines' zero guards, 5 (two readings of the clock around a pass never
  coincide); `read_file`, 2 (a failing size is -1, never another negative; `pread` of 0 bytes); an
  added token with id 0 in `--pieces`, 1 (the synthetic vocabulary's are 256-259); the NUL after a
  `--batch` buffer, 2; the chat's halving of its context under a memory refusal, 6 (no test
  machine is short of RAM, and the session's guard reads the real machine by design); position 0
  reused after `/reset`, 1 (the cache holds the same token there); the reply buffer's growth, 3
  (one step early, one byte of room more or less: the reply has no terminator); the system
  prompt's copy skipped, 1 (the synthetic model's chat answers "j\n" whatever it reads, and one
  byte is one token, so neither the reply nor the count sees the content); a rendered conversation
  of no tokens, 1 (the template always writes its markers); a control token other than EOS ending
  a chat reply, 1 (the synthetic chat never draws one; `run` has its test, on a model whose EOS is
  257). The timeout is a real loop (a malformed `--batch` record that does not advance); the
  other four loops die first at an earlier check now that a test stops at its first failure. The
  run slowed by the orphans of #122 had "killed" three of these (the last three), failing twice
  under load: the reason every file ran again.

`src/app/serve.c` (2026-09-24, `tools/mutate_files.sh serve`, whole file: it is not committed yet):
431 mutants, 205 killed, 73 alive, 153 same object, no timeout; 128 alive and 5 timeouts before
the new cases of `tests/test_serve.c` (raw clients, descriptors, another protocol, a second model),
which found LESSONS #137, #138 and #140. The 73 are not read yet.

**Hand-written mutations of the read ahead and of the load bar (2026-09-23).** `tools/mutate_bar.sh`:
63 of 63 red. `tools/mutate_prefetch.sh` (the I/O thread of the layer-major prompt, gcc and ASan):
27 of 31 red on the first run, 28 of 31 after the fix below; of the four alive, one was a flavour mistake and three are named.
A second `tr_experts_prefetch_start` without its guard starts a second thread and leaks the
first one's queue: only LeakSanitizer sees it, so that mutant now runs under ASan. Equivalent:
`break` to `continue` when no victim is left (the victim search depends on the layer and the
kept layer, not on the unit, so every later unit finds none either: more iterations, the same
store); a failed demand read not moved to the cold end (the victim already is the coldest slot
not in flight, and reserved slots sit at the hot end until taken in, so the free slot is the next
victim either way); the prompt passing no kept layer (it reads ahead after the layer's own
acquire, which makes that layer's units the most recent, and the margin of 2·n_expert + n_used
slots leaves at least n_expert + n_used colder slots outside it: the guard never fires in this
call order, and the store's own test of it is red).

**2026-09-24: the campaign faster, the same verdicts.** Three changes to `mutate_auto.py`: a gcov
pass first (every check once at -O0, beside the builds) lists the mutants on lines of code no
check runs as UNCOVERED instead of building them; with `--asan` the plain build judges every
mutant and the sanitizers only its survivors; the trees are built once and copied, mtimes kept,
not once per job. On `prof.c` (46 mutants, 12 jobs, ASan, the container at the same load, one run
each): **30 → 9 s, 36 killed and 10 survived in both**. The four files never mutated before, 20
minutes for all of them (`build/mutate/<name>.txt`):

| file | mutants | killed | survived | on lines no check runs | same object | timed out | s |
|---|---|---|---|---|---|---|---|
| `gguf.c` | 221 | 166 | 40 (the reader's error paths) | 3 | 5 | 7 | 228 |
| `experts.c` | 167 | 138 | 18 | 3 (out of memory) | 4 | 3 | 277 |
| `threads.c` | 84 | 45 | **22** (the spin's timing and the pinning: speed, not bits) | 0 | 9 | 7 | 391 |
| `platform.c` | 172 | 28 | 19 | 20 (EINTR, a failed dlopen, a line reader) | **104** (the Windows branches: never compiled in the container) | 0 | 293 |

`platform.c`'s Windows half is mutated nowhere: the container compiles its POSIX half only, and
native mutants meet Smart App Control (LESSONS #12). A debt, in STATUS.

**2026-09-25: `gguf.c`'s 40 survivors → 12, every one read.** The agent's seventeen cases
(LESSONS #172) corrected one by one, each green on the real code, and five more written against
what was still alive (`tests/test_gguf.c`): the truncations checked by the part of the file they
end in (header: a read's "unexpected end of file" or the length checks made before a read;
padding; data), the magic alone failing at `@4`, MAX_ARRAY told apart by the message in a file of
a few bytes (not 256 MiB), the exact boundaries of the two "array longer than file" checks, the
element-count and byte-size boundaries by the message they must give, general.alignment 0 and 3
alone, a string array cut inside its last item, and the last key cut at the alignment (where a
failed read taken for a success would open the file). `mutate_auto.py --asan --lines <the 39
lines>` in the container: 92 mutants, then the 17 left, then the 7 lines with cases: **12
survive, none that changes a result**:
- memory or speed only: the arena's rounding and its "does it fit" (75, 76), the hash table's
  capacity (252), the read buffer refilled at its own start (137), a read of exactly READ_CHUNK
  buffered instead of direct (140: the same bytes), the scalar array's spare byte no reader uses
  (233);
- the same verdict another way: `vsnprintf` with room 0 writes nothing (124); a tensor at
  `offset == data_size` is refused by the size clause beside it with the same message (389);
- unreachable without fault injection: the reads after a length check against the file (169, 235:
  only an I/O error fails them) and the two out-of-memory branches of `tr_gguf_open` (323, 350).

## Speed — Trochilus vs llama.cpp again (2026-09-24)

The race of 2026-09-17 (above) on the binary of commit `55c31ff`: `sh tools/race_llama.sh 5`, both
engines in the trochilus-dev container on the same OLMoE-1B-7B Q8_0 all in RAM (volume
`trochilus-models`), llama.cpp `b49650a` (`llama-bench`, `-d` = prompt for the decode), Trochilus
`generate` with the decode width forced to its thread count. Median of 5 runs after one warm-up,
per series; order A B B A per prompt, so each engine has two series (its A/A). The machine's
marker held throughout; a still machine before every series (below 3.5 busy processors); declared
background load 4.1 before and 4.5 after, of 32 logical processors, ~1 of them the kernel's System
process (as always here, LESSONS #85), the rest an idle OpenEMR stack and a browser of another
window. Tokens/s, series a / b; results in `build/race_llama/`.

| prompt | threads | Trochilus prefill | llama.cpp prefill | Trochilus decode | llama.cpp decode |
|---|---|---|---|---|---|
| 512 | 16 | 199.6 / 196.0 | 349.6 / 356.2 | 30.9 / 29.6 | 32.4 / 30.7 |
| 512 | 8 | 171.5 / 166.2 | 253.5 / 247.6 | 29.9 / 29.0 | 32.2 / 32.0 |
| 2048 | 16 | 183.5 / 185.0 | 331.5 / 327.0 | 23.6 / 24.3 | 28.4 / 27.0 |
| 2048 | 8 | 161.8 / 143.9 | 229.2 / 233.3 | 22.3 / 20.1 | 26.6 / 26.8 |

Where we lose, llama.cpp over Trochilus (means of the two series; the A/A gap in brackets, the
largest of the two engines):
- **prefill 1.8× at 16 threads, 1.5× at 8**, the same at 512 and at 2048 (A/A 2–12%). On
  2026-09-17 it was 13× at 512: the block prefill and the grouped attention closed most of it. What
  is left is mostly the int8 activations (VNNI) llama.cpp uses there, which are not exact: the
  declared int8 mode of Marcello's order (c).
- **decode at context 512: 1.04× at 16 threads, 1.09× at 8** (A/A 4–5%): at 16 threads within the
  noise. On 2026-09-17 it was 1.19× and 1.15×.
- **decode at context 2048: 1.16× at 16 threads, 1.26× at 8** (A/A 5–10%): the cost of a long
  context is still ours (question 18, the attention over the KV), and it is now the largest gap in
  decode.

## Speed — Trochilus vs llama.cpp after x8 (2026-09-25)

The same race (`sh tools/race_llama.sh`, 5 runs a series, order A B B A, both engines in the
container on OLMoE-1B-7B Q8_0 all in RAM) on commit `f6a8b4c`: the two-row eight-token kernel and
everything since `55c31ff`. Marker held, a still machine before every series, background load 0.65
before and 1.86 after (the kernel's System process, Memory Compression, the idle WSL VM). Tokens/s,
series a / b; results in `build/race_llama/`:

| prompt | threads | Trochilus prefill | llama.cpp prefill | Trochilus decode | llama.cpp decode |
|---|---|---|---|---|---|
| 512 | 16 | 294.0 / 295.4 | 377.1 / 386.1 | 33.6 / 33.4 | 34.7 / 35.0 |
| 512 | 8 | 250.6 / 257.2 | 256.3 / 257.3 | 32.2 / 32.1 | 34.1 / 33.8 |
| 2048 | 16 | 267.3 / 265.3 | 359.8 / 357.7 | 26.5 / 26.3 | 29.6 / 29.6 |
| 2048 | 8 | 241.5 / 237.9 | 249.4 / 247.4 | 25.4 / 25.5 | 29.1 / 29.0 |

- **Prefill at 8 threads: level** (llama.cpp 1.01× at 512, 1.03× at 2048, inside the A/A of 1–3%);
  at 16 threads theirs 1.30× at 512 and 1.35× at 2048 (the day before: 1.8×). Ours 1.45–1.49× the
  2026-09-24 race at 16 threads, 1.50–1.57× at 8: x8, the SIMD lane tree and what came in between.
- Where 16 threads still lose: our prefill scales 1.11–1.16× from 8 to 16 threads, theirs
  1.44–1.49×. The race
  gives no split by zone (experts against the dense projections, where tinyBLAS runs int8 4 × 4):
  the next measurement of piece 1's gap, if it is reopened, is `tools/race_llama.sh` with a
  profile of each engine by zone.
- **Decode: theirs 1.04–1.05× at 512, 1.12–1.14× at 2048** (the KV's bytes, F32 against F16;
  the exact levers of STATUS §Next steps).

## Q4_K on the CPU (2026-09-24)

M2's first step: Q4_K weights against float activations, exact (the dequantized weight in every
product), scalar as the definition and AVX2 and AVX-512 bit for bit the same (what was taken from
llama.cpp and ik_llama.cpp: docs/ORIGINS.md §Q4_K on the CPU).

**Prediction, written before measuring.** Microbenchmark (rows in cache, compute only): AVX-512
Q4_K 0.8–1.2× Q8_0's elements per second (the 16-value lookup saves the conversion to float, the
shift and the scales cost); AVX2 Q4_K ~0.7× Q8_0 (a mask or a shift and a subtraction more per
element). Real model: the Q4_K file is ~3.9 GB against 7.0, and decode reads every weight it
uses once per token, so decode 1.3–1.5× Q8_0 if the kernel keeps up with the memory; prefill,
where compute counts, 0.8–1.0×.

**Microbenchmark** (`bench_kernels`, native, a copy of the binary one byte longer: LESSONS #12;
the machine's marker, still machine; declared load 2.9 before, 4.1 after, ~0.9 of it System;
median of 7 runs of 200 ms; one core; rows in cache). Millions of elements per second, n = 2048 /
4096; `build/bench_q4k.txt`.

| tier | dot_row Q8_0 | dot_row Q4_K | Q4_K / Q8_0 | x4 Q8_0 | x4 Q4_K | Q4_K / Q8_0 |
|---|---|---|---|---|---|---|
| scalar | 1748 / 1848 | 853 / 848 | 0.46–0.49 | 2279 / 2050 | 1562 / 1557 | 0.69–0.76 |
| avx2 | 16097 / 17076 | 10157 / 9101 | 0.53–0.63 | 28905 / 32473 | 22274 / 21978 | 0.68–0.77 |
| avx512 | 15970 / 17479 | 13456 / 15685 | **0.84–0.90** | 34872 / 32613 | 31880 / 28312 | **0.87–0.91** |

Spread 5–35% per line (the machine), so a ratio holds to ±10%. Against the prediction: AVX-512 in
the range (0.8–1.2), AVX2 below it (0.7): the conversion path costs a mask or a shift, a convert,
a multiply and a subtraction per element where Q8_0 has a convert and a multiply. Per byte of
weight read, AVX-512 Q4_K goes through 15.7 G elements × 0.5625 B = 8.8 GB/s on one core, far
above one core's share of the memory (~60 GB/s over 16): decode stays bound by memory, where Q4_K
reads 0.53 of Q8_0's bytes.

**Tried and rejected: the lookup on AVX2.** The 16 values in two registers of 8, `vpermps` on the
index's low 3 bits and `blendv` on its fourth, the 8 quant bytes widened once for both nibbles;
bit-identical (gcc and clang). Order conversion, lookup, lookup, conversion, in the container (the
native copy was blocked by Smart App Control), marker held, load 2.4 before and 2.3 after; M
elements/s at n = 1024 / 2048 / 4096: dot_row 11390–11946 conversion against 11548–12216 lookup
(+0.5 to +1.5%, inside the conversion's own A/A of ~1%), x4 27650–28884 against 26616–27914
(−3.5%). Two permutes and a blend cost what a convert, a multiply and a subtraction cost on Zen 4:
the conversion stays. `build/attempts/avx2_q4k_lut_bench.txt`.

**Exactness on the real model.** The Q4_K file made by `tools/quantize_q4k.sh` (llama-quantize
`--pure --allow-requantize` from our Q8_0: 3.72 GiB of tensors, 4.51 bits per weight), cut to 2
layers, against transformers on its own dequantized weights (`make oracle-real` with
`REAL_MODEL_Q4K`, in `make check`): 27 + 32 and 1024 + 32 tokens, greedy 32/32 on both, logits
within 1.5e-5 and 1.9e-4 (tolerance 1e-3), argmax 1056/1056, every batch size bit-identical to one
token per pass. The whole model writes the same first 24 tokens as the Q8_0 on "The capital of
France is".

**Real model, Q4_K against Q8_0** (`sh tools/race_q4k.sh 5`: container, all in RAM, prompt 512, 128
generated, median of 5 after a warm-up, order Q4_K Q8_0 Q8_0 Q4_K, then llama.cpp on the Q4_K;
marker held, still machine before each series, declared load 3.6 before and 4.8 after, ~0.9 of it
System). Tokens/s, series a / b; `build/race_q4k/`.

| threads | Q4_K prefill | Q8_0 prefill | Q4_K decode | Q8_0 decode | llama.cpp Q4_K prefill | llama.cpp Q4_K decode |
|---|---|---|---|---|---|---|
| 16 | 203.8 / 200.4 | 202.4 / 195.1 | **45.6 / 46.0** | 29.5 / 30.6 | 415.8 | 48.8 |
| 8 | 171.1 / 175.4 | 152.4 / 148.4 | **43.5 / 42.6** | 29.6 / 27.7 | 309.2 | 50.3 |

- **decode 1.52× at 16 threads, 1.50× at 8** (A/A ≤ 7%): at the top of the prediction (1.3–1.5×),
  the bytes read per token nearly halved and the kernel keeping up with them;
- prefill 1.01× at 16 threads, 1.15× at 8 (A/A up to 3%; the 8-thread Q8_0 series is the noisy
  one): no loss, where 0.8–1.0× was predicted, the dequantization hidden behind the x4 kernel;
- against llama.cpp on the same Q4_K file: their decode 1.07× ours at 16 threads, 1.17× at 8 (one
  series each), their prefill 2.1× and 1.8× (their 8-bit activations).

## Q6_K on the CPU, and the Q4_K_M model (2026-09-24)

M2's second step: Q6_K weights against float activations, exact as Q4_K is (scalar the definition,
gguf-py bit for bit in `tools/check_dequant.py`, AVX2 and AVX-512 bit for bit the scalar), so that
Q4_K_M, the file people download, runs: for OLMoE's 64 experts llama-quantize puts Q6_K in
`output.weight` and in `attn_v` and `ffn_down_exps` of 8 of the 16 layers (17 tensors), Q4_K
elsewhere. The SIMD kernels unpack a block's 256 quants once into q − 32 as int8 (ik_llama.cpp's
DequantizerQ6K), then go as Q8_0 does: widen, convert, multiply by the sub-block's scale. No lookup:
a sub-block is 16 weights of 64 possible values, so building the table costs 4 multiplies per 16
weights before any lookup, against one conversion and one multiply without it.

**Prediction, written before measuring.** Microbenchmark (rows in cache): AVX-512 and AVX2 Q6_K
0.85–1.0× Q8_0's elements per second in dot_row and x4 (the per-element path is Q8_0's, the unpack
~20 instructions per 256 elements and 16 scale products per block on top), so above Q4_K on AVX2
(0.53–0.63×) and about equal to it on AVX-512; scalar 0.4–0.5× Q8_0 (the bit picking per element).
Real model: Q4_K_M reads per generated token ~10% more bytes than Q4_K (output.weight 80.6 against
58 MiB, half the layers' ffn_down at 0.82 against 0.56 bytes per weight: ~730 against ~660 MB), so
decode 0.90–0.95× Q4_K; prefill, where compute counts, 0.95–1.05×.

**Microbenchmark** (`sh tools/bench_kernels.sh`: native, a copy one byte longer, the machine's
marker, still machine; declared load 3.6 before and 2.5 after, ~0.25 of it System; median of 7 runs
of 200 ms; one core; rows in cache). Millions of elements per second, n = 2048 / 4096;
`build/bench_kernels/`.

| tier | dot_row Q8_0 | dot_row Q4_K | dot_row Q6_K | Q6_K / Q8_0 | x4 Q8_0 | x4 Q4_K | x4 Q6_K | Q6_K / Q8_0 |
|---|---|---|---|---|---|---|---|---|
| scalar | 1660 / 1921 | 888 / 827 | 553 / 573 | 0.30–0.33 | 2688 / 2705 | 1844 / 1836 | 1526 / 1527 | 0.56–0.57 |
| avx2 | 18521 / 19024 | 10296 / 10844 | 13792 / 15191 | 0.74–0.80 | 41812 / 37521 | 28673 / 28884 | 36798 / 36179 | 0.88–0.96 |
| avx512 | 21399 / 21711 | 17336 / 17902 | 18130 / 18527 | **0.85** | 44234 / 37448 | 37455 / 36934 | 39342 / 37436 | **0.89–1.00** |

Spread under 1.2% on the AVX-512 lines, 19–56% on AVX2's (a ratio there holds to ±20%). Against the
prediction: AVX-512 in the range (0.85–1.0), AVX2 a little below it on the single row (0.74–0.80,
**noise: remeasured 0.84–0.86**, §Two weight rows, LESSONS #150;
the 256-byte unpack, stored and reloaded, weighs on a row that has only one input to spread it
over; on x4 it is spread over four, 0.88–0.96), scalar below it (0.30–0.33: the bit picking per
element costs more than guessed; scalar is the definition, not a path any x86-64 CPU runs). Q6_K is
faster than Q4_K per element on both SIMD tiers (AVX2 1.3–1.4×, AVX-512 1.05×): Q4_K's AVX2 path
pays a shift or a mask, a conversion, a multiply and a subtraction per element, Q6_K's a conversion
and a multiply.

**Exactness on the real model.** The Q4_K_M file (`sh tools/quantize_q4k.sh m`, llama-quantize from
our Q8_0 without --pure: 4.21 GB, 17 tensors Q6_K), cut to 2 layers (where attn_v, ffn_down_exps
and output are all Q6_K), against transformers on its own dequantized weights (`make oracle-real`
with `REAL_MODEL_Q4KM`, in `make check`): 27 + 32 and 1024 + 32 tokens, greedy 32/32 on both, logits
within 1.3e-5 and 1.96e-4, argmax 1056/1056, every batch size bit-identical to one token per pass.
The whole model writes what the Q4_K writes on "The capital of France is".

**Real model, Q4_K_M against Q4_K** (`sh tools/race_q4k.sh 5 m`: container, all in RAM, prompt 512,
128 generated, median of 5 after a warm-up, order Q4_K_M Q4_K Q4_K Q4_K_M, then llama.cpp on the
Q4_K_M; marker held, still machine before each series, declared load 2.5 before and 3.2 after).
Tokens/s, series a / b; `build/race_q4km/`.

| threads | Q4_K_M prefill | Q4_K prefill | Q4_K_M decode | Q4_K decode | llama.cpp Q4_K_M prefill | llama.cpp Q4_K_M decode |
|---|---|---|---|---|---|---|
| 16 | 210.2 / 207.5 | 211.1 / 214.3 | **45.1 / 44.6** | 48.0 / 47.8 | 418.7 | 49.0 |
| 8 | 184.4 / 181.2 | 180.4 / 180.5 | **44.7 / 42.8** | 45.9 / 45.0 | 317.1 | 49.6 |

- decode **0.93–0.94× Q4_K at 16 threads**, 0.95–0.97× at 8 (A/A 0.5% and 2% for Q4_K, 1.2% and
  4.4% for Q4_K_M): inside the prediction (0.90–0.95×), the ~10% more bytes per token of the Q6_K
  tensors;
- prefill 0.97–0.99× at 16 threads, 1.00–1.02× at 8: the same, as predicted;
- against llama.cpp on the same Q4_K_M: their decode 1.09–1.10× ours at 16 threads, 1.11–1.16× at 8
  (on the Q4_K it was 1.07× and 1.17×), their prefill 2.0× and 1.7× (their 8-bit activations).

## Two weight rows at a time in the prefill's matmul (2026-09-24)

Marcello asked to measure again, reread the code and look for optimizations. What came out:

**The kernels measured again** (`sh tools/bench_kernels.sh`, second series, same rules; declared
load 2.5 before and 3.0 after): every AVX-512 line within 3% of the first series (the A/A). The
AVX2 dot_row lines of the first series were noise (spread 19–56%, now ≤ 2%): **Q6_K on AVX2 is
0.84–0.86× Q8_0 on the single row** (17089 / 17476 against 20142 / 20416 M elements/s), not
0.74–0.80×; x4 0.86–0.97×.

**The whole matrix** (new lines of `bench_kernels`: 1024 × 2048, 64 tokens through `tr_matmul`,
ms per token): Q8_0 0.0626, Q4_K 0.0629, Q6_K 0.0618 on one core. Three kernels 0.85–1.0× apart
in the microbenchmark cost the same in the matmul: decoding the weights is not what bounds the
prefill's matmul. What does: each `dot_row_x4` loads its four input rows (32 KB at 2048 columns)
for one weight row, 5 vector loads per 16 elements for 4 products each; the input comes from L2.

- **Rejected without building: a row decoded once per tile, then the F32 x4 kernel.** The F32 x4
  kernel itself (a new line of the microbenchmark) runs at 44779 / 33135 M/s at n = 2048 / 4096,
  Q8_0's x4 at 43926 / 36771: the decoding saved is worth nothing where the loads bound.
- **Kept: the sixteen Q6_K scales in SIMD** (one widen, convert and multiply instead of sixteen
  scalar ones; the same single rounding). Whole matrix, Q6_K on one core: 0.0551 → 0.0523 ms per
  token on the two-row path below, 0.0602 → 0.0590 on x4; Q8_0, untouched, the same 0.0491 in both
  sessions (the control).
- **Kept: two weight rows against the same four input rows** (`dot_row2_x4`, AVX-512, Q8_0, Q4_K,
  Q6_K; `tr_matmul` takes rows in pairs): each input vector loaded once for eight products, each
  sum still its own `dot_row` bit for bit. A/B in the container, the same binary with the kernel
  on and off (a switch that existed only for this measurement), order on off off on twice, marker
  held, declared load 2.7 before and 2.6 after; medians of 4, ms per token (the session before,
  without the SIMD scales, gave Q8_0 1.23× and 1.26×):

  | type | 1 core off | 1 core on | ratio | 16 cores off | 16 cores on | ratio |
  |---|---|---|---|---|---|---|
  | Q8_0 | 0.0607 | 0.0491 | **1.24×** | 0.0084 | 0.0065 | **1.29×** |
  | Q6_K | 0.0590 | 0.0523 | 1.13× | 0.0084 | 0.0071 | 1.18× |
  | Q4_K | 0.0625 | 0.0606 | 1.03× | 0.0081 | 0.0075 | 1.08× |

  Q4_K gains least: its 16-value lookup costs a permute per 16 weights of each row, and with two
  rows the permutes, not the loads, are what is left.

**Prediction for the engine, written before measuring.** The matmuls are ~85% of the prefill
(§Decode at context 2048, prefill zones); Q8_0 prefill 1.15–1.22× at 16 threads, Q4_K_M
1.04–1.08×; decode unchanged (one token takes `dot_row`).

**The engine** (`tools/ab_modes.sh 6`, in the container, `generate -p 512 -n 16 -t 16`, the new
binary against the one of commit e113d4f built in a worktree, the new one twice as A/A, first
mode rotating; marker held, declared load 2.6 before and 2.5 after; `build/ab_row2_engine/`):

| model | prefill new / new (A/A) | prefill old | ratio | decode new / old |
|---|---|---|---|---|
| Q8_0 | 252.9 / 241.3 tok/s | 200.6 | **1.20–1.26×** | 30.75 / 30.70 |
| Q4_K_M | 236.7 / 233.8 | 211.0 | **1.11–1.12×** | 41.9 / 41.4 |

Q8_0 at the top of the prediction, Q4_K_M above it (its Q6_K tensors, `ffn_down` of half the
layers and the output, gain 1.18×); decode the same, as predicted. Against llama.cpp's Q8_0 prefill
(415.8 at 16 threads in the container, §Speed — again) the gap goes from 1.8× to ~1.5× (not raced
again). Open: the same kernel on AVX2 (16 registers: eight accumulators leave no room, it would take
the lanes in two passes), and four rows at a time on AVX-512.

**The race with llama.cpp again** (`sh tools/race_llama.sh 5` on commit 2d709e0, the same rules as
§Speed — again; declared load 2.3 before and 2.6 after; `build/race_llama/`, the morning's series in
`build/race_llama-prev/`). Tokens/s, series a / b:

| prompt | threads | Trochilus prefill | llama.cpp prefill | Trochilus decode | llama.cpp decode |
|---|---|---|---|---|---|
| 512 | 16 | 252.5 / 255.2 | 365.1 / 348.9 | 31.0 / 31.6 | 31.9 / 30.8 |
| 512 | 8 | 223.9 / 210.1 | 244.8 / 247.5 | 30.6 / 29.9 | 32.4 / 32.1 |
| 2048 | 16 | 232.0 / 232.5 | 342.9 / 340.6 | 24.9 / 24.5 | 27.8 / 27.5 |
| 2048 | 8 | 196.9 / 198.3 | 242.5 / 239.4 | 23.3 / 23.9 | 27.5 / 26.8 |

llama.cpp over Trochilus (means of the two series; A/A at most 6.4%, Trochilus's prefill at 8
threads and 512): **prefill 1.41× at 16 threads and 1.13× at 8 with a 512 prompt, 1.47× and 1.22×
with 2048** (this morning 1.8× and 1.5×; our prefill 1.26–1.28× the morning's at 16 threads, a
little more than the engine A/B above: the morning's series ran on a busier machine); decode 1.00×
and 1.06× at context 512, 1.12× and 1.15× at 2048 (this morning 1.04–1.09× and 1.16–1.26×; the
long-context gap is the KV's bytes, §Decode at context 2048).

## Decode at context 2048: where the gap with llama.cpp is (2026-09-24)

Question 18, before writing any code: llama.cpp decodes 1.16× faster than us at 16 threads and 1.26×
at 8 at context 2048, 1.04–1.09× at 512 (§Speed — again). Profile by zone on the current binary
(`tools/profile_suite.py` on the two 2048 scenarios of `bench/scenarios-decode-context.json`,
native, a copy one byte longer; marker held, still machine, declared load 2.4 before and 2.2
after; median of 5; tokens identical in every run and thread count; `build/prof2048.txt`).

| zone (decode, Q8_0, context 2048) | 8 threads: ms/token | share | MiB read/token | GB/s |
|---|---|---|---|---|
| attention | 11.26 | 30.6% | 518 (KV) | 48 |
| expert_gate_up | 11.07 | 30.1% | 544 | 51 |
| expert_down | 5.58 | 15.1% | 272 | 51 |
| qkv_proj | 4.24 | 11.5% | 204 | 50 |
| lm_head | 2.11 | 5.7% | 104 | 51 |
| attn_out_proj | 1.83 | 5.0% | 68 | 38 |
| token | 36.83 (27.15 tok/s) | | 1718 | 49 |

At 16 threads the same within 2% (26.73 tok/s). Every zone reads memory at 38–51 GB/s of the ~57
the machine gives (question 4): the decode is the bytes it reads. Our KV cache is F32, 518 MiB per
token at 2048; llama.cpp's is F16 by default, 259. Halving the attention's bytes at the same speed
takes 5.6 ms of 36.8: **1.18×, the whole measured gap** (1.16–1.26×). The exact side has little left:
attention at 48 GB/s against the weights' 51 (≤ 0.7 ms, 2%), attn_out_proj at 38 (≤ 0.5 ms, 1.3%).
No code written: the lever is the KV at 16 bits, a declared mode that is not exact (Next steps
point 6, question 36, Marcello's call). With Q4_K weights (~660–730 MB per token instead of 1200)
the KV is ~43% of the decode at 2048 and the same lever gives ~1.25×.

## Skipping cached positions exactly: the premise on the real model (2026-09-24)

Marcello's decision of 2026-09-24: no KV at 16 bits with rounding; the long-context decode goes
faster only with logits identical to the byte. The exact way to read fewer KV bytes would be to
skip the positions that cannot change a bit: where `tr_expf(s − max)` is exactly 0, or where the
position's `e` is absorbed by its lane's running partial of the sum **and** every `a·v[d]` by the
running output (below half an ulp, in the definition's order), with cheap upper bounds on the
score (a norm per position, or the key's high 16 bits read for every position and the low 16 only
where needed) to decide without reading everything. Before any kernel, the premise.

How: `make attn-probe` builds a diagnostic engine (`build/probe/`, `-DTR_ATTN_PROBE`,
`tools/attn_probe.c`, never the engine) that writes every decode token's queries, keys and
values; `tools/attn_skip_report.py` replays the attention in float32 as `src/kernels/kernels.c`
defines it (16-lane dot, max, exp, 16-lane sum, division, output in increasing position) and asks
of every position whether each criterion holds; the bounds are rigorous (the bf16 high half with
the width of what was cut, plus 12 roundings of the dot), checked against the exact scores. Six
runs of 16 decode tokens on all 16 layers × 16 heads, native: `docs/ARCHITECTURE.md` as prose
(prompts of 1896 and 3993 tokens), `src/models/olmoe.c` as code (1965, 3970), the benches'
synthetic prompt (`-p 2048`, `-p 4000`); 7.8–17.4 M positions a run.

| run | s − max in (−2, 0] | (−5, −2] | (−10, −5] | (−15, −10] | (−20, −15] | (−25, −20] | ≤ −25 | exp exactly 0 | oracle: bytes left |
|---|---|---|---|---|---|---|---|---|---|
| prose ~2000 | 7.0% | 7.8% | 63.1% | 20.3% | 1.5% | 0.2% | 0.0% | 0 | 1.000 |
| prose ~4000 | 6.7% | 7.3% | 55.8% | 26.9% | 2.7% | 0.5% | 0.1% | 0 | 1.000 |
| code ~2000 | 6.9% | 9.5% | 61.3% | 20.4% | 1.8% | 0.2% | 0.0% | 0 | 1.000 |
| code ~4000 | 6.9% | 6.3% | 58.5% | 24.2% | 3.4% | 0.6% | 0.1% | 0 | 0.999 |
| synthetic 2048 | 6.5% | 10.9% | 62.8% | 17.2% | 1.9% | 0.5% | 0.1% | 0 | 0.999 |
| synthetic 4000 | 6.3% | 6.0% | 65.8% | 18.6% | 2.4% | 0.7% | 0.2% | 0 | 0.998 |

- **No score is ever 30 below its max** (exp reaches 0 at −103.97): the QK-norm keeps OLMoE's
  attention flat, two thirds of the positions sit between −10 and −5. Not one exact zero in
  65 M positions.
- **The oracle** (a position whose `e` leaves its lane's partial unchanged and whose `a·v` leaves
  all 128 outputs unchanged, known with every value in hand) skips at most 0.2% of the bytes. It
  bounds every scheme of the family: absorption with the largest |v| of a position (≤ 0.2%), keys
  split in high and low halves (the high half decides nothing: 0.3–0.6% of the bytes the other
  way), values split (the high half decides the new output for ≤ 0.6% of the positions that need
  V), norm, block and low-rank bounds.
- **Closed for this model, before any kernel.** A model without QK-norm, with sharper attention,
  may differ: the probe and the report run on any model with this attention, so a new model gets
  the same check (`docs/COMMANDS.md`).
- **An adversarial review of the proofs** (an Opus agent, its own float32 emulation on the same
  dumps, 4.6 M positions: oracle 0.07–0.37% of the bytes, largest gap max − s in a head median
  13.1, p99 31, max 40.2) found the skips exact only with these conditions, the ones a kernel
  would need if a model with sharp attention brings the idea back: a sticky non-finite flag per
  layer and head set at KV write (a skipped position hides the reference's `0·inf = NaN`); the
  default MXCSR asserted (under FTZ/DAZ `out` can become −0 and adding +0 is no longer the
  identity); absorption as `fl(P + e_hi) == P` (the "≤ half an ulp" form is wrong:
  P = 1 + 2^-23, e = 2^-24 gives 1.0000002; 0 violations of the right form in 53 M brute-force
  pairs); the V test two-sided (one-sided fails at a power of two: 173 cases); the largest |v|
  taken as an integer max of the bits (a `max_ps` drops a NaN depending on operand order); the
  lane of a position its absolute `t % 16` (compacting the kept positions changes the sum); a
  position absorbed in the sum but not in V re-reads its key's low half for the exact `a`; the
  bound from the high half of a key widened by γ12 · Σ|q_i|·max(|H_i|, |T_i|) + 128·2^-150 and
  `scale` rounded outward (without the margin the bound undershoots the computed score in 3.5%
  of adversarial cases, by up to two floats), and a NaN whose payload sits in the low 16 bits
  treated as unbounded (it truncates to ±inf). Scripts: the session's scratch `opus3/brute.py`,
  `emu.py` (not kept: the conditions are here).
- **A second review on the whole grid** (a Sonnet agent, 76 M query-position pairs): the largest
  gap max − s anywhere is 44.5; a per-position norm bound certifies 2 of the 452 475 positions
  below max − 20 (norms vary little: coefficient of variation 0.118); block min/max envelopes
  catch 0–0.03% of them; a low-rank bound (the top-32 singular subspace of the keys plus
  Cauchy-Schwarz on the rest; 0 violations in 76 M) catches 57% of them but costs 132 bytes a
  position against a 25.8% break-even, and only two heads (L0H11, L8H10) skip 8–40% there — at
  max − 20, which is not an exact threshold. The context stops at 4096 on this model
  (`olmoe.context_length`): the 4000-token runs are its longest.
- **Lossless compression, measured on the same dumps** (both reviews): sign and exponent carry 3.5
  of their 9 bits of entropy per channel (the exponent given its channel 2.56–2.58 bits), the
  mantissa none (6.97 of 7 and 7.8–7.95 of 8 bits); zlib on byte planes 13.6–13.9% on K, ~17% on
  V (up to 83% on layer 0 of code: repeated tokens, see the per-token row below).
  Entropy coding would take ≤ 17% and does not decode at memory speed; a fixed 28-bit layout
  (3 bytes of sign and mantissa, a 4-bit exponent offset per channel in blocks of 64 positions,
  code 0 for zero and subnormals, an escape for the 0.7% of block-channels whose exponents span
  more than 15, and for inf and NaN) reads 12.5% − 0.4% of headers = **12.1% fewer KV bytes**:
  ~1.4 ms a token at 2048, ~2.8 at 4000 (**~1.04× and ~1.06×**), if the decoding (~8 AVX2
  instructions per 8 floats) stays under the memory time. Not built yet: the CPU-only lever,
  after the GPU's numbers (question 51).

The other ideas of the session (orchestrator's, and two agents' asked for different hints), by
bytes first; a KV at context 2048 is 518 MiB a token:

| idea | exact? | bytes or factor at 2048 | estimate | where |
|---|---|---|---|---|
| skip positions (every variant above) | yes | ≥ 0.998 of the bytes | none | closed above |
| KV at 16 or 8 bits, per-head Q8 | no | 0.5, 0.25 | 1.18×, 1.25× | excluded by Marcello's decision |
| lossless compression of K and V | yes | sign and exponent per channel only, the mantissa has no entropy: ≤ 12–17% less | ≤ 1.04–1.05× | measured below |
| speculation from the prompt | yes (batch = token by token) | KV read once per pass: bytes ÷ tokens per pass | depends on the task | question 52 |
| attention on the GPU (RTX 4070 Laptop, ~256 GB/s) | yes, if every float op carries `.rn` (PTX never fuses those) and `tr_expf` is ported whole | same bytes, 4-5× the bandwidth | 1.28× at 2048, ~1.55× at 4000 | question 51 |
| a slice of the KV kept in L3 across tokens (weights streamed non-temporal) | yes | 64 MB of L3 holds ≤ 12% of the KV at 2048, and a cyclic read of 518 MiB through LRU hits nothing unless a fixed slice is pinned | ≤ 1.02–1.03× | not built |
| attention at 48 → 53 GB/s | yes | same bytes | ≤ 3% (§Decode at context 2048) | not built |
| self-speculation with a cheap draft (layers skipped) | yes | a draft still reads the 1200 MB of weights, a pass of k rows the union of their experts (~2.8× measured at ~4 rows, below) | a loss on an MoE | not built |
| the dense weights (QKV, output projection, output head: ~395 MB a token) in VRAM beside the KV | yes, same `.rn` rule | ~7.7 ms a token off the RAM at 2048 | with the KV on the GPU, ~22 ms a token, **~1.6×** (agent's estimate) | M3's first piece, after question 51 |
| the prompt lookup extended cyclically (a draft that overlaps the tail continues with its period) | yes (drafts are verified) | KV ÷ tokens per pass, in loops only | on the dumps' 87 decode steps 138 drafts accepted instead of 66, all in loops (the synthetic prompt falls into a period-2 loop by its 2nd-3rd token, prose ~4000 into a near period 4) | cost small; not built |
| drafts from a copy pointer (the argmax of a few induction heads) | yes | same | 74 accepted instead of 66 on 87 steps, same precision: weak | not built |
| layer 0's keys (before RoPE) and values from one row per distinct token (a pure function of the token) | yes (RoPE on the fly is the same computation) | 0.95–0.96 of the KV on prose and code (728 distinct tokens of 1896, 505 of 1965) | ~1.4% | not built |
| a key's top 3 bytes always, the low byte on demand (graded precision) | yes | with a ≈ 1/N every product needs ~13 bits: 4–12% of elements undecided, so almost no row is | ≤ 1–2% | dead by arithmetic |
| keys and values recomputed from the layer's input (8 KiB a position instead of 16) | yes | half the bytes, ~34 GFLOP a layer a token | a loss | dead |

## The decode's attention on the GPU: the premise (2026-09-24)

Question 51, measured before the engine by an agent on `tests/bench_gpu_attn.c` (native; the
driver `nvcuda.dll` loaded at run time, no toolkit; the kernels PTX written in C and compiled by
the driver; RTX 4070 Laptop, 8 GB GDDR6, sm_89, WDDM, driver 595.97). Three kernels a layer:
scores (one thread per position, keys stored position-minor so a warp reads in a row, the new key
and value appended in the same launch), exponentials (`tr_expf`'s own double arithmetic), values
(32 dims a block, the 16-lane sum by shuffles in `tr_lane_combine`'s tree, the values streamed
through shared memory, one warp adding them in position order). Every float operation carries
`.rn`; no `fma`, `.approx` or `.ftz`.

**Bits**: all six probe runs, every layer, head and query: 25 344 head outputs × 128 floats against
`tr_attention_group` (scalar table), **0 differ**; 164 synthetic cases (1 to 4096 positions across
every block border, −0, equal scores, scores landing on `tr_expf`'s exception arguments; counters:
90 exceptions, 361 605 underflows, 93 222 subnormal `e`) 0 differ; the GPU's `tr_expf` equals the
CPU's on all 2^32 floats. Mutations patched into the PTX at run time, all red: `.rn` dropped
(321 124 floats differ), the dot's tree swapped, the sum's shuffle order reversed, the output
started at −0 (red on the synthetic −0 cases), the exception table removed.

**Time a layer** (16 heads, the prose ~4000 dump, 1 GiB of VRAM rotated so L2 is always cold;
median of 64 calls, or 64 tokens × 16 layers for the bursts):

| a layer | 2048 positions (CPU 704 µs) | 4000 positions (CPU 1415 µs) |
|---|---|---|
| kernels only | 156.7 µs (214 GB/s) | 290.8 µs (225 GB/s) |
| a plain read of the same bytes | 140.3 µs (239 GB/s) | 266.4 µs (246 GB/s) |
| round trip with the copy engine, back to back | 214.6 µs | 348.8 µs |
| round trip in zero copy (the kernels read and write pinned memory), back to back | 157.9 µs | 292.5 µs |
| **the real pattern**: 1.6 ms of CPU between calls, no keep-warm | 390 µs (p90 754), and rising over a 6 s run | 390–919 µs |
| **the real pattern with keep-warm + zero copy** | **184.6 µs** (flat over 6 s) | **318.6 µs** |

- **A laptop GPU sleeps between two layers.** At 2048 its utilization is 9–15%: the driver steps
  down to P3–P4, the SM clock from 1605 to 255–345 MHz, the memory from 8101 to 6001 MHz, and the
  attention gets 2–3× slower. One warp on a second stream spinning on `%globaltimer` through the
  CPU's gap keeps P0 at 2580 / 8101 MHz: +12–17 W while decoding.
- **Projected token**: 36.8 → 28.5 ms at 2048 (**1.29×**), 48.2 → 30.6 ms at 4000 (**1.57×**) —
  the prediction was 1.28× and ~1.55×. With the copy engine 1.25× and 1.53×; without keep-warm
  ~1.14× and unstable.
- The floor of any GPU call (an empty kernel and its sync) is 6.4 µs; the exponentials cost 9–14 µs
  a layer (FP64 at 1/64 rate): an FP32 correctly rounded exp with the double one as the fallback
  of the hard cases would take most of it back.

**The dense weights too** (M3's first piece, an agent on `tests/bench_gpu_q8.c`): the Q8_0
matrix-vector product with `dot_row`'s bits (16 threads a row, thread l takes elements l and
l + 16 of every block, the 16 lanes combined by shuffles in the tree's order) on the GGUF layout as
it is: **0 of 1 091 008 floats differ** over 486 products (all 64 attention matrices, the output
head, synthetic rows with subnormal and zero scales, −128 quants, ±0, inf, NaN); three mutations
red (no `.rn`: ptxas does fuse the pair into FFMA, 75% of the floats change). 97% of a plain VRAM
read on the large shapes (head 50304 × 2048: 452 µs, 242 GB/s). A token's dense weights (16 fused
QKV, 16 output projections, the head): **2.38 ms on the GPU** with compute-engine-only round trips
against ~7.7 ms on the CPU. The copy engine costs 35–60 µs a call on small shapes; a gap over
~1.5 ms slows the next call (2048 × 2048: 38 → 67 µs after 5–20 ms).

**Where it leads** (an agent's model, `m3plan/model.py` in the session's scratch, from the numbers
above; ms a token, × against today's CPU on Q8_0):

| placement | 512 | 2048 | 4000 |
|---|---|---|---|
| CPU today | 27.5 | 36.8 | 48.1 |
| (a) the attention on the GPU | 25.8 (1.07×) | 28.7 (1.28×) | 31.2 (1.54×) |
| (b) + the dense weights | 20.5 (1.34×) | 22.5 (1.64×) | 25.0 (1.92×) |
| (c) + the experts that fit in VRAM, the rest on the CPU | 7.3 (3.8×) | 9.3 (4.0×) | 11.8 (4.1×) |
| (d) Q4_K_M whole on the GPU | 4.8 | 6.8 | 9.4 |

The model's own falsifiers: the exact GEMV at 180–235 GB/s (measured 242 on the head), the
per-token overhead of ~146 dependent kernels (0.4–1.2 ms), the round trip (20–80 µs host-driven,
5–15 through mapped flags), the burst penalty, usable VRAM (~7 GiB), cold experts per token at
87–95% residency. The exact mix of the experts stays in one place in the definition's order (the
8 outputs in increasing id); NaN payloads differ between x86 and NVIDIA, so the bytes are the same
while every value is finite.

## The decode's attention on the GPU, in the engine (2026-09-24)

`src/backend/gpu_attn.{h,c}` in the engine (DONE), exact through it (`tools/gpu_exact.sh full`:
logits of 2000 decode positions, tokens after 4000, a speculative run, all identical; the
`change` stage of `decode_context.sh`: logits of 600 positions and tokens after 4000 identical to
commit 718a84c). Then `PROF_BEFORE=build/trb.exe sh tools/decode_context.sh change build/trb.exe 6`
(native, marker held, background load 2.57 then 1.25 logical processors, 6 rounds, before =
718a84c, after = the GPU by default).

**First A/B, without the warm-up below** (decode tok/s at 8 threads forced, median of 6; A/A ≤ 2.2%
at 2048 and 4000, 6.0% at 512):

| context | before | after | after / before (the four pairs) |
|---|---|---|---|
| 32 | 36.81 / 36.87 | 37.23 / 36.88 | 1.000–1.011×: not distinguishable |
| 512 | 32.60 / 34.56 | 36.28 / 35.64 | 1.031–1.113× |
| 2048 | 26.71 / 26.52 | 32.77 / 33.42 | **1.226–1.260×** |
| 4000 | 21.51 / 21.12 | 27.61 / 27.02 | **1.256–1.307×** |

The prefill not distinguishable (A/A up to 7.3%). The profile by zone (median of 5, 8 threads, the
two profiles not alternated): the attention 11.76 → **3.18 ms a token at 2048** (199 µs a layer,
170 GB/s, as the premise), but 19.72 → **10.43 ms at 4000** (652 µs a layer, twice the premise's
305–319).

**Where the other half went at 4000: the GPU was cold.** Per call of the engine's attention zone
(`generate -p <n> -n 24..48 --profile`): 195–198 µs from 2040 to 2100 positions, then 382 at 2300,
795–951 at 2600, 574–1079 at 3000, 685 at 4000, and 214 or 359 at 2048 in two runs of the same
command: not a function of the context. The module alone (a scratch program through the public API,
the decode's pattern, 12 tokens a context) is linear and steady: 166 / 183 / 205 / 236 / 302 µs at
2048 / 2300 / 2600 / 3000 / 4000 — the same on the probe's real prose and synthetic data, with 8
threads burning cores (303 at 4000), with 8 threads reading 64 MiB of RAM in every gap (+20 µs), and
the engine the same with 16 or 8 pool threads. What differs is the start: after a prompt of seconds
the GPU has idled at P5–P8, and its clocks take hundreds of ms to climb back (sampled: 1605 MHz for
~400 ms, then 2565); over 40 decode tokens that is most of the run (436–694 µs a call), over 400 it
is not (**315 µs a call at 3000–3400**). The decode_context runs generate 48 tokens: they measured
the climb. **The fix** (`tr_gpu_attn_warm`): the last pass of every multi-token eval (the prompt's
last, a speculative check) launches the keep-warm warp after each layer's write, at most 200 ms at a
time (Windows resets a kernel near 2 s), ending at the first decode call. 40 tokens after the
prompt, alternated with the binary without it: **3000: 508–578 → 260–264 µs a call, 29.1–30.7 →
33.3–35.1 tok/s; 4000: 698–812 → 322–323 µs, 27.0–28.2 → 34.1–34.3 tok/s.** Bits unchanged
(`test_gpu_attn` runs its cases with that warp napping; `gpu_exact.sh quick`).

**The A/B again, with the warm-up** (the same command, 6 rounds; background load 1.23 logical
processors after the last run; results in `build/decode_context-gpu2/`):

| context | 8 threads forced: after / before (the four pairs) | measured width: after / before | prediction |
|---|---|---|---|
| 32 | 0.960–1.008× (A/A 2.6%): not distinguishable | 0.982–1.003× | — |
| 512 | 1.039–1.068× | 1.044–1.069× | — |
| 2048 | 1.195–1.276× | **1.307–1.324×** | 1.28× |
| 4000 | 1.431–1.508× | **1.535–1.584×** | 1.57× |

Worst A/A 4.0% (8 threads) and 2.1% (measured width). The profile by zone (8 threads, median of
5): at 2048 the token 35.21 → 27.92 ms, the attention 10.51 → 3.20 ms (170 GB/s); at 4000 the
token 44.78 → **29.75 ms (1.51×)**, the attention 20.29 → 5.20 ms (203 GB/s, 325 µs a layer: the
premise's). The queue fix of LESSONS #159 changes no decode time (4000, 40 tokens, alternated:
329–334 µs a call before it, 323–330 after).

## The decode's attention, one position at a time (2026-09-24)

Why the CPU's decode attention reads its KV at 46-48 GB/s when a plain read of the same bytes gets
52-54 (an agent on `tests/bench_attn_bw.c`: `bench_mem`'s cache shape, 64 MiB of other memory
through the caches between layers, every run on a different layer so no layer survives in L3 —
two runs on the same layer back to back read at 55 GB/s instead of 47; the machine was loaded by
other agents, so steps paired one by one and only the quiet ones kept). Timed inside each worker at
2048 on 8 threads (µs a layer): the engine K 351.9, softmax 18.9, V 319.4, 36.4 waiting at the end;
a plain read K 323.0, V 297.1, 31.2 waiting. **The gap is in the K and V passes, and it is the x4
kernels' order**: `dot_f32_x4`/`axpy_f32_x4` read four rows 512 bytes apart a cache line of each
in turn (lines 0, 8, 16, 24, 1, 9, ...), which the prefetcher does not follow as one stream; the
same reads in that order with no arithmetic run at 0.82-0.86× of the engine.

| variant (8 threads, 2048 unless said; ratio = engine time ÷ variant time, median of the quiet steps) | ratio |
|---|---|
| engine against its own copy (A/A) | 0.995-1.007 |
| **one position at a time** (`tr_attention_head`) | **1.101** [1.066-1.134]; 1.130 in a busier session; AVX2 tier 1.093; at 4000 **1.133** |
| x4 kernels + a prefetch of every line 4 KiB ahead | 1.084-1.145; at 4000 1.124-1.151 (2 and 8 KiB, `nta`: the same; 16 KiB, `t1`/`t2`: worse) |
| one position at a time + prefetch | 1.095: nothing more |
| exponentials hidden (fused passes; two heads interleaved) | no gain: other threads keep the bus busy meanwhile |
| one position at a time on 4 / 6 / 8 / 12 / 16 threads | 1.079 / 1.096 / 1.100 / 1.102 / 1.103 |
| the K pass split over threads, then softmax and V per head | worse (1.068-1.085) |
| a plain read of the same bytes (the ceiling) | 1.100-1.160 |

Every variant gave the engine's bits in every layer at 1, 5, 65, 2047, 2048 and 4000 positions on
the AVX-512 and AVX2 tiers; a mutant (V blocks reversed) differs in every layer. Huge pages need
SeLockMemoryPrivilege (`VirtualAlloc(MEM_LARGE_PAGES)` fails with 1314): not available. What is
left after the fix (51.4-52.1 GB/s against 52.3-53 for the plain read) is the end-of-layer wait
and the 1 MiB streams of a head. **Built**: `tr_attention_group` takes a group of one position by
position (`src/kernels/kernels.c`; test: `test_attention_group` covers both branches). Prediction
for the engine on a still machine, written first: the attention zone 11.26 → 10.0-10.2 ms at 2048
and 22.64 → 20.0-20.2 at 4000, the token **~1.03× and ~1.05×**.

**Measured: not distinguishable on a still machine.** `TR_GPU=0 PROF_BEFORE=build/trb.exe sh
tools/decode_context.sh change-short build/trb.exe 6` (the short protocol's first run; logits of
600 positions and tokens after 4000 identical to 718a84c; background load 1.12 logical processors
after the last run; `build/decode_context/`), decode at 8 threads, after / before over the four
pairs: 512 1.004–1.047×, 2048 0.959–1.002×, 4000 0.978–1.007×, worst A/A 2.9%. The profile by zone:
the attention 2.68 → 2.62 ms at 512, **11.00 → 10.02 ms at 2048** (49.4 → 54.2 GB/s), 20.67 →
20.48 at 4000. On a still machine the engine before already read its KV at 49–52 GB/s, not the
46–48 of the bench's loaded sessions: the x4 order costs little when nothing else fights for the
memory, and the bench's 1.10–1.13× (1.15–1.25 per repetition under more load) was mostly that
contention (LESSONS #160). Kept: exact, never slower, and it helps on a loaded machine.

## The KV packed in 28 bits: the premise (2026-09-24)

An agent on `tests/bench_kvpack.c` built the lossless format of §Skipping cached positions exactly
(blocks of 64 positions: a base exponent per channel, a 4-bit code per value — 0 for zero and
subnormals, 15 an escape to a list of raw bits —, 16 low bits and 8 of sign and high mantissa),
decoded four positions at a time into a scratch that the tier's own `dot_f32_x4`/`axpy_f32_x4`
then read. **Bits**: every stream of the six dumps packs and unpacks to the same bytes; 25 344
decode queries through the packed attention give `tr_attention_group`'s bits (also under
`TR_CPU_MAX=scalar` and `avx2`); a mutation of the decoder (base off by one) red in 4096 of 4096;
a synthetic block with every float class unpacks exactly, 5 escapes as designed. **Bytes**:
3.53-3.55 a value, **1.128-1.132× fewer** than F32; 0.38-0.43% of the values escape. **Time, on a
still machine** (question 57, `sh tools/bench_native.sh bench_kvpack time --run all`, 8 threads,
marker held, load 2.0–2.4 logical processors): one decode token's attention over all 16 layers,
packed against F32, **0.97–1.05×** (prose 1.00 / 0.97, code 1.03 / 1.05, synthetic 1.01 / 1.02 at
~2000 / ~4000 positions; spreads 7–64%). The packed layout reads 44–45 GB/s where F32 reads 49–51:
decoding eats the 1.13× in bytes. Prediction (1.00–1.08×) held at its low end. **Closed as no** on
the CPU; the GPU does the decode's attention anyway.

## An exact exp in float32 only, for the GPU (2026-09-24)

`tr_expf` computes in double: on this GPU FP64 runs at 1/64 rate, fine for a decode token's 32 K
exponentials, not for a GPU prefill's ~2.5 G (softmax and SiLU of a 4000-token prompt). An agent
wrote, in `tests/bench_expf32.c`, a correctly rounded exp with float32 and int32 operations only
(so, being correctly rounded, the same function as `tr_expf`): reduction on 256 intervals with
`fma` (or ln2/256 in three 8-bit pieces without it), the 2^(j/256) table as 256 float pairs (2 KB),
a degree-3 minimax polynomial, the result as a float-float pair and a rounding test with an
absolute margin D = 2^-39 on y (the analytic worst case 2^-39.9 with fma, 2^-39.5 without);
results below 2^-126 rounded on the subnormal grid; the ambiguous ones to a slow path in 64-bit
fixed point on 32-bit limbs (Taylor of degree 9, margin 8 units of 2^-62 against an error under
3.4). Every constant computed at 300 bits (mpmath) and checked by the bench.

- **All 2^32 floats, both variants: 0 differ from `tr_expf`** (run as 26 one-thread slices of
  ≤ 15 s while this session's measurement held the machine); the slow path alone over its whole
  range [−104, 88.72], 2 239 853 081 floats: 0 differ; the largest error of the fast pair 0.32–0.37
  of D. **9.4e-6 of the arguments** fall back (2 in 65 536 softmax arguments).
- CPU, one thread (indicative: the machine was measuring): **3.10–3.22 ns with fma against
  `tr_expf`'s 3.59–3.60**, 5.6 ns without fma; the slow path 82 ns. On the GPU ~31 instructions
  (21 FP32) against `tr_expf`'s ~16 FP64 ops, ~1000 FP32 issue slots at 1/64.
- Mutations, all seen: a table entry one ulp up (8864 wrong), D = 2^-45, the slow path's 1/6!
  term dropped, the subnormal grid off by a binade, the low product dropped; the table's low half
  one ulp up is an equivalent mutant (inside D's slack, every result still correctly rounded),
  seen only by the constants check, which exists for that (LESSONS #83).
- Not in the engine: the GPU prefill will use it (with `fma.rn`, the table in shared memory; NaN
  payloads differ between x86 and NVIDIA). The gate builds the bench; its exhaustive run
  (`--no-timing --slow-all --error`, ~40 s on 8 threads) enters the gate with its first user.
- **In SIMD (question 58, 2026-09-24)**: the fma variant as an AVX-512 tier (16 lanes) and an AVX2
  tier (8), lane by lane the scalar operations in the same order, the table by gathers, the
  unsettled lanes through the scalar slow path. **All 2^32 floats: 0 differ from `tr_expf` in
  both tiers**; 21 119 lanes through the slow path (the scalar fma variant's 21 114 + 5 below
  2^-126: the same decisions). Mutations seen red: the slow path skipped (7: 492 wrong on 1/64 of
  the floats), a table entry one ulp up (1: 8 871 wrong). **Cost on a still machine, one thread,
  softmax arguments (−12..0): AVX-512 0.733 ns a value, AVX2 0.802, against `tr_expf`'s 3.52
  (4.8× and 4.4×)**; over −104..88.7 1.62 and 1.73 ns (the subnormal branch and the special lanes).
  Prediction (the softmax and SiLU zones 3–8× faster, the prefill 1.03–1.06×) fell: with question
  39's zone times, the exponentials take ~74% of the softmax zone and ~54% of the SiLU zone, so the
  zones would run 2.4× and 1.7× faster, and **the prefill ~1.02× at 2048 and ~1.03× at 4000** —
  the range Marcello closed question 39 on. Kept in the bench as the GPU's reference; not wired
  into the CPU engine.

## Speculation at long context (2026-09-24)

Question 52. The verification of a speculative pass reads the KV once (`tr_attention_group`, up
to 16 rows, `OLMOE_ATTN_QUERIES` = `TR_LOGIT_ROWS_MAX`), so tokens per pass divide the KV bytes of
a token. Prediction, written first: 1.5–2.5 tokens a pass on tasks that quote or rework the
prompt, ~1.1 on free prose. Measured by an agent, native, `run -n 256 -t 8 -c 4600` with and
without `--spec 8` (one run each: the pass counts are deterministic, the tok/s were taken on a
machine shared with other work and are noisy); every `--spec` output identical to its twin
(`cmp`, 7 of 7). The two task prompts are the session's prompts with an instruction appended.

| prompt | context | tokens / pass | drafts accepted | tok/s without → with (noisy) |
|---|---|---|---|---|
| prose, free continuation | 1896 | 1.04 | 10 of 48 | 26.03 → 24.94 |
| prose, free continuation | 3993 | 1.13 | 29 of 64 | 19.38 → 20.60 |
| prose + "summarize section by section" | 4034 | 1.08 | 18 of 57 | 18.47 → 20.56 |
| code, free continuation | 1965 | 1.27 | 55 of 125 | 24.92 → 26.58 |
| code + "rewrite renaming every identifier" | 2037 | 1.57 | 95 of 151 | 24.36 → 27.78 |
| code, free continuation (repetitive loading boilerplate) | 3970 | **3.29** (3.39 with `--spec 15`) | 179 of 227 | 20.57 → **34.12** (1.66×) |

- The prediction held on the rewrite (1.57) and on free prose (1.04–1.13); summarizing quotes
  little (1.08). Where the text repeats, speculation already gives what the GPU would: 1.66× at
  4000 on the boilerplate.
- **The union of the experts eats the gain, not the KV**: on the clean pair (code at 4000), a pass
  of ~3.9 rows reads the KV once but ~2.8× one token's expert weights (from the net 1.66× against
  3.29 tokens a pass, bytes over bandwidth); `--spec 15` the same (2.76×, 1.73× net). Consistent with
  §Adaptive draft (one more row 13.7–17.6 ms of a 31.3 ms pass). On free prose the lever is nil.
- `--spec 16` is refused by design: a pass holds at most 16 logit rows (1 + 15 drafts).
- The synthetic prompt of the benches falls into a period-2 loop by its 2nd–3rd decode token
  (found on the probe's dumps): speculation measured on it measures the loop, not the lever.

## A Q4 draft against the exact Q8_0: how often the same token (2026-09-24)

Question 53: a Q4 copy of the model drafts, the exact engine verifies a pass (the accepted tokens
are exact by construction). Prediction, written first: top-1 agreement 95–98% on prose, 97–99% on
code; 6–7.5 tokens a pass of 8 on prose. `tools/draft_agreement.sh` (container, correctness only):
three real texts of this repo (prose: `docs/ARCHITECTURE.md`; code: `src/models/olmoe.c`; Italian
prose: MEASUREMENTS lines 1633–1760), the first 1024 tokens as prompt, the Q8_0's greedy 512
after it, then the logits of all 1536 positions (`logits -b 1`) of the Q8_0, the Q4_K and the
Q4_K_M; `tools/draft_agreement_report.py` compares them. The Q8_0's greedy through `logits -b 1`
is `generate`'s token for token (the report fails otherwise).

| draft | text | agreement: real text / Q8_0 trajectory | KL(Q8_0‖draft), trajectory | tokens a pass of 8 (trajectory / real text) |
|---|---|---|---|---|
| Q4_K | prose | 84.2% / **88.7%** | 5.2e-2 | 5.52 / 5.04 |
| Q4_K | code | 88.5% / **94.7%** | 4.3e-2 | 7.23 / 5.79 |
| Q4_K | Italian | 82.8% / **90.6%** | 5.8e-2 | 6.40 / 4.69 |
| Q4_K_M | prose | 86.8% / 90.4% | 4.0e-2 | 6.04 / 5.33 |
| Q4_K_M | code | 90.7% / 93.6% | 3.4e-2 | 6.92 / 6.36 |
| Q4_K_M | Italian | 84.3% / 92.8% | 4.3e-2 | 6.74 / 4.85 |

- **The prediction fell**: 88.7–94.7% on the trajectory, 83–91% on real text; KL 3.4–5.8e-2,
  4–6× llama.cpp's int8 activations (9e-3): 4.5-bit weights move a distribution much more than
  8-bit activations. The Q4 is not worse, it chooses differently where the Q8_0 is unsure: next-
  token accuracy on the real text is the same (Q8_0 35.7 / 54.5 / 38.6%, Q4_K 36.0 / 54.5 / 38.7%);
  agreement is 35–58% where the Q8_0's top-2 margin is under 0.1 nats, 54–65% at 0.1–0.5, 99.6–100%
  above 2 nats. The code trajectory repeats itself (66.6% of its 4-grams seen before; prose 8.6%,
  Italian 18%): its 7.23 is inflated.
- Cutting the draft where its own margin falls (τ = 0.5 / 1 / 2 nats) trades tokens for rows at
  about the same ratio: prose 4.28 tokens in 5.03 rows, 3.35 in 3.78; it does not pay in bytes
  below.
- **The lever survives the missed threshold**, because a pass's experts overlap more than predicted
  (§Experts read by a pass of k rows: 9 rows read 3.52× one token's experts, not ~5×). A token reads
  ~1.26 GB (experts 855 MB, attention 285, output 109); a verification pass of 9 rows ~0.40 + 0.855
  × 3.52 = 3.41 GB: at 5.52 tokens a pass **0.62 GB a token, 2.0× fewer bytes** on prose (2.4×
  Italian, 2.7× code). The draft costs its own time: the Q4 whole on the GPU (3.9 GB), 8 serial
  draft tokens ~30–40 ms against a ~80 ms pass (28 ms + 2.52 × 21 ms of experts, §Adaptive draft):
  ~1.3× serial, up to ~1.9× with the next draft overlapped with the verification.

## Experts read by a pass of k rows, and which fire together (2026-09-24)

Questions 54 and 62 (routing). `tools/route_union_report.py` on the six real-text route traces of
question 44 (`build/route/`: C ×2, Python, shell, English and Italian prose; 6 770 tokens). Union
of the experts of k consecutive tokens per layer, in units of one token's 8, mean over windows and
layers; prediction, written first: 1.7–1.8× at k = 2, 2.8× at 4, 4.2–5× at 8, 5.5–6.5× at 16.

| k | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| union (mean of 6 traces) | 1.00 | **1.55** | 1.97 | **2.33** | 2.89 | **3.34** | 3.52 | 3.97 | **4.42** |
| range over the traces | | 1.50–1.62 | | 2.18–2.59 | | 3.09–3.86 | | | 4.00–5.11 |
| k random 8-subsets of 64 | 1.00 | 1.88 | 2.64 | 3.31 | 4.41 | 5.25 | 5.59 | 6.39 | 7.06 |

- **Below the prediction at every k** (3.34× at 8 against 4.2–5): consecutive tokens reuse experts
  far more than chance, code more than prose (C 3.09–3.24 at 8, English prose 3.86). The 2.8× at ~4
  rows of §Speculation at long context was a time ratio, not a count: it carries the rows' compute.
  A pass of k rows costs ~0.4 GB + 0.855 GB × union(k) in bytes: this table is the cost side of
  every speculative or batched decode on this model.
- **Co-activation is strong and grows with depth**: per layer, the pairs that fire together more
  than twice as often as independence gives are 6–8% at layers 0–1 and 17–21% at 8–15; 18–50% of
  pairs fire together under half as often; an expert's top-3 partners hold 18–28% of its
  co-firings (uniform 4.8%); the strongest pair's lift 6 at layer 0, 17–67 at layers 8–15. Ground
  for placement and file order (M1 point 4, mbolt): experts that fire together read together.

## The model read like a genome: weights, KV, routing (2026-09-24)

Questions 56 and 62. `tools/weights_genome.py` reads every block of a GGUF once: entropy of the
codes (order 0 per tensor; given a per-block context: Q8_0's 16 magnitude classes with their own
cost, Q4_K's 6-bit sub-block scale; given the input column, one table per column, table not
counted), entropy of the f16 scales, all-zero blocks, blocks repeated (whole, codes only, codes up
to sign, anywhere in the file), rows repeated. Predictions, written first: Q8_0's codes 6.3–7.0
bits of 8 (12–20% of the code bytes), Q4_K's 3.5–3.8 of 4 (5–12%); ≤ 0.1% of blocks repeated.

| | OLMoE Q8_0 (7 009 MiB scanned) | OLMoE Q4_K (3 711 MiB) |
|---|---|---|
| code bits, order 0, per tensor | **7.59–7.69 of 8** | **3.83–3.87 of 4** |
| gain of the per-column table | experts ≤ 0.015 bits; attention 0.10–0.30, layer 0's `attn_q` 1.01 | experts ≤ 0.01; layer 0's `attn_q` 0.70 |
| gain of the per-block context | none (the class costs more than it saves) | 0.005 |
| scale bits (f16) | 7.1–8.9 of 16 | 8.8–11.9 of 16 (d and dmin) |
| whole file at these entropies | **93.1%** (codes 96.0%) | **96.1%** (codes 96.6%) |
| all-zero code blocks | 0 | 2 140 (token_embd 1 976, layer 0 gate and up 82 each) |
| blocks repeated anywhere | 16 686 (0.008%): `output` 16 086, token_embd 88, `attn_q`/`attn_k` of layers 5, 9, 11–14 up to 242 | 4 174 (0.015%) |
| rows repeated | **220, all in `output`** | 504: `output` 244, token_embd 246, layer 0 gate and up 7 each |

- **Both predictions fell on the entropy**: the quantizers already spend their code space almost
  whole (a Q8_0 block's scale puts its largest weight at ±127, so 32 roughly Gaussian weights fill
  the range: ~7.7 bits; Q4_K fits a min and a scale per 32 weights). Lossless coding gives **~7%**
  of a Q8_0 file (half of it the scales) and **~4%** of a Q4_K, the experts (94% of the bytes)
  nothing beyond order 0. At memory speed that needs an entropy decoder faster than ~57 GB/s on
  16 cores: not worth a format. Closed.
- **The structure that exists is in the vocabulary and in a few attention columns**: 220 rows of
  the output head are byte-identical to another row (Q4_K: 244, and 246 embedding rows whose
  codes are all zero): the tokens the model never learned, whose logits come out identical; a few
  attention `q`/`k` blocks repeat, and layer 0's `attn_q` carries 1 bit a code of per-column
  structure (a handful of input dimensions dominate). In Q4_K, 7 rows of layer 0's experts' gate
  and up are constant (dead rows). Repeats prediction (≤ 0.1%) held. Worth ≤ 0.5% of the head:
  closed, noted for the next model.

- **The KV** (`tools/kv_repeats_report.py` on the six probe dumps, 16 layers × 16 heads): layer 0's
  values repeat exactly wherever the token repeats (61.6–81.6% of positions on real text, 0.8–1.1%
  on the synthetic prompt), in every head at once: they depend on the token alone, as the
  architecture says. **Layers 1–15: 0 repeated rows in every run; keys 0 everywhere** (RoPE).
  Prediction held. What it would give: layer 0's values are 1/32 of the KV; storing them once per
  distinct token saves ≤ 2.5% of the KV bytes, and summing by token instead of by position changes
  the order of the output's sum (not the definition's bytes). Closed.

## The CPU's peak, and how far the prefill's matmul is from it (2026-09-24)

Question 55, the premise of an assembly microkernel. `tests/bench_peak.c` through
`tools/bench_native.sh bench_peak --runs 15` (marker held, load 2.0–2.5 logical processors,
Defender and the WSL VM; median of 15). Prediction, written first: the matmul at 35–50% of the
no-FMA peak (0.7–1.0 of ~2 TFLOP/s).

| GFLOP/s | 1 thread | 16 threads |
|---|---|---|
| peak, zmm multiplies and adds (inline assembly, 12 chains) | **166.5** (= 32 FLOP/cycle at 5.2 GHz) | 2 309 (spread 26%) |
| peak, ymm | 166.0 | 1 730 (21%) |
| the kernel's instruction stream in asm, L1, 2 rows × 4 tokens (as `dot_row2_x4` today) | 106.6 (64%) | 1 484 (8.5%) |
| the same, 2 rows × **8** tokens (16 accumulators, 26 zmm) | **119.4 (72%)** | 1 674 (11%) |
| `dot_row2_x4` Q8_0 on 2048 columns (L2) | 89.0 | 1 016 (12%) |
| `tr_matmul` 1024 × 2048, 64 tokens (an expert's gate/up) | **87.4 (52%)** | 949 (51%) |
| `tr_matmul` 2048 × 1024, 64 tokens (down) | 82.4 (49%) | 932 (19%) |
| `tr_matmul` 2048 × 2048, 512 tokens (attention) | 87.5 (53%) | 1 027 (38%) |
| peak with FMA (question 59; a later run, load 1.6–1.8) | 166.1 | 2 500 (16%) |
| the 2 × 8 stream with FMA instead of mul + add (q. 59) | **141.1** (the same run's without: 118.8) | 2 005 (5.5%; without: 1 710) |

- **The prediction held**: 49–53% of the no-FMA peak on one core, ~40–45% on 16 (those lines are
  noisy: every run of four put them at 0.80–1.20 TFLOP/s; the one-core lines are clean, 1.3–3%).
- **The margin is real and measured**: today's kernel runs at 82–84% of its own instruction
  stream's ceiling (89 of 106.6); **the same stream with 8 tokens a weight instead of 4 reaches
  119.4, 1.34× today's matmul on one core** (1.6–1.8× on 16 cores, where the matmul loses more to
  memory than the stream does). The widened codes and the scale multiply are paid once for 8
  tokens instead of 4. Every output stays its own `dot_row` sum in the same order, so exact. The
  next step: a `dot_row2_x8` microkernel in assembly (26 zmm, no spills), prefill ~1.15–1.25×
  (the matmul is 56–77% of it, question 37).
- **Order matters on Zen 4, even between independent ops**: the peak's 12 ops strictly
  alternating (mul, add, mul, add…) run at 139 GFLOP/s, grouped (6 mul, then 6 add, or gcc's own
  order) at 167; the same on every repeat (LESSONS #162). The microkernel is measured in more than
  one order.
- **For question 59 (Marcello's decision), on this Zen 4**: FMA raises no peak (166.1 against
  165.9: FMA runs only on the two multiply pipes), but the kernel's stream with FMA is **1.19×** the
  one without on a core (141.1 / 118.8) and 1.17× on 16: the multiply pipes are the bottleneck, and
  a fused op takes one slot where mul + add took a multiply slot and an add slot. On a CPU whose
  pipes all do FMA (Intel since Haswell, Apple) the ratio should be larger; not measured here.
- The peak loop is inline assembly: written in C, gcc merged its twelve chains into two, and the
  "peak" came out 2× too high (LESSONS #161). A mutant with dependent chains (`-DBENCH_PEAK_MUTATE`)
  makes the bench fail: a peak below a real kernel is not a peak.

## Two rows against eight tokens (2026-09-24)

The step question 55 pointed to: `dot_row2_x8` (two weight rows against eight input rows, sixteen
accumulators, AVX-512; Q8_0, Q4_K, Q6_K), each sum still its own `dot_row`. **Prediction, written
before the native runs**: the matmul 1.15–1.22× on one core (the container's numbers below),
1.1–1.3× on 16; the engine's prefill 1.08–1.15× on Q8_0 (the matmul is 56–77% of it), more on
Q4_K_M (its permute paid once for 8 tokens); the decode unchanged (one token takes no block road).

**Taken apart in the container first** (`bench_peak`'s new lines, one core, median of 5, every
variant against the same binary's `-x8` line, two rounds; the 16-core lines of the container are
not this machine's, a WSL VM):

| variant of the x8 kernel, Q8_0 | kernel, 2048 cols (GFLOP/s) | matmul 1024 × 2048, 64 tokens | its `-x8` |
|---|---|---|---|
| first cut: intrinsics, scalar lane tree (16 stores, 240 scalar adds) | 97–99 | 94–97 | 86–88 |
| the lane tree in SIMD (`avx512_pair_sums`, 45 instructions) | **100–105** | **97–102** | 84–88 |
| + scales by vcvtph2ps (on the FP pipes) | −1% | −1% | |
| + scales converted ahead, 16 blocks at a time, scalar | 89 | 87 | |
| + the same by a gather and vcvtph2ps | 92–94 | 91 | |
| + the next block's scale through memory (broadcast load) | 100–101 | 95–99 | |
| + two tokens' loads, then 4 multiplies, then 4 adds (bench_peak's order) | 102–104 | 99–100 | |
| timing only: every block with block 0's scale (no conversion) | 99 | 97 | |

- **The lane tree was a tenth of the call**: sixteen `tr_lane_combine` per call (a store, sixteen
  scalar loads, fifteen adds each) against 128 vector steps at 2048 columns, twice that share at
  1024. In SIMD every level adds each even lane to the odd one after it, the even on the left as
  the scalar tree does: the same adds, the same bits (1 920 x8 calls compared with scalar's, special
  values included). The x4 kernels (`TR_ROW2_OUT`) now end the same way.
- **The scale's conversion is free where it is**: its scalar code runs on the integer units beside
  the FP pipes that bound the loop; every way of moving it (vcvtph2ps, ahead in a loop of its own, a
  gather, a ring in memory) was equal or slower. The order of independent ops, unlike the peak's
  (LESSONS #162), moved nothing here (two orders, within the A/A).
- **An experiment the compiler changed**: the first "no conversion" run used a scale of 1.0f, gcc
  dropped the multiplies by one, and the kernel read 119 (the stream's ceiling), a 15% "cost" that
  was not the scale's. With a scale it cannot fold, 99 (LESSONS #165).
- gcc keeps the sixteen accumulators and both weight vectors in registers, no spill (disassembly:
  the Q6_K kernel's two stack stores are its scale arrays); no inline assembly needed.

**Native** (`sh tools/bench_native.sh bench_peak --runs 15`, marker held, load 2.9 before and 1.9
after; the `-x8` lines are the same binary with x4 two-row kernels that already end in the SIMD
tree; "this morning" is §The CPU's peak's run):

| GFLOP/s, one core | x8 | `-x8` | ratio | this morning | ratio |
|---|---|---|---|---|---|
| two-row kernel, 512 columns (L1) | 102.0 | 87.8 (x4) | 1.16× | 77.6 (x4) | 1.31× |
| two-row kernel, 2048 columns (L2) | 103.3 | 91.8 (x4) | 1.13× | 89.0 (x4) | 1.16× |
| `tr_matmul` 1024 × 2048, 64 tokens | **102.3** | 90.1 | 1.14× | 87.4 | **1.17×** |
| `tr_matmul` 2048 × 1024, 64 tokens | **101.0** | 88.6 (spread 11%) | 1.14× | 82.4 | **1.23×** |
| `tr_matmul` 2048 × 2048, 512 tokens | 93.2 (spread 19%) | 84.0 (16%) | 1.11× | 87.5 | 1.07× |

- **The prediction held on one core**: 1.14× against x4 with the new tree, 1.17–1.23× against this
  morning on the expert shapes; the matmul now at 61% of the no-FMA peak (102 of 166), the kernel at
  87% of its own stream (103 of 118.4). The attention shape's lines are noisy (16–19%).
- The sixteen-core lines are not readable: spread 38–81% on every kernel and matmul line of this
  run (four earlier runs put them anywhere in 0.8–1.2 TFLOP/s, §The CPU's peak). The engine's
  prefill below is the measurement for many cores.

**The engine** (container, `tools/ab_speed.sh` before = 59e0af8 against after, runs alternated,
prompt 512, round 0 dropped; declared load 3.2 before and 4.4 after, so the ratios are indicative:
`tools/prefill_context.sh change` could not start natively, 10 GiB free of the 12 it wants while the
other windows' containers ran). First the bits: the real models' logits **identical byte for
byte** before and after, Q8_0 and Q4_K_M, 600 positions one token a pass and in passes of 64, 2000
in passes of 512 and of 100.

| tok/s, medians | before | after | ratio |
|---|---|---|---|
| Q8_0 prefill, 16 threads (n = 6) | 217.5 | 258.5 | **1.19×** |
| Q4_K_M prefill, 16 threads (n = 6) | 215.4 | 256.0 | **1.19×** |
| Q4_K_M prefill, 8 threads (n = 4) | 140.3 | 182.3 | **1.30×** |
| decode, both models, 16 and 8 threads | 28.97 / 39.86 / 38.36 | 28.74 / 39.99 / 38.63 | 1.00× |

- **Above the prediction** (1.08–1.15× on Q8_0): the x4 kernels' new tree and the x8 kernel add
  up, and at 16 threads the matmul loses less to memory with half the activation loads per weight
  decode. Q4_K_M gains as much as Q8_0 now: its permute is paid once for 8 tokens (the morning's
  two-row kernel gave it only 1.03–1.08×). The Q8_0 8-thread cell did not finish (the script's
  tail cut it); not rerun.
- **Native, 2026-09-25** (`sh tools/prefill_context.sh change build/before-x8/build/trochilus.exe`,
  8 rounds, marker held, load 1.6 before and 0.7 after; logits identical byte for byte before and
  after, one token a pass, passes of 64, 512 and 100, and the same tokens after a prompt of 4000).
  Prefill tok/s, series a / b: **512: 381.9 / 397.6 → 456.2 / 456.2 (1.15–1.19×)**; **2048: 370.7 /
  367.9 → 426.6 / 425.3 (1.15–1.16×)**; **4000: 327.2 / 328.4 → 372.8 / 371.5 (1.13–1.14×)**.
  Decode unchanged (512: 40.0 / 40.2 → 40.0 / 40.3; 2048: 37.5 / 36.9 → 37.3 / 37.6; 4000: 34.7 /
  34.4 → 34.0 / 34.0, within the spreads). The prediction (1.15–1.25×) held at 512 and 2048, a
  point under it at 4000, where the attention's share of the prompt grows.

**Per type** (`sh tools/bench_native.sh bench_kernels --matrix --runs 9`, 1024 × 2048, 64 tokens,
ms per token, each type with and without x8 in turn; declared load 2.4 / 2.1, but another window
ran a build and tests from 22:05 to 22:12, inside this run, without the marker: indicative):
one core Q8_0 0.0471 → 0.0415 (**1.13×**), Q4_K 0.0599 → 0.0493 (**1.21×**), Q6_K 0.0495 →
0.0422 (**1.17×**), spreads 1.4–4.3%; the sixteen-core lines 12–90% spread, not read. Q4_K gains
most, as predicted: its lookup (a permute per 16 weights of each row) is paid for 8 tokens.

## The prompt's matmul against the four references (2026-09-24)

Piece 1 of ORIGINS §Every piece, read at the pinned commits before building on it again. Where
each reference keeps weights, activations and sums for a prompt (x86, AVX-512 VNNI, this Zen 4):

| reference | tile in registers (weight rows × tokens) | activations | the weight's decode, how often | exact against F32? |
|---|---|---|---|---|
| llama.cpp `mul_mat`, Q8_0 (attention projections, head): llamafile `tinyBLAS_Q0_AVX` | 4 × 4, ymm, 16 accumulators | **int8**, Q8_0 per 32, quantized once per op by all threads | none: int8 × int8 → int32 (`sign` + `vpdpbusd`), one FMA per block with the two fp16 scales | no |
| llama.cpp `mul_mat_id`, Q8_0 experts (no repack on x86) | **1 × 1**, one ymm accumulator per dot; cache blocks of 16 rows × 16 tokens | int8, Q8_0 | none, per dot | no |
| llama.cpp `mul_mat_id`, Q4_K experts, repacked at load (`q4_K_8x8_q8_K`) | 16 rows (two interleaved 8-row blocks) × 4 tokens, zmm | int8, Q8_K per 256, 4 tokens interleaved | nibbles → int8 once per 4 tokens, then `vpdpbusd` | no |
| ik_llama.cpp `iqk_mul_mat` | R8/R16 row-interleaved gemm | int8 | **from 32 tokens: 32 rows at a time converted on the fly to a simple int8 form (Q8_0_R8, Q8_K_R8/R16, Q8_1), then one gemm over every token**: paid once per row, not once per token group | no |
| ds4 `matmul_q8_0_batch` | 1 × 2; threads split output rows | int8 per 32 | none | no |
| colibri `xf_moe_run` (its own planar int4) | 1 × 1; (expert, row chunk) items keep the chunk in cache for all the expert's tokens | **F32** (default), int8 opt-in | per dot, one FMA per element | its own order, with FMA |
| Trochilus `dot_row2_x8` | 2 × 8, zmm, 16 accumulators | F32 | once per 8 tokens | **bit for bit, scalar = SIMD** |

- **Every other engine's speed is int8 activations**: all four quantize them (colibri as an
  option); one instruction does 64 multiply-adds against our 16. That is mode (c): a declared
  mode, never the default (CLAUDE.md, invariants).
- **What is exact and can be taken: ik's order of work.** Decoding a weight into its F32 value
  (`d * q` rounded, the float `dot_row` multiplies by) once per row and keeping it for every
  token changes no bit: the same products in the same lanes, summed in the same order. Today the
  x8 kernel decodes each weight vector once per 8 tokens (two widenings, two conversions, two
  scale multiplies per 16 multiply-adds, on the FP pipes the arithmetic needs); an expert sees
  ~64 tokens in a 512-token pass, so a panel decoded once pays it 8× less.
- llama.cpp's Q8_0 experts run one dot at a time on one accumulator: an FMA chain of 4 cycles a
  block, at most ~83 GFLOP/s a core on paper, below our 102. Its lead in the engine (1.41× at 16
  threads before x8) would then come from the attention projections (tinyBLAS, 4 × 4 in int8) and
  from 4× fewer activation bytes when 16 cores share L3: the isolated race below says which.

**Prediction for question 63, written before the runs** (container, one core, `bench_peak`): the
x8 stream with F32 weights loaded instead of decoded 135–150 GFLOP/s (the decoded one 119); the
matmul by panels 115–130 on Q8_0 (1.13–1.27× today's 102), more on Q4_K (its decode is heavier: a
permute and a shift per vector, the scales unpacked per block); the same outputs bit for bit.
Cost: a panel of 32 rows × 2048 columns is 256 KiB a thread, in L2 (1 MiB on Zen 4).

**Question 63 measured** (`bench_peak --one-core --runs 11`, container, load 4.8 before and 3.8
after; the panel line and `tr_matmul` alternated run by run, every panel output byte for byte
`tr_matmul`'s, red under an FMA). A first run decoded the panel with the scalar `dequant_row`, one
weight at a time against the x8 kernel's sixteen, and lost 0.58–0.70× at 64 tokens: that priced
a slower decode, not the premise (LESSONS #170). With the decode in AVX-512 (the same floats):

| GFLOP/s, one core | `tr_matmul` | panel P = 16 | panel P = 32 | ratio |
|---|---|---|---|---|
| x8 stream, decoded / F32 weights loaded (L1) | 115.9 | 163.6 | | **1.41×** |
| Q8_0 1024 × 2048, 64 tokens | 69.9 / 70.6 | 70.0 | 70.3 | 1.00× |
| Q8_0 2048 × 1024, 64 tokens | 82.6 / 77.8 | 85.0 | 73.5 | 0.94–1.03× |
| Q8_0 2048 × 2048, 512 tokens | 95.9 / 99.5 | 99.7 | 102.8 | 1.03–1.04× |
| Q4_K 1024 × 2048, 64 tokens | 91.2 / 86.6 | 100.7 | 96.7 | **1.10–1.12×** |

(spreads 2–35%: the 64-token Q8_0 lines are not distinguishable from 1.00×.)

- **Below the prediction, and why**: the decode is 29% of the stream in L1 (1.41×), but the real
  matmul reads its eight input rows from L2 at every step and is bound there, as §Attempts' tile
  of 16 tokens had already found; a panel of F32 weights adds 64 bytes a vector to that traffic
  where int8 added 16. *(2026-09-25: question 64 refuted "bound by the input loads": tiles with
  half of them per product lose in the matmul, §More weight rows per input load.)* **Closed for Q8_0**; Q4_K's heavier decode leaves 1.1×, not enough to
  carry a panel of 128–256 KiB a thread.
- **The next idea, from these numbers and from the references' tiles**: every input load must feed
  more weight rows. We run 2 rows × 8 tokens (8 input loads per 32 FP ops); llama.cpp's repacked
  Q4_K runs 16 rows × 4 tokens, tinyBLAS 4 × 4. A 4 × 6 tile (24 accumulators, 29 of 32 zmm)
  loads 6 input vectors per 48 ops, half as many per op: question 64.

**Their kernels alone, on bench_peak's shapes** (`sh tools/bench_ggml.sh`: ggml's public API
against `ref/llama.cpp/build-trochilus`'s static libraries, their activation quantization inside
the timing; container, median of 15, **load 7.0 before and 8.2 after**: indicative, the one-core
lines 5–23% spread, the sixteen-core ones 7–66%). Ours: §Two rows against eight tokens (native,
one core) and its per-type line.

| GFLOP/s, one core | llama.cpp | its road | Trochilus | ratio |
|---|---|---|---|---|
| Q8_0 dense 1024 × 2048, 64 tokens | **162.9** | tinyBLAS 4 × 4, int8 | 102.3 | theirs 1.59× |
| Q8_0 dense 2048 × 1024, 64 tokens | 161.5 | tinyBLAS | 101.0 | theirs 1.60× |
| Q8_0 dense 2048 × 2048, 512 tokens | 155.9 | tinyBLAS | 93.2 | theirs 1.67× |
| **Q8_0 experts**, 64 × (1024 × 2048), 512 tokens top-8 | **71.8** | `mul_mat_id`, one dot at a time | 102.3 (the same expert shape) | **ours 1.42×** |
| Q4_K dense 1024 × 2048, 64 tokens | 91.7 | vec_dot, int8 | 85 | theirs 1.08× |
| Q4_K experts | 90.4 | vec_dot | 85 | theirs 1.06× |
| Q4_K dense, repacked (`q4_K_8x8`) | **194.2** | 16 rows × 4 tokens, int8 | 85 | theirs 2.3× |
| Q4_K experts, repacked | 106.9 | the same through `mul_mat_id` | 85 | theirs 1.26× |

- **On the experts, two thirds of OLMoE's prefill (§Prefill profile), ours is ahead on Q8_0**: 102
  against 72 a core, as the reading predicted (one FMA chain a block, ≤ 83 on paper). The
  sixteen-core lines agree in direction (their experts 614, ours ~950 before x8), too noisy to
  quote as a ratio.
- **They win where int8 meets a tile**: tinyBLAS on the dense projections (1.6×) and the repacked
  Q4_K (2.3× dense, 1.26× on experts). In the engine that is the attention projections (~23% of
  the prefill) and all of a Q4_K model: where the 1.41× of the race before x8 came from, and what
  the native race (`tools/race_llama.sh`) must now split by zone.
- Exact answers to both are the same idea: decode once (question 63), then F32 with a tile large
  enough; the int8 road stays mode (c).

## More weight rows per input load (question 64, 2026-09-25)

Four rows × six tokens (24 accumulators, 4 weight vectors, 29 of 32 zmm): 6 input loads per 48
multiply-adds against x8's 8 per 32, half as many a product; but 4 weight vectors decoded for 6
tokens where x8 decodes 2 for 8, a third more decode a product. Three rows × eight tokens (24
accumulators, 28 zmm) keeps x8's decode a product and loads two thirds of its inputs: measured
beside it, the same question from the other side.

**Prediction, written before any run** (container, one core, `bench_peak --one-core`), from
§The CPU's peak's streams: the f32 stream (no decode) 16.3 cycles a step of 32 ops, the x8 stream
22.3, so a weight vector's decode costs ~2–3 cycles beside the arithmetic:
- the 4 × 6 stream in L1 **0.92–1.02× the x8 stream** (arithmetic 24 cycles a step, decode 8–12);
  the 3 × 8 stream 0.97–1.03× (the same decode a product);
- `dot_row4_x6` in the matmul's loop, Q8_0 1024 × 2048, 64 tokens: **1.00–1.12× x8's
  `tr_matmul`**, bit for bit. The gain can only come from the L2 side: x8's matmul runs at 0.86 of
  its own stream (102 of 119), so 1.17× is the ceiling if the input loads were the whole gap;
- Q4_K (a permute a weight vector, x8 gained most from sharing it): 0.90–1.05×.
If the matmul line stays under 1.05×, question 64 closes as no and the tile stays 2 × 8.

**Measured** (`sh tools/bench_native.sh bench_peak --q64 --one-core --runs 11`, native, marker
held, load 1.1 before and after; the streams are one asm statement each, registers named, their
loops read in the disassembly, LESSONS #174; the kernels `row4_x6` and `row3_x8` in intrinsics,
checked byte for byte against `tr_matmul` on every shape before timing, red under an FMA each;
`tr_matmul` and the two tiles timed in turn, run by run):

| GFLOP/s, one core | today's 2 × 8 | 4 × 6 | 3 × 8 |
|---|---|---|---|
| stream in L1 (scale broadcast from memory, the same decode for all three) | 116.9 | 119.1 (1.02×) | 121.4 (1.04×) |
| Q8_0 1024 × 2048, 64 tokens | 104.1 | 93.4 (0.90×) | 97.3 (0.93×) |
| Q8_0 2048 × 1024, 64 tokens | 100.9 | 91.1 (0.90×) | 94.5 (0.94×) |
| Q8_0 2048 × 2048, 512 tokens | 104.3 | 92.2 (0.88×) | 96.8 (0.93×) |
| Q4_K 1024 × 2048, 64 tokens | 86.9 | 73.4 (0.84×) | 78.2 (0.90×) |

(spreads 0.5–3.6%, two lines 7.4 and 9.5%.)

- **The prediction failed, and with it question 63's explanation.** The streams held (the arithmetic
  of the three tiles is the same within 4%), but in the matmul both larger tiles lose 6–16%: halving
  the input loads per product bought nothing, so the matmul is **not** bound by its input rows'
  loads from L2, as §The prompt's matmul against the four references concluded. What the 13%
  between x8's stream (119) and its matmul (104) is stays open: the scalar scale conversion
  (`row4_x6_q8_0` spills 73 general registers around four fp16 conversions a block, x8's kernel 5),
  the lane tree, the stores.
- **Closed as no**: no tile shape has more than ~4% to give in its own arithmetic (the streams), and
  the real kernels of the two lose. The tile stays 2 × 8; the bench keeps the lines (`--q64`).

## The decode's matmul against ggml's (question 65, 2026-09-25)

Piece 2 of ORIGINS §Every piece: one token, every weight read once. The references (ORIGINS row 2):
llama.cpp skips tinyBLAS at one column (`llamafile_sgemm` returns at n < 2) and runs `vec_dot` one
row at a time on int8 activations quantized once per matmul, rows claimed 64 at a time by an atomic
counter; repacked Q4_K runs a `gemv` of 8 rows; ik has `nrc_y = 1` kernels with two block chains a
row; colibri is one row x one vector on int8 (`dpbusd`); ds4's CPU path has no x86 SIMD at all.

`sh tools/bench_ggml_decode.sh` (container, marker held, load 1.8 before and 2.0 after, 5 rounds x
2 passes): ggml's MUL_MAT / MUL_MAT_ID (b49650a, its pool pinned with `strict_cpu` to our first T
slots) and `tr_matmul` / `tr_matmul_grouped` in one binary, the same weight bytes (copies filling
~1 GiB, every pass from RAM) and cores, alternating, beside a plain read of the same bytes by our
pool (the ceiling; ~48 GB/s at 4 threads in the container, 57 native). Both engines checked on the
first node: ggml differs by 0.3-0.45% of the largest output (its int8 activations). GB/s; the
lines are the median over the five shapes (1024x2048, 2048x1024, 2048x2048, the 50304x2048 head, 8
of 64 experts), the range in brackets; spreads 2-25%:

| threads | Q8_0 ours | ggml / ours | ours / read | Q4_K ours | ggml / ours | repacked / ours | ours / read |
|---|---|---|---|---|---|---|---|
| 1 | 21.2 | **1.54** (1.43-1.69) | 0.60 | 9.3 | **2.70** (2.60-2.75) | 3.20 (3.07-3.36) | 0.27 |
| 2 | 39.4 | 1.04 (0.94-1.09) | 0.86 | 17.9 | 2.01 (1.83-2.13) | 2.26 (2.24-2.53) | 0.40 |
| 4 | 46.0 | **1.00** (0.97-1.01) | 0.96 | 33.6 | **1.37** (1.34-1.40) | 1.39 (1.33-1.44) | 0.70 |
| 8 | 47.7 | 1.02 (1.02-1.06) | 0.95 | 43.5 | 1.07 (0.94-1.09) | 1.06 (1.01-1.12) | 0.88 |
| 16 | 48.0 | 1.01 (0.95-1.04) | 0.93 | 42.1 | 1.08 (0.99-1.12) | 1.06 (1.01-1.12) | 0.81 |

- **Q8_0, the model's type: level from 4 threads**, both at the ceiling (0.93-0.96 of the read); the
  decode runs 4 threads at short context and 8 at long. So the race's decode gap (theirs 1.04x at
  512, 1.13x at 2048) is not this matmul: it is attention (row 3, the KV's bytes). ggml wins alone
  at one thread, 1.4-1.7x: a core of ours computes 21 GB/s, one of theirs reads its 31-36.
- **Q4_K: ours is bound by its arithmetic**, 9.3 GB/s a core (0.27 of the read, 17 G weights/s),
  so at 4 threads ggml is 1.34-1.40x (repacked alike), and level from 8 (0.94-1.12, inside the
  spreads). The decode's width is picked by measuring: a Q4_K model likely takes 8 (not measured).
- Prediction (question 65): Q8_0 at one thread 1.2-1.3x (measured 1.4-1.7, more), at two 1.3-1.5x
  (measured 1.0: two of our cores already read 0.86 of the ceiling), from four 0.95-1.05x (held);
  the 2048x2048 projections losing 5-10% to sync (not seen: 0.96 of the read); Q4_K 1.2-1.4x at 4
  (held), <= 1.1x from 8 (held).

**Three premises for a faster exact Q4_K dot, each timed and closed as no** (the kernel alone in
L1, one core, 2048 columns, best of 5, ns a row; today's `avx512_dot_row_q4_k` 120 ns):
- **four rows against the token at once** (`dot_row4_x1`, four independent chains of adds, the
  token loaded once for four; Q8_0, Q4_K, Q6_K, bit for bit, tested and wired): in L1 only
  1.04-1.08x four `dot_row`s (the core already overlaps consecutive rows: the dot is bound by its
  throughput, not the add's latency), and **from RAM 0.5-0.9x** the engine (Q8_0 at 1-4 threads
  0.49-0.88x): four interleaved rows of 1-2 KiB are four short streams the prefetchers do not
  follow, one row after another is one long stream (LESSONS #178). Removed;
- **the scales as vector** (`avx512_q4_k_scales`: a dozen vector instructions instead of sixteen
  scalar conversions, the same floats): 140 ns, 0.86x (two lane permutes a sub-block cost more
  than two broadcasts from memory);
- **the weight by arithmetic instead of a lookup** (`scale * cvt(q) - min`, no `vpermps`): 156 ns,
  0.77x (the float pipes, not the shuffle unit, are what binds: the exact weight costs a multiply
  and a subtract where the lookup costs one permute).
- ~~So an exact dot at the F32 definition has ~5% left in Q4_K's kernel on this core~~ **refuted
  the same day** (question 66, LESSONS #179: an inference; sequenced, the scalar scale decode was
  24% of the row, and the exact kernel went 1.25x, two rows 1.38x); ggml's 2.7x a
  core comes from int8 activations (`dpbusd`: 64 multiply-adds an instruction), i.e. from giving
  up the exact definition: question 60 (Marcello's), not a kernel. Row 2 of ORIGINS closed.

## Softmax and exp against the references (2026-09-24)

Piece 4 of ORIGINS §Every piece, read: llama.cpp (b49650a) computes softmax and SiLU with
`ggml_v_expf` (vec.h, "adapted from arm limited optimized routine", declared error 1.45 + 0.5
ulp), the softmax's sum in double after a 16-lane `reduce_add` (so its order is the tier's); its
CPU flash attention calls the C library's `expf` once per score. ik_llama.cpp (f3d6e6e) has the
same routine (`iqk_utils.h`). ds4 (8db1d1d) and colibri (a90bed9) call the C library's `expf`,
one value at a time: their bits are the platform's. Ours: `tr_expf`, correctly rounded, proven on
every float in the gate.

`sh tools/bench_expf_refs.sh 8` (container; every one of the 2^32 floats against `tr_expf`; then
one core, 4096 arguments in [-20, 0], median of 15; **a mutation campaign ran beside it**: the
counts are exact, the times indicative):

| exp | floats rounded otherwise than the exact exp | worst | ns a value |
|---|---|---|---|
| llama.cpp / ik `ggml_v_expf`, AVX-512 | **144 478 066 (3.364%)** | 2 ulp | **0.157** |
| glibc `expf` (ds4, colibri, llama.cpp's flash attention on Linux) | 170 648 (0.004%) | 1 ulp | 3.20 |
| Trochilus `tr_expf`, scalar (the engine's) | 0 | 0 | 5.42 here, 3.52 native |
| Trochilus float-only exact exp, AVX-512 (question 58, not wired) | 0 | 0 | 0.73 native |

- **Where we stand**: the only exact exp of the four; ggml's is 4.6× faster than our exact SIMD
  one and wrong on one float in thirty. The C library's is right almost always, and "almost" is
  the platform's: MinGW's gave other bits (question 37), which is why Windows and Linux agreed to
  the byte only after `tr_expf`.
- **What it is worth**: the exp is 2–3% of our prefill at 2048–4000 once in SIMD (question 58);
  the 4.6× between exact and approximate is at most ~2% of a prompt. Nothing to take.

## The Q4_K decode dot sequenced (question 66, 2026-09-25)

Marcello's task: reach or beat llama.cpp's decode at 4 threads on Q4_K, keeping the exact
definition (question 65 had left ggml 1.34-1.40x ahead at 4 threads, "ours bound by its exact
arithmetic, ~5% left", an inference: LESSONS #179). Method: sequence the kernel before changing
it, every premise with its prediction written first (`build/q66/predictions.txt` of the day),
timed in L1, then from RAM, then bit for bit, then in the engine, then the race.

**Step 0, the instructions on this Zen 4** (`sh tools/bench_q4k_genome.sh`, container, one core,
clock measured by a chain of integer adds, 5.1-5.2 GHz; cycles an instruction, twelve independent
ones a loop, or dependent for "lat"): `vmulps` / `vaddps` zmm 1.0 (latency 3), `vpermps` zmm 1.0
(latency 5; ymm 0.5), `vpermt2ps` zmm 1.0, `vpmovzxbd` zmm from memory or register 1.0, `vpsrld`
zmm 1.0, `vbroadcastss` zmm from memory or register 1.0 (not free), `vcvtsi2ss` 1.1, `vcvtph2ps`
xmm 0.5, `vmulss` 0.5, a zmm load 1.06. Mixed: any two zmm operations a cycle (permute + multiply +
add 0.5 an op, multiply + add 0.5, the kernel's own 16-op mix 0.5), but a permute and a widen share
one pipe (1.0 an op together). Predictions held for all but the broadcast (predicted 0.5 from
memory) and the widen/permute sharing.

**The kernel with one piece removed at a time** (a row of 2048 in L1, called row after row as
tr_matmul does; cycles a row, median of 11, spreads 1-5% unless said):

| kernel | cycles | what is gone |
|---|---|---|
| tier (`avx512_dot_row_q4_k`) / its copy | 618-623 / 620 | — |
| f16c | 586 | the halves converted by `vcvtph2ps` instead of in software |
| prescale | 472 | the scale decode (**146 cycles, 24%**: sixteen `vcvtsi2ss` + `vmulss` and the bit fiddling) |
| pretable | 421 | and the tables' build (51) |
| prescale + constant indices | 413 | the indices' widen and shift (59) |
| prescale + no permute | 359 | and the permutes (54) |
| pretable + no permute | 329 (spread 32%) | only the loads, the 16 multiplies and 16 adds a block |
| notree | 615 | the lane tree (3) |

The floor of a one-row kernel is its one chain of adds (16 a block, latency 3: 384 cycles a row,
partly hidden across rows). Predicted: the scale decode the largest piece (held: predicted 15-25%),
the chain the floor (held).

**Candidates in L1** (cycles a row; every exact one equal to scalar bit for bit before timing):

| candidate | cycles | ns | vs tier |
|---|---|---|---|
| scales in a vector, stored, broadcast from memory in the same block | 598-600 | 116 | 1.03x (the broadcast waits for the store: LESSONS #180) |
| the same, the whole row's scales first | 527-534 | 103 | 1.17x |
| the same, **one block ahead** | 497-502 | 97 | **1.25x** |
| the same, scales broadcast by lane permutes | 731-739 | 142 | 0.85x (as question 65's vector scales) |
| **two rows a call**, 8 blocks' scales first | 443-450 | 85-87 | **1.38x** |
| two rows, scales one block ahead | 454 | 88 | 1.37x |
| four rows | 440 | 85 | 1.41x (the pipes, not the chain, bind from two) |
| tables by a fused multiply-subtract (exact: scale * q has at most 21 significant bits) | 481 (x2) / 512 (x1) | — | 0.94x / 0.97x of the same without: gcc needs a second broadcast and a copy |
| FMA into the accumulator (question 59, another definition), today's scalar scales | 589 | 114 | 1.05x |

Premise e, **a dictionary of tables** (`tools/q4k_tables.py`, the real Q4_K file, 35 tensors of
layers 0, 7 and 15 and the attention outputs): 93.8-99.6% of the sub-blocks have a (scale, min)
pair of their own, and the 16384 most common pairs (a 1 MiB L2 of tables) cover 1.1-18.7%. Predicted
> 90% and < 20%: closed as no. Not built (their premises answered by the lines above): the scales
pre-decoded at load (the ideal "prescale" is 5% under the vector decode, for +44% bytes a block
from RAM), `vpermt2ps` (costs a permute and still yields 16 weights), panels repacked at load (the
prepass below gives the pair one stream without them), an exactly rounded dot (question 60, a new
definition: not needed to reach level), asm ordering (the compiler's order is at 75-80% of the
two-ops-a-cycle budget).

**From RAM** (`--ram 4`: ~1 GiB of Q4_K rows, 4 threads on 4 cores, one contiguous chunk a thread,
outputs compared with the tier's; GB/s, medians of 10-18 passes; the container's machine was noisy,
spreads 7-45%):

| kernel | container | native |
|---|---|---|
| read (ceiling) | 47.5-47.9 | 55.6 |
| the tier before today | 34.7-35.2 | — |
| vector scales one block ahead | 40.4-40.8 | — |
| the same with a prefetch 4 rows (4608 bytes) ahead: **the engine's `dot_row`** | 42.0-42.8 | 43.7 |
| two rows, scales one block ahead | 25.4-26.3 (0.72x: LESSONS #181) | — |
| two rows, 8 blocks' scales first | 40.9-41.6 | — |
| the same with the prefetch: **the engine's `dot_row2`** | 43.3-45.3 | 46.8 |
| four rows, one block ahead | 27.0-28.1 | — |
| ggml's way, 64 rows claimed at a time by an atomic counter | 0.93-0.97x of the chunks | — |

**SMT** (native; the container's logical processors are virtual): in L1 a core with two threads on
its two siblings does a row every 87.0 ns against 97.4 alone (1.12x) with the one-row kernel, 84.3
against 86.3 (1.02x) with the pair, which already fills the pipes. From RAM, 4 cores with 8 threads:
read 54.7, one-row 49.5 (1.13x its 4 threads), pair 49.3; in the container 49.1 / 46.8 / 48.2. Not
wired: the pool puts one thread a core first (next step).

**Wired and raced.** `avx512_dot_row_q4_k` now computes the next block's scales in a vector
(`avx512_q4_k_scales_store`) and prefetches 4608 bytes ahead; `avx512_dot_row2_q4_k` (the new table
entry `dot_row2`, kernels.h) takes two rows against one token, the scales of 8 blocks of both rows
first; `matmul_rows` and the one-token remainder of `matmul_tiled` take rows in pairs. Tests:
test_kernels (every tier's dot_row2 against scalar's dot_row, special values, a wrong pair seen, the
decode matmul counting pairs), test_tier_used (the pair required on AVX-512, every product counted);
eleven mutations in `tools/mutate_row2.sh`, all red. `sh tools/bench_ggml_decode.sh 5 q4_k`
(container, medians over the five shapes; the load 3.2 processors busy):

| threads | ours GB/s | ggml GB/s | ggml / ours | before today | ours / read |
|---|---|---|---|---|---|
| 1 | 12.5 | 25.0 | 2.01x (1.96-2.06) | 2.70x | 0.37 |
| 2 | 24.4 | 35.3 | 1.42x (1.34-1.58) | 2.01x | 0.55 |
| 4 | 42.8 | 44.6 | **1.04x** (1.01-1.04) | 1.37x | 0.91 |
| 8 | 45.2 | 46.0 | 1.02x (0.95-1.05) | 1.07x | 0.92 |
| 16 | 44.3 | 44.1 | 0.96x (0.95-1.03) | 1.08x | 0.86 |

**The real model** (`RACE_THREADS=4 sh tools/race_q4k.sh 5`, prompt 512, 128 generated, median of 5,
container): Trochilus Q4_K decode **49.91 and 50.90 tok/s**, llama.cpp on the same file **50.16**:
level. An A/B against HEAD's engine in turn (`build/q66/ab_q4k.sh`, the same settings): **40.57,
39.98 -> 50.15, 50.56 tok/s, 1.25x**; the prefill unchanged (129-139 both), llama.cpp's 222 at 4
threads (1.6x: not this question). The Q8_0 A/A of the race: 30.60 and 27.48 (runs 23.6-31.4: the
machine's noise, the Q4_K series held 46.7-51.0).

## SMT in the decode (question 67, 2026-09-25)

Marcello's task: beat llama.cpp on the same 4 cores by running two threads on each (the genome
bench had +13% from RAM natively for the one-row kernel, +5% for the pair the engine runs).
Predictions in `build/smt/predictions.txt`, written before every run.

**The container cannot test it.** `sh tools/race_smt.sh 5` (taskset 0-7: "4 cores and their
siblings" by lscpu, Q4_K 512 + 128, series Trochilus, llama.cpp, llama.cpp, Trochilus): our prefill
at 8 threads ran **1.8x** its 4 threads, which two threads on one core never give; the VM's
processors land wherever Windows puts them, so this is 4 against 8 virtual processors (LESSONS
#184). There: Trochilus 48.0-49.0 -> 48.6-48.8 tok/s, llama.cpp 48.5-50.5 -> 53.3-53.6 (+6-10%).
That gap led to §The pool's tail below.

**Natively** (`start /affinity`, CPU attention `TR_GPU=0`, `TR_POOL_PIN=1` for two threads a core,
Q4_K 512 + 128, runs alternated by `tools/ab_modes.sh`, still machine: 1.7-1.9 processors busy;
`build/smt/ab_native.sh`, `ab_widths.sh`, `ab_tuner.sh`):

| cores | one thread a core | two threads a core | prefill, two against one |
|---|---|---|---|
| 4 (LPs 0-3, 16-19) | 53.0-54.7 (A/A 54.05) | 55.6-56.0 (**+1.6-5.7%**) | 1.06-1.08x |
| 8 (LPs 0-7, 16-23) | **60.01** | 60.06 (+0%) | 1.08x |
| 16 (all) | 58.26 | **8.43** (0.15x, question 68) | 0.42x |

The prefill's small gain at 4 and 8 is SMT's signature (the mask did pair siblings). The engine's
own measured width: **8 cores in 6 runs of 6**, 59.75 tok/s against 60.17 with 8 forced and 53.23
with 4: the width that matters is 8 cores, where a second thread adds nothing. Predicted P1-P3
1.03-1.12x; held only at 4 cores, at its low end. **Closed as no: not wired** (a pool with
sibling threads, a tuner with SMT widths, a forced flag: nothing any default width would gain).

## The pool's tail (2026-09-25)

`tr_parallel_for` gave each thread one contiguous chunk of equal size. A research build
(`make BUILD=build/linux-trace EXTRA_CFLAGS=-DTR_POOL_TRACE`, natively `build/win-trace`) writes
every call's dispatch and end and each chunk's start and end (`TR_POOL_TRACE_FILE`);
`tools/pool_trace.py` sums them by call shape: "balanced" is the mean chunk's work over the wall,
what a perfect split of the same work would take.

| where | threads | balanced, the decode's matmuls | the last chunk to end |
|---|---|---|---|
| container | 4 | 0.94-0.97 | any |
| container | 8 | **0.77-0.81** | a different one each call |
| native | 8 | 0.84-0.91 | any |
| native | 16 | 0.76-0.85 | chunks 2 and 7 more often |

Built: `tr_parallel_for_balanced` (threads.h): each thread runs its chunk in `TR_POOL_BLOCKS` = 64
blocks, front to back, then takes the blocks other chunks have not started (an atomic add a block;
each chunk's first block is its owner's); every index still runs once, so the bits cannot move.
Used by the matmul when every group has one input row (the decode; its experts only from 2026-09-26,
the condition missed their 64 groups: LESSONS #192) and by the decode's attention; the prompt keeps the static chunks its tiles want. Tests: test_base (every index once
over widths, n and min_chunk; a held worker's chunk finished by the others, `helped > 0`: red with
`TR_POOL_BLOCKS=1` and with the help loop cut to the own chunk), test_phase and test_tier_used
unchanged (every logit as one thread).

The decode calls' wall a token, from the traces (ms, two runs each):

| where | threads | static | blocks 16 | blocks 64 | blocks 128 |
|---|---|---|---|---|---|
| container | 4 | 20.81, 21.01 | 20.94, 21.17 | 20.60, 20.86 | — |
| container | 8 | 21.37, 20.34 | 18.61, 19.47 / 19.17, 21.31 | **18.82, 19.64** | 19.62, 20.28 |
| native | 8 | 16.96, 17.26 | — | 17.32, 16.62 | — |
| native | 16 | 18.32, 18.14 | — | 18.39, 18.61 | — |

In the container at 8 the projections' calls 55.6 -> 49.5 us, the experts' down 214.7 -> 200.2 (noise: they
still ran static, LESSONS #192);
natively at 8 the projections 47.1 -> 45.5 and the rest level; at 16 the stolen blocks' extra
stream starts cost 1-2% (the mean chunk's work rises), a width the engine does not pick. By
tokens a second (container, 8 threads, alternated with tools/ab_modes.sh: the A/A spread was 11%)
no difference could be read. Kept for machines whose processors are taken away mid-call (virtual
machines: the container's case), at no cost on the width chosen natively. Predicted P5 1.08-1.12x
at 8 in the container (reached in the calls' wall, ~1.08x), P6 (SMT gains more with it) refuted.

## The attention against llama.cpp's (row 3, 2026-09-26)

Marcello's task: pass llama.cpp where it still leads, the decode at long context, without moving a
bit. Predictions in `build/attn/predictions.txt`, written before every run.

**How they do it** (ORIGINS row 3, read 09-25): llama.cpp runs flash attention on the CPU by
default (AUTO resolves on), its KV in **F16**, `[position][head × dim]` a layer. A decode token at
≥ 512 positions splits the positions over every thread; each thread runs every head over its chunk
(a 256-byte slice every 4 KiB: sixteen strided passes), with an online softmax, and **sums the
values in an F16 accumulator rounded at every position** (`ops.cpp` 8755-8809), the chunks merged by
rescaling. ik_llama.cpp: the same family, positions split for MLA only. colibri and ds4 keep F32
rows and never split a query's positions.

**Theirs against ours, as a line.** The race of 09-25 (`f6a8b4c`, container, Q8_0, §Speed —
Trochilus vs llama.cpp after x8) at two contexts gives each engine's decode a token as an intercept (no context) plus a slope
(each cached position):

| threads | Trochilus | llama.cpp |
|---|---|---|
| 8 | 28.1 ms + **5.31 µs** a position | 27.6 ms + **3.24 µs** |
| 16 | 26.8 ms + 5.24 µs | 26.8 ms + 3.34 µs |

Level with no context: the whole gap is the slope. A position is 256 KiB of F32 KV for us, read at
49 GB/s, 128 KiB of F16 for them, read at 40 (their strided passes).

**Ours taken apart** (`bench_attn_bw 2048`, native, 11 repetitions of paired steps, 3.9-4.0
processors busy elsewhere): the engine's attention reads at **the speed of a plain read of the same
bytes** at every width (engine 52.4-52.8 GB/s at 4, 6 and 8 threads; plain reads 53.0-53.5; 5
threads 49.5, sixteen heads over five). The products, the softmax and the tail are all hidden
under the reads: the decode's attention is its bytes.

**P1, one front instead of T: refuted.** The RAM gives 57.6 GB/s to 4 readers and 52-53 to 8 or more
(§Decode at long context and RAM bandwidth); the premise was that T threads reading T places far
apart cost the drop. `bench_mem streams` (new: the same 2 GiB as T contiguous shares, or as one
front the threads take turns on; native, 4-6 processors busy elsewhere, indicative), GB/s:

| threads | T fronts | turns of 256 B | 4 KiB | 32 KiB | 256 KiB |
|---|---|---|---|---|---|
| 1 | 24.9 | 23.6 | 24.7 | 24.6 | 24.9 |
| 4 | 55.7 | 9.5 | 32.3 | 44.7 | 54.0 |
| 8 | 52.4 | 18.2 | 35.8 | 43.6 | 53.1 |
| 16 | 50.7 | 25.8 | 36.4 | 44.0 | 51.0 |

Turns cost, up to 256 KiB: a thread's reads must be long runs (each short one restarts its
prefetcher, as ggml's 64-row claiming did in §The Q4_K decode dot sequenced), and one front
gains nothing at any length. The attention with one front (`bench_attn_bw front`: K in blocks
taken in turns, V cut by dimensions, the same bits) ran 0.78-0.81x the engine.

**What that leaves.** To match their slope with exact F32 keys and values, 256 KiB in 3.24 µs is
**81 GB/s**: the DDR5-5200's paper peak (83), against the 53-57 this machine reads. Lossless
packing (question 57, 28 bits) would need 71. On the CPU an exact F32 KV cannot pass an F16 one at
long context; our decode wins there where the KV is not on the CPU's bus: the GPU attention, the
same bits, 1.31x at 2048 and 1.54-1.58x at 4000 (§The decode's attention on the GPU, in the engine).

## The prompt's attention in tiles (2026-09-26)

The other half of row 3. A prompt's attention is not bound by memory (a group of 16 queries reads
a block of keys once): **taken apart on one core** (`bench_attn 2048 --threads 1 --heads 1`, ns a
(query, position) pair, 7 repetitions):

| variant | ns a pair |
|---|---|
| one query at a time (`tr_attention_head`) | 14.98 |
| the engine's group (`tr_attention_group`, x4 kernels) | 12.09 |
| products only, x4 kernels (blkx nosm) | 7.36 |
| the softmax alone (a row as long as the context) | 4.45 |

A pair is 512 flops: 7.36 ns is 70 GFLOP/s, where the core's no-FMA peak is ~150-167. The x4
kernels store each score's sixteen lanes and add them in scalar, sixteen trees a tile.

**Built: tiles of 4 queries x 4 positions** (`dot_f32_4x4`, `axpy_f32_4x4` in the kernel table;
AVX-512: sixteen accumulators, every key and query vector loaded once for four products, the
sixteen lane trees in four levels of `avx512_pair_sums`, as the x8 matmul; four outputs against
four values, each value vector loaded once for four queries; AVX2: four x4 calls). In
`tr_attention_group` the queries of a block go four at a time wherever the quad's first query sees
all four positions; every query's positions stay in increasing order, so every score and every sum
is the same operations. Predicted P2 1.5-1.7x on the products, P3 9.0-9.8 ns a pair in all:

| one core | before | tiles |
|---|---|---|
| products only, 2048 | 7.36 | **4.55** (1.62x) |
| the whole attention, 2048 (the engine's function) | 12.09 | **9.18** (1.32x) |
| the whole attention, 4000 | 13.07 | 9.44 (1.38x) |
| 16 threads, 16 heads, 4000, one layer (spreads 17-26%) | 143 ms (blkx) | 116 ms (1.23x) |

The bits: every variant hashes to the one-query line's; `test_kernels` compares 16 400 tiles a
tier with scalar's and with the tier's own x4 kernels (two mutants of the definition and two of the
AVX-512 kernel, a tree's pair swapped and two values' order swapped, seen red); `test_tier_used`
counts the 4x4 calls a synthetic model's prompt makes.

**In the engine** (native, 16 threads, Q8_0). Logits identical byte for byte before and after
(`prefill_context.sh change`'s exact stage: 600 positions one token a pass and in passes of 64, a
prompt of 4000 in passes of 512 and of 100, the same tokens after it). Its speed stage stopped at
round 5 of 8 (the engine's memory guard: 3.2 GB free for a minute) and read nothing: A/A gaps of
4-8%, every after/before inside them. Then `tools/ab_zone.sh` (new: the two binaries alternated
run by run, the zone's seconds from `--profile-json`, 8 pairs a prompt, 2.8-3.5 processors busy
elsewhere); predicted P5 the zone 0.75-0.82 at 4000, 0.77-0.85 at 2048, the prefill 0.92-0.95 and
0.96-0.98 of its time:

| prompt | attention zone, s (before → after) | after / before, per pair | prefill, after / before |
|---|---|---|---|
| 2048 | 0.576 → 0.443 | **0.751** [0.735-0.772] | 0.971 [0.957-0.986] (**1.03x**) |
| 4000 | 2.352 → 1.815 | **0.762** [0.736-0.813] | 0.944 [0.910-0.995] (**1.06x**) |

The zone 1.31-1.33x, as on a core; at 4000 it was 21% of the prefill, now 17%. The softmax is now
half of a pair (4.45 of 9.2 ns on a core): question 70, the exact exponential in a vector.

## The engine read as entangled pairs (2026-09-26)

Marcello's task: pass llama.cpp keeping every bit, with the quantum lens (CLAUDE.md): a byte the
receiver could deduce exactly from what it already holds need not travel. The unit is **MiB read
per exact token**, OLMoE Q8_0: weights 1200 (dense N = 385: attention 272, head 104, router and
norms 9; one token's experts E1 = 815), and 0.25 MiB of F32 K and V a cached position: ours
1200 + 0.25 L, llama.cpp's 1200 + 0.125 L (F16): 1328 / 1264 at 512, 1712 / 1456 at 2048, 2200 /
1700 at 4000. Predictions in `build/entangled/predictions.txt`, written before every run. The
evening ran beside two other busy windows (9.6-11 logical processors busy): those timings were read
as structure only, and the deciding runs waited in `build/entangled/when_free.sh`, which started by
itself on the free machine at 03:02 (background 1.6-2.5 processors, 0.7-0.9 of them the kernel's
System; Marcello: timings only then). Results in `build/entangled/free/`.

**The inventory** (what determines what; the most a token could save; the premise and its state):

| pair | entangled | most saved a token | premise | state |
|---|---|---|---|---|
| a coarse model inside the exact bits | a Q8_0 code's high nibble (with the block's scale) is a 4-bit model; a float's high 16 bits a bf16 | a draft at 18/34 of the weights and half the KV, verified k tokens a pass | `tools/draft_probe.c`: agreement from the exact state | 82-98%; 0.81-1.09x llama.cpp's bytes (model): not built |
| the bus's silence inside a token | the next call's weights are known at load: entangled with nothing | the RAM idle share of a token: 5-15% | `tools/token_timeline.py` | a bug found: decode **1.04-1.08x**, the RAM idle 1-4% |
| gate and up, used together | row r of gate and row r of up, 2 MB apart | one stream and one barrier where two | `bench_mem gateup` | 1.006-1.023x from RAM: no |
| the page tables | 512 pages of 4 KiB are one of 2 MiB | TLB walks under 1.4 GB of streams | THP in the container | no, on the free machine too |
| the same value at another position | layer 0's values depend on the token only | <= 1/32 of the KV (question 62) | - | closed by 62's number |
| the next token inside the present | neighbours share 3.6-4.0 of 8 experts; the next router 92-95% | no bytes: reads ahead of time | questions 54, 62 | known; the verify pass's union uses it |
| two memories | the KV written in RAM and VRAM | the KV off the CPU's bus: 1712 -> 1200 at 2048 | built (GPU attention) | another race: the whole machine |
| a thread's work inside its index | 322 pool calls a token, 145 of them inline | the calls' tails and dispatches | the timeline | measured below |
| the head's argmax inside its high plane | greedy needs the argmax, not every logit | <= 49 MiB (4%) | bounds from the high plane, exact on the candidates | not measured |

### The bus's silence: a token's timeline

A new instrument: the traced pool (`-DTR_POOL_TRACE`) now records, per call, the bytes it reads and
the shape that names it (`TR_TRACE_NOTE` in tr_matmul_grouped and the decode's attention; inline
calls traced too), and `tools/token_timeline.py` lays a decode token out call by call against a
plain read's GB/s. Container, Q8_0, 8 threads, context 808-842, 35 tokens (P1 predicted the calls
90-94% of a token, the RAM idle >= 12%):

| | per token |
|---|---|
| wall | 33.7 ms, 1406.6 MiB, 43.8 GB/s (a plain read 50.0 in the same session) |
| in calls / between calls | 99.4% / 0.6% (0.2 ms) |
| calls | 322: 64 dense projections, 48 expert matmuls, 16 attentions, 16 swiglu, 145 inline n = 1 |
| **the RAM idle** | **4.18 ms (12.4%)**: experts' gate/up 1.12, down 0.55, q/k/v/o 0.32, inline calls 0.29, swiglu 0.20, attention 0.20, between calls 0.20, head 0.06 |

The gaps between calls are nothing; the loss is inside the calls, and `tools/pool_trace.py` says
where: the dense projections' tail 0.85 us, **the experts' 41-60 us**. The dense calls ran balanced
(`tr_parallel_for_balanced`, §The pool's tail), the experts never did: the matmul chose balanced when
`offsets[n_groups] == n_groups`, and a decode token's experts are 8 rows in 64 groups (LESSONS
#192). Fixed (`tr_matmul_one_row_per_group`, tested on the experts' shape, red with the old
condition); logits identical byte for byte on the real model (600 positions one token a pass, and
passes of 64), 64 generated tokens identical. Natively (indicative, loaded): the down's tail 28.1 ->
2.5 us, gate/up's 15.8 -> 5.3. **On the free machine** (native, 8 threads, CPU attention, context
~800, 64 tokens, the traced builds alternated, 4 pairs; no prediction was written for the fix's
size, only P1's idle share):

| | before (static experts) | after (balanced) |
|---|---|---|
| a token | 29.4-31.3 ms | **28.2-29.0 ms** (after / before per pair 0.925-0.958) |
| read over the token | 47.2-50.3 GB/s | **51.0-52.4** (a plain read 53-55) |
| the RAM idle | 1.5-3.4 ms (5.2-10.9%) | **0.3-1.1 ms (1.1-3.7%)** |
| the experts' tails, gate/up and down | 13.4-53.0 and 23.8-67.6 us | 1.6-1.7 and 1.6-1.7 us |
| tokens a second | 31.8-33.9 | 34.1-35.2 |

`tools/ab_zone.sh` (the untraced binaries, 8 pairs a prompt): the experts' zones 0.964 [0.926-0.987]
at 512 and 0.958 [0.932-0.988] at 2048, the decode 0.976 [0.946-1.006] at both. The decode now reads
at 97% of a plain read's speed; what the RAM still waits for is the attention's tail (16 heads over 8
threads, 13-20 us a layer) and the 145 inline calls.

**Taken apart, one fusion.** Gate, up and the activation in one call (one tail a layer instead of
three) was built bit for bit (a test on three types, three tiers, three widths, red with gate and
up swapped) and removed after two native pairs on the loaded machine ran it 10-20% slower; in those
pairs every call was slower, the untouched head 9-13% too (LESSONS #195). Taken apart since: Q8_0
has no two-row kernel (`dot_row2` is Q4_K's), so the fused call read gate and up as two streams a
thread. The four dimensions of the question, which bytes (a row of gate and a row of up are always
used together, 2 MB apart), when they arrive (two streams or one), who reads them (the thread's
share), in what sequence (three calls, three barriers): rows interleaved at load are one stream,
one call, one barrier. `bench_mem gateup` prices it alone from RAM (three calls, fused on two
streams, fused interleaved, the same without the activation, a plain read; P4 predicted interleaved
1.03-1.08x of three calls); its loaded run spread 15-124%. **On the free machine** (21 repetitions,
twice), GB/s at 8 threads: three calls 53.9 / 53.5, fused on two streams 53.0 / 54.2, **fused
interleaved 54.2 / 54.8**, the same without the activation 54.7 / 54.9, a plain read 54.9 / 55.2; at 4
threads 55.9 / 55.5 against 57.6 / 56.9. Interleaving gives 1.006-1.023x, under P4: three calls
already read at 97-98% of a plain read once the experts are balanced, and the activation hides under
the reads. **Question 72 closed as no**: no layout change for ≤ 2% of one zone.

### 2 MB pages in the container

The container's kernel gives huge pages on request (`transparent_hugepage` = madvise). The weights
and the KV with `madvise(MADV_HUGEPAGE)` (a research build, `-DTR_THP`) land on them (8.36 GB of
AnonHugePages), but three alternated pairs of `bench_mem` gave nothing above the noise (plain read at
8: 42.6 / 46.5 / 44.3 against 44.6 / 42.3 / 46.6 GB/s; the engine's matmul 44.6 / 47.0 / 46.3 against
46.3 / 45.6 / 48.0; spreads 7-30%, loaded). P3 predicted +0-4%. **On the free machine**, four
alternated pairs: a plain read at 8 threads 49.6 / 50.0 / 50.7 / 51.2 against 52.2 / 50.9 / 50.8 / 50.9
GB/s, the engine's matmul 51.3 / 53.5 / 51.7 / 53.3 against 48.8 / 53.8 / 45.2 / 49.5: nothing, the
matmul if anything worse. Closed as no; the code stays out.

### A draft inside the exact bits

The draft reads each Q8_0 block's high plane (its scale and the codes' high nibbles, read as
16 hi + 8: 18 of 34 bytes) and the KV's high 16 bits (a bf16 read at its midpoint), and runs from
the exact state: `tools/draft_probe.c` keeps the exact session E and a draft session D whose cache
is E's cut, and at every position runs the same token once per variant (the tier's own kernels on
rows cut on the fly), rewound, then takes E's row cut. Variants: the KV cut alone (`kv16`); the
planes (`planes`); the planes with the top 4 of the 8 experts, rescaled (`top4`); with a window of
the first 4 and last 256 positions (`win256`). 256 positions a run on the exact greedy continuation
of the three texts of question 53 (prose, code, Italian: the old ARCHITETTURA.md), container. The
agreements are measured; the bytes a token (model, not measured) come from simulating the passes along the sequence (the draft proposes up to k tokens,
the exact pass keeps the leading agreements and adds its own; the pass reads N + E1 U(k + 1) + KV
once, U from question 54), `tools/draft_probe_report.py`:

| text, context | rep4 | kv16 | planes | where the exact margin < 1 nat (share) | >= 1 nat | best k | MiB a token | llama.cpp / it |
|---|---|---|---|---|---|---|---|---|
| prose 512 | 13% | 100% | **94.9%** | 78.7% (24%) | **100%** | 6 | 1189 | 1.06x |
| code 512 | 31% | 100% | **95.3%** | 73.2% (16%) | 99.5% | 6 | 1189 | 1.06x |
| Italian 512 | 6% | 99.6% | **88.3%** | 64.7% (33%) | **100%** | 1 | 1305 | 0.97x |
| code 2048 | 9% | 100% | **94.5%** | 80.3% (24%) | 99.0% | 10 | 1374 | 1.06x |
| prose 2048 | 57% | 100% | 97.7% | 71.4% (8%) | 100% | 6 | 1330 | 1.09x |
| Italian 2048 | 77% | 100% | 97.3% | 74.1% (11%) | 100% | 14 | 1345 | 1.08x |

- **The KV's low 16 bits decide nothing**: 1 token of 1792 moved. **The planes are sure where the
  exact model is sure**: 99.0-100% wherever its top-2 margin is at least 1 nat, 64-80% where it is
  under; the overall figure is how unsure the text is. Above the separate Q4_K of question 53
  (88.7-94.7%): nothing drifts, the draft restarts from the exact cache. P2 predicted 86-91% on
  prose, 92-96% on code: above it on prose, inside on code, and the KV alone as predicted.
- **The bytes: 0.97-1.09x of llama.cpp's**, not the win: a pass's own drafts are 56% of its bytes
  (prose 512: 6 x 699 MiB against a verification of 3040). The runs at 2048 on prose and Italian
  loop (57% and 77% of their 4-grams seen before): their agreement is inflated, as question 53's
  code trajectory was (LESSONS #194).
- **Cheaper drafts lose the sureness**: 4 experts or a window keep 80-98% where the model is sure,
  not 100%, and fall to 18-59% where it is not; in bytes 0.88-1.12x, the best ones on the looping
  runs.

**Along the real text** (`--text 1`: the exact model reads the file's own tokens, no loop; the
draft's own margin recorded; the bytes a token (model, not measured) from passes simulated at fixed
k or stopping where the draft's margin falls under tau = 0.5-3 nats, the best of both):

| text, context | rep4 | kv16 | planes | margin < 1 nat (share) | >= 1 nat | best k, tau | MiB a token (model) | llama.cpp / it |
|---|---|---|---|---|---|---|---|---|
| prose 512 | 0% | 98.4% | **85.2%** | 67.0% (44%) | 99.3% | 2, - | 1329 | 0.95x |
| code 512 | 8% | 100% | **93.4%** | 74.2% (24%) | 99.5% | 9, 0.5 | 1218 | 1.04x |
| Italian 512 | 0% | 99.6% | **85.5%** | 66.4% (43%) | 100% | 1, - | 1320 | 0.96x |
| prose 2048 | 2% | 100% | **87.9%** | 71.2% (41%) | 99.3% | 11, 0.5 | 1599 | 0.91x |
| code 2048 | 15% | 100% | **93.0%** | 77.0% (29%) | 99.5% | 15, 0.5 | 1426 | 1.02x |
| Italian 2048 | 0% | 98.8% | **83.6%** | 64.8% (42%) | 97.3% | 3, 0.5 | 1661 | 0.88x |
| prose 4000 | 0% | 100% | **82.4%** | 66.9% (52%) | 99.2% | 1, - | 2099 | 0.81x |
| code 4000 | 25% | 100% | **94.9%** | 52.0% (10%) | 99.6% | 9, 0.5 | 1687 | 1.01x |
| Italian 4000 | 3% | 99.2% | **85.2%** | 63.4% (39%) | 99.4% | 9, 0.5 | 1983 | 0.86x |

- Real text is less sure than the model's own continuation (39-52% of prose and Italian positions
  under 1 nat, against 8-33% on the greedy runs): the draft stays at 97.3-100% where the model is
  sure, and the whole falls to 82-88% on prose and Italian, 93-95% on code.
- **In bytes 0.81-1.04x of llama.cpp's, worse with the context**: a draft token reads the KV's high
  halves, 0.125 MiB a position, k times a pass, while the exact pass reads the whole KV once. The
  draft stopping where it doubts helps only code (tau = 0.5). **Pair 1 closed on the CPU in this
  form**: no plane layout. What the numbers leave open (question 71): at a position where the draft
  doubts, its second choice as an extra leaf of the verification (+1 row, where most passes end); a
  draft whose own bytes do not grow with the context (the window lost the sureness: 48-96% where
  the model is sure).

## The draft's KV in 8 and 4 bits (question 75, 2026-09-26)

The idea bounce's second lever (`build/entangled/bounce.md`): the draft reads the history from a copy
written once as each row leaves the last 64 positions (as KIVI: K per channel over groups of 32
positions, V per position, each group's min and step in fp16, 8 or 4 bits a value; the last 64 at
16 bits), and a draft head of the first 16384 rows (+h16k). Container, `--text 1`, 128 positions a
run (`build/entangled/run_kv.sh`, data in `build/entangled/kv/`); bytes from the report's model;
each cell: agreement, then the leaf's best in sample / **out of sample** (k and tau chosen on one
half of the positions, scored on the other, both ways) against llama.cpp's bytes a token:

| text, context | planes (16-bit KV) | kv8 | kv4 | +h16k (planes) |
|---|---|---|---|---|
| prose 512 | 85.9%, 1.02 / 0.97 | 85.9%, 1.04 / **1.00** | 89.1%, 1.09 / 1.09 | 80.5%, 0.96 / 0.82 |
| Italian 512 | 85.2%, 1.02 / 0.99 | 85.2%, 1.04 / **1.01** | 87.5%, 1.05 / 1.00 | 71.9%, 0.89 / 0.90 |
| code 512 | 93.0%, 1.16 / 1.12 | 93.0%, 1.21 / **1.14** | 94.5%, 1.23 / 1.16 | 87.5%, 1.02 / 0.95 |
| prose 2048 | 85.9%, 1.04 / 1.01 | 85.9%, 1.12 / **1.09** | 85.9%, 1.08 / 1.00 | 76.6%, 0.91 / 0.85 |
| Italian 2048 | 84.4%, 0.95 / 0.90 | 84.4%, 1.01 / **0.95** | 85.2%, 1.05 / 1.01 | 79.7%, 0.94 / 0.87 |
| code 2048 | 93.8%, 1.15 / 1.16 | 93.8%, 1.25 / **1.26** | 94.5%, 1.31 / 1.33 | 85.2%, 0.93 / 0.90 |
| prose 4000 | 80.5%, 0.89 / 0.87 | 80.5%, 0.96 / **0.93** | 83.6%, 1.02 / 0.99 | 71.1%, 0.78 / 0.73 |
| Italian 4000 | 88.3%, 1.04 / 0.99 | 88.3%, 1.18 / **1.12** | 88.3%, 1.22 / 1.19 | 76.6%, 0.83 / 0.80 |
| code 4000 | 96.1%, 1.02 / 1.00 | 96.1%, 1.15 / **1.14** | 96.1%, 1.25 / 1.21 | 88.3%, 0.92 / 0.92 |

- **The 8-bit copy costs the draft nothing**: its token equals the 16-bit draft's at all 1152
  positions, and it lifts every cell by 0.02-0.14× (the history's bytes 256 → 142 MiB a draft token
  at 2048). The in-sample best overstates by 0.00-0.08× (the out-of-sample column is the one that
  counts; 64 positions a half, itself noisy).
- **Prose stays at 0.93-1.09× and Italian at 0.95-1.12× out of sample** (kv8): 1.1× only on Italian at
  4000 and on code (1.14-1.26×). What is left of a draft token is its weight plane (635 MiB against
  the copy's 80-270): the next lever is the draft's weights, or another bus (question 74).
- **The 16k-row head is out**: the lowest 16384 ids hold 85-93% of the exact tokens, and a miss costs
  more than the 37 MiB saved (0.73-0.95× out of sample).
- **The 4-bit copy agrees more often than the 16-bit one** (question 76): it changes 27 of 1152 tokens,
  and where one of the two drafts is right and the other not, the 4-bit one wins 18 and loses 3,
  mostly where the exact margin is under 1 nat (a sign test ~0.001). The path is the same (the 8-bit
  copy changes none), so the gain is the 4-bit rounding itself; not used until it is understood.

## The race after the balanced experts (2026-09-26)

`build/entangled/race_when_free.sh` started by itself at 13:59 (`tools/race_llama.sh`: container,
Q8_0, 8 threads, prompts 512 and 2048, 128 generated, 3 runs a series, A B B A; the engine of the
day: the balanced experts and the vector exp). Background 3.6-3.7 logical processors before and after
(0.96-1.06 of them the kernel's System; the free machine of 03:02 had 1.6-2.5), and a 3-second
native build of mine (16 jobs) at 14:09 fell on the last 2048 series (LESSONS #204): indicative.

| | 512: trochilus a / b | llama.cpp a / b | theirs / ours | 2048: trochilus a / b | llama.cpp a / b | theirs / ours |
|---|---|---|---|---|---|---|
| decode tok/s | 33.27 / 32.50 | 33.47 / 33.41 | 1.006-1.028 | 26.29 / 25.55 | 25.37 / 27.03 | 0.965-1.058 |
| prefill tok/s | 257.3 / 237.3 | 268.2 / 260.4 | 1.042-1.097 | 236.5 / 238.2 | 243.1 / 238.0 | 0.999-1.028 |

Read as structure only (LESSONS #205). **Again on a free machine** (`build/entangled/strict_rerun.sh`:
started at 14:28 once the machine stayed at <= 2.5 processors for 30 s; background 2.04 before the
first series, 2.93 after the last; the series still at the old limit of 3.5, the guard was tightened
two minutes later):

| | 512: trochilus a / b | llama.cpp a / b | theirs / ours | 2048: trochilus a / b | llama.cpp a / b | theirs / ours |
|---|---|---|---|---|---|---|
| decode tok/s | 33.02 / 32.35 | 33.06 / 32.46 | **1.001-1.003** | 25.70 / 25.67 | 26.84 / 26.66 | **1.039-1.044** |
| prefill tok/s | 264.8 / 237.8 | 255.1 / 258.3 | 0.963-1.086 | 245.6 / 244.2 | 235.1 / 246.1 | 0.957-1.008 |

**The decode is level with llama.cpp at 512 and theirs is 1.04× at 2048** (on 09-25: 1.04-1.05× and
1.12-1.14×): the balanced experts (1.04-1.08×) and, at 2048, the decode attention's softmax in a
vector. The prefill at 8 threads is level at both. What is left at 2048 is the KV's bytes (their F16,
§The attention against llama.cpp's): the GPU attention passes it.

## The softmax's exp in a vector (question 70, 2026-09-26)

Question 58's exact float-only exp is a kernel-table entry: `expf_f32(x, y, n)`, y[i] =
`tr_expf(x[i])`, y may be x, the scalar definition a loop over `tr_expf`. AVX-512 (16 lanes) and
AVX2 (8) run question 58's variant without FMA (ln2/256 in three 8-bit pieces, Dekker's product): a
lane is settled when an error of 2^-39 cannot move its rounding (on the subnormal grid below
2^-126), and the others, about 1 in 33 000, take `tr_expf` itself. `tr_softmax` (the prompt's and
the decode's attention, the router) subtracts the max in float and calls it in place; `tr_swiglu`
calls it in chunks of 64 on the stack. `test_expf` compares every tier with `tr_expf` on all 2^32
floats, in place and not, tails 0-16 (12 s with gcc on 4 threads, 41 s under ASan): **0 differ in
each tier**; softmax and SiLU bit for bit under every tier. Seen red (`tools/mutate_expf.sh`): lane
3 of the AVX-512 tier one ulp up (268 376 318 floats), the unsettled lanes keeping the fast guess
(10 204 avx512, 10 422 avx2: only the exhaustive loop sees it).

| `bench_attn 2048 --threads 1 --heads 1`, ns a (query, position), 3 rounds alternated, loaded machine | before | after |
|---|---|---|
| group, whole (the softmax in it) | 11.87 / 13.30 / 12.57 | 7.68 / 8.86 / 10.01 |
| tile without the softmax (the control) | 5.90 / 6.80 / 7.73 | 5.11 / 6.83 / 7.32 |

The softmax's share of a pair ~5.8 → ~2.0 ns (median minus the control), the bits' hash the same
before and after: **1.42×** on the prompt's attention against 1.5-1.8× predicted, indicative.

**In the engine, natively** (`build/entangled/after_race.sh`, started by itself after the race:
`tools/ab_zone.sh`, Q8_0, 16 threads, the launch snapshot against the vector exp, 8 pairs alternated
run by run; background 2.26-2.79 processors, 0.84-0.89 of them System; predicted first: the zone
0.67-0.77, the prefill +3-5% at 2048 and +4-7% at 4000):

| prompt | the attention zone, after / before | the prefill, after / before |
|---|---|---|
| 2048 | **0.753** [0.729-0.773] (0.42-0.45 → 0.32-0.34 s) | **0.975** [0.948-0.990] |
| 4000 | **0.742** [0.725-0.778] (1.80-1.91 → 1.30-1.44 s) | **0.948** [0.930-0.975] |

Not a completely free machine (it idles at 1.8 with the containers stopped; LESSONS #205). **Again
on a free machine** (`build/entangled/strict_rerun.sh`, after the race; background 2.42 before, 2.30
after; each run at <= 3.0; data in `build/entangled/strict/ab_zone/`):

| prompt | the attention zone, after / before | the prefill, after / before |
|---|---|---|
| 2048 | **0.732** [0.710-0.772] | **0.951** [0.943-0.970] |
| 4000 | **0.747** [0.634-0.859] | **0.953** [0.815-1.038] |

At 4000 three of the eight pairs ran through a burst elsewhere (the before binary 11.3-12.7 s against
10.3-10.4); the five clean pairs give the prefill 0.934-0.968. **The prompt's attention 1.34-1.37×,
the prefill 1.05× at 2048 and at 4000**, inside the prediction (+3-5% and +4-7%). Open: the SwiGLU's
share, a NEON tier.

## The draft's two levers (question 71, 2026-09-26)

Container, `--text 1` (the real text), 128 positions a run, one run a cell, bytes from the report's
model (not measured); llama.cpp 1264 / 1456 / 1700 MiB a token at 512 / 2048 / 4000. Two levers
on the Q8_0 high nibbles (`planes`, the whole KV at 16 bits): (a) the draft's second choice as one
more leaf of the verification, at the first position where the draft's own margin is under tau (+1
row, the union of experts grown by one row); (b) a KV of constant bytes: `nx96` keeps the sinks (4),
the last 64 and the 96 positions the exact previous token weighted most, head by head, plus the
position after each (<= 260, ~33 MiB); `sel192` the top 192 without the successors. Data in
`build/w3-draft/` (`report.txt`: every variant).

| text, context | planes: agreement (sure) | planes + leaf: tokens a pass, vs llama.cpp | nx96: agreement (sure) | nx96 + leaf: tokens a pass, vs llama.cpp |
|---|---|---|---|---|
| prose 512 | 85.9% (98.6%) | 3.35, 1.02× | 84.4% (97.2%) | 2.93, 0.98× |
| code 512 | 93.0% (98.9%) | 6.28, 1.16× | 91.4% (100%) | 6.05, **1.20×** |
| Italian 512 | 85.2% (100%) | 4.07, 1.02× | 81.2% (98.6%) | 2.60, 0.95× |
| prose 2048 | 85.9% (100%) | 5.38, 1.04× | 82.8% (97.6%) | 3.23, 1.03× |
| code 2048 | 93.8% (99.0%) | 6.82, 1.15× | 89.1% (96.0%) | 5.27, **1.20×** |
| Italian 2048 | 84.4% (97.5%) | 3.37, 0.95× | 77.3% (90.0%) | 3.08, 1.02× |
| prose 4000 | 80.5% (100%) | 3.02, 0.89× | 63.3% (87.5%) | 2.19, 0.90× |
| code 4000 | 96.1% (99.1%) | 4.88, 1.02× | 96.1% (98.2%) | 9.38, **1.46×** |
| Italian 4000 | 88.3% (100%) | 5.35, 1.04× | 69.5% (84.1%) | 3.03, 1.04× |

- **Only code passes llama.cpp by 1.1×** (1.15-1.20× at 512 and 2048, 1.46× at 4000 with nx96), and
  the code text repeats (rep4 9-12%): the easy case, where the prompt's lookup already gave 1.66× net
  on repetitive code (question 52). Prose and Italian stay at 0.89-1.04×.
- **The leaf is the lever that works**: +0.07-0.10× on prose and Italian (predicted at most +2%:
  wrong), tokens a pass 1.80 → 3.02 on prose at 4000 and 1.87 → 4.07 on Italian at 512; the draft's
  top two hold the exact token at 92-100% of the positions.
- **In the selection, the successors are what works**: the position after each one the previous
  token looked at, which a copying head reads next. Without them (sel192) the sure positions fall to
  61-93% (predicted >= 97%: wrong); nx96 keeps them at 512 and on code, and loses them on prose and
  Italian at 2048-4000 (84-90%).
- What the draft may cost for 1.1× (planes' agreement, with the leaf): 582-801 MiB a token on prose,
  598-1034 on Italian, and the weight plane alone reads 635: a KV of constant bytes that loses nothing
  passes at prose 2048 and 4000 and Italian 4000; at 512 the weight plane itself must shrink.
- Upper bounds: the best (k, tau) is chosen on the same 128 positions it is scored on; the probe
  hands the draft the exact selection of position p-1, where a chain's j-th token has one j steps old
  (not measured); the leaf's chain stops at the fork; the union of experts is extrapolated past 16
  rows.

Open: nx96 on prose and Italian at 2048-4000 (the exact pass hands the draft the part of the softmax
the draft skips, max, sum and weighted V a head: head_dim + 2 floats, constant; or nx192, ~56 MiB);
the staleness of the selection; bytes measured instead of modelled.

## The prompt from 8 to 16 threads (2026-09-26)

The race's gap (§Speed after x8, container, Q8_0): our prefill 8 → 16 threads 1.11-1.16×,
llama.cpp's 1.44-1.49×. Natively on Q4_K (the traced build, 2048 tokens of olmoe.c, `generate -n
1`, one run a width, twice with the order swapped; another agent's container ran beside it, so
structure, not decisions; `tools/prompt_scale.sh`, `tools/prompt_timeline.py`, data in
`build/w2-scale/`):

| run | order | 8 threads | 16 threads | 8 → 16 |
|---|---|---|---|---|
| a | 8 first | 154 tok/s | 229 | 1.49× |
| b | 16 first | 163 | 245 | 1.50× |

**Natively on Q4_K the prompt scales as llama.cpp's did in the race**: the 1.11-1.16× belongs to the
container (the WSL2 VM), to Q8_0, or to both; not separated today (the VM had 2.3 GB free and the
engine refused the Q8_0's store). Where 16 threads lose against a perfect 2× (the pool's time over
a perfect scaling of the 8-thread run: 2763 / 2047 ms of 8831 / 8268):

1. **every unit of work slower, 61-67% of the loss**: GFLOP/s a busy thread 8 → 16, dense q/k/v/o
   61.0 → 49.1, experts' gate/up 57.9 → 47.2, down 54.8 → 45.5, attention 40.3 → 29.2. The dense
   projections are compute-bound (512 tokens a weight) and slow down 1.24× like the experts: an
   all-core effect (the clock with 16 busy cores, the shared L3), not the partition. Inferred, not
   measured (a fixed-FLOP loop at 8 and 16 busy cores says it); llama.cpp pays it on this chip too,
   and 1.24× caps 2× at 1.61×;
2. **the tails, 33-39%**: 12.2-12.3% of the pool's time at 8 threads, 17.4-20.5% at 16. The
   prompt's matmul gives every thread one fixed region (`tr_parallel_for`) and some cores are
   slower: chunks 0, 2, 8, 12 (cores 0, 1, 4, 6 of the first CCD) take 3-19% longer a unit, chunk 0
   (the calling thread) finishes last in 25-44% of the big calls at 16. The regions are equal (32
   tokens × 2048 rows in the dense calls): the cores' speed, not the cut;
3. the per-row calls without weights (norms, adds, RoPE, KV writes; 580): 293-336 ms at 16
   against 273-301 at 8, 0.5-0.7% of the prompt;
4. between calls 91-98 ms at both widths: nothing.

**Proposed, not built** (the largest piece that is ours): the prompt's grouped matmul stealing work
in blocks that keep whole 8-token tiles (a group, 16 output rows, all its tokens; regions cut by
cost, then `run_chunk`'s atomic stealing). The balanced call's contiguous ranges hold at most 4
whole tiles in a 32-token region, too few to absorb a core 1.17× slower. Same bits: every element is
one dot of a kernel bit-identical to scalar, the partition only chooses the thread. Predicted
natively: the tails 17-20% → ~4% of the pool's time, 16 threads −13 to −16% (~245 → 285 tok/s), 8
threads −7 to −8%; less on a still machine if part of the slow cores was the other load. Open: the
container's Q8_0 at 8 and 16, pinned and not (`PS_BIN=build/linux-gcc/trochilus TR_POOL_PIN=0|2 sh
tools/prompt_scale.sh <out> 16 8`: if the pin inside the VM is the cause, since Hyper-V places the
vCPUs as it likes and llama.cpp does not pin, pin 0 at 16 threads gives >= 1.3× pin 2); the native
Q8_0 with 10 GB free.

## Phase-major: the prompt's matmul at the float's peak, the same bits (2026-09-26)

Marcello's task: beat llama.cpp where it still led, the Q4_K prompt at 4 threads (theirs 1.6×, 222 against
139 tok/s, their activations rounded to 8 bits) and the Q8_0 prompt at 16 threads (theirs 1.30-1.35×,
container). A thinker (Opus) and the orchestrator bouncing ideas in rounds; predictions in
`build/e24/predictions.txt`, prototypes in the session's scratch.

**The core's genome** (native, one core, 5.23 GHz, inline asm, 8 independent chains): `vpdpbusd` zmm one a
cycle (64 byte-MACs), latency 4. Every multiply takes its slot: `vpmulld`, `vpmullw`, `vpmaddwd`,
`vpdpwssd`, `vmulps`, `vmulpd` (8 dp + 4 of them = 12 cycles). Adds, logic, shifts, the conversions
`vcvtdq2ps`/`vcvtqq2pd`/`vpmovsxdq`, moves and broadcast loads are free beside one dp a cycle, up to about
one zmm op each (8 dp + 12 ALU = 1.25 cycles a dp); `vpshufb` and `vpermb` are not. F32 without FMA: one
`vmulps` and one `vaddps` zmm a cycle, 16 MAC/cycle, 167 GFLOP/s at 5.23 GHz.

**First idea, E24 (exact integers): measured, set aside.** Activations in 24-bit block fixed point (3
signed 8-bit digits), `vpdpbusd`, exact int32 sums, F64 across super-blocks, one rounding: a new
definition, free of order. Kernel 145 → 157.6 GFLOP/s-eq a core, bit-identical to its own scalar
definition; Q8_0 144.5. Accuracy on the engine's real inputs against the exact dot (TwoProduct + fsum), in
F32 ulps, median / p99: today's F32 definition 1.8-2.3 / 108-166; E24 with one shift per 256 2.0-5.2 /
132-525; with a shift per 32-sub-block (at most 3 finer, folded into the rescale) 1.3-2.4 / 81-212, the
same class as F32 and not better (the thinker's forward: 41% of outputs worse, 22% twice worse); E32 (4
digits) 0.25 / 0.5-1.2, correctly rounded 96-99%. In the engine (a scratch build, weights repacked in a
side cache): 4.29 M outputs equal to its scalar definition, the generated text identical, the prompt 193
tok/s at 4 threads (1.38×), the experts at 70% of the kernel. Set aside: it changes every byte for
today's accuracy. E32 stays a question (more exact than the float; exact on int8 tensor cores).

**Phase-major (the thinker's idea).** The lane contract's lane p is the chain k = p, p + 16, ... in
increasing k: a sequence in time, not a place. Sixteen weight rows can run it side by side, one row a SIMD
lane, each lane adding dot_row's products in dot_row's order, then the same tree on the 16 phase vectors:
the same bits by construction, and no scale, decode or reduction in the loop (a step is one panel load, T
multiplies with the input broadcast from memory, T adds).
- The stream alone: 157.4 GFLOP/s, 15.05 MAC/cycle, 0 of 768 differ (predicted 15.0-15.9). With GCC's
  inline asm taking the broadcast as an "m" operand the 24 accumulators were stored every step: 7.6.
- A whole Q4_K matmul (dequantized and transposed into a panel per 16 rows): 130 GFLOP/s at 1024 × 2048
  × 64, 150 at 2048 × 2048 × 512; 0 of 65 536 and of 1 048 576 outputs differ from `dot_row`.
- The engine's own pieces alone (one core, n = 2048): the tile 164.7 GFLOP/s at T = 24 (98.6% of the
  peak), 162-164 from T = 12, 138-154 at T = 4-10. The panel 3.48 µs (11.3 token-equivalents) → 2.74 (the
  scales by `avx512_q4_k_scales_store`: scalar they were 0.69 of 3.29 µs, one piece removed at a time) →
  1.83 (the quants transposed as bytes, 8 `vpermt2b` for 512 weights, VBMI: the float transpose was 0.38 of
  the panel). The inputs' interleave 61.7 µs at T = 24 (stores 8-12 KB apart in one L1 set) → 7.9 (16 × 16
  transposes in registers and 16 floats of pad after each phase's run; the pad on the panel too), once a
  call for 64-128 row groups.
- **The engine, a 512-token prompt, native, a free machine (background 1.6-1.9), alternated**: Q4_K at 4
  threads 139.9-142.4 → **218.3-223.2 tok/s** (median ~220; llama.cpp 222 in the container); 8 threads
  241.4 → 352.5; 16 threads 368.8 → 575.2. Q8_0: 8 threads 266.5-287.1 → 374.4-376.6; 16 threads
  390.4-418.9 → 550.4-569.0. The decode unchanged (4 threads 54.6-58.6 against 55.0-58.5): a group under
  4 input rows keeps the x4 and x8 kernels.
- **Exactness**: the real model's logits byte-identical before and after (Q4_K, 160 tokens, at 1, 4, 8 and
  16 threads; Q8_0, 96 tokens, at 16); 48 generated tokens identical; `test_phase_major` (every tile width
  4..24, n 256 / 512 / 2048, ordinary and special blocks, Q4_K and Q8_0) red on a regrouped tree (6 941
  checks).
- The zones at 4 threads (Q4_K): the dense q/k/v/o 141-143 GFLOP/s a core (87% of the tile alone), the
  experts 124-127. What is left on the experts is the panel (1.83 µs per expert and 16 rows, against a
  median of 28 tokens an expert) and the small tiles.
- The thinker's plan, from OLMoE's real routing (1024 tokens, 16 layers): tokens an expert in a 512-token
  pass median 28, p10 2, p90 165, max 511; tiles of every width 4..24 cut evenly (no padding, 1.7% of the
  work under 12); items (expert, 16 rows) costed by tokens with stealing (a static split by count leaves
  38% tails at 4 threads: the E24 hack's 70%).

Open: the panel toward the thinker's P ≈ 3 token-equivalents (the phase vectors' decode is what remains),
the interleave shared by gate/up and by q/k/v, the race in the container (Q4_K at 4 threads, Q8_0 at 16).
The float-transpose Q4_K panel now runs here under `TR_CPU_MAX=avx512-novbmi` (LESSONS #213).

## Exact sums at the float's speed: Marcello's challenge (2026-09-26)

The question: a prompt matmul that loses no bit in its sums (activations as ≥ 31.5-bit block fixed point,
exact integer sums, one rounding: E32's class or better, correctly rounded 96-99% on the real inputs) and
runs at least as fast as phase-major's float. Every idea with its prediction first
(`build/e24/predictions.txt`), prototypes in `build/e24/wino/` (wino.c, e32w.c, gpu_tc.c, results.txt).

**The budget.** This core issues 2 zmm ops a cycle, at most 1 of them multiply-class (§Phase-major's
genome). The float kernel spends both: a `vmulps` and a `vaddps` per 16 MACs, 16 MAC a cycle. An exact
32-bit activation times a 4-bit weight is 128 bit-products; `vpdpbusd` gives 64 byte products (8 × 8) a
cycle and the weight fills half of each: 4 digit passes per 64 MACs, 16 MAC a cycle again. Exactness at
the float's price, no better, unless the byte's free half does work.

**Spirit quantum: the activation rides in the weight's free nibble (Winograd 1968, FFIP 2023).**
Σ_j (q_2j + D_2j+1 + 120)(q_2j+1 + D_2j − 8) = Σ_k q_k D_k + a row term (at load) + a token term (once a
call): with digits D in [−120, 120] (base 241) both factors fit a byte (u8 and s8), so one `vpdpbusd`
carries 128 digit-MACs for 2 byte adds on the free ALU slot. Predicted and measured to the cycle (wino.c,
one core, cycles per block of 8 dp): 8 plain 8.00 (64 digit-MACs a cycle), 2 Winograd + 6 plain 8.00
(80), **4 + 4 8.00 (96: 1.5×)**, 5 + 3 9.00, 6 + 2 10.00, 8 Winograd 12.00 (85). E32 at 4 + 4 is 24 exact
MACs a cycle in the pure loop, 1.5× the float's 16.
- Spirit pioneer, the scalar core beside the vector: `mulx` of a 32-bit activation by two weights 40 bits
  apart gives 2 exact MACs; alone 1.8 cycles each, 4 of them free per 8 dp, 8 cost 14 cycles: +1 exact MAC
  a cycle (+4-6%), not pursued.

**The tile (e32w.c): one definition, two kernels, 0 of 64 outputs differ from its scalar definition.**
16 rows × 4 tokens, K = 2048, Q4_K rows repacked 16 at a time (e24.c's bytes); T and M exact in int64,
the f64 steps of E24's definition. Plain E32 (4 digits base 256): **127.0** GFLOP/s-eq (L1), 125.3 (L2).
Winograd (W0, W1 base 241 + P2, P3 base 256): **136.6** / 135.1. The float tile: 164.7. The deletion
series of the Winograd kernel, one piece out at a time: the formation 6%, the per-sub-block init 0.7%,
pairing the digits and the sub-block scale 12%, the nibble unpack 3%, **the super-block's end 23%** (the
mins per digit by `vpdpwssd`, T and M combined by `vpmullq` with extracts). That end rewritten in f64
only (every value an integer under 2^53: T = lo + 58081·hi by one exact fma, M = Σ m_j B_j by 8 exact
fmas from the token's exact sub-block sums): **149.0**, still bit-identical; the scale and pairing are then
the largest piece (13%). The ports' model of that kernel: ~28.75 multiply-slot and ~30 ALU ops per token
and sub-block, 17.4 exact MAC a cycle = 182 GFLOP/s-eq (1.09× the float's peak); measured 82% of it.
Q4_K's per-32 scales and mins, which the float folds into its panel once for every token, cost the
integer kernel per token: about a third of its ops. 2 + 2 digits is the optimum (3 + 1 and 4 + 0 are
ALU-bound).
- **W16, the scale inside the weight (the thinker's lens, built and measured):** the 16-bit word
  w = sc_j·q_k (10 bits) carries its sub-block's scale into a panel, the digits are 16 bits (base 64591,
  2 of them ≈ 31.96 bits), one `vpdpwssd` takes two Winograd pairs a lane ((w_k + V_k' − 473)(w_k' + V_k
  − 473) fits s16 × s16). The int32 lanes wrap; a 64-column window's true sum stays under 2^31, so once
  its row and token terms are subtracted the lane is exact; windows go to f64 by even and odd lanes
  (shifts and `vcvtqq2pd`, no shuffle). Port genome: 8 `vpdpwssd` 7.99 cycles (32 word-MACs a cycle),
  8 Winograd 12.00 (42.7), 4 + 4 9.33. The tile, bit-identical (0 of 1024): **158.7** GFLOP/s-eq (L1),
  150.5 (16 groups, L2: the panel is 64 KB a group); same run, plain 126.2, Winograd in bytes 151.2.
  0.96× the float tile; the model gives 182 (87% reached), the formation is 64 of its ~118 ops a window.
  Its deletion series (L1): full 156.0, without each window's flush into f64 177.7 (+14%), without the
  mins and the super-block's end 173.1 (+11%), without the window's init 159.7, without all three
  **198.6**: the W16 loop itself runs at 1.21× the float tile. The window (64 columns, so the true sum
  stays under 2^31) and the mins (per row and per token, as the float pays them only in its panel) are
  what is left between exact and faster.
- **The thinker's round on W16, each idea with its prediction (all bit-identical, 0 of 1024):** (1) each
  window into int64 by a biased even/odd split: the 16 lanes read as 8 qwords, the odd lane is the qword's
  high half (`vpsraq`), the even lane the whole qword with +2^31 riding in the window's init, the bias
  and the odd part taken out once a super-block: 14 ops a window and token → 6. Predicted 165-170:
  **167.2** (L2; 16 groups 156.9). (3) the mins' 8-deep fma chains split in two (exact: integers under
  2^53): predicted +3-5%, measured +0.5%, not latency. (2) the 4W+4P anomaly isolated in wino.c: 4W+4P
  9.05 cycles, plain ops in register form 8.00, one broadcast a Winograd op 8.00, `vpaddd` for `vpaddw`
  9.28, 3W+5P 7.99: the broadcast loads (12 per 8 dp) are not free above about one a cycle; a broadcast
  shared by two row groups (32-row tiles) would take the loop from 42.7 to 48 word-MACs a cycle, not built.
- **So on the tile, the challenge is met:** the exact W16 tile at T = 4 runs **167.2 GFLOP/s-eq against
  the float's 164.7 at T = 24** (both panels in L2), and against the float's ~138 at T = 4, 1.21×. Its
  panel (the 16-bit words sc·q and each window's row terms) is half the float panel's bytes and needs no
  conversion; not yet timed, nor the activations' digits, nor the engine.

**Four dimensions, which unit: the GPU's int8 tensor cores (gpu_tc.c, PTX through the driver).** RTX 4070
Laptop, 36 SMs, registers only: `mma.sync` m16n8k32 s8 × s8 → s32 (exact) **59.10 T-MAC/s**; the float
contract `mul.rn` + `add.rn` **3.29**; `fma.rn` 6.46; `dp4a` 14.66. E32's 4 digit passes on the tensor
cores: 14.8 T-MAC-eq/s, **4.5× the float contract on the same GPU** (2.3× even an fma the contract
forbids), ~12× this CPU's whole float peak. The first run gave mul + add 6.39: the 16 chains shared one
product and the driver's compiler computed it once; each chain now has its own weight.

**Verdict so far.** On this CPU's vector unit the exact E32 tile by W16 is at least as fast as the float
(167.2 against 164.7; bytes-Winograd 149.0, plain digits 127.0), with the float's own ceiling 167 and
W16's loop alone at 198.6; Q8_0 has no exact kernel faster than the float (the thinker's model: its per-32
F16 scale cannot ride in a 16-bit word, ~0.9×). Where int8 matrix units exist (the GPU's tensor cores
here; AMX, SME, NPUs elsewhere) the exact integer definition is also the fast one, and the float contract
cannot use them. Order-free sums also make any split, width or device give the same bits.

**W16 on the engine's own layouts: stage 1 of question 77's plan (Marcello's ok, 2026-09-26 night).**
Each piece timed alone before building anything in `src/` (`build/e24/wino/w16.h`, `w16_panel.c`,
`w16_digits.c`, `w16_tile.c`, `zone_model.py`; predictions in `build/e24/predictions.txt`). One change to
the prototype's layout: the Winograd pairs are the low and high nibble of one quant byte, columns
(64c + i, 64c + 32 + i), so the panel needs no nibble shuffle and the input's digits stay in natural
column order (the tile broadcasts the dwords at 64c + 32 + 2u and 64c + 2u). The panel per super-block:
d, dmin and the eight m_j as f64 (even rows, then odd: the order the windows' int64 split gives), then
four windows of a row term (int32, the even rows biased by 2^31) and 16 pairs of 16-bit weight vectors
sc·q: 9 728 bytes, 76 KB for 16 rows × 2048 against the float panel's 128 KB. Native, one core, best of 7,
background 1.7-2.0:
- **the panel** (a 16 × 32-byte transpose in 24 `vpermt2b` a window, the nibbles widened by `vpmovzxbw`,
  times the rows' scales by `vpmullw`, each window's row term by `vpdpwssd`): byte for byte its scalar
  definition over 64 panels; hot **1.10 µs** against the float VBMI panel's 1.83 (0.61×; predicted
  0.45-0.65 µs: wrong), **cold 1.95 against 3.76-3.81 (0.52×)**. Its deletion series: the row term 18%
  (0.90 without it), its 16-deep `vpdpwssd` chain split in four 0%;
- **the input's digits** (a shift per super-block by `vscalefps`, X by `vcvtps2dq`, V1 = round(X/64591)
  in f64, V0 by `vpmulld`, the windows' token terms by one `vpdpwssd` each, the sub-blocks' exact sums in
  f64): its scalar definition on every token (a zero block, a block of subnormals, a large negative
  maximum); **0.65 µs a token of 2048** against the float interleave's 0.23 (2.8×). Computed once a
  token for gate and up (shared by the token's 8 experts) and once a (token, expert) for down: 0.5% of
  the experts' work;
- **the tile**, 16 rows × T tokens on those two, the windows into int64 by the biased even/odd split:
  **0 of 1 536 outputs differ** from the definition computed straight from the Q4_K bytes and the floats
  (T = 1..4, the same special blocks). GFLOP/s-eq, the panel in L2:

| | T = 1 | 2 | 3 | 4 | 8 | 12 | 24 |
|---|---|---|---|---|---|---|---|
| W16 tile | 122.2 | 156.0 | 165.6 | **168.2** | | | |
| float tile | | | | 127.2 | 140.6 | 155.9 | 161.5 |

- **the experts zone** (d): OLMoE's real routing (32 (layer, pass of 512) pairs), items (expert, 16 rows)
  as the engine cuts them, cold panels, the tiles' measured rates: float 141.1 GFLOP/s-eq a core (the
  engine measures 124-127, 0.89 of the model), W16 155.8: **1.10× on the experts zone**, and at least as
  fast on the dense projections (168.2 at T = 4 against the float's 161.5 at T = 24). The estimate wins:
  stage 2 (the definition in the kernels, prompt and generation together) is next.

**Stage 2: Q4_K's definition is the integer one, in the engine (2026-09-26 night).** kernels.h `q4x_*`: an
input row prepared once (`q4x_prep`), the decode's pairs (`q4x_dot2`), the W16 panel and tiles of 1-4
rows (`q4x_panel`, `q4x_tile`); the scalar definition, AVX-512 (VBMI, VNNI, DQ: W16) and AVX2 (by rows,
also the avx512 tier without VBMI); the driver `matmul_q4x` prepares every input row once, groups of 2
input rows or more take panels and tiles, one-row groups the pairs. The float Q4_K kernels are gone
(419 lines). Exactness: the real model's logits identical at 4 and 16 threads over 600 positions, and
position 599 the same by one-token passes and by passes of 64; the 2-layer cut against its reference,
max |logit diff|, float → integer: Q4_K 1.50e-05 → 1.11e-05 (dante), 1.850e-04 → 1.835e-04 (long);
Q4_K_M 1.29e-05 → 1.05e-05, 1.96e-04 → 1.94e-04 (every greedy token the same). Speed, native, Q4_K,
`tools/ab_zone.sh` HEAD against W16, 6 alternated pairs, 4 threads:

| | predicted | measured |
|---|---|---|
| the prompt of 512, experts zone after/before | 0.88-0.93 | 0.992 [0.987-0.997] |
| the prompt of 512 | 0.90-0.95 | 1.001 [0.995-1.008] |
| the decode's matmul zones, first `q4x_dot2` | 0.95-1.02 | 1.080 [1.064-1.102] |
| the decode's matmul zones, final `q4x_dot2` | 0.95-0.99 | **1.006 [0.989-1.016]** (decode 1.004) |

- **The decode's pair, sequenced** (`build/e24/wino/dot2_ram.c`, `dot2_genome.c`: two rows of 2048 in
  L1, and 256 MB of rows from RAM): the first kernel 188 ns a pair against the float's 172 (1.09×; RAM
  1.09× on one thread, 1.12× on four); the block end's four reductions joined in f64 (exact: integers
  under 2^53) 177 (1.03×); its chains split in two 183 (worse: not latency, reverted). The deletion
  series: no header pre-pass −16%, no quant loads and widening −11%, no nibble split −6%, no `vpmullw`
  −4%, no digit loads −3%, the scales' broadcasts 0. The headers of both rows in one ymm (a row a
  128-bit lane) and d, dmin of both by one `vcvtph2ps`: **161.6 ns = 0.937× the float's** hot, RAM
  0.954× on one thread and 0.971× on four (51.0 against 49.5 GB/s); in the engine level (above).
- **The prompt did not gain**: per item (expert, 16 rows; 68 tokens on average, not the median's 28)
  the model gives W16 28.5 µs and the float 31.5; the engine takes about 34.5 for both. The float's
  tile keeps 0.90 of its microbench rate in the engine, W16's about 0.83: the microbench fed the tile
  the same four tokens every call, the engine feeds it four others each time (their digits from L2 or
  L3), and at T = 4 the 76 KB panel is read again every four tokens (the float's 128 KB every 24).
  Not yet measured piece by piece in the engine (the next step: a deletion series of the item there).

## The race for the README (2026-09-26 night)

Both engines in the container on the same GGUF, a free machine (the guard's load 2.03-2.12 before the
first series, 2.09-2.49 after the last), each engine two series of 5 runs alternated with the other's
(`tools/race_llama.sh`; the Q4_K rows with `RACE_MODEL=<Q4_K> RACE_PROMPTS=512 RACE_THREADS=4`), the
median of the 10 runs. `tools/race_q4k.sh`, which loads the Q4_K and the Q8_0 file in turn, ended twice
as NOT FREE (2.51, 3.02) with Windows' Memory Compression at 0.49 right after its last series: its two
models (11.6 GB) press the memory; one model at a time left 0.25 (LESSONS #217).

| tok/s | llama.cpp | Trochilus | Trochilus / llama.cpp |
|---|---|---|---|
| Q4_K, prompt 512, 4 threads | 221.2 (221.0 / 221.4) | 215.0 (216.6 / 214.7) | 0.97 |
| Q4_K, generation after it, 4 threads | 51.2 (51.1 / 51.6) | 52.6 (52.8 / 52.6) | 1.03 |
| Q8_0, prompt 512, 16 / 8 threads | 370.7 / 260.2 | 492.9 / 363.6 | **1.33 / 1.40** |
| Q8_0, prompt 2048, 16 / 8 threads | 341.3 / 239.1 | 485.2 / 347.8 | **1.42 / 1.45** |
| Q8_0, generation at 512, 16 / 8 threads | 33.7 / 33.1 | 32.7 / 33.1 | 0.97 / 1.00 |
| Q8_0, generation at 2048, 16 / 8 threads | 27.7 / 27.2 | 26.4 / 26.3 | 0.95 / 0.96 |

**The Q4_K rows again on the exact engine** (Q4_K in integers, §Exact sums at the float's speed, stage 2;
the same command, 2026-09-27, background 2.44 before (a transient python at 1.00), 2.00 after): prefill
llama.cpp 220.6 (219.9 / 221.2), Trochilus 220.0 (219.5 / 220.1), **1.00**; decode 50.9 (51.0 / 50.1)
against 52.4 (52.1 / 52.8), 1.03. These are the README's Q4_K rows.

Predictions (written first): the Q4_K prompt 0.97-1.01 (0.97, the bottom), its decode level (1.03);
the Q8_0 prompt at 16 threads 0.95-1.10 (**1.33-1.42: wrong, phase-major scales to 16 threads in the
container too**), at 8 threads 1.2-1.35 (1.40-1.45); the decode level at 512 and theirs 1.04× at 2048
(1.00 and 1.04: right). What is left to llama.cpp: the Q4_K prompt at 4 threads (their activations in
8 bits; W16 is the exact lever, above) and the decode at 2048 (the KV's bytes, §The attention against
llama.cpp's).

## A draft without a model: an n-gram table and a grammar on new code (2026-09-27)

**The idea** (Marcello's thread on the quantum spirit, 2026-09-26/27). Moving data costs, computing
does not: every token moves ~1 GB of weights from RAM, the same bytes every time, known in advance,
while the new information of a token is a few KB. So something small that lives in the cache guesses
the next tokens, and the exact model verifies them all in one pass: several tokens for one trip of the
weights, and the same bits (`make spec-check`). It need not be a network: a table of the token
sequences most frequent in public code plus the language's grammar, built like q/kdb+ (its ~800 KB
interpreter lives in L2). The question: how often does it guess on **new** code, where the prompt
lookup gets 13% and loses (0.60x fixed, 0.953x adaptive, §Adaptive draft)? Decision rule written
first: near 50% accepted build it, 25-50% try a confidence gate, below 25% stop.

**How.**
- Text: the greedy continuation of 18 new-code prompts in 9 languages (`bench/prompts/new-code`: C,
  Python, JavaScript, TypeScript, Go, Rust, shell, SQL), 256 tokens each, 4 608 tokens, Q4_K, native,
  8 threads (`sh tools/draft_table_gen.sh`, ~20 min).
- Table: 78 MiB of public code already on this machine (llama.cpp, ds4, colibri, the venv's
  site-packages, node_modules, Go, shell, SQL; no Rust), 26.7 M tokens by the engine's tokenizer
  (`tools/draft_corpus.py`, then `tokenize --batch`). Contexts of 4 down to 1 token, each with its most
  frequent next token and that token's share (the confidence); pruned to the most frequent contexts
  within a budget, 16 bytes an entry. The draft chains the table and stops when the product of the
  confidences falls below a threshold.
- Replay: `tools/draft_table_sim.py` replays `tr_greedy_step` (fixed and adaptive policies,
  `src/gen/greedy.c`) with each draft source, the lookup copied line by line from `src/gen/lookup.c`.
  **Validated**: on three prompts stopped at 200 tokens the replay gives the engine's own counters
  exactly (129 passes 74/402, 138 passes 62/271, 129 passes 71/376).
- Speed: a model, from the native Q8_0 costs of §Adaptive draft (a one-row pass 31.3 ms, an extra row
  17.6 ms alone, 13.7 ms each at eight, linear between). On the prompt lookup it gives 0.68x fixed and
  0.97x adaptive where 0.60x and 0.953x were measured: slightly optimistic.
- Grammar: not built, its ceiling measured. A perfect grammar proposes the true next tokens while they
  are structural (no letter, digit or underscore: whitespace, brackets, operators, punctuation).

| Draft source, new code | Accepted | Tokens a pass | Estimated speed |
|---|---|---|---|
| prompt lookup (today), fixed 8 | 16.3% | 1.37 | 0.68x |
| prompt lookup (today), adaptive | 44.7% | 1.18 | 0.97x |
| table 4 MiB, proposes only at >= 70% confidence | **64.2%** | 1.10 | **1.01x** |
| table 16 MiB, >= 70% | 59.8% | 1.13 | 1.01x |
| table 16 MiB, >= 50% | 40.7% | 1.19 | 0.95x |
| table 85 MiB, >= 70% | 41.8% | 1.17 | 0.96x |
| full table (212 MiB), >= 70% | 26.2% | 1.20 | 0.86x |
| lookup, else the 16 MiB table (>= 50%), adaptive | 45.7% | 1.24 | 0.97x |
| perfect grammar (ceiling) | 100% | 1.79 | **1.25x** |

Readings:
- **No cheap draft passes break-even on new code**: the best is the small table proposing only when
  sure, 60-64% accepted but 1.10-1.13 tokens a pass, 1.01x. The rule's "near 50%" was reached and is not
  enough: a guess that is right but alone saves less than the extra row costs.
- **A bigger table is worse**: it knows more long contexts, seen a few times each, whose confidence is
  noise (the full table at 70% accepts 26%).
- **Structure is 44.2% of the tokens but scattered between names**: even a grammar that is never wrong
  gives 1.79 tokens a pass and at most 1.25x. Names are the program's own and no table holds them.
- Under the adaptive policy every mix lands at 0.96-0.97x, as the lookup alone: where the lookup fails,
  the table adds nothing.
- **The lever is the cost of a verified row, not the guess**: a drafted token routes to other experts
  (§Experts read by a pass of k rows), so a row costs about half a pass. The table at 30% confidence
  gives 1.21-1.35 tokens a pass: with a row at a fifth of a pass it would pay. The draft inside the
  exact bits (question 71, 82-98% agreement) remains the strong source; this closes the cheap one.
- **Reopened 2026-09-27** (§The post-it taken apart): the replay's row price was nine days older than the
  engine; at today's prices a calibrated gate gives 1.077x on the same texts.

## The post-it taken apart: the verified row's price today, and a gate that pays (2026-09-27)

**The doubt** (Marcello: the section above closed too early). Taken apart with the method: the replay priced
every pass with the constants of 2026-09-18 (an extra row 17.6 ms alone, 13.7 each at eight, of 31.3), older
than x8, the balanced experts, phase-major and the exact Q4_K; and already then the extra row's new experts
ran at 29.8 MiB/ms against the one-row pass's 49.8, "the why is not measured" (§Revisione). Predictions in
`build/rp/predictions.txt`, written before the run.

**The price today** (`sh tools/row_price.sh`, native, attention on the CPU for every mode, code-edit with the
draft fixed at 1, 2, 4, 8: every pass then has 2, 3, 5, 9 rows; three rounds alternated after a warm-up;
background 1.59 logical processors before, 1.57 after: free). Pass ms, then one more row's ms and share of a
pass; the new experts' MiB/ms (the one-row pass reads its experts at 53.1 on Q8_0, 49.5 on Q4_K):

| rows a pass | Q8_0, 8 threads | new experts | Q4_K, 8 threads | new experts | Q8_0, 16 threads |
|---|---|---|---|---|---|
| 1 | 25.71 | | 15.90 | | 26.52 |
| 2 | 40.41: 14.70 (0.57) | 36.3 MiB/ms | 23.89: 7.98 (0.50) | 41.5 | 45.51: 18.99 (0.72) |
| 3 | 62.53: 18.41 (**0.72**), dense +6.71 a row | 36.3 | 28.72: 6.41 (0.40) | 42.1 | 69.06: 21.27 (0.80) |
| 5 | 59.13: 8.36 (0.32) | 47.4 | 38.64: 5.68 (0.36) | 39.0 | 59.81: 8.32 (0.31) |
| 9 | 79.62: 6.74 (0.26) | 45.4 | 54.03: 4.77 (0.30) | 36.3 | 75.97: 6.18 (0.23) |

- **Long drafts got cheap without anyone measuring it**: at 9 rows a row costs 0.23-0.30 of a pass (0.44 on
  09-18), and the prompt copy at a fixed draft of 8 on code-edit runs **1.95x** on Q8_0 (38.7 -> 75.5 tok/s)
  and **1.78x** on Q4_K (62.4 -> 111.2), against 1.34x on 09-18.
- **Short passes are where the price is wrong, and they are new code's**: 2 rows cost 0.50-0.57 of a pass
  (0.72 at 16 threads), and on Q8_0 3 rows cost more than 5. Read in the code: while no group reaches 4 rows
  (a pass of 2-3 rows) the grouped matmul leaves phase-major and the balanced pool
  (`tr_matmul_one_row_per_group` fails): static chunks cut by index, i.e. by compute, a shared expert's rows
  split between threads, the new experts at 28-36 MiB/ms; and 3 tokens have no kernel that decodes a weight
  row once for all of them (x4 and x8 need 4 and 8): +6.7 ms of dense a row on Q8_0. P1 said 12-15 ms
  (0.45-0.52): 14.7 (0.57), above; P2 7-10 ms at 9 rows: 6.7; P3 0.50-0.60 on Q4_K: 0.50; P4 wrong: 16
  threads make the short passes worse.
- The row at its bytes (its new experts at the one-row pass's MiB/ms, 1.2 ms of the rest): 0.36 of a pass at
  2 rows, 0.31 at 3 (Q4_K): question 78.

**The drafts priced right** (`tools/draft_gate_sim.py`: the texts and the replay of the section above; the
measured Q4_K passes; the row at its bytes at 300, 2048 and 4000 positions, the KV read once a pass and the
attention's compute a row: the context is a model, not measured). Each cell: speed against no draft:

| new code, 18 texts | 09-18 price | measured today, Q4_K | bytes, 300 | bytes, 2048 | bytes, 4000 |
|---|---|---|---|---|---|
| prompt lookup, adaptive (the engine) | 0.972x | 1.014x | 1.048x | 1.082x | 1.100x |
| table 4 MiB >= 0.7 (the best above) | 1.013x | 1.027x | 1.046x | 1.059x | 1.066x |
| llama.cpp's lookup decoding, draft 15 | 0.681x | 0.803x | 0.877x | 0.969x | 1.023x |
| llama.cpp's lookup decoding, draft 3 | 0.921x | 1.021x | 1.092x | 1.170x | 1.214x |
| **gate: context cache + table, calibrated, out of sample** | 1.038x | **1.077x** | **1.116x** | **1.170x** | **1.210x** |
| perfect grammar (ceiling) | 1.252x | 1.334x | 1.427x | 1.510x | 1.556x |

- **The loss is in knowing when, not in guessing**: on the true context at least one cheap source is right
  on 68.3% of the tokens (structural, 44.3%: the table's top-1 34.3%, the lookup 26.8%; names, 55.7%: table
  21.6%, word completion from the text 11.5%, lookup 28.7%), while the drafts above add 0.1-0.5 tokens a pass.
- **The gate**: each source's P(right) by bin (llama.cpp's context cache of the text itself by n, occurrences
  and share; the table by order and confidence), learned on 9 prompts and scored on the other 9, both ways;
  token j is proposed while P(tokens 1..j right) > the price of row j. Predictions: 0.99-1.03x at the 09-18
  price (1.038x), 1.06-1.10x at bytes 300 (1.116x), 1.10-1.15x at 4000 (1.210x): each at or above.
- **Structure is cheaper to verify**: in the code route traces the second of a pair brings 4.25-4.43 new
  experts of 8, **2.99 when both tokens are structural** (U(2) 1.37 against 1.52-1.55).
- **llama.cpp's lookup drafts better than the table** (1.46 tokens a pass at draft 3 against 1.10-1.27): its
  cache of the text itself, n from 1 to 4, validated by corpus counts. At its default length (16) it loses
  here as the fixed draft does on ours: a MoE row is not free (ORIGINS row 8).
- The verdict above came from the price, not from the guess: at today's prices the cheap draft gains 7.7%
  on new code; with the short pass at its bytes 11.6%, 17-21% at 2048-4000 positions (modelled).
- Two claims of an outside review (DeepSeek, via Marcello), measured on the same replay: **a grammar that
  also knows keywords** (9.6% of the tokens) lifts the ceiling from 1.79 to **2.15 tokens a pass**, 1.334x ->
  1.460x at today's price (1.583x at bytes, 1.774x at 4000; predicted 2.0-2.2 and ~1.45x): a ceiling, since
  a grammar knows that a keyword may come, not which. **A suffix automaton** on prompt + generated text (the
  longest earlier match, up to 32 tokens, as one more calibrated source) adds nothing: 1.062x against the
  context cache's 1.077x (predicted +0.00-0.02x): long repeats are rare in new code, and the engine's lookup
  already reads the generated text. What the review rightly finds missing: prompts with a real context (the
  18 are file beginnings), question 81.

## The short verify pass at its bytes (question 78, 2026-09-27)

**The deletion series, by call** (a `TR_POOL_TRACE` build whose notes now name every road and carry each
call's input rows, LESSONS #226; code-edit with the draft fixed at 1 and 2, 120 tokens, 8 threads, native;
`tools/token_timeline.py --rows R`; background 2.16 before, 2.64 after: structure). One more row, call by
call, against the one-row pass, Q8_0:

| call | 1 row | 2 rows | 3 rows |
|---|---|---|---|
| experts, gate/up and down | 15.6 ms: 816 MiB at 55 GB/s | +15.7 ms: 1313 MiB at 44 GB/s | +26.6 ms: 1616 MiB at 40 GB/s |
| q, k, v, o | 5.4 ms at 52.5 GB/s | +1.3 ms (42 GB/s) | **+10.3 ms (18 GB/s)** |
| head | 2.0 ms | +0.3 ms (48 GB/s) | **+4.3 ms (17 GB/s)** |
| attention | 2.3 ms | +0.6 | +0.8 |
| the calls with no weight (norms, RoPE, routing, mixes) | 0.5 ms | +0.5 | +0.9 |

- **The largest piece at 3 rows: the dense weights read three times** (LESSONS #227). No group reaches 4 rows,
  so `tr_matmul_grouped` cuts static chunks of p*rows+r: each weight row of q, k, v, o and the head sits under
  2-3 threads at different times. At 2 rows the halves run in lockstep and the L3 serves the second read.
- **Then the experts**: the same static chunks, by compute (a shared expert's rows split between threads), and
  each Q8_0 weight row decoded once a token (x4 needs 4).
- Q4_K's integer road was balanced already: its experts at their bytes at 2 rows (52 GB/s), 47.6 GB/s at 3;
  its dense at 39 / 35 GB/s (the W16 panel and a tile of 2-3 rows): +0.9 / +1.3 ms; no weight +0.5 / +1.0.
- The references (read 09-27): llama.cpp's tinyBLAS takes a dense Q8_0 matmul from 2 columns (n < 2 returns),
  tiles of up to 4 weight rows by 1-4 columns, each block decoded once for all of them (int8 activations);
  `mul_mat_id` (the experts) goes by `vec_dot`, one token at a time.

**The change** (each output one `dot_row` of its own, the same bits by construction; `make check`, test_spec's
verify rows at every thread count, eight mutations of `tools/mutate_pm.sh` red):
- a short pass (no group of 4 input rows, one of 2 or 3) takes phase-major's plan on any tier: items of 16
  weight rows, balanced, every input row of the group against each weight row while it is in cache
  (`matmul_phase_major`); a one-row-per-group call (a decode token) stays on its own road;
- `dot_row_xt` (Q8_0: scalar, AVX2, AVX-512): one weight row against 2 or 3 input rows, decoded once, each sum
  in its own named accumulator, as x4 does with 4.

**The price after** (`sh tools/row_price.sh <after> 3 <before>`: the two binaries alternated run by run, 3
rounds after a warm-up, native, TR_GPU=0, code-edit; background 2.53 before, at the rule's edge, 1.58 after;
the new binary ran first in every pair, LESSONS #232: the large effects stand, the small ones are the second
session's below). Pass ms before -> after, and one more row's share of a pass:

| rows | Q8_0, 8 threads | Q8_0, 16 threads | Q4_K, 8 threads (its road untouched) |
|---|---|---|---|
| 1 | 25.73 -> 26.04 | 25.98 -> 25.96 | 15.99 -> 16.38 |
| 2 | 41.45 -> **37.15** (0.896): a row 0.61 -> **0.43** | 47.27 -> **38.47** (0.814): 0.82 -> 0.48 | 24.06 -> 23.15: 0.50 -> 0.41 |
| 3 | 64.59 -> **42.36** (0.656): a row 0.76 -> **0.31** | 71.58 -> **46.24** (0.646): 0.88 -> 0.39 | 28.68 -> 29.10: 0.40 -> 0.39 |
| 5 | 60.55 -> 58.60 (0.968) | 59.57 -> 59.09 (0.992) | 39.03 -> 39.25 |
| 9 | 80.24 -> 78.17 (0.974) | 78.09 -> 77.14 (0.988) | 54.59 -> 54.81 |

**Again, the first binary swapped every round, and an A/A** (`ROW_PRICE_CFGS="Q8_0 8" ROW_PRICE_AA=1 sh
tools/row_price.sh <after> 4 <before>`, 4 rounds, background 1.64 before and 1.61 after: free):

| rows | Q8_0, 8 threads: before -> after | the A/A (before against itself) |
|---|---|---|
| 1 | 25.79 -> 25.78 (**1.000**: the decode did not move) | 0.996 |
| 2 | 41.33 -> **35.93** (0.869): a row 0.60 -> **0.39** of a pass | 0.989 |
| 3 | 62.69 -> **42.97** (0.685): a row 0.72 -> **0.33** | 1.004 |
| 5 | 59.57 -> 58.27 (0.978) | 0.996 |
| 9 | 79.87 -> 78.34 (0.981) | 1.007 |

- **Q8_0's 3-row pass is at its bytes**: its new experts at 52.0 MiB/ms (the one-row pass's 52.9), the dense
  -0.1 ms a row (the target was <= ~42 ms: 42.4-43.0). The 2-row pass nearly: 35.9 against <= ~35, its new
  experts at 47.1 MiB/ms, attention +0.34 and the rest +0.52 a row (first session's split).
- The first session's one-row 1.012 was the order (#232): with the order swapped the one-row pass is 1.000,
  the A/A within 1.1% everywhere; the 5- and 9-row gains (2.2%, 1.9%) sit above the A/A.
- **The drafts at these prices** (`tools/draft_gate_sim.py --prices build/rowprice-q78b/<side>/prices.json
  --config Q8_0-t8`, the measured column): new code, the engine's lookup 0.969x -> **1.040x**, the gate 1.013x
  -> **1.103x**; with a real context (question 81's 20 texts) 0.998x -> **1.109x** and 1.061x -> **1.191x**
  (1.169x and 1.274x with the context's cost modelled). On Q8_0 the engine's own lookup stops losing on new
  code; Q4_K's prices, and so the Q4_K rows of §Drafts with a real context, are unchanged.

**The width, forced** (question 82: `--decode-threads` 4, 8, 16 on a pool of 16, alternated run by run, 3
rounds after a warm-up, native; background 1.46 after). Pass ms:

| rows | Q8_0: 4 / 8 / 16 threads | Q4_K: 4 / 8 / 16 threads |
|---|---|---|
| 1 | **25.69** / 27.04 / 28.48 | 17.84 / **15.84** / 16.64 |
| 2 | 37.40 / **36.13** / 37.81 | 34.44 / 23.26 / **23.12** |
| 3 | 46.47 / **43.15** / 45.72 | 44.57 / 28.83 / **27.66** |

- **The one-row pass's width is not the short passes'**: on Q8_0 the decode wants 4 threads, a 3-row pass 8
  (the tuner's 4 would cost it 1.08x, the whole pool 1.06x); on Q4_K 4 threads double the short passes (the
  W16 panel and tiles are compute). A short pass needs its own measured width (question 82).
- Q4_K's column is the session's noise on an unchanged road: a pass moves ±3%, a row price 0.09 of a pass
  (LESSONS #230): a price is compared inside one alternated session only.
- 16 threads are still 1.04x (2 rows) and 1.09x (3 rows) slower than 8: the short passes never get a tuned
  width when every pass drafts (LESSONS #228).
- Predictions (`build/rowprice-q78/predictions.txt`): 3 rows 40-44 ms: 42.36, in; 2 rows 33-36: 37.15, above;
  16 threads 34-38 and 42-47: 38.47 and 46.24, in; 5 and 9 rows 1-3% faster: 2.2% and 1.9% (the second
  session), in; Q4_K and the one-row pass within 2%: 1.025 and 1.012 in the first session (the order and the
  noise above), 1.000 for the one-row pass in the second, in.
- Left, by size: Q4_K's short groups (the dense +1.2 ms at 2 rows, +0.9 a row at 3; its experts 42.8 MiB/ms
  at 3), the short passes' width (#228), the 2-row pass's rest.

## Drafts with a real context (question 81, 2026-09-27)

The 18 new-code prompts are file beginnings (100-200 positions); real use continues a function mid-file.
20 prompts cut from this repo (`bench/prompts/real-context`: 11 C, 6 Python, 3 shell; a real file up to the
start of a function's body, 1770-3798 tokens, mean 2852; `sources.tsv`, `cut.py`), each continued greedily
for 256 tokens (Q4_K, in the container, `tools/draft_ctx_gen.sh`), replayed by `tools/draft_gate_sim.py --set
build/draft_ctx --ctx real` (validated: the replay's counters equal the engine's `--spec 8` on c05_olmoe and
py03_route_trace, SAME, and its pass counts on all 20). A repo table: the n-gram table of this repo's code
without the file under test (leave one out). Speed against no draft (tokens a pass, accepted), Q4_K's prices
of 09-27 (about 300 positions; "real ctx" adds the byte model's context terms at each pass's position):

| 20 texts with a real context | measured | measured, real ctx | bytes, real ctx | new code (18), measured |
|---|---|---|---|---|
| prompt lookup, adaptive (the engine) | 1.067x (1.37t, 52%) | 1.164x | 1.200x | 1.014x |
| public table 4 MiB >= 0.7 | 1.017x | 1.044x | 1.054x | 1.027x |
| repo table, leave one out | 1.056x (1.30t) | 1.134x | 1.164x | 0.991x |
| llama.cpp's lookup, draft 3 | 1.096x (1.75t) | 1.273x | 1.338x | 1.021x |
| **gate: context cache + public table, out of sample** | **1.146x** (1.71t, 55%) | **1.271x** | **1.324x** | 1.077x |
| gate + the repo table | 1.146x | 1.256x | 1.311x | 1.079x |
| perfect grammar (ceiling); counting whitespace runs | 1.299x; 1.375x | 1.427x; 1.532x | 1.478x; 1.594x | 1.334x; 1.423x |

- **With a real context the cheap drafts pay**: the engine's own lookup 1.014x -> 1.067x, the gate 1.077x ->
  1.146x at the measured prices, 1.27-1.32x with the context's cost modelled (a native row price at ~3000
  positions is owed). Per text the gate runs from 0.967x (py03) to 1.616x (py04).
- The repo table beats the public one alone (1.056x against 1.017x) and adds nothing as the gate's third
  source: it is right where both others are wrong on 4.3% of the positions.
- Predictions (`build/draft_ctx/predictions.txt`, before any run): the engine's lookup and the gates in range;
  the repo table as a source predicted +0.01-0.04x, measured 0 to -0.015x.
- Found on the way: the replay's token classes count 23 whitespace-run tokens as names (LESSONS #229): the
  grammar ceilings of §The post-it taken apart are low (new code 1.423x, not 1.334x).

## The Q4_K short passes: the panel's rows brought ahead (question 78's remainder, 2026-09-27)

Q4_K's verify passes of 2-3 rows read the dense weights at 35-39 GB/s through the W16 panel and the new
experts at 42.8 MiB/ms at 3 rows (§The short verify pass at its bytes). The targets at 8 threads (the
bytes): 2 rows <= ~21.6 ms, 3 rows <= ~26. The predictions, mine and the Opus design agent's, are in
`build/rowprice-q79/predictions.txt`, each written before its run. All native, a free machine,
`tools/row_price.sh`, 4 rounds, the first binary swapped every round, pass ms at 8 threads.

**(a) A group of 2-3 rows by pairs** (`Q4X_MIN_TILE_ROWS` 4: `q4x_dot2` a token at a time, the weight row
read again from L1/L2 for the next token), against today, with an A/A:

| rows | today | by pairs | A/A | one more row: dense, new experts |
|---|---|---|---|---|
| 1 | 15.59 | 15.78 (1.012) | 1.011 | |
| 2 | 22.99 | **22.20 (0.966)** | 0.999 | +1.32 -> +0.80 ms; 44.0 -> 48.0 MiB/ms |
| 3 | 28.44 | 28.99 (1.019) | 1.000 | +0.91 -> +1.28 ms; 42.2 -> 44.0 MiB/ms |

Two `q4x_dot2` streams beat a panel built for two tokens; three do not (a panel's cost per token is under
half a dot2's). The panel was also losing more than its arithmetic: `avx512_q4x_panel` read its 16 rows on
demand, 16 short interleaved streams (a 144-byte block of each row a step) that no hardware prefetcher
follows, and at 2-3 rows a panel serves a single tile (LESSONS #233; the design agent's road (e)).

**The panel's rows brought ahead**: seven binaries in one alternated session (background 1.41 / 1.33).
pf1 hands the panel the next item's rows (the worker's following item, when its group is short: 9 lines per
64-column window of the current build); pf2 brings each row's block s + 2 during block s (12 prefetches a
window); pf3 does both; m3 is `Q4X_MIN_TILE_ROWS` 3 (a 2-row group by pairs).

| variant | 1 row | 2 rows | 3 rows | prompt, 411 tokens (tok/s) |
|---|---|---|---|---|
| today (pf0) | 15.81 | 23.07 | 28.32 | 405.2 |
| today again (A/A) | 0.990 | 0.996 | 0.988 | 408.9 |
| pf1: the next item's rows | 0.989 | 22.07 (0.957) | 27.08 (0.956) | 407.4 |
| pf2: block s + 2 | 0.981 | 22.58 (0.979) | 27.54 (0.972) | 408.0 |
| pf3: both | 0.994 | 22.14 (0.960) | **26.65 (0.941)** | 409.2 |
| m3: 2-row groups by pairs | 0.990 | 22.00 (0.954) | 28.10 (0.992) | 405.6 |
| pf1 + m3 | 0.992 | **21.89 (0.949)** | 26.81 (0.947) | 405.3 |
| (a): every 2-3-row group by pairs | 1.004 | 22.18 (0.961) | 28.56 (1.009) | 406.0 |

- **The panel's extra time was mostly latency**: pf1 took 1.24 ms off the 3-row pass (the premise's disproof
  was <= 0.3). With pf3, one more row at 3 rows costs 5.46 ms (6.25 today): new experts at 46.9 MiB/ms
  (42.6), dense +0.47 ms (+0.84), attention 0.08, the rest 0.40.
- pf3 beat pf1 by 0.43 ms at 3 rows (predicted a tie): block s + 2 covers each worker's first item and the
  items another worker takes over. pf1 is 0.3-0.7 ms short of the agent's prediction at both sizes.
- The prompt (411 tokens, 8 threads, 12 runs a binary) did not move: 405-409 tok/s everywhere, the A/A 0.9%.
- Predictions: (a) 2 rows 22.4-22.9 (mine) and 22.2-22.7 (the agent's): 22.20; 3 rows 28.6-29.4 and
  28.2-29.2: 28.99. In the series, m3 and pf1 + m3 in the agent's ranges; pf1 and pf2 above them (less gain).

**Kept: pf3 and m3 together**, measured as the final binary (question 82's tuner in it, so every pass is
forced onto 8 threads: `ROW_PRICE_FORCED=1`), against today, with an A/A (background 1.49 / 1.24):

| rows | today | final | A/A | one more row: dense, new experts |
|---|---|---|---|---|
| 1 | 15.50 | 15.44 (0.996) | 0.999 | |
| 2 | 22.92 | **22.19 (0.968)** | 1.000 | +1.32 -> +0.92 ms; 44.1 -> 47.1 MiB/ms |
| 3 | 28.00 | **26.53 (0.947)** | 1.006 | +0.88 -> +0.51 ms; 42.9 -> 46.5 MiB/ms |

- A row's share of a pass: 0.48 -> 0.44 at 2 rows, 0.40 -> 0.36 at 3 (Q8_0: 0.39 and 0.33).
- Predictions: 1 row 1.000 +-1%, in; 2 rows 21.8-22.1, 22.19 (0.09 above: the session gave 2 rows less than
  pf1 + m3's, LESSONS #230's spread between sessions); 3 rows 26.4-26.8, in.
- Left, by size: the dense, +0.5 ms a row at 3 rows and +0.9 at 2 (each weight row decoded once a token:
  a kernel decoding it once for 2-3 prepared tokens, road (c), whose design is in the next session's
  post-it); the rest, +0.35-0.41 a row; the new experts, 3-4 MiB/ms under the one-row pass's 50.8.

## The verify passes' own width (question 82, 2026-09-27)

The tuner measured the width of one-token passes only (LESSONS #228): a verify pass took the decode's width,
or the whole pool while the decode had measured nothing (a session drafting on every pass). Forced on a pool
of 16 (§The short verify pass at its bytes), a 3-row Q8_0 pass wants 8 threads where the decode wants 4. Now
every size of verify pass (2 to `TR_DECODE_ROWS` rows, a logit row each) has a measurement of its own, with
the decode's rules (the fastest within its own noise, again at every doubling of the context, a switch on
two votes; model.c `tune_state`); a short prompt (one logit row) and a pass past `TR_DECODE_ROWS` take the
decode's width. The same final binary three ways on a pool of 16, drafts fixed, 200 tokens, 4 rounds
(`tools/row_price.sh` with wrapper scripts, `build/q79/fin*/`): tuned; the pool (`TR_DECODE_ROWS=1`: the
verify passes on 16, the engine before when every pass drafts); the decode's width forced (4 threads for
Q8_0, 8 for Q4_K: the engine before once the decode has measured). Pass ms:

| | pool | decode's width | tuned, every width probed | tuned, the narrowest skipped |
|---|---|---|---|---|
| Q8_0, 2 rows | 36.97 / 37.08 | 36.61 / 36.30 | 35.95 (0.972): 8 in 4 runs of 4 | **35.62 (0.961)**: 8, 4 of 4 |
| Q8_0, 3 rows | 44.37 / 44.31 | 45.30 / 45.43 | 43.39 (0.978): 8 or 16 | **42.82 (0.966)**: 8, 4 of 4 |
| Q4_K, 2 rows | 22.01 / 22.11 | 22.08 / 22.07 | 22.38 (1.017): 16 | 22.21 (1.005): 8, 4 of 4 |
| Q4_K, 3 rows | 27.06 / 27.00 | 26.35 / 26.77 | 27.72 (1.024): 8 or 16 | **26.51 (0.982)**: 8, 4 of 4 |

(two sessions, each with its own pool and decode's-width columns; background 1.23 / 1.17, then 1.48 / 1.13)

- **Every width probed lost on Q4_K** (LESSONS #236): three probes on 4 threads cost a Q4_K 3-row pass +60%
  each, more than the right width wins back in a 200-token run; and the pick wavered between 8 and 16.
- **The narrowest skipped** (a verify pass does more work per byte read than the decode: in every
  measurement the narrowest width lost it, by 3-61%): 8 picked in 16 runs of 16; Q8_0 3.4-3.9% faster than
  the pool and 1.9-5.7% faster than the decode's width; Q4_K 1.8% faster at 3 rows, 0.5% slower at 2 (its
  three probes on 16). On a pool of 8 a verify size has one width left, the whole pool.
- Since the prefetched panel (§The Q4_K short passes) a Q4_K 3-row pass wants 8 threads, no longer 16 (26.35
  against 27.06).
- Predictions (`build/rowprice-q79/predictions.txt`): every width probed, Q8_0 in and better, Q4_K 1-3%
  slower (in); the narrowest skipped, Q8_0 better than predicted, Q4_K in, no size more than 1% slower than
  the pool (in).

## A Q4_K weight row decoded once for 2-3 tokens (question 78's road (c), 2026-09-27)

Until this step a Q4_K group of 2 input rows went by `q4x_dot2` pairs, a token at a time (each weight row decoded
twice), and a group of 3 by the W16 panel and one tile (§The Q4_K short passes). `q4x_dot_xt` (kernels.h) takes two
weight rows against 2 or 3 prepared rows: per 64 columns and row the weight side once (the quants widened, the two
nibbles, two `vpmullw` by the sub-block scales), 4 `vpdpwssd` a token; its header pre-pass and block end are
`q4x_dot2`'s statements (`q4x_headers2`, `q4x_end2`), so every value is dot2's. The driver cuts a group under 4 rows
(`Q4X_MIN_TILE_ROWS` 4) into runs of 3, or of 2 when 2 or 4 are left, 8 calls a run; a lone row keeps `q4x_dot2`.
Scalar and AVX2 take the pairs one input row after the other (the same bits; AVX2's own kernel is owed).
Predictions in `build/rowprice-q80/predictions.txt`, written before any code or run. The kernel alone:
`tools/bench_q4x.sh` (tests/bench_q4x.c, every output checked against scalar first).

**In L1** (container, one core, 11 runs; two rows and T prepared rows hot, then an item of 16 rows of 2048 with its
T prepared rows in L1/L2), ns and the ratio to the same work by `q4x_dot2`:

| line | T = 2 | T = 3 |
|---|---|---|
| 2 rows of 1024: T x `q4x_dot2`, `q4x_dot_xt` | 158.8, 122.4 (**0.771**) | 237.3, 170.1 (**0.717**) |
| 2 rows of 2048 | 310.8, 238.8 (0.768) | 468.1, 333.2 (0.712) |
| an item of 16 rows: dot2, xt, the panel and a tile | 2492.7, 1914.9 (0.768), 1935.3 (0.776) | 3743.4, 2666.6 (0.712), **2269.5 (0.606)** |

- Predicted 0.80-0.88 (T = 2) and 0.73-0.83 (T = 3): both better. The task's first drop line (xt(3) above 0.8 of
  three dot2) is passed. In cache the panel is the fastest road at 3 tokens.
- The disassembly: the window loop unrolled, 96 `vpdpwssd` and 16 `vpmullw` a call at T = 3, the 12 accumulators in
  registers, no zmm on the stack (LESSONS #208, #209).

**From RAM** (native, `--ram P`: 1 GiB of Q4_K rows of 2048 (256 MiB on one thread) in items of 16 rows, one
contiguous chunk of items a thread, the same prepared rows every item; GB/s of weight bytes, medians of 14-22
passes; spreads 2-10% on one thread, 3-20% on 4, 7-17% on 8; background 1.2-1.5):

| line | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| a plain read (the ceiling) | 26.6 | 58.5 | 55.6 |
| T = 1: `q4x_dot2` (the decode) | 13.3 | 50.6 | 55.7 |
| T = 2: dot2 a token at a time (the road before) | 6.7 | 27.5 | 49.1 |
| T = 2: **xt** | **9.4** | **35.9** | **54.1 (0.97 of the read)** |
| T = 2: the panel and a tile | 9.0 | 28.8 | 47.1 |
| T = 3: dot2 a token at a time | 4.5 | 18.5 | 34.7 |
| T = 3: **xt** | 6.8 | **26.4** | **49.0 (0.88)** |
| T = 3: the panel and a tile (the road before) | **7.7** | 23.6 | 42.2 |

- New prepared rows every item (a pool of 48, LESSONS #219): within 3% of the same rows on every line.
- **The panel is the faster road on one core and the slower one from 4 threads** (LESSONS #240): from 1 to 8
  threads xt keeps 0.90 of its one-thread rate a thread, the panel 0.69 (16 short streams a thread, 128 at 8).
- Predictions (1 thread / 8 threads): xt(2) 6.3-7.8 / 46-54, xt(3) 4.7-5.9 / 38-46, the panel 4.0-5.2 / 33-40,
  dot2 x 3 3.7-4.4 / 30-35: every road faster than predicted on one core; at 8 threads xt and the panel above my
  ranges. The task's second drop line (xt(3) at 8 threads under 0.95 of the bytes' rate) trips, as predicted: 0.88.
  xt(3) went to the engine for what it gains on the panel there (1.16x at 8 threads, 1.12x at 4).

**The engine** (`tools/row_price.sh`, Q4_K, every pass forced on 8 threads, code-edit with the draft fixed at 0, 1, 2,
the first binary swapped every round, an A/A; pass ms). First A/B (4 rounds, background 2.19 / 1.57, `build/rp-q80`):

| rows | before | after | A/A | one more row: dense, new experts |
|---|---|---|---|---|
| 1 | 15.42 | 15.91 (1.032) | 0.985 | |
| 2 | 22.31 | **21.22 (0.951)** | 0.999 | +0.87 -> +0.07 ms; 45.4 -> 51.7 MiB/ms |
| 3 | 26.68 | **26.28 (0.985)** | 1.001 | +0.49 -> +0.39 ms; 45.6 -> 49.3 MiB/ms |

- The one-row pass (the decode, which stays on `q4x_dot2`) read 1.032: 16.31-16.38 ms in rounds 0-2, 15.45-15.52
  in rounds 3-4, against 15.30-15.75 and 15.17-15.49 for the two copies of before; every zone +2-6%, the attention
  (+3.9%) and the F32 router (+6%) as much as the Q4_K matmuls. In that binary `q4x_dot2` shared the new kernel's
  header pre-pass and block end (the same instructions, other registers). It went back to the committed text, its
  instructions identical to HEAD's but for addresses (LESSONS #239), and the final binary was measured again.

**The final binary** (`q4x_dot2` as committed) against HEAD's, 6 rounds, a busier machine (background 1.65 / 1.85; the
still check waited six times at 3.1-4.9 logical processors busy; `build/rp-q80b`):

| rows | before | final | A/A | one more row: dense, new experts |
|---|---|---|---|---|
| 1 | 15.75 | 15.85 (**1.006**) | 0.996 | |
| 2 | 22.23 | 21.97 (0.988) | 1.018 | +0.77 -> +0.27 ms; 47.5 -> 47.8 MiB/ms |
| 3 | 27.32 | 26.90 (0.985) | 1.026 | +0.51 -> +0.45 ms; 44.7 -> 46.7 MiB/ms |

- The one-row pass is level (1.006, A/A 0.996).
- Against the mean of the two copies of before: 2 rows 0.979 (the first session 0.952), 3 rows 0.972 (0.984). The two
  sessions agree on the sign and differ at 2 rows by more than a session's A/A (LESSONS #230).

**The final binary again, on a free machine** (6 rounds, background 1.62 / 1.29; the still check waited four times,
three for spikes gone in the next window and once for another project's `node` at 1.0 core, LESSONS #241;
`build/rp-q80c`):

| rows | before | final | A/A | one more row: dense, new experts |
|---|---|---|---|---|
| 1 | 15.73 | 15.90 (1.011) | 0.988 | |
| 2 | 22.95 | **21.45 (0.934)** | 0.980 | +0.99 -> +0.19 ms; 44.0 -> 51.0 MiB/ms |
| 3 | 27.66 | **26.45 (0.956)** | 0.983 | +0.58 -> +0.44 ms; 43.9 -> 49.1 MiB/ms |

- Against the mean of before's two copies: 2 rows 0.944, 3 rows 0.964. Over the three sessions: **2 rows 0.94-0.98,
  3 rows 0.96-0.985**.
- **The one-row pass** of the final binary: 1.008 and 1.017 of the mean of before's copies in sessions 2 and 3
  (their A/A 0.996, 0.988), spread over every zone, the attention (+1.1%, +1.7%) and the F32 router as much as the
  Q4_K matmuls: not `q4x_dot2`, whose instructions are HEAD's. Every function after the new AVX2 kernel moved by
  some tens of bytes, the attention's kernels too: the hot loops' alignment, or the machine; open (a HEAD binary with
  the same bytes inserted and no new code, or the hot kernels aligned so that no insertion moves them).
- A row's share of a pass: 0.45 -> 0.33 at 2 rows and 0.37 -> 0.33 at 3 in the first session; 0.41 -> 0.39 and 0.37
  -> 0.35 in the second; 0.46 -> 0.35 and 0.38 -> 0.33 in the third.
- The prompt (411 tokens, 12, 18, 18 runs a binary): 405.0 / 407.6 (A/A) / 404.9 tok/s, then 403.5 / 404.3 / 404.6,
  then 406.9 / 404.8 / 402.3: unchanged.
- Predictions (the first A/B): 1 row 1.000 +-1%: out; 2 rows 21.6-21.95: better (21.22); 3 rows 25.8-26.3: in
  (26.28); one more row's dense at 3 rows +0.2-0.45: in (+0.39); the prompt 1.00 +-1%: in. The final binary, second
  session: 1 row 1.000 +-1%: in (1.006); 2 rows 21.1-21.5: above (21.97, a slower session: before's own 3-row pass
  27.32 against 26.68); 3 rows 26.1-26.5: above (26.90). Third session: 1 row 1.000 +-1%: out (1.011); 2 rows
  21.2-21.7: in (21.45); 3 rows 26.2-26.5: in (26.45); the prompt: in (0.989, 0.994 of the A/A copy).
- The targets: 2 rows <= ~21.6 reached in the first and third sessions (21.22, 21.45), not in the second (21.97);
  3 rows <= ~26 not reached (26.28, 26.90, 26.45).
- Left at 3 rows, one more row: the dense +0.39-0.45 ms (xt(3) reads at 0.88 of the bytes at 8 threads: its block
  end, per token, is the largest piece after the `vpdpwssd`; batching it across the tokens is the lever), the rest
  +0.42; at 2 rows the dense is near its bytes (+0.07-0.27) and the rest +0.30-0.41.
- Correctness: every tier's xt against scalar's dot_row (the prepared rows at the end of an exact buffer, canaries past
  the outputs), three witnesses on every Q4_K kernel (a signed zero, cancelling mins, the blocks' order: LESSONS #237,
  #238), a short pass's calls counted exactly (`test_short_pass_q4k`), 47 mutations in `tools/mutate_q4x.sh` all red;
  the real model's text identical with drafts of 1 and 2; `make check` green.

## The decode's +1% and the code's layout (LESSONS #242, 2026-09-27)

Road (c)'s binary read 1.006-1.017 of HEAD's on the one-row pass in three sessions, spread over every zone, the F32
router too, though `avx512_q4x_dot2` is HEAD's byte for byte. **The layout, read first** (nm, objdump): every object's
.text is 64-aligned, and GCC 15 aligns each tight loop to the power of two above its size (`.p2align 5` / `6`, the
"align tight loops" tuning): each such loop is an anchor, and an insertion shifts the code mod 64 only up to the next
anchor (a pad before `avx2_axpy_f32_x4` was absorbed by it). In the final binary the chunk `avx512_q4x_tile ..
avx512_axpy_f32` moved 16 -> 0 mod 64 (the new 48-byte dispatcher before it); `q4x_dot2`'s 1080-byte main loop spans
18 64-byte windows at both places; its 47-byte "loop" at +1715 is the early exit for n < 256, never run. Data: only
`g_pm_tiles` and `xexp_tab` moved mod 64.

**The race** (`build/rp-q81b`, one-row passes, Q4_K 8 threads forced, 16 rounds, 9 binaries alternated, the order
reversed every other round; predictions in `build/rowprice-q81/predictions.txt`). Variants of the final binary with
the dispatcher lengthened (it never runs in the decode): F08 puts the chunk back at HEAD's places, F16 and F32 at the
two others (and `avx512_expf_f32` at 32), F48 keeps F's places and moves all the code after it by 64 bytes. Hc and Fc:
HEAD's and F's bytes plus one byte (another file). Rounds 1-2 dropped: 8 runs in a row at 20.4-24.9 ms (against
~15.6) while the CPU guard was satisfied (LESSONS #244). Median paired ratio to HEAD over 14 rounds:

| binary | total | qkv | attention | attn_out | router | gate_up | down | lm_head |
|---|---|---|---|---|---|---|---|---|
| F (the final) | **0.997** | 1.000 | 1.005 | 0.996 | 1.006 | 0.992 | 0.999 | 0.998 |
| HEAD again (A/A) | 0.995 | 0.987 | 0.994 | 0.994 | 0.988 | 0.995 | 0.999 | 0.994 |
| Hc (HEAD's bytes, another file) | 0.997 | 0.998 | 1.004 | 1.004 | 0.983 | 0.994 | 0.996 | 0.997 |
| **Fc** (F's bytes, another file) | **1.018** | 1.020 | 1.027 | 1.025 | 1.041 | 1.017 | 1.017 | 1.019 |
| F08 (chunk at HEAD's places) | 1.000 | 1.004 | 1.004 | 1.000 | 1.002 | 0.993 | 0.996 | 1.007 |
| F16 | 1.003 | 1.005 | 1.008 | 1.009 | 1.026 | 0.993 | 1.006 | 1.003 |
| F32 | 1.019 | 1.032 | 1.021 | 1.022 | 1.047 | 1.018 | 1.021 | 1.026 |
| F48 (F's places, 64 bytes on) | 0.995 | 0.996 | 1.006 | 0.999 | 0.999 | 0.995 | 0.999 | 1.010 |

Pairs that ran in the same slots: F48/F 0.999, F32/HEAD 1.019, F16/HEAD-again 1.013, F08/Hc 1.004; Fc/F 1.010.

- **The final binary is level with HEAD** (0.997; F08 and F48 too): #242's +0.8-1.7% does not reproduce.
- **Two files of the same bytes differ by 1.0-1.8%** (Fc against F and HEAD, every zone alike, the router +4%): a
  file's own effect (its image base, its pages; or its slot, the middle one) is as large as what was chased. One file
  a side with the slots fixed cannot resolve under ~2% (LESSONS #243); F32's +1.9% and F16's +0.3-1.3% are inside it.
- Predictions: F 1.000-1.012 (0.997: in at the low end); F08/F, F48/F within +-0.5% (in); Fc/F within +-0.7% (out:
  1.010, 1.018 against HEAD); the router within +-1.5% everywhere (out for Fc, F16, F32).
- The design's critique came from the Opus thinker (a pad between objects moves nothing mod 64: every object is
  64-aligned; the same file as its own A/A cannot see a file's effect; the forward/backward order pins the middle
  slots: before2/before read 0.984-0.995 in earlier sessions; paths of different lengths move the CRT's heap).

## An A/B inside one process (2026-09-27)

What the race above cannot resolve (a file's own 1-2%, the slots), one process can: `generate --ab <switch>` runs
the decode's passes (or the verify passes: a step is a pass) A B B A A B B A..., each arm's zones in its own phase of
the profile ("decode", "decode_b"), every pass's wall kept (`ab_pass_ms`); both arms run in one file, one memory,
one set of threads, one routing history. `tools/ab_inproc.sh <binary> <switch> [runs]` takes the runs under the
native guards, `tools/ab_inproc.py` reports: each run's arms, the A B B A blocks' B/A, and the estimate, the arms'
means from the walls without the first block (the first pass after the prompt is always A's). `--ab none` is the
A/A: both arms the same code, the tool's own noise. Q4_K, 8 threads, one row a pass, 200 tokens, 6 runs:

| session | runs' speed | arms' means B/A | without the first block | blocks' median |
|---|---|---|---|---|
| ab-aa1 | 15.5-22.3 ms a pass (a load the guard does not see in runs 1, 5, 6) | 0.988, every run < 1 | - | - |
| ab-aa2 | 15.2-17.4 | 1.0002 +- 0.0029 | **1.0004 +- 0.0028** | 0.9972 +- 0.0053 |

- **The resolution: 0.28% over 6 runs** (a run's B/A spreads 0.7%), against 1-2% between two files of the same bytes:
  a 1% change reads at 3.5 standard errors in about two minutes.
- ab-aa1's bias came from the first pass (15-24 ms against a median of 16, always arm A) and chance; the estimate
  leaves the first block out. A load that slowed whole runs by 40% left their B/A in place: both arms carry it.
- A pass spreads 5-7% with its experts, so the blocks' median is robust but noisier than the arms' means. Finer:
  the same token twice (arm A, the KV rewound, arm B, the order swapped every token), which would also compare
  every pass's logits bit for bit between the arms.
- Predictions (`build/rowprice-q81/predictions.txt`): a run within +-0.5% (ab-aa2: 5 of 6); the estimate within
  +-0.3% (in); the first pass >= 1.3x the median (3 of 6).

## The prep once a row (2026-09-27)

Q4_K's integer road prepares every input row before a matmul (`q4x_prep`: the shifts, the two digits, the sums;
~8.8 KB for 2048 columns). A call prepared its own rows: q, k and v the same attention-normed rows three times, gate
and up each the 8 gathered copies of a token's ffn-normed row (16 preps a token for one row). The references prepare
once a distinct row (ORIGINS §Every piece, row 12): ds4 once a token for every expert with gate and up in one pair
kernel, colibri once a layer with the experts' int8 rows copied, ik_llama.cpp with q/k/v merged at load and up/gate
fused; llama.cpp once an op. Marcello's question was half a prep a token; the Opus thinker's floor is 7 rows of 2048
a token and layer against 24 (0.29), ~0.2 with a cheaper prep (the scalar tail's `frexp`/`ldexp`, the token terms
only a tile reads). Predictions in `build/prep/predictions.txt`, each written before its run.

**Step 0, the traces** (`build/prep/traces`, the traced binary, native, 8 threads; a traced pass ~10% slower): a
decode layer spends ~33 us on preps, the gather and the swiglu (gate's prep 7.1, up's 3.2, down's 2.4 in the pool;
q, k, v, o 1.2-2.4 inline; the gather 4.4 inline; the swiglu 9.4), 3% of a pass; a 3-row layer ~36 us removable (the
gather 13-18, gate/up preps 8.4-12.2); a 512-token prompt 91-96 ms of preps (6.9-7.2%) and 23 ms of gather (1.7%).

**Built** (P1 and P2 of the thinker's order): `tr_q4x_prepare` prepares a pass's token rows once into the session's
own buffer (never `pm.xq`, which a phase-major call between two readers would overwrite), `tr_matmul_q4x_prepared`
runs a call on them, each group's row p reading prepared row map[p]: q, k and v from the attention-normed rows, gate
and up from the ffn-normed rows through `xmap` (grouped row -> token), with no gather and no copy. A matrix of another
type takes its road from the floats (the gather stays for those). `q4x_dot_xt` and `q4x_tile` take a pointer a
prepared row instead of a stride (a group's tokens are not consecutive rows). The old road stays as the switch `prep`
of `generate --ab` (arm B), for the A/B inside one process and for the tests.

**Correctness**: `test_tier_used` pins the preps exactly, 412 on the Q4_K model where every call preparing its own
rows made 888 (the thinker's count), and runs the same passes on prep's arm B: every pass's logits the same bits;
`test_kernels` hands xt and tile their rows in reverse (pointers, not a stride; the xt rows at the end of an exact
buffer); the real model's logits of 160 positions identical to the last binary's, one pass and one row a pass; 7 new
mutations (the map ignored, a run's and a tile's rows in reverse, every token reading the first row in xt and the
tile, the model's map by slot, q/k/v on rows nobody prepared) among 54, all red.

**The A/B inside one process** (`tools/ab_inproc.sh build/q82/p2a.exe prep`; B/A = the old road over the new):

| pass | B/A | predicted | background |
|---|---|---|---|
| the prompt, 411 tokens (8 evals a run, 4 runs) | **1.0577 +- 0.0027** | 1.04-1.07 | 1.20 / 2.04 |
| a verify pass of 3 rows (6 runs) | **1.0262 +- 0.0040** | 1.012-1.025 | 2.22 / 1.63 |
| the decode, one row (4 runs, the prompt's session) | **1.0103 +- 0.0013** | 1.007-1.017 | 1.20 / 2.04 |
| the decode (6 runs, headless Chrome at 6.4 cores between runs) | 1.0044 +- 0.0048; blocks 1.0128 +- 0.0041 | | 1.68 / 5.28 |

- **The prompt 5.8% faster, a verify pass of 3 rows 2.6%, the decode 1.0%**, every bit the same. The decode's arms were aliased with the KV's fresh pages (LESSONS #247: arm A, the new road, paid them): with the arms aperiodic, P1 + P2 + P4 read **1.0252 +- 0.0044** on the decode (blocks 1.0220; 6 runs, background 1.30 / 1.12), kv_write level between the arms.
- When a load comes and goes, the blocks' median holds (1.0128) where the arms' means move (1.0044).

**P4, the swiglu and the down's prep in one call**: `tr_swiglu_prepare` splits the swiglu by whole rows and prepares
each row while it is in its core's cache, into `pm.xq` (the down reads it right after, no call between); the switch
`act` puts back the swiglu apart and the down preparing its own rows. `test_swiglu_prepare`: its floats tr_swiglu's and
its bytes scalar's q4x_prep of them on every tier and pool (a -0 from exp's overflow, zeros, subnormals, NaN); the tier
test's 412 preps on both roads and every pass's logits the same bits; the real model's logits identical. In one process
on a free machine (8 runs): **the prompt 1.0087 +- 0.0027** (every run 1.004-1.015; predicted 1.006-1.015), the decode
0.9982 +- 0.0027 (level). A first session with another window's node at 3-4 cores read 1.0045 +- 0.0042: structure only.

**The prep alone** (`bench_q4x`'s new line, one row of 2048 in L1, one core): 699.6 ns (spread 1.6%; a run inside
another window's node job read 703 at 30%). Its bit arithmetic (3 UCRT calls a block) and the token terms no road but
the tile reads are 10-25% and ~16% of it: ~0.1 us a prep, 0.04% of a decode token, 0.1-0.2% of the prompt: last.
- Left, by size: gate and up in one call, the calls' tails by countdowns, the idle workers prefetching; then the prep's
  bit arithmetic.

## The argmax in parallel (2026-09-27)

The Opus thinker read the decode's zones per arm on the untraced engine (build/ab-act-prompt2) and found the greedy
token's argmax on the main thread: 116-120 us a token (0.78%), the serial scan `if (x[i] > x[best])` over the 201 KB of
logits the head's 8 workers had just written into their own caches; a 3-row verify pass pays it once a row, inside its
step. `tr_argmax_f32` splits the scan over the pool by chunks on the head's 16-row items, each chunk from (-inf, none)
with strict > on eight interleaved lanes, the chunks merged in order from (x[0], 0) with the same strict >: the scan's
index by construction (a NaN never wins, the first of equal values, index 0 for a NaN there or all -inf).
`tr_session_argmax(s, back, n)` runs it on the model's pool for main.c and tr_greedy (whose vocabulary may be narrower:
LESSONS #248); the switch `argmax` (the session's own, arm B the serial scan).

- **Correctness**: `test_argmax` (2480 calls: sizes 1 to 50304, pools of none to 16, a NaN at 0 and inside, all -inf,
  +-0 ties, every value tied, +inf twice, the largest planted at every chunk border with a later tie); the tokens of
  every generate and speculation test unchanged; four mutations (ties to the later index in a lane and across lanes,
  the merge from -inf, the chunks merged backwards), all red.
- **In one process** (6 runs each, background 1.13-1.19): the sample zone 116-118 us -> **5.6-6.9 us** (17x; predicted
  12-25x), the decode's pass walls level (0.998 +- 0.003), so a token ~0.7% faster; a 3-row verify pass **1.0175 +-
  0.0044** (predicted 1.008-1.015). The serial scan also slowed the next kv_write by 18-20% (the main thread's caches
  filled with logits).
- Outside `generate --ab` it did not run until 2026-09-27 evening: the session's switch field was never set, and a
  heap's leftover byte kept the serial scan (every process race read 118 us; LESSONS #252). Fixed: every run 5.5 us.

## The KV's pages touched in time (2026-09-27)

The KV cache is allocated untouched: a page becomes memory at its first write. A decode token writes a 512-byte row
into each of the 512 streams (16 layers x 16 heads, K and V), so every 8 positions each stream enters a fresh page:
512 faults inside kv_write on the main thread, ~250 us a crossing, ~31 us a token (the Opus thinker's piece 1). Every
prediction in `build/prep/predictions.txt`, written before its run.

**The faults alone** (`bench_mem faults`: the engine's cache allocated anew each repetition, window 0 written, window
1 of 64 positions timed): 4096 faults a window by the OS's count, none on touched pages; the decode's writes 80.9 us a
position on fresh pages against 27.0 on touched ones (0.84 us a fault); `tr_kv_touch` alone 787 ns a page on one
thread and only 1.58x faster on 8 (1.62x on 16). That serialization is the bench's own: it churned 4.3 GB of fresh
pages in 2.4 s and emptied the zeroed list (LESSONS #249); the engine's decode ran the same faults ~7x on 8 threads.

**A window of 64 touched ahead** (the design given: one parallel call when a pass reaches untouched positions;
`sh tools/ab_env.sh`, process against process, 12 rounds, `TR_KV_TOUCH=0` off):

| decode, a token | off | touched ahead |
|---|---|---|
| kv_write | 80.2 us | 51.1 |
| kv_touch | - | 4.1 (12288 faults in 820 us: 0.5 us a fault a thread) |
| attention | 2475 | **2535 (+2.4%)** |
| the pass | 15.24 ms | 15.32: the touch costs 0.4% |

The prompt (411 tokens, 8 threads): kv_write 16.1 -> 4.5 ms (its faults serialized over 8 threads, 0.40 us a fault of
wall), the touch 7.1 ms (~1.8x: right after the load the zeroed list is short), the prompt 0.5% faster.

**The attention's 2.4%, taken apart** (12 rounds each, the attention against off): the same touch position-major, as
kv_write's faults place the pages, +1.8%; only the decode's windows touched, the prompt's pages left to kv_write,
+1.5% (they are ~13% of what the attention reads: no cost per touched page explains it); the touch on the main thread
alone +2.2%; the pool's call with nothing touched +0.1%. The cause is pages faulted **ahead** of their writes, whoever
faults them, in whatever order. `bench_mem kvend` isolates it: the engine's cache written to 500 positions, then 0, 8
or 64 positions past the streams' ends touched, a new cache each repetition, four runs of 15-31: the attention 51.85 /
51.34 / 50.80 GB/s, a plain read of the same streams 54.17 / 53.31 / 52.85. **Pages present past the streams' ends
slow the streaming reads by 1-2.4%** (a prefetcher running on into present memory, by all appearances: the decode's
kernel reads exactly positions [0, n)); the spread of a single run (26-152%) hides it, four runs agree.

**Built: the pages a pass enters, touched in time.** Every stream starts a page (the blocks page-aligned, a stream
padded to whole pages: under 4 KiB a stream), so the positions whose rows enter a fresh page are the same in every
stream (`tr_kv_fresh_page`); a pass whose new positions enter one touches exactly them over the pool
(`tr_kv_touch_pass`), never past its end, and a pass that enters none calls nothing. `TR_KV_TOUCH=0` turns it off.

| 12 rounds, on / off / on again | off/on | on2/on (the A/A) |
|---|---|---|
| decode, a token | 1.0023 +- 0.0022 | 1.0009 +- 0.0018 |
| the decode's attention | 1.0032 +- 0.0028 | 0.9986 +- 0.0019 |
| the prompt, 411 tokens | 1.0029 +- 0.0038 | 0.9931 +- 0.0028 |

- By the zones: the decode's kv_write 80.2 -> 49.7 us a token, kv_touch 4.9 (25 crossings of 512 faults on 8
  threads): **26 us a token, +0.17%**; the prompt's kv_write 15.9 -> 5.1 ms and kv_touch 7.0 ms: **3.8 ms, +0.4%** of
  a first prompt. The attention level with off. Every bit the same: `test_session` evaluates a model whose pages hold
  8 positions, as the real one's, pass by pass with TR_KV_TOUCH on and off (a prompt, a decode, a rewind, the
  context's end): the logits identical, 24 touches exactly; `test_kv` checks every byte of a touch, a pass's pages
  never past its end, and the fresh-page rule against every stream's pages at their own addresses.
- Predictions: the bench's cost a fault and its scaling out (its churn); the window ahead: kv_write and kv_touch in,
  the decode out (the attention); the placement out (the position-major touch), my guess of the other cores out
  (the main thread alone); kvend in; the pass's own pages: in (the decode 1.0010-1.0020, the attention level).
- The references: llama.cpp clears the whole KV when it creates a context (`ggml_backend_buffer_clear`: every page
  faulted before the prompt, the whole context present); colibri allocates it to the prompt plus the tokens asked,
  grown when needed (`kv_alloc`). Neither touches in time; llama.cpp's cleared context is all "present past the end".

## The routers as bf16 (2026-09-27)

Every F32 tensor of OLMoE's GGUFs has zero low 16 bits (the thinker's `f32bits2.py`: 81 tensors, 8.5 MiB, not one
value with a nonzero low half): the model was converted from bf16. Its 16 routers (64 x 2048, 512 KiB each) are read
every token, 8.4 MB at 46 GB/s: 179-186 us. `tr_f32_to_bf16_exact` narrows an F32 matrix at load when every value
fits (in place, the top halves), and the BF16 rows of every tier (`dot_row`, `dot_row_x4`: scalar, AVX2, AVX-512)
widen each half by a shift into the same lanes as `dot_f32`: the F32 row's bits by construction. `TR_BF16_EXACT=0`
keeps F32. BF16 stays an internal type: a GGUF tensor stored as BF16 is still refused (LESSONS #253).

| 12 rounds, on / off / on again | router, us a token | the decode, off/on | the A/A |
|---|---|---|---|
| build/bf16/ab1 (the serial argmax still in, #252) | 129.9 / 186.3 / 123.5 (1.396 +- 0.031) | 0.9984 +- 0.0033 | 0.9979 +- 0.0038 |
| build/bf16/ab2 (the argmax fixed) | 120.1 / 178.8 / 122.8 (**1.483 +- 0.025**) | 1.0027 +- 0.0024 | 1.0016 +- 0.0034 |

- **The router 1.40-1.48x, 56-63 us a token: 0.37-0.4% of a token by the zone**; the pass itself sits inside the
  process race's noise (the two sessions' A/A 0.998 and 1.002). The prompt level (its router 12.8 -> 11.4 us a
  token, 0.6 ms of 943). Predictions: the router 100-125 (in, ab2; ab1's first arm 130 just out), the decode
  1.004-1.006 (the zone in, the pass unresolved), the prompt level (in).
- Correctness: `test_kernels` (BF16 rows against the F32 rows they widen to on every tier and on scalar, special
  values, every tail, `dot_row` and `dot_row_x4`; the narrowing refused for a nonzero low half at every position,
  done otherwise); `test_tier_used` (a Q8_0 and a Q4_K model whose router fits bf16: its 544 products through the BF16
  entries, counted, every pass's logits the bits of the same model loaded with the router F32); 17 mutations red
  (`tools/mutate_bf16.sh`).
- The references keep the router F32: colibri ("norms, router, bias stay f32: small and sensitive"), ds4 on the CPU
  (`tensor_expect_layout(..., DS4_TENSOR_F32, ...)`; an F16 router kernel on its GPU for its own format), llama.cpp
  (ggml's F32 dot). Narrowed only where exact, it is neither small nor sensitive: the same bits at half the bytes.
- **The fixed argmax**: this race's first session read the sample zone at 118 us a token in every mode, where the
  argmax in parallel measures 5.6-6.9 (§The argmax in parallel): outside `generate --ab` the session's switch field
  was never set (LESSONS #252). With the struct from `calloc`, every run: **5.45-5.53 us, -113 us a token, +0.74%**.

## The idle workers, and what lies past a region's end (2026-09-27 night)

Question 84, then piece 3 of the decode in the thinker's order: while the calling thread runs a serial step, the
other threads bring their first bytes of the next call toward their caches. Every prediction in
`build/prep/predictions.txt`, written before its run; the runs in `build/q84/`.

**A bench that compares layouts races them** (LESSONS #254): the first two runs of question 84 (whole passes of each
layout, the order rotated by round) read spreads of 10-60% round to round, the RAM's speed drifting faster than a
round. `bench_q4x --ends` and `--idle` now race two layouts or two arms set by set (a pair of 0.2-0.7 ms sets, the
order by a hash, a round's value the median pair): standard errors of 0.05-0.2%.

**Question 84: present memory past a thread's region.** `bench_q4x --ends 8`: the decode's kernel (`q4x_dot2`, one
token) and a plain read, 8 threads, sequences of 4 calls from RAM, a region a thread a call. guard: the page after
every region dropped (nothing present past an end) against present memory there (next); own: a thread's regions of
consecutive calls one after the other (its run past an end is its next call's first lines) against the same slots
in reverse (ownrev) and against today's layout (next0). 21 rounds (`ends-race1.txt`):

| shape, a region | guard/next | own/ownrev | own/next0 |
|---|---|---|---|
| the dense call, 288 KiB | **1.0079 +- 0.0010** (read 1.0168) | **1.0102 +- 0.0009** (read 1.0024) | **1.0175 +- 0.0009** (read 1.0243) |
| the experts' call, 1152 KiB | 1.0012 +- 0.0016 (read 1.0070) | 1.0018 +- 0.0020 (read 1.0048) | 1.0020 +- 0.0012 (read 1.0074) |

- The weights pay what the KV paid, a region end at a time: ~0.8% of a dense call, nothing measurable on an expert's
  (one end for 4x the bytes). The dense calls are 3.2 ms of a token: a guard page at each thread's rows would give
  0.17%, a thread's regions of q, k, v, o laid one after the other 0.2-0.37%. **Question 84 closed: a lever of
  0.2-0.4% of a token in the dense calls' layout, none in the experts'**; it goes with piece 4 (q, k and v in one
  call lays the dense matrices out again at load).

**Piece 3, B: the idle workers.** `tr_pool_hint` (threads.h): the dispatcher posts, for chunk t of the next call,
the bytes its thread will read first; the worker, spinning for work, brings them in two lines at a time between
checks of its slot (T0 or T2) and leaves the rest the moment a call comes. `tr_matmul_hint` (kernels.h) computes
the hints as the call will split it (the Q4_K road's items of 16 rows, tr_matmul_grouped's rows; one input row a
group only): at most max_bytes, half the region, never past its first group. `tr_pool_region` gives a chunk's region
as `pool_run` cuts it.

The bench (`bench_q4x --idle 8`: a serial step of W us on the calling thread, then a call from RAM, hints off
against on, raced set by set; the serial step of three kinds: the clock alone, sums in L2, reads of a cold buffer;
`idle-grid1.txt`, `idle-grid2.txt`, 11 rounds a race): **every one of 180 races faster with the hints**, none slower.

| call, X a worker (T2) | W 1.5 us | W 3 us | W 6 us |
|---|---|---|---|
| dense 2.25 MiB, 64 KiB | 1.021-1.038 | 1.053-1.065 | 1.056-1.064 |
| dense, 128-256 KiB | 1.032-1.033 | 1.067-1.070 | **1.118-1.142** |
| experts 9 MiB, 64 KiB | 1.006-1.011 | 1.015-1.016 | 1.016-1.019 |
| experts, 128-256 KiB | 1.009-1.011 | 1.017-1.019 | 1.027-1.038 |

- The gain a step reaches 0.85-0.96 W once X >= 128 KiB: the RAM works through the whole serial step. T0 and T2
  alike, T2 ahead at large X; a serial step reading a cold buffer takes nothing away.

**In the engine it is level.** Hints before q, the router, gate and the output matrix, 256 KiB a worker at most
half a region; the in-process A/B `idle` (arm B no hints; `TR_POOL_HINT=2 sh tools/ab_inproc.sh <bin> idle N`), us a
token by the zones (the calls' gain, the serial steps' loss):

| race | hints | THE ESTIMATE B/A | the calls gained | the serial steps paid |
|---|---|---|---|---|
| ab-idle1, 6 runs | early (q after the down, the router after o) | 1.0053 +- 0.0038 | qkv 88, gate_up 102, router 14 | the prep 66, the mix 47, the add 15 |
| ab-idle-x64, 6 | early, 64 KiB | 0.9931 +- 0.0026 | nothing | the prep 22, the mix 14, the add 15 |
| ab-idle-late, 6 | late (q after the mix, the router after the add) | 0.9990 +- 0.0053 | qkv 41, gate_up 70, router 14 | attn_norm 54, the prep 67 |
| ab-idle-early12, 12 | early | **0.9962 +- 0.0039** | qkv 62, gate_up 85 | the prep 75, the mix 52, the add 27 |

- The serial step after a hint pays about what the call gains (LESSONS #255): the calling thread's steps read the
  rows the workers wrote (attn_out for the add, h3 for the mix) and store over lines they share (normed, the prepared
  row); with seven cores filling their caches from the RAM those messages wait. The bench's serial steps touched no
  line of theirs. Predicted 1.006-1.012 (the thinker's 1.010-1.028): out.
- Kept, off: the pool starts with its hints off (`TR_POOL_HINT=1|2` turns them on), the engine posts them (a call
  that returns at once while they are off), the `idle` switch races them, `TR_HINT_KB` sizes them (research).
  Written and never raced (Smart App Control held its three binaries for over an hour), then taken out (its
  `prefetchw` was a string in the hot zone, which the lint refuses): the calling thread asking for its own next lines
  before each hint (the norms' weights, prefetchw over normed and the prepared row), predicted 1.008-1.015 if the steps
  pay their own lines' latency, level if the workers' busy caches are slow to answer any probe (question 85).
- **pretouch raced, 2026-10-02: level, out.** Written again without a string in the hot zone (`__builtin_prefetch`,
  prefetchw through a per-function `target("prfchw")` where CPUID 0x80000001 ECX bit 8 says so; the patch kept in
  `build/q102/pretouch.patch`, disassembly checked: `0f 0d`). Before q's hint the norm's weight, normed and the
  prepared row; before the router's the ffn_norm's weight, normed and the router's row; before gate's the prepared
  row; before the output's the output_norm's weight and the row. `TR_POOL_HINT=2 ab_inproc idle 8` (build/q102,
  build/q84/ab-idle-touch; free machine: 1.90 and 1.77 logical processors busy before and after, 0.75 of them the
  kernel's System; A 16.4-17.8 ms a pass): **THE ESTIMATE B/A 1.0027 +- 0.0022** (median 1.0015; the blocks pooled
  1.0029 +- 0.0024): under 1.005 at two standard errors. The zones, us a pass, B - A against the late hints without
  pretouch (q98): attn_norm **-54.4 (was -54.1): pretouch moved nothing there**, so the norm does not pay its own
  lines' latency (its weight's 128 lines asked before the hint, normed and the prepared row owned): the prediction's
  second branch; qkv_proj (the prep inside it) +95.5 (was +40.7); the prep of gate and up (expert_gather) -56.3 (was
  -67.3); gate_up +66.9 (+70.2); router +16.5 (+14.4); the pass +27.1 (-33.5). A serial step slows by ~3.4 us a layer
  with the workers flooding whatever lines it has: not its misses, the core's or the fabric's share under the flood.
- Tests: test_base (each chunk's region as the call runs it, a static and a balanced call; a hint taken whole, left
  the moment a call comes 3 of 3, ignored for chunk 0, past the width and with hints off, the workers awake for
  every case: LESSONS #256), test_kernels (each worker's hint where its first block reads: the dense Q4_K matrix at 8
  and at 3 threads, the experts' 8 of 64 groups, the float road at 64 and 16 rows, the caps; nothing for two rows a
  group or with hints off), test_tier_used (a decode's hints taken; none with the idle switch's arm B, the same 412
  preps and every pass's logits the engine's bits); `tools/mutate_idle.sh`.
- The references (read 09-27): llama.cpp's workers spin on `ggml_barrier` with a pause and no budget (ggml-cpu.c
  576-610), no prefetch in the thread pool nor in `vec_dot_q4_K_q8_K`; ik_llama.cpp spins 100 000 pauses then yields
  (ggml.c 4778-4810) and faults the MoE's expert pages in from a pool of its own (`ggml-moe-prefetch.cpp`,
  `MADV_POPULATE_READ`, Linux), not from the idle workers; colibri keeps OpenMP's team hot (`OMP_WAIT_POLICY=active`),
  no prefetch; ds4's workers sleep on a condition variable, no spin. None brings the next call's weights with the
  idle workers.

## Piece 4: q, k, v and gate, up in one call (2026-10-02)

The decode ran q, k and v as three calls of the pool on one prepared row, and gate and up as two: five dispatches,
joins and tails a layer. `tr_matmul_q4x_prepared_n` (kernels.h) takes 2-3 weights of one shape on the same groups and
prepared rows in one call; the engine fuses q, k, v when they share a shape (no GQA: OLMoE's 16 and 16 heads) and gate,
up always on Q4_K's road. Every output is the same `q4x_dot2` or tile on the same inputs: the bits do not move
(test_kernels at 1-16 threads, test_tier_used's `fuse` arm B, every pass's logits).
- The references (read 10-02): ik_llama.cpp merges q, k, v into one tensor at load (`-mqkv`, `wqkv` and views) and
  runs gate and up as one op per expert (`GGML_OP_MOE_FUSED_UP_GATE`: chunks of the same rows of both, claimed by an
  atomic counter, the activation applied in the kernel); llama.cpp keeps them apart; colibri fuses gate and up per
  (expert, row chunk) in `xf_moe_run` (ORIGINS row 2).
- First form, each thread's items on w[0], then the same items on w[1], ... (the plan's blocks carry 2-3 weights):
  `generate --ab fuse` 8 and 12 runs, **neither at a free machine** (2.59 busy after, 3.04 before): 0.9914 +- 0.0069,
  then 1.0025 +- 0.0028; qkv_proj +13 and +24 us a pass, gate_up -27 and +17. Predicted 1.008-1.018 from a call's
  fixed cost guessed at 3-6 us: out. A call's dispatch and join cost under ~1 us; a block 2-3 times heavier makes
  the tail longer by what the dispatch saved (LESSONS #260).
- **The form kept: the items of every weight as one flat range**, w[0]'s first, balanced by the pool as finely as one
  weight's; a thread's rows cross one or two weights, one region where it had one a weight (question 84: a region's
  end ~0.8% of a dense call). Each piece of a weight marks the thread's panel stale (the weights share their groups
  and rows: LESSONS #261). 12 runs at a free machine (1.31 and 1.18 busy; build/q106/ab-fuse-flat, A 14.3-15.1 ms a
  pass): **THE ESTIMATE B/A 1.0054 +- 0.0007** (median 1.0043; the blocks pooled 1.0053 +- 0.0011); qkv_proj
  1.0283 +- 0.0014 (-55.6 us a pass), expert_gate_up 1.0077 +- 0.0009 (-38.5 us). Predicted 1.002-1.006: in.
- Open: the idle workers' hints (off by default) still describe q's plan alone, not the flat range (with
  `TR_POOL_HINT=2` they bring other rows than the call reads first); the activation inside the gate-up call (ik's
  fusion, question 85's messages); q, k, v merged at load (one stream of rows, ik's `-mqkv`), the layout question 84
  priced at 1.0-1.75% of a dense call.

## The disk at its limit: the request's size, and BitLocker's processors (2026-10-02)

Question 41 reopened (Marcello: "is there something better than the GGUF? measure every possibility, to the max").
`tests/bench_disk.c`, native, by `tools/bench_native.sh` (marker, still machine, load declared: 1.30-1.95 busy except
the one line marked), the Q8_0 file (6.85 GiB, BitLocker XTS-AES-128 in software), direct reads
(`FILE_FLAG_NO_BUFFERING`), median of 5 runs, files in `build/disk41/`. New switches: `--seq` (a run's blocks in file
order, claimed by whichever reader is free: one stream, `readers` requests in flight), `--handles` (one handle per
reader), `--scatter <bytes>` (one request whose pieces land a page apart, `ReadFileScatter`; checked byte for byte
against a plain read). Predictions in `build/prep/predictions.txt`, written before each run.

| request | 1 reader | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 2.125 MiB random (one expert matrix: the engine today) | 1.85 GB/s | 1.95 | 1.95 | 1.94 | 1.92 |
| 2.125 MiB in file order (`--seq`; 2.6-2.8 busy) | 1.88 | 1.98 | 1.89 | 1.93 | 1.96 |
| 2.125 MiB random, one handle a reader | 1.76 | 1.87 | 1.86 | 1.81 | 1.86 |
| 6.375 MiB random (an expert's three matrices) | 2.35 | 2.52 | 2.53 | 2.46 | 2.39 |
| 16 MiB random | 2.53 | 2.70 | 3.00 | 2.94 | 3.15 |
| 64 MiB random | 3.11 | 3.36 | **3.49** | 3.42 | 3.42 |
| 64 MiB in file order | 3.20 | 3.43 | **3.59** | 3.53 | 3.38 |
| 64 MiB random, one handle a reader | 3.04 | 3.39 | 3.46 | 3.50 | 3.51 |
| 64 MiB in file order, scattered in 2.125 MiB pieces | 3.11 | 3.45 | **3.48** | 3.42 | 3.43 |
| 96 MiB random | 3.35 | 3.58 | 3.50 | 3.52 | 3.48 |

- **The request's size is the only lever**: 1.9 GB/s at 2.125 MiB, 3.5 at 64 MiB (1.8x), flat past 64 MiB. Neither
  the order (`--seq` = random), nor more requests in flight (1-16 readers within ~10%), nor one handle a reader moves
  it. A fit: each request ~0.5 ms that others in flight do not overlap, plus its bytes at ~3.4 GB/s. 3.5 GB/s is 78%
  of the card's 4.5 GB/s spec. The 09-20 numbers (1.40-1.51 at 2.125 MiB) were lower by a third: same file, same
  bench; the sessions differ (load, the drive's state), so only same-session ratios are compared.
- **One request can land in the store's slots**: a 64 MiB read scattered in 2.125 MiB pieces a page apart runs level
  with a plain one (3.48 against 3.59 in file order, within the spread), every byte where a plain read puts it. The
  GGUF already stores each layer's experts as three tensors of 64 x 2.125 MiB in a row (136 MiB each): **a layer can
  be read in three requests without a format of our own**, where the engine issues 192 today.
- **BitLocker's processors** (`tools/background_load.ps1` during a sustained read; at rest the System process holds
  0.71-0.81): at 64 MiB and 3.5 GB/s **4.47 logical processors** (+3.8); at 2.125 MiB and 2.0 GB/s 1.48 (+0.8). The
  decryption runs in the System process, and the large requests that reach the bandwidth cost 2.7x more processors a
  byte (1.07 against 0.39 a GB/s). On a 4-core machine a read at full speed would take the whole CPU: the engine's
  read ahead competes with its own compute on an encrypted volume, and the plan has to count it.
- The file is not the cause: 116 extents, runs of ~100 MiB on average (`fsutil file queryExtents`).
- Not measured, and why: IoRing (a request's own call costs 2.5 us, LESSONS #94: nothing to remove at this size); an
  unencrypted volume (not removed for a measurement, question 41: Windows 11 ships encrypted, 3.5 GB/s is the
  target); the system cache's first read (`cached first seq` 6.5 GB/s is pages the system already held; the store
  reads direct by design).
- **What it gives**: the first prompt under budget reads the whole model (question 45: 4.44 s of disk in 6.11 s at
  512 tokens, half budget, 09-21). At 3.5 GB/s instead of 1.9-2.0 the disk part shrinks ~1.75x. The decode's misses
  (0.03-0.6 a token) gain at most an expert's three matrices in one request (2.5 against 1.9 GB/s), and only with a
  format of our own: not worth one now.
- **Built** (same evening, 10-02): `tr_file_preadv`, the store's runs (`cfg.readv`: a layer's consecutive experts one
  request a part, at most 96 MiB, at load, ahead and on demand) and `tr_experts_touch` (the slots' pages faulted
  over the pool before any read: inside a direct request each fresh page's first fault is the disk's time).
- **The guarded race** (`sh tools/disk_race.sh build/q107b/trochilus.exe 8`, `build/disk_race/`): Q8_0, `TR_GPU=0`,
  8 tokens, four arms a process each, rounds rotating, median of 8 (round 0 dropped), the native guards (load 2.09 ->
  1.28 busy, 0.72-0.79 the System process). **old** = `TR_EXPERT_RUNS=0 TR_EXPERT_TOUCH=0` (every part one request),
  **runs** = runs, no touch, **new** = both (the default), aa = new again. The same tokens in all 36 runs of every
  scenario. Prompt times from its tok/s; "whole" is the process, start to end (load included).

| scenario | what | old | runs | new | old/new | A/A |
|---|---|---|---|---|---|---|
| resident, 321 tokens, 8 threads | the experts' read at load | 4.07 s (3072 requests) | 2.58 (96) | **2.02** (96) | **2.015x** | 1.000 |
| | whole run | 5.49 s | 4.11 | **3.48** | **1.576x** | 1.011 |
| half budget (3264 MiB), 321 tokens (one pass, reads on demand) | prompt | 4.39 s (3531 requests) | 3.12 (918) | **2.81** | **1.564x** | 1.001 |
| | whole run | 5.73 s | 4.48 | 4.18 | 1.371x | 0.999 |
| half budget, 2048 tokens (layer-major, the I/O thread reads ahead) | prompt | 7.18 s (3507 requests) | 6.68 (762) | **6.59** | **1.090x** | 1.003 |
| | whole run | 8.40 s | 7.89 | 7.83 | 1.073x | 0.998 |
| the 8 GB machine (`-t 4`, `TR_CPU_MAX=avx2`, half budget), 321 tokens | prompt | 6.59 s | 5.32 | **4.88** | **1.350x** | 0.986 |
| | whole run | 7.92 s | 6.69 | 6.27 | 1.263x | 1.009 |
| the 8 GB machine, 2048 tokens | prompt | 20.75 s | 20.45 | 20.17 | 1.029x | 0.997 |
| | whole run | 21.94 s | 21.72 | 21.43 | 1.024x | 1.002 |

- **Read**: the runs give 1.58x on a resident load's read and the touch 1.28x more (2.0x together, 3.4 GB/s: the
  disk's one-request ceiling); a first prompt under budget that reads on demand gains 1.35-1.56x. **The touch costs
  a sixth of what it saves**: 0.093 s for the resident 6540 MiB (0.091-0.094, 8 runs, `build/disk_race_touch`),
  0.047 s for half (0.045-0.050), predicted 0.2-0.5 s: fell; the read takes 0.55 s less (the wall's 0.63 is inside
  the runs arm's 3.7% spread). The second session gave every ratio again within 1.2%, the same tokens in 40 runs. Where the I/O thread reads ahead (2048 tokens) the disk was already mostly
  behind the compute: 1.03-1.09x. **The 8 GB machine's long prompt is compute** (101 tok/s at 4 threads AVX2 against
  311 at 8 threads AVX-512; the disk, 2.77 s of reads, hidden): that is its phase 3. Not emulated: its own disk, and
  BitLocker's 3.8 processors at full speed (above) that a 4-core machine would take from its compute.
- **Two requests in flight, the premise** (`bench_disk --block 71303168 --seq --scatter 2228224`: a run of 32
  experts' parts, the engine's own request, `build/bench_native/bench_disk/`, load 1.20-1.24 busy): 1 in flight
  3269 MB/s, 2 3516 (+7.6%), 4 3525, 16 3535 (predicted +4-8%: in). The engine's resident read with one in flight
  is already 3.41 GB/s (6540 MiB in 2.01 s), so at load the room is ~3%; the on-demand runs of a one-pass prompt
  (918 requests of ~8 MiB, 2.8 GB/s) have more (the bench's 6.375 MiB: +8% at 2 readers).
- **Built** (10-02 night): `tr_file_preadv_n` (platform.h: up to 4 scattered requests, on a direct Windows file
  all issued before the first is waited for; elsewhere one after another), and the store's readv takes a run's
  three parts in one call, so they are in flight together at load, ahead and on demand.
- **Raced** (2026-10-03, `DISK_RACE_ARMS=inflight sh tools/disk_race.sh build/q115/trochilus.exe 8 res half321`,
  build/disk_inflight; arms new, serial = `TR_EXPERT_INFLIGHT=0` one call a part, aa; load 1.37-1.40 busy, 0.68-0.75
  of it the System process). Prediction (build/prep/predictions.txt, 00:09): res read serial/new 1.02-1.05, wall
  1.01-1.03; half321 read 1.05-1.08, wall 1.02-1.05.

  | scenario | read_s new | serial | serial/new | wall ms new | serial | serial/new | aa (read, wall) |
  |---|---|---|---|---|---|---|---|
  | res (6528 MiB, 96 requests) | 1.95 | 2.03 | **1.041** | 3419.5 | 3496 | **1.022** | 1.005, 0.997 |
  | half321 (7503 MiB, 918 requests) | 2.73 | 2.79 | **1.026** | 4098.5 | 4168 | **1.017** | 1.006, 1.000 |

  res inside the prediction; half321 below it: the guess "smaller runs, in flight helps more" fell (8.2 MiB a
  request gains less than ~68 MiB, LESSONS #269). The prompt there 114.0 -> 116.5 tok/s (321 tokens in 2.82 ->
  2.76 s). The same tokens in all 27 runs of each scenario. **Kept** (both above 1.01, A/A inside 1%).
  The hand mutation of the page lists (every request's list over the first's) stays green: Windows takes the list
  at the call (LESSONS #268).

## The three machines, first measured (2026-10-03)

R1's block `r1-profiles` (docs/ARCHITECTURE.md §The roadmap): `sh tools/machines.sh build/q119/trochilus.exe 8 q4k
q8` (build/machines; load 1.2-1.4 busy, 0.65-0.75 of it the System process). The same 321-token prompt
(bench/prompts/code.txt) and 64 generated tokens on: **weak** (`-t 4`, `TR_CPU_MAX=avx2`, `TR_GPU=0`, the plan
seeing 8 GiB with 4.5 free: `TR_MEM_TOTAL_MIB=8192 TR_MEM_AVAILABLE_MIB=4608`, the new switch: the plan's reserve is
a tenth of the total, and the real 31 GiB gave 3.1 GiB where an 8 GB machine has 2, so the Q4_K was refused),
**avg** (`-t 8`, `TR_GPU=0`, 16 GiB with 11 free), **pc** (as it is). Medians of 8; the bytes a token from one
profiled run a machine; the limit = the RAM's nominal bandwidth (21 / 51 / 83 GB/s) over the RAM bytes a token, plus
the disk's bytes at 3.5 GB/s. The prediction (build/prep/predictions.txt, after a 1-round smoke on q4k): q8 weak
3-4 tok/s with ~300 MiB a token from disk, prompt 15-25 tok/s; avg 40-45 tok/s at ~100% of its limit; pc 42-48.

| model | machine | experts in RAM | prompt tok/s | decode tok/s | RAM MiB/token | disk MiB/token | GB/s drawn | limit tok/s | of limit |
|---|---|---|---|---|---|---|---|---|---|
| Q4_K | weak | 224/1024, 759 MiB | 32.8 | 6.76 | 723.5 | 166.3 | 5.1 | 11.6 | 58% |
| Q4_K | avg | all, 3468 MiB | 442.2 | 69.48 | 723.5 | 0 | 52.7 | 67.2 | **103%** |
| Q4_K | pc | all | 652.5 | 72.73 | 723.5 | 0 | 55.2 | 109.4 | 66% |
| Q8_0 | weak | **83/1024, 530 MiB** | 65.9 | **2.14** | 1284.6 | **816.0** | 2.9 | 3.2 | 66% |
| Q8_0 | avg | all, 6540 MiB | 398.0 | 40.84 | 1284.6 | 0 | 55.0 | 37.9 | **108%** |
| Q8_0 | pc | all | 562.5 | 41.39 | 1284.6 | 0 | 55.8 | 61.6 | 67% |

The same tokens in all 78 runs, on the three machines (another tier, other thread counts, the GPU or not: the same
bits). What the numbers say:
- **avg** draws more than an average machine's RAM gives (103-108%): there it is the RAM's, at its limit if it
  draws its whole bandwidth (not provable here: this PC's RAM is faster).
- **pc** draws 55-56 GB/s, 66-67% of the nominal 83; the practical ceiling of this RAM, measured again the same
  day (`build/tests/bench_mem.exe ram`): 56.3 GB/s sequential at 8 threads: the engine is at 98-99% of it (#273). The engine settles on 8 decode threads on Q4_K, 4 on Q8_0.
- **weak, Q8_0: 0 hits in 9010 accesses** (the prediction said ~300 MiB a token from disk: 816). The plan keeps 83
  units, fewer than the 128 a token uses (8 experts x 16 layers), and an LRU over a cycle longer than itself evicts
  every unit just before it is wanted again: 57 439 MiB read for 385 tokens (LESSONS #271). Keeping the first layers
  and evicting the most recent (STATUS M1 item 2, the eviction during a sweep) would hit ~83/128 of a token.
- **weak's budget is eaten by the session allowance**: the plan sets aside a 4096-token F32 KV (1 GiB) plus 512 MiB
  before the experts; the run uses ~400 positions (100 MiB). On 8 GB that is 1.4 GiB, 2.7 times the experts' 530.
- **weak, the Q4_K prompt** (32.8 tok/s) is half the Q8_0's (65.9) on the same machine: AVX2 has no Q4_K tile
  (STATUS backlog: AVX2's own tile), and its 3.2 GB of reads in the prompt. Not separated yet.
- `tools/ab_modes.sh` died on the zero median (hits 0) and lost the weak rows after it: fixed, tested (#270).

## The 8 GB machine's store: its disk, its plan, its eviction (2026-10-03 night)

R1 phase 3 on the weak machine (LESSONS #271-#279): the arithmetic first (#272), then the deletion series, then the
race. Predictions in build/prep/predictions.txt, each before its run.

**The limit, measured not nominal; each machine's disk** (#273, closed). `tools/machines.sh` now takes each
machine's disk in its environment (`TR_EXPERT_DISK_MBPS`: the expert store's reads at that rate, one disk shared by
the calling thread and the I/O thread, measurement only; `tests/test_experts.c` branch `disk`, 4 mutations red) and
two limits: at the RAM's nominal bandwidth and at its practical ceiling (this PC's measured 56.3 GB/s, 68% of 83;
the others estimated at the same 68% until a real PC, question 88). Disks, estimates: the 8 GB laptop's SATA-class
SSD 500 MB/s, an average one's NVMe 2000. The 10-03 runs again (`sh tools/machines.sh table build/machines`): this
PC decodes at **98-99% of its RAM's measured ceiling** (Q4_K 72.73 tok/s, Q8_0 41.39), and the average machine's
emulation at 152-159% of an average machine's estimated ceiling: the emulation draws this PC's RAM, so its decode
is an upper bound, not an average machine's.

**The eviction, replayed before it is built** (`tools/evict_replay.py`; `make lint` runs its 9 hand-made cases):
the store's own calls on a route trace (bench/prompts/code.txt, 321 tokens + 256 generated, build/evict), eight
policies at every store size. It reproduces both measured numbers: the LRU at 83 slots 0 hits of 128 (#271's 0 of
9010), at 224 slots on Q4_K 166.3 MiB a token (the 10-03 profile: 166.3). Hits a token, of the 128 units:

| slots | LRU (ours) | MRU | random (an mmap's page cache) | per layer balanced | LFU | colibri (an LRU a layer) | **ds4** (hotness halved every 16 tokens) | Belady (the ceiling) |
|---|---|---|---|---|---|---|---|---|
| 83 (weak Q8_0, 10-03) | **0** | 10.4 | 17.1 | 36.4 | 46.7 | 36.4 | **47.3** | 65.4 |
| 128 | 59.6 | 16.9 | 35.1 | 59.6 | 59.9 | 59.6 | 61.7 | 82.9 |
| 270 | 84.8 | 38.6 | 73.5 | 87.5 | 86.5 | 86.1 | **90.2** | 107.7 |
| 400 | 102.3 | | | | | | 105.0 | 118.0 |
| 512 | 113.4 | | | | | | 114.2 | 122.7 |
| 768 | 124.3 | | | | | | 124.4 | 127.0 |

- **The premise of #271 falls on paper** (#274): "keep the first layers, evict the most recent" (MRU) hits 10 of
  128, not ~83: the routes move from token to token, and what a store keeps by arrival order goes stale. Under
  one token's units the lever is the policy (ds4's: 0 -> 47); above it, the store's size (the LRU 0 -> 85 from 83
  to 270 slots), the policy adding 0-6%.
- **ds4's policy is the best practical one at every size, and never below the LRU**; colibri's (a fixed LRU a
  layer) equals the balanced one; the page cache an mmap leans on (llama.cpp; ds4 by default) sits between the LRU
  and the per-layer ones. Belady stays 18-38% above ds4 at 83-270 slots: what a policy that knew the next token's
  routes would add.

**The plan's session allowance** (#275): 1.5 GiB set aside (a 4096-position F32 KV and a flat 512 MiB) for a run
of 353 positions that allocates ~0.19 GiB: on 8 GB, 2.7x the experts' own 0.53. Now the plan takes the session's
own bytes (olmoe.c `session_bytes`, the one sum the session's guard asks for too) at the positions the run will
hold (`tr_model_load_plan`; generate and run pass prompt + -n + the draft; chat and serve the default context).
The weak machine's store: Q8_0 83 -> 290 slots, Q4_K 224 -> 615.

**The deletion series** (`MACHINES_LIST="weak weakr aa" MACHINES_N=32 sh tools/machines.sh build/q121/trochilus.exe
8`, build/machines_q121; load 1.86 busy, 0.76 the System process). weakr = the weak machine's cores with every
expert in RAM: its compute alone. Medians of 8, the same tokens in all 58 runs:

| model | arm | prompt tok/s | decode tok/s | disk MiB/token | prompt: disk share | decode: disk share |
|---|---|---|---|---|---|---|
| Q4_K | weak (the new plan, disk 0.5 GB/s) | 20.82 | 8.03 | 42.9 | 44% (3203 MiB in 6.7 s) | 72% |
| Q4_K | weakr | 36.95 | 31.91 | 0 | | |
| Q8_0 | weak | 20.52 | 1.78 | 252.5 | 82% (6031 MiB) | 94% |
| Q8_0 | weakr | 114.72 | **42.39** | 0 | | |

- On the 8 GB machine with its disk, **the decode is the disk's** (72% Q4_K, 94% Q8_0) and so is a first Q8_0
  prompt (82%); the Q4_K prompt is half compute (step 4, AVX2's own Q4_K tile, is its lever).
- weakr's Q8_0 decode, 42.4 tok/s, is this PC's whole RAM (57 GB/s) drawn by four AVX2 cores: predicted 16-21
  from a guess never measured (#277 a). A real 8 GB machine's RAM (21 GB/s nominal) would hold it to ~11.
- With the store partial, a decode token's compute costs 34.8 ms against 31.3 resident (Q4_K: the slots' bytes
  arriving by DMA, the pointers refreshed each layer).

**The race** (`MACHINES_BIN_weakold=build/q120b/trochilus.exe MACHINES_LIST="weakold weak weakhot aa"`, 5 rounds;
weakold = the engine of 10-03 with the emulated disk, weak = the new plan with the LRU, weakhot = with ds4's
eviction; build/race_weak_q8 at -n 16, build/race_weak_q4k at -n 32; load 1.26-1.91 busy). The same tokens in all
54 runs:

| model | arm | slots | decode tok/s | disk MiB/token | whole run | prompt |
|---|---|---|---|---|---|---|
| Q8_0 | weakold | 83 | 0.57 | 816.0 | 42.1 s | 20.50 |
| Q8_0 | weak | 292 | 1.65 (**2.90x**) | 274.5 | 25.2 s | 20.39 |
| Q8_0 | weakhot | 292 | 1.70 (**2.98x**) | 266.1 | 25.0 s | 20.23 |
| Q4_K | weakold | 224 | 2.52 | 171.6 | 27.9 s | 20.85 |
| Q4_K | weak | 615 | 8.02 (**3.18x**) | 42.9 | 19.5 s | 20.81 |
| Q4_K | weakhot | 615 | 8.02 (3.18x) | 42.9 | 19.6 s | 20.68 |

- Predicted (02:10 and ~03:00): weakold 0.55-0.60 and 2.4-2.8, in; the plan 2.9-3.3x and 2.9-3.4x, in; the
  prompts level, in. ds4's eviction 1.08-1.16x on Q8_0: **1.03x, below**: the replay's 8-16% was for 256 tokens,
  the race ran 16, mostly the first tokens after the prompt (#277 c). Counted at the replay's horizon (one run an
  arm, `MACHINES_COUNTS=1`, build/counts_test): at 64 tokens 259.6 MiB a token with the LRU, 222.1 with ds4's
  (816.0 before): **3.67x fewer bytes a token than the engine of 10-03**; at 128 tokens 24% fewer misses a token.
  At every other size ds4's never misses more (the replay above; this PC at half budget, 128 tokens: 2456 misses
  against 2518). **Kept, and made the default** (`TR_EXPERT_EVICT=lru` for the LRU).

**The requests in flight, and the references' ways to read a miss** (#269, #273's points b and c):
`tests/bench_disk_misses.c`, the store's own request shapes on the Q8_0 file (`BENCH_BUILD=build/q123 sh
tools/bench_native.sh bench_disk_misses <Q8_0> <fresh copy> --runs 5`; a copy made by `xcopy /J`, which no page of
is held in RAM, for the two ways that go through the system's cache, never a unit twice). Medians of 5, load
1.45-1.79 busy. The first run of it fell under an agent's checkout (#278) and was run again. Predictions at ~04:00.

A decode token's misses, 8 units a layer (51 MiB), 4 layers a run:

| how | GB/s | ms a layer | against ours | predicted |
|---|---|---|---|---|
| **ours**: a unit's three parts in flight together, the units one after the other | 1.87 | 28.6 | 1 | 1.9-2.2 GB/s |
| ours, every unit of the layer in flight (24 requests) | 1.84 | 29.0 | 0.98 | 0.95-1.10 |
| ds4: the parts as tasks of 9 threads, a direct read each | 1.87 | 28.6 | 1.00 | 0.95-1.05 |
| colibri direct (its DIRECT=1): one 6.375 MiB request a unit | 2.13 | 25.1 | **1.14** | 1.10-1.25 |
| colibri as it ships: one request a unit through the system's cache | **2.49** | 21.5 | **1.33** | 0.5-1.0: fell |
| llama.cpp: the file mapped, 4 threads faulting its pages | 0.92 | 57.8 | 0.49 | 0.2-0.6 |

A prompt's runs, a whole layer (408 MiB), `--run k`:

| request | 1 run in flight (3 requests) | 2 runs (6) | 4 runs (12) |
|---|---|---|---|
| k = 4 experts, 8.5 MiB (a one-pass prompt's on-demand runs) | 3.05 GB/s | **3.29 (+8%)** | 2.39 (-22%, 130-190 ms) |
| k = 32 experts, 68 MiB (a resident load's) | 3.25 | **3.58 (+10%)** | 3.54 |

- **The count of requests in flight is not the lever on a miss; the request's size is** (24 in flight = 3 in
  flight = ds4's nine threads, all at 2.125 MiB). One request a unit (colibri's own format) reads 1.14x faster,
  and through the system's cache 1.33x: the cache manager splits and reads ahead a 6.375 MiB request better than
  three direct ones of 2.125 (not yet separated: buffered reads of our three parts were not timed). The mmap
  llama.cpp leans on reads half as fast.
- **A run more in flight is worth +8-10%** at both sizes (predicted +0-5% and +0-3%: fell, #277 b); four are worse
  at 8.5 MiB. The engine issues one run at a time today: the second one is a lever for the load and the prompt.
- On the 8 GB machine's SATA-class disk all of this flattens: 0.5 GB/s is reached by any request of 2 MiB; there the
  bytes are the lever (the plan and the eviction above), not the shape.

**Today's table, the final binary** (q124: the plan and ds4's eviction by default; `MACHINES_LIST="weak avg pc aa"
MACHINES_N=32 sh tools/machines.sh build/q124/trochilus.exe 5 q4k q8`, build/machines_q124; load 1.29 busy after;
the same tokens in all 54 runs; the predictions of ~05:00 all in):

| model | machine | prompt tok/s | decode tok/s | disk MiB/token | decode of the RAM's ceiling |
|---|---|---|---|---|---|
| Q4_K | weak (its disk) | 20.74 | 8.05 | 42.9 | (the disk's) |
| Q4_K | avg | 444.77 | 69.87 | 0 | 152% (this PC's RAM) |
| Q4_K | pc | 678.53 | 72.49 | 0 | **97%** |
| Q8_0 | weak (its disk) | 20.40 | **1.92** | 233.8 | (the disk's) |
| Q8_0 | avg | 400.30 | 41.50 | 0 | 161% (this PC's RAM) |
| Q8_0 | pc | 582.30 | 41.53 | 0 | **99%** |

The weak Q8_0 at 32 tokens 1.92 (1.70 in the race's 16: ds4's eviction gains with the horizon, #277 c). This PC's
prompts 3.5-4% above 10-03's (652.5, 562.5): a run's session is now its own 353 positions, not 4096 (a smaller KV
and score rows); not raced on its own.

## AVX2's own Q4_K tile (2026-10-03)

R1 phase 3, the 8 GB machine's first gap (LESSONS #280-#282). Predictions in build/prep/predictions.txt, each
before its run.

**Measured before changing.** The weak machine's Q4_K prompt with every expert in RAM (weakr: `-t 4`,
`TR_CPU_MAX=avx2`) was 37.2 tok/s, and 98.8% of it the Q4_K matmuls (8.5 of 8.6 s, build/machines_q121): 345 G weight
MACs in 34 core-seconds, 10 G a core-second, ~2.1 a cycle. AVX2 had no W16 panel: a prompt's groups went by rows,
`q4x_dot2` a token at a time (`q4x_dot_xt` is T calls of it on AVX2), every weight decoded once a token, a scalar
header, an 8-lane horizontal sum and the f64 steps per row, block and token.

**The references' AVX2 ways** (read, not raced; ORIGINS §Q4_K): ik_llama.cpp converts Q4_K to 8-bit rows from 32
tokens (`iqk_convert_q4_k_q8_1_r8`, d sc and dmin m folded to fp16 per 32) and runs an 8 rows x 8 tokens GEMM on
`vpmaddubsw`; llama.cpp's `ggml_gemm_q4_K_8x8_q8_K` runs 8 rows x 16 tokens on rows repacked at load, 32 float
accumulators that spill. Both ~9 weight MACs an instruction, on 8-bit activations: not the definition's bits. Ours
pays two exact 16-bit digits: ~4 an instruction.

**Built**: `avx2_q4x_panel` writes the W16 panel in scalar's bytes (16 rows' words transposed 8 x 8 by unpacks and
128-bit permutes, the row terms by `vpmaddwd` over the stored vectors); `q4x_tile2_t` runs AVX-512's
arithmetic in two halves of 8 rows: `vpmaddwd` + `vpaddd` for `vpdpwssd` (mod 2^32, the same), the odd row's int64
by a blend of two shifts for `vpsraq`, int64 to f64 by the 1.5 2^52 constant for `vcvtqq2pd` (every value under
2^51), the f64 steps by mul and add (exact until d and dmin). The panel and every tile equal scalar's bit for bit
the first time (test_kernels: 15 panels, 60 tiles on the avx2 tier).

**One core hot** (`bench_q4x --prompt`, 16 rows of 2048, free machine): dot2 3138 ns a token (10.4 GMAC/s); the
panel 2957 ns (a dot2 token); the tile T = 1-4: 1082 / 1532 / 2076 / 2934 ns (30 / 43 / 47 / 45 GMAC/s). A token
at groups of 4 / 8 / 40 / 320 rows: 2.1x / 2.8x / 3.9x / 4.2x dot2 (predicted 4-6x at T = 4).

**Raced** (q125 against q124, `MACHINES_LIST="weakrold weakr weakold weak aa"`, 5 rounds, 32 tokens, the same tokens
in all 34 runs): weakr prompt **36.36 -> 126.98 tok/s (3.49x)**, weak (its disk at 0.5 GB/s) **20.59 -> 34.62
(1.68x)**, the decode level (31.0 / 31.4, 8.0 / 8.0). Predicted 120-170 and 30-40. On weak the prompt now waits for
the disk 72.6% of its time (weight_read 6.7 of 9.25 s, 3203 MiB read): its next gap is the disk's (STATUS).

**The tile sequenced** (`tests/bench_q4x_genome.c`, one piece removed at a time, clock 5.09 GHz, free machine):

| T = 3 | ns a call | MACs a cycle |
|---|---|---|
| full (= the tier's) | 2125 | 9.09 |
| without the block's f64 tail | 1781 | 10.84 |
| and without the windows' int64 split | 1730 | 11.16 |
| and without the broadcasts (digits from registers) | 1418 | 13.62 |
| only the tails | 430 | |

The Winograd core alone runs 3.4 vector ops a cycle (of ~4); the block's f64 tail costs 16%, the broadcasts 15%, the
split 2.5%. Candidates, each checked bit for bit: the odd rows biased in the tile (a split of 3 ops, not 5) 1.002x,
fma in the exact sums 1.013x, both 1.013x (noise: 3-5%); both halves in one pass at T = 2 (each broadcast for 16
rows) 0.975x. Rejected. **T = 4 runs 8.09 MACs a cycle against T = 3's 9.09** (T = 2 8.54): its eight accumulators
spill, its windows' split 565 ns a call against 51 (#280).

**Tiles of 3 on AVX2**: the table's `q4x_tile_max` (AVX2 3, AVX-512 4), the plan's room for tiles of 3. One core: a
token at G = 40 907 -> 810 ns (-11%), G = 320 837 -> 735 (-12%; not a free machine, structure). Raced (q126 against
q125, 5 rounds, the same tokens in all 34 runs, load 2.3-2.5): weakr prompt **124.81 -> 137.89 tok/s (1.105x)**, weak
**34.57 -> 35.09 (1.015x)**, the decode level. Predicted +3-6% and +0-2%: the matmul's tiles weigh more than guessed.

**Since this morning** (two races, each against the binary before): the 8 GB machine's Q4_K prompt 20.6 -> 35.1 tok/s,
its cores' 36.4 -> 137.9, every token the same. Left on AVX2's tile: the block's f64 tail (16%) and the broadcasts
(15%); the next lever on weak is the disk.

## A pass reads the next layer ahead (2026-10-03 afternoon)

**The premise first, and it gave zero.** The task was a second run in flight (+8-10% on this PC's NVMe at 8.5 and
68 MiB, §The disk at its limit). On the 8 GB machine it is nothing by construction: its emulated disk is one queue
(`disk_emulate`, one clock for both threads), and a SATA-class disk reaches its 0.5 GB/s with any request of 2 MiB.
What its prompt waited for was not the disk's width but the compute's turn: the 321-token prompt fits one pass
(n_batch 512), which read every layer's experts on demand and nothing ahead (`weight_read` 6.72 of 9.13 s; the other
zones 2.39 s, ~150 ms a layer, never overlapped). The layer-major prompt's I/O thread (09-23) needs a prompt longer
than a pass.

**The arithmetic** (build/ahead/predictions.txt, written before any run): the trace of the same prompt
(build/evict/Q4_K.trace) asks 949 of 1024 units, 56-64 a layer, so reading the whole next layer ahead wastes ~7% of
it; read ahead while this layer computes, the compute hides behind the disk, and what the next layer does not ask can
be dropped while still queued, if the requests are short enough to leave it queued (the store's runs took the whole
layer, 216 MiB, in one request). A layer's time becomes its disk time plus the unasked units read before it names its
own: predicted 7.0-7.4 s, 43-46 tok/s (1.23-1.30x), Q8_0 1.13-1.19x.

**Built** (src/models/olmoe.c `forward_pass`, src/memory/experts.c): a pass whose layer named more than half its
experts reads the next layer ahead (`tr_experts_prefetch_n`, requests of at most 8 MiB a part: 7 Q4_K units, 3 Q8_0;
`TR_AHEAD_RUN_KIB`), and before the next layer acquires, `tr_experts_prefetch_cancel` drops every unit of it still
queued that it does not name (slot freed, unit absent, never counted as read). The layer-major prompt keeps its rule
and its long runs (a later block may ask what the first did not). The read ahead now queues a layer under one lock and
one wake: the I/O thread used to wake at the first unit and read it alone (#284). Every bit the same (test_prefetch's
`pass`, `cancel`, `major_kept`, `fail_pass`, `lru` branches, test_experts' `hot` and `cancel`; 39 mutations red, #283).

**Counted** (`MACHINES_COUNTS=1`, the emulated disk kept: the drop depends on how far the disk got, #285): Q4_K 31
units read and not asked, 43 dropped; Q8_0 22 and 56; the decode's bytes a token unchanged (42.9, 233.8 MiB).

**Raced** (`MACHINES_LIST="weak weakold" MACHINES_N=32 sh tools/machines.sh build/q127/trochilus.exe 5 q4k q8`,
build/machines_q127, the binary before as weakold; Q4_K 5 rounds, Q8_0 stopped steady at 3; the same tokens in all 24
runs; load 2.56 before and 2.97 after, just over the 2.5 limit, MsMpEng and the kernel's System, the arms interleaved
round by round with spreads of 0.0-1.0% on the prompt):

| model | prompt tok/s before | after | | prompt bytes before | after | of the disk's time for them | decode |
|---|---|---|---|---|---|---|---|
| Q4_K | 35.26 | **45.49** | **1.290x** | 3203 MiB | 3294 (+2.8%) | 97% | 8.01 -> 7.97 |
| Q8_0 | 20.29 | **24.55** | **1.210x** | 6031 MiB | 6158 (+2.1%) | 98% | 1.91 -> 1.91 |

- Q4_K inside its prediction; Q8_0 just above it (24.55 against 23.1-24.3 tok/s: the prediction took its hidden
  compute from one profiled run, 2.8 s; the race's runs before were 0.3 s slower than that run).
- The 8 GB machine's first prompt is now at its disk's limit for its bytes (97-98%): what is left there is bytes, 2-3%
  read unasked (the units read before the next layer names its own; an order by likelihood, the next layer's router on
  this layer's state as colibri's PILOT does in decode, would read the asked ones first), and the first prompt's
  ~950 units themselves.
- On this PC (NVMe, 3 GB/s) a whole layer is read ahead before the next one asks: nothing is dropped, every unasked
  unit read (the counts at this disk: 76). A half-budget store's prompt is not raced yet; there the second run in
  flight (+8% at 8.5 MiB) would join these short requests.
- The read ahead now returns once the I/O thread has taken its first run (#286: under the gate's load the thread woke
  after the next layer had named its own, every time): the disk starts before the compute holds the cores, which a
  real 4-core machine's spinning pool would otherwise delay; one wake a layer, the weak prompt level (one profiled run
  each: 7.11 s against 7.15, the same counts, build/ahead/counts3). The later runs of a layer still need the thread
  scheduled between them: on a real 4-core machine that stays to measure (question 88).

## The prompt's routings, told to the store (2026-10-03 evening)

**The question** (R1 phase 3, the 8 GB machine's next gap): its decode missed 12.7 units a token in the first 31
tokens after the prompt (Q4_K, 615 slots) against 7.8 later. **What the store held**: a prompt's pass names each unit
once, and the store of 10-03 added 1 a call, so every unit the prompt read stood at hotness 1; 949 units through 615
slots left the last ~600 read, the last layers, and the decode's misses evicted by age, the earliest layers first:
the ones its next token asks first again.

**ds4 does not do that** (read for this piece's verdict, LESSONS #289): its prefill adds a batch's every token row to
the units it chose (ds4_metal.m:17437-17447, 14359-14362) and halves nothing before the first decode token
(14292-14306). "ds4's eviction" had been adopted from its decode path only, and the replay's `ds4` modelled ours.

**The replay first** (`tools/evict_replay.py`, build/passage: the prediction written before every run but the first,
#288). Misses a decode token by window, the Q4_K trace at 615 slots, the decode's hotness as ds4's, the prompt's
call adding:

| a prompt's call adds | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | 128-256 | first 32 | all 256 |
|---|---|---|---|---|---|---|---|---|
| 1 (the store of 10-03, `once`) | 24.88 | 13.12 | 5.94 | 5.38 | 4.80 | 7.32 | 12.47 | 7.09 |
| its tokens' routings c, whole (ds4's own) | 6.25 | 2.38 | 4.25 | 4.44 | 4.25 | 7.31 | 4.28 | 5.81 |
| c x 16 / n | 11.12 | 7.38 | 5.00 | 5.34 | 4.80 | 7.32 | 7.12 | 6.42 |
| c x 32 / n | 7.38 | 3.62 | 3.88 | 5.09 | 4.80 | 7.32 | 4.69 | 6.08 |
| **c x 64 / n** | 6.50 | 2.38 | 3.25 | 5.00 | 4.77 | 7.32 | **3.84** | 5.96 |
| Belady (the ceiling) | 0.00 | 0.00 | 2.19 | 3.53 | 1.53 | 2.98 | 1.09 | 2.45 |

Over the 7 traces (the two weak ones at their slots, Q8_0 at 290; five prompts of 860-1000 tokens at 615), the sums
of misses a token, first 32 / all 256: `once` 129.4 / 102.3; c x 16/n 105.9 / 99.4; x 32/n 89.5 / 97.2; **x 64/n
81.4 / 95.7**; x 128/n 82.8 / 95.4; x 256/n 85.2 / 96.1; whole (ds4's) 87.3 / 96.7. The prompt's last 64 or 128
tokens weighted more: 81.4-85.1 / 95.4-96.3, nothing more. Whole counts lose most on the longest prose prompt
(prose-en's first 32: 24.1 against 19.7 scaled): a prompt's counts of hundreds outlive many halvings.
- Predicted (build/passage/predictions.txt): Q8_0 at 290, first 32 15-30% fewer: 36.66 -> 28.66, 22%, in; over 256
  3-8%: 4%, in. The other traces 2-3x in the first 32: 1.2-3.8x, partly in (prose least).

**Built** (src/memory/experts.c `tr_experts_acquire_counts`, src/models/olmoe.c `olmoe_refresh_experts`): a call
adds to each unit c x min(1, 64 / n_tok), rounded half up, at least 1 (`TR_EXPERTS_HOT_PROMPT`): a decode token 1 as
before, a pass its routings scaled to 64 tokens, twice ds4's steady state (32 x a unit's rate between halvings: the
prompt is fresh at its end). The pass's counts were already there (`s->offsets`, the routing's counting sort).
Every bit the same (only which units are read moves). Tests: test_experts' `hot` against a reference with counts
(a third of 800 random calls a pass of 1-300 tokens), its `prompt` case (the liked unit kept), test_prefetch's `heat`
(`hot_extra`, the hotness beyond one a unit, > 0 under ds4's and 0 under the LRU); 7 mutations red. test_prefetch's
on/off equality of the decode's misses held only while every hotness was equal (the read ahead never evicts the
computing layer, a call on demand may): now checked under the LRU, the logits under both (#287).

**Counted** (`MACHINES_COUNTS=1`, MACHINES_N=32, the emulated disk kept, build/passage/counts.log): disk MiB a
decode token **Q4_K 42.9 -> 14.0** (predicted 12-19), **Q8_0 233.8 -> 184.9** (180-200); the prompt's bytes +0.8-1%.

**Raced**: (`MACHINES_LIST="weak weakold" MACHINES_BIN_weakold=build/q127/trochilus.exe MACHINES_N=32 sh tools/machines.sh build/q128/trochilus.exe 5 q4k q8`, build/machines_q128; both stopped steady after 3 rounds; the same tokens in all 20 runs; load 2.44 before and 2.49 after, under the 2.5 limit, a Windows service (Appinfo) holding one core throughout):

| model | decode tok/s before | after | | prompt tok/s before | after | whole run |
|---|---|---|---|---|---|---|
| Q4_K | 8.00 | **15.50** | **1.94x** | 45.22 | 45.14 | 11.24 -> 9.39 s |
| Q8_0 | 1.91 | **2.38** | **1.25x** | 24.50 | 24.48 | 29.63 -> 26.43 s |

- Predicted from the counts (the old arm's compute and rest, 34.8 and 33.3 ms a token, plus the new bytes at 0.5 GB/s): 15.6 and 2.375 tok/s, ranges 14.5-16 and 2.3-2.45: both in. The prompts level, in.
- The 8 GB machine's Q4_K decode is now 29 ms of disk and ~35 of compute a token: the disk's share fell from 72% to 46%. Its Q8_0 decode is still the disk's (92%): 184.9 MiB a token, 117 in Belady's replay (256 tokens).
- What is left between the policy and Belady (replay, 256 tokens): Q4_K 5.96 against 2.45 misses a token, Q8_0 33.75 against 18.35: a policy from the routes (the next layer's router: STATUS §Next steps).

## Attempts

| Date | What | Before | After | Spread | Outcome |
|---|---|---|---|---|---|
| 2026-09-17 | `dot_row q8_0` AVX-512, bit-identical | 1 957 M el/s | 19 837 M el/s | 15% / 22% | kept: on by default on AVX-512 CPUs |
| 2026-09-17 | `dot_row q8_0` AVX2, bit-identical | 1 957 M el/s | 19 532 M el/s | 15% / 5% | kept: on for AVX2 CPUs without AVX-512 |
| 2026-09-17 | OLMoE-1B-7B Q8_0 decode, AVX-512 kernel | 6.87 tok/s (16 threads) | 21.08 tok/s (8 threads) | one run | kept; 16 threads 17.72: the thread default needs revisiting |
| 2026-09-17 | thread pool: 2 ms busy-wait, one slot per thread (Linux/Docker) | expert matmul 16 threads 0.376 ms | 0.014 ms | 9% / 16% | kept; to be measured on native Windows and the real model (median of 5) |
| 2026-09-17 | OLMoE-1B-7B Q8_0 decode, new pool (native Windows, median of 5) | 21.08 tok/s (8 threads, one run) | 27.02 tok/s (8 threads) | 2.4% | kept; 16 threads = 8 threads (26.90) |
| 2026-09-17 | rope from a table, activations in parallel, attention per head + `axpy_f32` AVX-512 (logits identical to the bit) | 26.34 tok/s (16 threads) | 32.78 tok/s (16 threads) | 1.4% / 2.8% | kept; 4 threads = 16 threads (32.93): memory limit |
| 2026-09-17 | batched prefill in exact C: 512-token passes, tokens per expert, matrices in blocks of 16 tokens, logits only for the last one (logits identical to the bit) | prefill 30.1 tok/s (16 threads, prompt 512) | 196.9 tok/s with the 4-token kernel | 14% / 27% | kept; decode unchanged; scales 1.16× from 8 to 16 threads (question 20) |
| 2026-09-17 | q8_0 row decompressed once per token block into a per-worker scratch, then `dot_f32` (exact by contract) | prefill 144 tok/s (16 threads) | 140 tok/s | 6% / 6% | **rejected**: no gain (reads 4 bytes per element instead of 1, and the measurement was not alternated); the good idea is keeping the weight compressed and splitting it across several tokens (row below) |
| 2026-09-17 | `dot_row_x4` kernel: one weight row against 4 tokens in registers (scalar, AVX2, AVX-512), results identical to the bit to `dot_row` | prefill 124 tok/s (16 threads, alternated runs) | 180 tok/s | 12% / 16% | kept; alone the kernel is worth 2.1× (44 against 21 G elements/s) |
| 2026-09-17 | the 4-token kernel's accumulators in an `__m512 acc[4]` array instead of named registers | — | — | — | **rejected**: the compiler keeps them on the stack and the gain disappears (LESSONS #45) |
| 2026-09-17 | `tr_matmul`'s token block at 32, 64, 128 instead of 16 | prefill 180.5 tok/s (16 threads, tile 16) | 176.9 / 183.3 / 172.5 | 10-13% | rejected: all within the noise, stays at 16 |
| 2026-09-18 | one weight row against **8** tokens (float, identical to the bit to x4), bench only | x4: 25.6 / 47.6 / 114.1 ns per row (n = 1024 / 2048 / 4096) | 1.13× / 0.79× / 0.93× | 3-8% | **rejected**: at n=2048 the 8 rows' activations (64 KB) fall out of L1 (§Adversarial review) |
| 2026-09-18 | int8 VNNI dot with the x4 structure (256 bit with ggml's sign trick; 512 bit in two blocks), bench only, not exact | x4 float as above | 256 bit 1.64× / 1.62× / 1.96×; 512 bit **1.83× / 1.79× / 2.26×** | 3-8% | measured, not adopted: int8 is not bit-identical, it can only be a declared mode. Decision open (§Adversarial review, LESSONS #65) |
| 2026-09-18 | adaptive draft's pause cap corrected (31 → 16) | exact replay: 172 passes, 65 draft rows (`code`) | 172 passes, 66 rows | — | kept: it is a correction, not a lever; the constants shift ±2-3% (§Adversarial review) |
| 2026-09-18 | threads per phase: prompt on the whole pool, short passes (up to 4 rows) on the first n slots, n measured by the session (16/8/4, the widest within 1% of the fastest, remeasured every 1024 tokens), `--decode-threads` forces it; tokens identical to the bit | decode 30.82 tok/s at context 512, 23.79 at 2048 (16 threads) | 33.45 (**1.085×**) and 24.43 (**1.027×**); with 8 forced 33.50 and 24.98 (1.050×); prefill unchanged; `--spec 8` worst case 1.064× | A/A 1.0% | **kept**, on by default. 3% and 2% margins measured and rejected (at 2048 they kept 16 threads in 10 and in 5 of 16 runs); boundary at 16 rows instead of 4: not distinguishable (§Threads per phase) |
| 2026-09-19 | KV with one head's positions in a row (`[layer][head][position]`, `src/kv/`), logits identical to the byte | decode at context 512 / 2048 / 4000: 32.30 / 24.18 / 17.91 tok/s (8 threads forced) | 35.48 (**1.06-1.10×**) / 27.32 (**1.12-1.14×**) / 20.84 (**1.15-1.18×**); prefill at 2048 **1.10×**, at 4000 **1.39-1.43×**; at context 32 not distinguishable | A/A 3.8% | **kept**: the `attention` zone goes from 32-35 to 47-48 GB/s read, of RAM's 54 (§Decode at long context and RAM bandwidth) |
| 2026-09-19 | width per zone: the decode's attention on 16 threads and the rest on 8 (exact), estimated from two profiles' zones | token at context 4000: 48.1 ms | 47.1 ms estimated (−2.1%); at 2048 −0.7% | A/A 3.8% | **not written**: below the session's threshold; question 34 |
| 2026-09-19 | the prompt's attention in groups of 16 tokens per head, 64-position blocks, `dot_f32_x4` and `axpy_f32_x4` (logits identical to the byte) | `attention` zone at 512 / 2048 / 4000: 116 / 1883 / 7420 ms | 106 / 1688 / 6568 (1.09× / 1.12× / **1.13×**) | A/A 2.1% | **kept**; in the bench 1.15× at 4000, 1.38× when L3 is under strain (§Prefill on long prompts) |
| 2026-09-19 | single-token work on the pool (norms, RoPE, KV, router, rows for the experts) and F32 rows with the tier's kernel | zones per token 121 / 455 / 880 ms, `router` 80 / 324 / 627 | 48 / 174 / 353 and 6 / 22 / 45 | A/A 2.1% | **kept**; with the row above the prefill runs **1.05-1.08× / 1.07-1.08× / 1.11-1.14×** at 512 / 2048 / 4000, decode not distinguishable (A/A 2.4%) |
| 2026-09-19 | **F16** weight rows with the tier's kernels (`vcvtph2ps`, exact conversion), bit-identical to the scalar; born from the `test_tier_used` check (LESSONS #78) | `dot_row f16` 466 M el/s in every tier (n=2048) | 22.8 G el/s AVX2, 22.7 AVX-512 (**49×**) | 3-7% | **kept**; no F16 model in the scenarios: the number is the kernel's, not a prefill's |
| 2026-09-19 | a group of 4 or 64 queries, a block of 16 or 256 positions instead of 16 × 64 (bench only) | 355 ms per layer at 4000 | 352-387 | 2-15% | rejected: all within the spread, stays at 16 × 64 |
| 2026-09-19 | **scalar `tr_expf`**, correctly rounded over all 2^32 floats (exhaustive proof in `make check`), in place of the library's `expf` in softmax and SwiGLU; Windows: logits identical to the byte; Linux: KL 3.9e-13, same bytes as the emulation | prefill 296 / 249 / 211 tok/s at 512 / 2048 / 4000; decode at 8 threads 35.1 / 28.0 / 21.4; `attention` 92 / 1537 / 5960 ms, `expert_act` 138 / 595 / 1201 | 306-315 / 303-308 / 276; 36.0 / 29.3 / 23.2; 41 / 591 / 2331 and 25 / 108 / 218 | A/A 3.0% prefill, 1.9% decode | **kept**: prefill 1.03-1.08× / 1.20-1.23× / 1.29-1.31×, decode 1.02-1.10×; Windows and Linux now give the same bytes (§`tr_expf`) |
| 2026-09-19 | `tr_expf` in SIMD (not written: the numbers to decide with) | after the scalar, softmax 9 / 130 / 610 ms and `expert_act` 25 / 108 / 218 ms of the prefill | estimate: prefill 1.01× / 1.03× / 1.04×, decode +1-2%, about 200 lines | — | **Marcello decides** (question 39): the scalar has taken almost everything |
| 2026-09-19 | clean-machine remeasure of threads per phase and decode at long context (four `yes` processes forgotten under the 09/18-19 measurements, LESSONS #84) | decode at 512, 8 against 16 threads: 1.10×; RAM ~54 GB/s | 1.00-1.02× (not distinguishable at any context); RAM ~57 GB/s with 4-6 threads; at 512 the decode wants 4 threads | A/A 2.5-4.6% | correct conclusion: few threads yes, the "+10%" no; the width estimator needs rewriting (question 31, §Clean machine remeasure) |
| 2026-09-19 | decode width estimator rewritten (question 31, LESSONS #88): margin from the noise measured in the passes themselves (no longer a fixed ceiling), picking in pairs (never a third width's noise), extension up to `TR_DECODE_TUNE_ROUNDS_MAX` on the two contenders, a new measurement at every context class and after `TR_DECODE_TUNE_REMEASURE_KEPT` passes kept with a change pending, two-vote hysteresis before switching pick | — | — | — | C tests green (`tests/test_phase.c`, `tools/mutate_tune.sh`); **to be validated on the real model**, at night on a quiet machine: no number yet |
| 2026-09-24 | `dot_row q4_k` AVX-512: a sub-block's 16 values computed once, `vpermps` on the nibble; bit-identical | Q8_0 15 970-17 479 M el/s | Q4_K 13 456-15 685 M el/s (0.84-0.90x per element, ~1.6x per byte) | 5-30% | kept: on for AVX-512 CPUs; the real model's decode 1.5x Q8_0 |
| 2026-09-24 | `dot_row q4_k` AVX2 with the same table (two `vpermps` of 8 + `blendv`) against the conversion | row 11 390-11 946, x4 27 650-28 884 M el/s | row 11 548-12 216 (+0.5-1.5%), x4 26 616-27 914 (-3.5%) | ~1% A/A | rejected: level or worse on Zen 4; the conversion stays |
| 2026-09-24 | `dot_row2_x4` AVX-512 (two weight rows against four input rows, eight accumulators; Q8_0, Q4_K, Q6_K), bit-identical; `tr_matmul` takes rows in pairs | whole matrix 1024x2048, 64 tokens: Q8_0 0.0607 / 0.0084 ms per token (1 / 16 cores); engine prefill Q8_0 200.6, Q4_K_M 211.0 tok/s | 0.0491 / 0.0065; prefill 252.9 / 241.3 and 236.7 / 233.8 | A/A 4.6% and 1.2% | kept: prefill 1.20-1.26x (Q8_0), 1.11-1.12x (Q4_K_M), decode unchanged |
| 2026-09-24 | the sixteen Q6_K scales in SIMD instead of scalar (same single rounding) | whole matrix Q6_K, 1 core: 0.0551 (two rows) / 0.0602 (x4) ms per token | 0.0523 / 0.0590 | control Q8_0 0.0491 in both sessions | kept |
| 2026-09-24 | a row decoded once per tile of 16 tokens, then the F32 x4 kernel (not built: measured the F32 x4 kernel alone) | Q8_0 x4 43 926 / 36 771 M/s (n 2048 / 4096) | F32 x4 44 779 / 33 135 | 3% | rejected: the matmul is bound by loading the input rows, not by decoding the weights |
| 2026-09-24 | skipping cached positions exactly (exp exactly 0; absorption in the lane sum and in the output; keys and values split in high and low 16 bits; norm, block and low-rank bounds), premise on the real model before any kernel | KV read per decode token: all of it | oracle (every value known): 0.998-1.000 of the bytes; no score 30 below its max in 65 M positions | six runs, all layers and heads | **rejected before code**: OLMoE's QK-norm keeps the attention flat (§Skipping cached positions exactly) |
| 2026-09-24 | the KV packed in 28 bits, lossless (premise bench, `tests/bench_kvpack.c`) | 4 bytes a value | 3.53-3.55 bytes a value (1.13× fewer); 25 344 queries give the F32 attention's bits | still machine (2026-09-24): 0.97–1.05× the F32 attention, spreads 7–64% | **rejected on the CPU** (question 57): decoding eats the bytes |
| 2026-09-24 | the decode's attention on the GPU (`src/backend/gpu_attn`, the same bytes; keep-warm between layers, warm-up in the prompt's last pass) | decode at 8 threads, 512 / 2048 / 4000: 35.0 / 29.1 / 22.1 tok/s | 37.4 / 34.8 / 33.4 | A/A ≤ 4.0% | **kept, on by default when there is a GPU**: 1.04-1.07× / 1.20-1.28× / 1.43-1.51× (measured width: 1.31-1.32× at 2048, 1.54-1.58× at 4000) |
| 2026-09-24 | an exact exp with float32 and int32 only, to run on the GPU at full rate (`tests/bench_expf32.c`, premise) | `tr_expf` 3.59-3.60 ns (double) | 3.10-3.22 ns with fma, 5.6 without; 0 of 2^32 differ; in SIMD (2026-09-24, still machine) AVX-512 0.733 ns, AVX2 0.802 (tr_expf 3.52), 0 of 2^32 differ | one thread | **not wired into the CPU engine** (question 58): prefill ~1.02-1.03×, question 39's range; the GPU prefill's reference |
| 2026-09-24 | the decode's attention one position at a time (`tr_attention_group` with one query runs `tr_attention_head`), the same bits | decode at 8 threads, 512 / 2048 / 4000: 35.5 / 28.3 / 22.6 tok/s; the attention zone at 2048 11.00 ms | 36.1 / 27.9 / 22.2; the zone 10.02 ms | A/A 2.9% | kept, but **not distinguishable** on a still machine (0.96-1.05×); the bench's 1.10-1.13× was measured under load (LESSONS #160) |
| 2026-09-24 | `dot_row2_x8` AVX-512 (two weight rows against eight input rows, sixteen accumulators; Q8_0, Q4_K, Q6_K) and the lane tree in SIMD (`avx512_pair_sums`, the x4 two-row kernels too), bit-identical | matmul 1024x2048, 64 tokens, one core: 87.4 GFLOP/s; engine prefill at 16 threads, container: Q8_0 217.5, Q4_K_M 215.4 tok/s | 102.3 GFLOP/s (1.17x); prefill 258.5 and 256.0 (1.19x), Q4_K_M at 8 threads 1.30x | kernel lines 1-3%; engine min-max 181-272 under load 3.2-4.4 | kept, on by default on AVX-512; native prefill_context to confirm (RAM) |
| 2026-09-24 | the Q8_0 scale by vcvtph2ps in `dot_row2_x8` (three instructions instead of a scalar conversion) | x8 kernel 2048 cols 98.6-98.8 GFLOP/s (container) | 97.3-97.6 | 0.4-4.7% | rejected: the conversion moves onto the FP pipes that bound the loop (LESSONS #167) |
| 2026-09-24 | the Q8_0 scales converted ahead, 16 blocks at a time (scalar, or a gather and vcvtph2ps), read back by broadcast loads; and a ring of the next block's scale in memory | x8 kernel 2048 cols 100-104 (container) | ahead scalar 89, gather 92-94, ring 100-101 | 0.2-3.9% | rejected: the interleaved scalar conversion is free on the integer units |
| 2026-09-24 | the x8 step in bench_peak's order (two tokens' loads, then four multiplies, then four adds) against per token | 100-104 (container) | 102-104 | 0.2-3.1% | not distinguishable: the compiler's order kept (LESSONS #162's 20% is the peak loop's, not this one's) |
| 2026-09-25 | Q4_K `dot_row` AVX-512: the scales in a vector one block ahead, a prefetch 4608 bytes ahead (bit-identical) | 618 cycles a row in L1; 34.7-35.2 GB/s at 4 threads from RAM | 497; 42.0-42.8 | 1-5% L1; 7-45% RAM | kept (LESSONS #179, #180) |
| 2026-09-25 | Q4_K `dot_row2` AVX-512: two rows against one token, 8 blocks' scales first, prefetch; tr_matmul's decode in pairs (bit-identical) | 497 cycles; 42.0-42.8 GB/s | 443-450; 43.3-45.3 | 1-5%; 15-45% | kept: the real model's decode at 4 threads 40.3 -> 50.4 tok/s with the row above, llama.cpp 50.2 |
| 2026-09-25 | the pair (and four rows) with the scales one block ahead, from RAM | 42.0-42.8 GB/s | 25.4-28.1 | 8-37% | rejected: short interleaved streams (LESSONS #181) |
| 2026-09-25 | Q4_K tables by a fused multiply-subtract (exact) | 454 / 497 cycles | 481 / 512 | 2-3% | rejected: a second broadcast and a copy |
| 2026-09-25 | decode rows claimed 64 at a time by an atomic counter (ggml's way) | contiguous chunks | 0.93-0.97x | 9-24% | rejected: short chunks, more stream starts |
| 2026-09-25 | SMT in the decode, natively: two threads on each of 4 / 8 / 16 cores (TR_POOL_PIN=1 under an affinity mask) | 53.0-54.7 / 60.01 / 58.26 tok/s | 55.6-56.0 / 60.06 / 8.43 | 1-6% | **rejected**: the engine's width is 8 cores, where it adds nothing; 16 x 2 collapses (question 68) |
| 2026-09-25 | the decode's matmuls and attention balanced at the tail (tr_parallel_for_balanced, 64 blocks a chunk; same bits) | decode calls' wall 20.3-21.4 ms a token (container, 8) / 17.0-17.3 (native, 8) | 18.8-19.6 / 16.6-17.3 | two runs each | kept: ~1.08x in the container at 8, level natively at 8, 1-2% worse natively at 16 (not a chosen width) |
| 2026-09-26 | the decode's experts balanced at the tail too (the condition missed their 64 groups, LESSONS #192; same bits) | a token 29.4-31.3 ms, tails 13-68 us (native, free machine) | 28.2-29.0 ms, tails 1.6-1.7 us | 4 pairs; ab_zone 8 pairs | **kept**: decode 1.04-1.08x (traced pairs), the experts' zones 0.958-0.964, the RAM idle 5-11% -> 1-4% |
| 2026-09-26 | gate, up and the activation in one call on the two matrices (two streams a thread; same bits) | three calls | head-normalised level; raw 10-20% slower | the whole run 10-15% | removed (LESSONS #195); rows interleaved at load 1.006-1.023x on the free machine: question 72 closed as no |
| 2026-09-26 | 2 MB pages for the weights and the KV in the container (madvise) | 42.6-46.5 GB/s | 42.3-46.6 | 7-30% | no: on the free machine too (plain read level, the matmul 0.93-1.01x); code removed |
| 2026-09-26 | the softmax's and SwiGLU's exp in a vector (the table's `expf_f32`, AVX-512 and AVX2 without FMA; same bits, 0 of 2^32 differ) | the prompt's attention zone 0.42-0.45 / 1.80-1.91 s at 2048 / 4000 (native, 16 threads) | 0.32-0.34 / 1.28-1.44 | 8 pairs, background 2.3-2.4 (free) | **kept**: the zone 0.732-0.747, the prefill 0.951 at 2048 and 0.953 at 4000 (1.05x) |
| 2026-09-26 | exact E32 Q4_K tile by Winograd's inner product (the digit in the weight's free nibble; bit-identical to its definition) | plain E32 127.0 GFLOP/s-eq, the float tile 164.7 (one core) | 149.0 after the f64 super-block end (136.6 before); W16 (scale inside a 16-bit weight) 158.7, with its windows into int64 167.2 | one core, background ~2.0-2.4 | W16 at T = 4 167.2 against the float tile's 164.7 at T = 24: exact and at least as fast on the tile; not yet in the engine (question 77, §Exact sums at the float's speed) |
| 2026-09-26 | the float contract's arithmetic against int8 tensor cores on the laptop's RTX 4070 (premise) | mul.rn + add.rn 3.29 T-MAC/s | mma.sync s8 59.10 T-MAC/s (exact); E32 = 14.8 eq, 4.5x | one run each, 7 launches, best | premise measured: question 74 (does the GPU count?) for Marcello |
| 2026-09-26 | the next item's 16 rows prefetched into L2 before the current item's tiles (the panel from cold weights 3.7-4.0 us against 1.83 hot, measured alone; same bits) | experts zone, Q4_K, 512 tokens, 4 threads | 0.990 [0.970-1.022], prefill 0.995 | 6 alternated pairs, load rising to 3.5 mid-run | reverted: inside the noise (predicted +4-6%); to retry spread over the tile's phases, at a free machine |
| 2026-09-26 | Q4_K's definition in exact integers in the engine (W16 panel and tiles for the prompt, `q4x_dot2` for the decode; question 77's stage 2) | float Q4_K: prompt 512 at 4 threads, decode at 4 threads | prompt 1.001 [0.995-1.008], experts zone 0.992; decode matmul zones 1.080 → 1.006 [0.989-1.016] after the pair's headers of both rows in one ymm (the pair 0.937x the float's, hot) | ab_zone, 6 alternated pairs, background 1.74-2.63 | kept: exact (logits identical at 4/16 threads and between prompt and decode; the 2-layer oracles 1-26% nearer their reference) at the float's speed in both phases; the modelled +10% on the experts zone did not come (§Exact sums at the float's speed, stage 2) |
| 2026-09-27 | a draft without a model on new code: an n-gram table from 78 MiB of public code (4-212 MiB), a grammar's ceiling (§A draft without a model) | prompt lookup 0.68x fixed, 0.97x adaptive (model; measured 0.60x, 0.953x) | best 1.01x (4 MiB table, >= 70% sure, 64% accepted); a perfect grammar 1.25x at most | replay = the engine's counters on 3 prompts; speed from measured row costs | not built: right guesses come one at a time, a row costs half a pass |
| 2026-09-27 | the same drafts priced with today's verified row (`tools/row_price.sh`) and at its bytes, a calibrated gate, llama.cpp's lookup replayed (§The post-it taken apart) | best 1.01x at the 09-18 price | gate (context cache + table) 1.077x at today's Q4_K prices, 1.116x at bytes (1.170x / 1.210x at 2048 / 4000, modelled); the prompt copy at draft 8 on code-edit 1.95x Q8_0, 1.78x Q4_K | row prices native, free machine, 3 alternated rounds; the gate out of sample | the 09-18 verdict reversed; not built: question 78 (the short pass at its bytes) first |
| 2026-09-27 | the short verify pass on phase-major's balanced items on every tier, and Q8_0's `dot_row_xt` (a weight row decoded once for 2-3 tokens), same bits (§The short verify pass at its bytes) | Q8_0 8 threads: 2 rows 41.33, 3 rows 62.69 ms; 16 threads 47.27 / 71.58 | 35.93 (0.869) / 42.97 (0.685): a row 0.60 -> 0.39 and 0.72 -> 0.33 of a pass; 16 threads 0.814 / 0.646; one row 1.000; drafts at Q8_0's prices on new code 0.969x -> 1.040x (the engine's lookup), the gate 1.013x -> 1.103x | native, free machine, 4 rounds, the order swapped every round, A/A within 1.1% | **kept**: Q8_0's 3-row pass at its bytes; left Q4_K's short groups and the width (question 82) |
| 2026-09-27 | the short passes' width forced to 4 / 8 / 16 threads on a pool of 16 (question 82) | the one-row pass's best: Q8_0 4 threads, Q4_K 8 | a 3-row pass: Q8_0 best at 8 (46.47 / 43.15 / 45.72 ms), Q4_K at 8-16 (44.57 / 28.83 / 27.66) | native, free, 3 rounds alternated | measured, not built: a short pass needs its own width |
| 2026-09-27 | Q4_K's groups of 2-3 rows by `q4x_dot2` pairs, a token at a time (`Q4X_MIN_TILE_ROWS` 4) | Q4_K 8 threads: 2 rows 22.99, 3 rows 28.44 ms | 22.20 (0.966) / 28.99 (1.019) | native, free, 4 rounds, order swapped, A/A within 1.1% | 2 rows kept (as `Q4X_MIN_TILE_ROWS` 3), 3 rows no (§The Q4_K short passes) |
| 2026-09-27 | the W16 panel's rows brought ahead: the next item's (pf1), block s + 2 (pf2), both (pf3); with 2-row groups by pairs (m3) | 23.07 / 28.32 ms | pf3 22.14 / 26.65 (0.941); pf1 + m3 21.89 / 26.81; the prompt 405 -> 405-409 tok/s | native, free, 7 binaries alternated, A/A 0.988-0.996 | **kept**: pf3 and m3 (§The Q4_K short passes) |
| 2026-09-27 | the final (pf3 + m3), every pass forced on 8 threads | 15.50 / 22.92 / 28.00 ms | 15.44 / 22.19 (0.968) / 26.53 (0.947) | native, free, 4 rounds, A/A within 0.6% | **kept** |
| 2026-09-27 | every size of verify pass measuring its own width, on every width (question 82) | pool of 16, the pool: Q8_0 2 / 3 rows 36.97 / 44.37, Q4_K 22.01 / 27.06 ms | 0.972 / 0.978, Q4_K 1.017 / 1.024 | native, free, 4 rounds, three ways alternated | no: the probes on 4 threads cost Q4_K more than they win (§The verify passes' own width) |
| 2026-09-27 | the same, the narrowest width skipped | 37.08 / 44.31, Q4_K 22.11 / 27.00 | 0.961 / 0.966, Q4_K 1.005 / 0.982; 8 picked in 16 of 16 | native, free, 4 rounds | **kept** |
| 2026-09-27 | `q4x_dot_xt` alone (road (c)): two Q4_K rows against 2-3 prepared rows, each weight decoded once for them, against `q4x_dot2` a token at a time and the W16 panel with a tile | L1: dot2 x 3 468.1 ns; RAM, 8 threads: dot2 x 3 34.7, the panel 42.2 GB/s | xt(3) 333.2 ns (0.712), 49.0 GB/s (0.88 of the read); xt(2) 0.768, 54.1 GB/s (0.97) | L1 1-7% (two lines 21-51%), RAM 7-17% | to the engine (§A Q4_K weight row decoded once for 2-3 tokens) |
| 2026-09-27 | Q4_K's groups of 2-3 rows by runs of `q4x_dot_xt` (`Q4X_MIN_TILE_ROWS` 4) | Q4_K 8 threads: 1 / 2 / 3 rows 15.42 / 22.31 / 26.68 ms (the second session 15.75 / 22.23 / 27.32) | 15.91 (1.032) / 21.22 (0.951) / 26.28 (0.985); with `q4x_dot2` as committed 15.85 (1.006) / 21.97 (0.988) / 26.90 (0.985), again on a free machine 15.90 (1.011) / 21.45 (0.934) / 26.45 (0.956) | native, 4, 6 and 6 rounds, A/A 0.980-1.026 | **kept**; `q4x_dot2` stays the committed text (LESSONS #239) |
| 2026-09-27 | the KV's pages touched a window of 64 positions ahead, one parallel call when a pass reaches untouched positions (the thinker's design) | kv_write 80.2 us a token, the attention 2475, the pass 15.24 ms | kv_write 51.1, kv_touch 4.1, **the attention 2535 (+2.4%)**, the pass 15.32 (0.9963); the prompt 1.0054 | native, 12 rounds, process against process, A/A 1.0014 | **no**: pages present past the streams' ends slow the reads (§The KV's pages touched in time) |
| 2026-09-27 | the KV's pages a pass enters, touched over the pool in time, never past the pass's end (the streams page-aligned) | as above | kv_write 49.7, kv_touch 4.9, the attention level (1.0032 +- 0.0028); the decode 1.0023 +- 0.0022 (zones +0.17%), the prompt 1.0029 (zones +0.4%) | native, 12 rounds, A/A 1.0009 | **kept**, on by default (`TR_KV_TOUCH=0` off) |
| 2026-09-27 | the routers (F32, every value's low half zero) narrowed to bf16 at load, widened by a shift into dot_f32's lanes | the router zone 179-186 us a token | 120-130 (1.40-1.48x); the decode 1.0027 +- 0.0024 and 0.9984 +- 0.0033 in two sessions (A/A 1.0016, 0.9979); the prompt level | native, 12 rounds twice, process against process | **kept**, on by default (`TR_BF16_EXACT=0` off): 0.37-0.4% of a token by the zone |
| 2026-09-27 | the session's struct from calloc: the parallel argmax outside generate --ab (LESSONS #252) | the sample zone 118 us a token in every process race | 5.45-5.53 us | native, 36 runs | **kept**: +0.74% of a token |
| 2026-09-27 | question 84: the page after each thread's region dropped (guard), a thread's regions of consecutive calls one after the other (own); `bench_q4x --ends 8`, raced set by set | a dense call from RAM, 288 KiB a region | guard 1.0079 +- 0.0010, own 1.0102-1.0175; the experts' calls 1.001-1.002 (level) | 21 rounds, SE 0.05-0.2% | **measured**: 0.2-0.4% of a token in the dense layout, taken with piece 4; nothing on the experts |
| 2026-09-27 | the idle workers bring the next call's first 256 KiB in during the serial steps (`tr_pool_hint`, T2) | the decode, no hints | the bench 1.02-1.14 a call after a 1.5-6 us step; the engine 1.0053, 0.9931 (64 KiB), 0.9990 (late), 0.9962 +- 0.0039 (12 runs) | in-process A/B, 6-12 runs | **off by default** (`TR_POOL_HINT=2` on): the serial step after a hint pays what the call gains (LESSONS #255, question 85) |
| 2026-10-02 | pretouch: before each hint the calling thread asks for its own next lines (the norms' weights; prefetchw over normed, the prepared row, the router's row) | the decode, no hints (arm B) | 1.0027 +- 0.0022 (8 runs); attn_norm still +54 us a pass, the prep inside qkv +55 us more | in-process A/B, native, free | **out** (patch in build/q102): the serial step's slowdown is not its own lines' latency (question 85) |
| 2026-10-02 | piece 4, first form: q, k, v and gate, up one call each, a thread's items on each weight in turn | the decode, one call a weight | 0.9914 +- 0.0069 (8 runs), 1.0025 +- 0.0028 (12); qkv_proj +13/+24 us | in-process A/B, native, not free | **replaced**: the blocks 2-3 weights heavy, the tail took the dispatch's gain (LESSONS #260) |
| 2026-10-02 | piece 4, the form kept: the weights' items as one flat range (`tr_matmul_q4x_prepared_n`) | the decode, one call a weight (14.3-15.1 ms a pass) | **1.0054 +- 0.0007** (12 runs); qkv_proj 1.0283, gate_up 1.0077 | in-process A/B, native, free | **kept**, on by default (`--ab fuse` arm B: the calls apart) |
| 2026-10-03 | AVX2's own Q4_K W16 panel and tile (two halves of 8 rows), bit-identical | weakr Q4_K prompt 36.36 tok/s (dot2 a token) | **126.98** (3.49x); weak 20.59 -> 34.62 | 0.2-1.1% (one weakr round 13%) | kept: AVX2 and the avx512 tier without VBMI |
| 2026-10-03 | AVX2's tile: the odd rows biased in the tile (the window's split in 3 ops) | T = 3 2125 ns | 2120 | 3-6% | rejected: level |
| 2026-10-03 | AVX2's tile: fma in the block's exact sums | T = 3 2125 ns | 2097 (1.013x) | 3-6% | rejected: inside the noise |
| 2026-10-03 | AVX2's tile: both halves in one pass at T = 2 (a broadcast for 16 rows) | T = 2 1508 ns | 1547 (0.975x) | 4-9% | rejected |
| 2026-10-03 | AVX2's tiles at most 3 wide (`q4x_tile_max`; T = 4 spills) | weakr prompt 124.81 tok/s | **137.89** (1.105x); weak 34.57 -> 35.09 | 1.1-4.5% | kept |

## The KV grown from the store's room (2026-10-03 night)

**The question** (R1 phase 3, the 8 GB machine's next gap, its first): chat and serve plan the default context, and
the plan set aside its whole KV at load (4096 positions F32: 1 GiB, with the working memory of passes of 512), for a
conversation that has written none of it. On the 8 GB machine that is ~150 Q8_0 slots of the 290 a run of its own
positions keeps.

**The references** (ORIGINS row 7): colibri takes the KV of max_ctx off its cache once (`cap_for_ram`,
colibri.c:10226-10259; its OLMoE engine the KV of prompt + new tokens, else 4096 F32, olmoe.c:488-524), and its
`rss_guard` (8412-8464) only lowers the cap, never raises it; ds4's automatic cache ignores the context
(`(void)ctx_size`, ds4.c:68145) and allocates its KV whole; llama.cpp allocates the KV whole at construction
(llama-kv-cache.cpp:233), its `--fit` a decision at load. Nobody grows the KV into the cache's memory.

**Built**: the plan leaves the KV out of what it sets aside, and gives the store and the sessions' KV pages one room
(olmoe.c `kv_room`): before each pass the session takes the pages of the positions it will write (`kv_hold`,
tr_kv_bytes up to them: the pages a pass faults, `tr_kv_touch_pass`), and the store keeps floor((room - every
session's KV pages) / slot) slots (`tr_experts_set_slots`): the policy's coldest units go, the ones above the new
count move below it (their bytes copied whole, their recency kept), and the pages of the slots given back go to the
system (`tr_pages_release`: MADV_DONTNEED, Windows' decommit; the slab now `tr_pages_alloc`). A session freed gives
its KV back and the store takes its slots again. A session whose whole context's KV would leave the store under its
minimum is refused by name, so chat's halving still finds the context that fits; the machine's guard is asked only
for what the store will not have given back. A forced budget and a resident store share no room (as before). Every
bit the same: only which units are in RAM moves.

**Counted** (`MACHINES_COUNTS=1`, weak, MACHINES_N=32, the emulated disk kept, build/kvroom; predictions in
build/kvroom/predictions.txt, written before each run):

| `generate --tokens <321> -n 32` on weak | slots in RAM | made / given / moved | disk MiB a decode token |
|---|---|---|---|
| Q8_0, `-c 4096` (a chat's plan), before | 131 | 131 / - / - | 375.9 |
| Q8_0, `-c 4096`, after | **278** | 291 / 13 / 1 | **193.7** |
| Q4_K, `-c 4096`, before | 314 | 314 / - / - | 87.4 |
| Q4_K, `-c 4096`, after | **591** | 617 / 26 / 2 | **16.3** |
| Q8_0, the run's own positions, before / after | 290 / 291 | | 184.9 / 183.6 |
| Q4_K, the run's own positions, before / after | 615 / 616 | | 14.0 / 14.0 |

- Predicted: before 125-145 and 310-350 slots (in), after 276-286 and 590-605 (in), made 290-300 and 615-630 (in),
  given ~14 and ~26 (13, 26: in); disk a token Q8_0 330-450 -> 180-205 (in), Q4_K 35-70 -> 13-18 (the before
  above: 87.4; the after in); the run's own positions level (in).
- The old Q8_0 chat store (131 slots) was under the 136 a read ahead needs (2 layers and a margin): its prompt read
  on demand. The new one reads ahead again.

**Raced** (2026-10-04 00:07, `MACHINES_LIST="weak weakold" MACHINES_BIN_weakold=build/q128/runnable2/trochilus.exe
MACHINES_N=32 MACHINES_ARGS="-c 4096" sh tools/machines.sh build/trochilus.exe 5 q4k q8`, build/kvroom/race_c4096b;
3 rounds an arm, stopped steady, every run behind the still-machine guard, load 3.12 before and 1.49 after; the first
try at 23:03 was stopped: a model download, not ours, held the disk, LESSONS #291):

| a chat's plan, weak | decode tok/s before | after | | prompt tok/s before | after | whole run |
|---|---|---|---|---|---|---|
| Q4_K | 4.55 | **14.77** | **3.25x** | 45.45 | 45.41 | 14.07 -> 9.41 s |
| Q8_0 | 1.22 | **2.29** | **1.88x** | 20.55 | **24.59** | 41.31 -> 26.87 s |

- Predicted (build/kvroom/predictions.txt, from the counts and the old arm's compute): decode Q4_K 4.3-4.9 -> 13.5-15.5,
  Q8_0 1.15-1.30 -> 2.20-2.40; prompt Q4_K level, Q8_0 ~19-21 -> 23.5-25.5: all in. The same tokens in all 20 runs.
- The Q8_0 prompt's 1.20x is the read ahead back (the old store's 131 slots were under its 136).
- A chat on the 8 GB machine now generates as fast as a run of its own length (15.50 and 2.38 tok/s at 290 and 615
  slots, §The prompt's routings): what the context it may reach costs is paid only when it reaches it.

**On colibri's own engine** (2026-10-04, the offer of UPSTREAM row 7; LESSONS #293): the fork's branch
`perf/olmoe-kv-room` (`kv_room_fit`: before a forward, each layer's cap down to what the room holds beside the KV's
pages, its coldest slots freed, `malloc_trim` on glibc) against `dev` `0813cf8`. OLMoE int8 (colibri's conversion),
`docker run --memory=5g` (the limit counts the page cache), `RAM_GB=5`, cap 0, 4 threads, the model's files dropped
from the VM's page cache before every run, 32 prompt tokens + 64 generated, 5 alternated rounds after a warm-up
(colibri `tmp/olmoe-kv-room/race2.log`, manifest validated by its `experiment_manifest.py`):

| colibri, 5 GB | slots a layer | hits / misses | request, median (min-max) | peak RSS |
|---|---|---|---|---|
| dev | 16 | 8133 / 4155 | 22.05 s (21.44-22.66) | 3.31 |
| the room shared | 26 | 9652 / 2636 | **18.52 s** (18.35-18.76), **1.19x** | 4.25 |

- Predicted 12 -> 22 slots, hits ~40 -> 60-65%, **1.4-1.8x**: the slots and hits came out higher (resident ~1.5 GB,
  not 2.1), the time lower. colibri reads experts through the page cache: the gigabyte dev sets aside holds expert
  pages there and serves part of its misses without the disk, so 37% fewer misses buy 16% of the time.
- Every forward's logits byte-identical between the arms (`DUMP`); the same tokens in all 12 runs. A first race was
  spoiled by another window's gate (dev 38-56 s, #294) and rerun.
- The freed slots' memory goes back (a scratch program, 16 OLMoE slots from glibc's heap, 14 freed): RSS -84 MB with
  `malloc_trim(0)`, -0 MB without; from mmap both give it back.

## Qwen3.8-Flash-Next FP8 on colibri's Vulkan tier, the 8 GB laptop GPU (2026-10-04)

What a 185 GB MoE does on this machine (Ryzen 9 7940HX, 2 x 16 GB DDR5-5200, Micron 1 TB NVMe, RTX 4070 Laptop 8 GB
over PCIe 4.0 x8, Windows 11, "Turbo" power mode, cooling pad), on colibri `dev` `0c4a751` built with its new
installer (`coli setup --backend vulkan`, MSYS2 UCRT64 gcc 16.2). The official FP8 checkpoint
(`Qwen/Qwen3.8-Flash-Next-FP8` rev `bcd9f01`, 131 shards), `tools/datapoint.py` persistent, 16 threads, the page
cache retained (`--no-evict`), every campaign from a copy of the same expert history, two campaigns per
configuration, the orders ABCCBA and ABBA. The full report is colibri issue **#1900**; raw data, scripts and
`analyze.py` in colibri `tmp/q38/` (`runs/`, `bench*.sh`, `telemetry.ps1`).

Decode, tok/s, median of 4 different prompts with 128 tokens each (both campaigns):

| colibri configuration | cap | decode | vs CPU |
|---|---:|---:|---:|
| CPU only | 16 | 0.75 · 0.74 | — |
| tier, dense chain on (the default on a discrete GPU), `COLI_VK_DENSE=0` or `=1` | 16 | 0.62-0.65 | −14 % |
| tier, chain off (`COLI_VK_CHAIN=0`) | 16 | **0.84 · 0.84** | +13 % |
| CPU only | 48 | 0.90 · 0.90 | — |
| tier, chain off | 48 | **1.01 · 1.01** | +12 % |

Prefill, tok/s = prompt tokens / time to the first token, prompts of 556 and 453 tokens, cap 48: CPU 2.39 · 2.40,
chain on 2.48 · 2.45, chain off **2.67 · 2.65**. A ~500-token prompt waits 3-4 minutes for its first token.

- **Disk-bound.** During the runs the NVMe reads ~0.6 GB/s at queue depth ~1.5 and the CPU is busy 19 % of the
  time; colibri's `iobench` on a shard of the same drive gives 2.12 GB/s (O_DIRECT, 1 thread, 4.69 MB = one FP8
  expert), 2.40 (16 threads) and 3.05 (8 threads, 19 MB). qwen38 reads experts through the buffered handle; its
  `COLI_TIMERS` banks cover about a third of a request.
- **The GPU barely works.** Chain off: 3.7 W mean (peak 14), 10 % utilization, 82 s of device compute in a 965 s
  campaign, 28 % of routed experts. VRAM is worth more as expert cache than as trunk compute: the chain puts the
  trunk's 773 matrices (4.17 GiB) in VRAM and leaves the tier 1.87 GiB (389 experts) against 6.00 (1248), and
  the CPU waits ~340 ms a forward for the device.
- **The cache is the lever that exists today.** Cap 16 → 48 (RSS 11.6 → 17.9 GB) raises the hit rate ~41 → ~60 %
  and decode by a fifth; the lowest available RAM fell to 2.2 GB (7.8 at cap 16).
- **Energy.** J/token, RAPL package + GPU board power (not the whole laptop): 91 CPU at cap 16, 85 chain on, 81.5
  chain off, 82 CPU at cap 48, 75 chain off at cap 48.
- **Heat was not a factor.** Tctl 63-71 °C on average, peak 85; GPU ≤ 44 °C, NVMe ≤ 45 °C; no thermal, power or
  current limiter in any HWiNFO sample, Windows' "% performance limit" at 100 throughout.
- **For Trochilus:** the 8 GB-GPU, 31 GB-RAM laptop can hold ~5 % of this model's experts in VRAM (1248 of 48 x 512) and ~9 % in RAM
  at cap 48 (48 a layer); each token routes 10 a layer, and the rest comes from the disk at whatever rate the read path gets. Here that path, not the drive, set
  the speed (0.6 GB/s used of 2-3 available). How to measure J/token on Windows without admin: LESSONS #295.

## The eviction told the future (2026-10-04)

R1 phase 3, the 8 GB machine: its Q8_0 decode is ~90% disk, and Belady's ceiling stood 1.8x (Q8_0 at 290 slots)
and 2.4x (Q4_K at 615) under the store's `heat` in misses a token. The premise (STATUS 10-03): a policy from the
routes, the next layer's router on this layer's state (92-95% of its experts within 12 candidates, question 13) as
an eviction hint. **Replayed only, nothing built in the engine** (build/routes_hint: `hint.py` and `lookup.py`, the
predictions written before every run in `predictions.txt`; the final numbers from `tools/evict_replay.py --see`,
whose `heat` and `opt` the ad hoc script reproduces to the hundredth). Traces build/evict (bench/prompts/code.txt,
321 tokens + 256 generated). Misses a decode token, the first 32 / all 256:

| the store's victim | Q8_0 at 290 | Q4_K at 615 |
|---|---|---|
| `heat` (the store today) | 28.66 / 33.75 | 3.84 / 5.96 |
| + the next call's units kept (a perfect next-layer hint) | 28.56 / 33.62 | 3.84 / 5.94 |
| + the next layer's router's top 8 / 12 / 16 kept (`pred_in`) | 28.59-28.62 / 33.64-33.65 | 3.84 / 5.95 |
| heat halved every 8 / 32 / 64 / 128 / never (16 today) | 29.47 / 34.69; 28.28 / 35.20-37.52 | 3.97 / 6.08; 4.16 / 6.25-7.49 |
| held to Belady's own slots a layer (its mean occupancy) | 28.28 / 31.88 | 4.19 / 5.93 |
| LIRS's key (max of recency and the last reuse gap) | 35.47 / 36.80 | 4.28 / 6.50 |
| the next use estimated as last + the mean gap | 38.59 / 42.35 | 9.50 / 9.71 |
| heat, ties by that estimate | 28.31 / 34.42 | 3.88 / 6.28 |
| **`see16`: Belady knowing the next token (16 calls), heat beyond** | 24.53 / 28.68 | 3.66 / 5.58 |
| `see64`: knowing 4 tokens | 20.31 / 23.09 | 2.81 / 4.96 |
| `see256`: knowing 16 tokens | 18.19 / 18.35 | 1.69 / 3.67 |
| `see1024`: knowing 64 tokens | | 1.09 / 2.48 |
| Belady (`opt`) | 18.19 / 18.35 | 1.09 / 2.45 |

- **The next layer's router as an eviction hint gives 0.3%**: a call evicts ~2 of 290 units, and the 6-8 the next
  call wants are hot already. Its coverage holds (top 8 / 12 / 16: 0.81-0.82 / 0.91-0.92 / 0.95-0.96 on these
  traces): the prediction is right, what it predicts is not what the store loses.
- **Belady's gap lives 4-16 tokens ahead**: the perfect next token takes 15% of Q8_0's misses (a third of the gap),
  4 tokens two thirds, 16 tokens all of it; Q4_K's larger store needs 64. heat's slots a layer are already
  Belady's within 1-2, and its 16-token half-life is the best period on all 256.
- **The text's own past sees 1-2 tokens** (`lookup.py`, the 7 traces): a suffix of >= 3 tokens matches an earlier
  position at 10-46% of the generated positions (code 18-46%, prose 10%), the next token right in about half; the
  matched position's routes share 6.4-6.8 of 8 a layer at t, 3.2-5.9 at t+2, and fall to a random earlier
  position's (1.6-3.3) by t+4 to t+8. Inside that horizon even the perfect next token gave 6-15%: not built.
- Predictions (build/routes_hint/predictions.txt): the next-layer ceiling 2-7%, out (0.3%); one token ahead half
  the gap, out (6-15%); the period 32-64 best, out (16); Belady's quotas 20-40% of the gap, out (6%, 0.5%); lookup
  matches 50-70%, out (10-46%); the past-only estimators within +-5% of heat, out (all worse, up to 1.6x).
- **Closed**: today's rule is the best one that needs no future, of every rule replayed (the tool's nine and
  the six here). What is left of the 8 GB
  machine's decode is in its bytes (a unit's size, the disk's request) and its slots, not in the choice of victim.
  `tools/evict_replay.py --see H` stays as the instrument for R2-R3's stores (question 89: how far ahead a VRAM
  tier's policy must see).

## The routes as time (2026-10-04)

R1 phase 3, the 8 GB machine, after the victim closed: **its decode's bytes and time, not its misses**. Its Q4_K decode
is about half disk (MEASUREMENTS §q128's race: 34.8 ms of compute a token, the rest the disk at 500 MB/s), Q8_0 92%
disk (33.3 ms of compute). The engine reads a layer's misses on the calling thread after its router, then computes
its experts (src/models/olmoe.c olmoe_refresh_experts); nothing is read ahead in decode. **Replayed only, nothing
built** (`tools/evict_replay.py --prefetch`, replay_time: `heat`'s store and one disk queue in time; build/evict's
traces, code.txt 321 + 256 tokens, Q4_K at 615 slots, Q8_0 at 290; a unit 3.54 MB = 7.08 ms, 6.68 MB = 13.37 ms; a
layer's compute 1/16 of the token's 95%, a quarter of it before the router, 5% after the last layer, the output
head; predictions written before each run in build/routes_time/predictions.txt). The read ahead: at layer L's
router, layer L+1's router on L's FFN input (`pred_in`, top k) names units; the ones not resident are queued, each
taking a slot, and read while the disk is free in chunks; a demand read waits only for the chunk in flight; at
L+1's router a queued unit it names finishes first, one it does not name is dropped (a half-read one frees its
slot). The arrival order (`--arrive 1`): a layer's experts computed as their bytes arrive, the resident ones first
while a miss is read.

| a decode token (gain against today's serial) | Q4_K ms | gain | MiB | Q8_0 ms | gain | MiB |
|---|---|---|---|---|---|---|
| today: the misses, then the experts | 76.96 | 1 | 20.1 | 484.46 | 1 | 215.1 |
| **the experts as their bytes arrive** (no prediction) | **71.00** | **1.084** | 20.1 | **466.94** | **1.038** | 215.1 |
| `pred_in` top 4 / 8 / 12 ahead, chunks of 512 KiB | 74.98 / 72.22 / 71.75 | 1.026 / 1.066 / 1.073 | 20.4 / 22.9 / 27.5 | 474.21 / 469.99 / 473.49 | 1.022 / 1.031 / 1.023 | 216-224 |
| **both: arrival order + `pred_in` top 8** | **68.15** | **1.129** | 22.4 | 464.35 | 1.043 | 219.0 |
| the true next call ahead (any next-layer predictor's ceiling) | 68.62 | 1.122 | 20.2 | 462.89 | 1.047 | 216.7 |
| arrival order + the true next call | 65.30 | 1.179 | 20.2 | 459.97 | 1.053 | 216.7 |
| full overlap (max of compute and disk, a bound) | 42.2 | 1.82 | | 451.2 | 1.074 | |

Without the head's 5% (every ms of compute inside a layer) the same table gives 1-2% more to every read ahead.
Sensitivity (no head): chunks of 2 MiB lose 1.5-5.6% against 512 KiB on Q4_K at k 8-16 (a chunk is 4.2 ms, the window
2.2); whole units lose: Q4_K 0.997 / 0.959 / 0.934 at k 8 / 12 / 16, Q8_0 0.965 / 0.944 / 0.933; `pred_out` (known
only after the layer: the window is the next layer's attention) at best 1.016 (Q4_K k 8) and 1.004 (Q8_0).

- **The window is the limit, not the prediction**: a Q4_K unit reads in 7.1 ms, a layer computes in 2.2; even the true
  next call read ahead hides one layer's compute where the next layer misses: 1.12x of a 1.82x bound. The router's
  top 8 already takes half of that ceiling (1.066), at +14% bytes; top 12 +37% bytes for 1%.
- **A layer's own resident experts are the larger window**: computing the 7 resident experts while the eighth is read
  gives 1.084 on Q4_K and 1.038 on Q8_0 with no prediction and no byte more, more than the router's read ahead
  alone; both together 1.129 (Q4_K), the oracle's level.
- **A wrong read ahead costs its victim more than its time**: chunks of 512 KiB waste 2-7 MiB a token on Q4_K, but
  a wrong unit read whole stays and has evicted a resident one; with a shorter window (the head's 5% moved out of
  the layers) fewer wrong units finish and k 12 gains more (1.054 -> 1.073).
- Predictions (build/routes_time/predictions.txt): the serial model to the hundredth, in; the oracle Q4_K +10-15%,
  Q8_0 +5-7%, in; `pred_in` Q4_K -3..+5% best k 4, **out** (better: 1.059-1.066 at k 8); 2 MiB chunks, in; whole
  units Q4_K -10..-30%, out (milder, -4%); bytes Q4_K +10-30% at k 12, out (+37-47%); the arrival order alone
  Q4_K +7-10%, Q8_0 +3-5%, in; with `pred_in` k 8 Q4_K +12-16%, in, Q8_0 +5-6%, out (+4.3%); with the oracle
  Q4_K +18-22% and Q8_0 1.06-1.07, out (just under: 1.179, 1.053).
- **Next (question 90)**: the arrival order built first (the same bytes, no prediction: the engine's expert stages
  over the resident experts while an I/O thread reads the misses, then over the late ones), raced on the 8 GB
  machine against the model's 1.084 / 1.038; the router's read ahead after, on top, only if the race keeps the
  model's word.

## The arrival order built (question 90, 2026-10-05)

R1 phase 3, the 8 GB machine. §The routes as time found a layer's own resident experts the larger window: computed
while its miss is read, 1.084 (Q4_K) and 1.038 (Q8_0) in the model, no byte more. **Built**: at the router the store
queues a layer's misses to its I/O thread (`tr_experts_acquire_async`: the same hotness, victims, runs and bytes as
`tr_experts_acquire_counts`, one wake, back once the thread has taken the first), the engine groups the used experts
present first, late after (`arrival_layout`: each row moved to its expert's group, keeping its place among that
expert's rows), computes the present groups' stages (gate and up, the activation, the down), then takes the late
units in waves (`tr_experts_acquire_take`: the first waited for, the next ones while already landed) and runs each
wave's stages; the combine unchanged, in slot order: the bits are the definition's (test_stream arrival: the
resident's logits in every arm, Q4_K's roads and F32's). A pass's read ahead of the next layer waits until the late
units are in (its own wait for its first unit would wait behind them). `--ab arrive` B: today's calling thread;
`--ab waves` B: one wave after the last lands.

**The references** (read 10-05, ORIGINS row 7): colibri reads a miss inside its expert loop, the experts after it
waiting (`moe`, olmoe.c:925-948); ds4 splits exactly this way on its GPU, resident experts in a masked pass while its
pread pool reads, the missing after, the down and sum once in slot order, but only from three misses
(`split_worthwhile`, ds4_metal.m:13095); llama.cpp's mmap stalls only the faulting thread (no barrier between experts
in `mul_mat_id`, ggml-cpu.c:1663-1729).

**The variants replayed first** (`evict_replay.py --arrive 0,1,2,3 --wave-ms`, predictions in
build/routes_time/predictions.txt, every one in its range):

| variant (head 0.05, no read ahead) | Q4_K | Q8_0 | with 0.05 ms a wave |
|---|---|---|---|
| a1: each late unit as it lands | 1.084 | 1.038 | 1.081 / 1.036 |
| a2: two waves (the resident, then every late one) | 1.079 | 1.029 | 1.076 / 1.028 |
| a3: a2 from three late units (ds4's threshold) | 1.004 | 1.009 | |
| a1 + the router's top 8 read ahead | 1.129 | 1.043 | 1.127 / 1.043 |

ds4's threshold costs nearly all of it on a CPU (a wave is three pool regions, not a GPU submit): ours splits at the
first miss, and takes each wave as it lands (a1).

**Counts** (`MACHINES_COUNTS=1`, build/q90/counts and counts_var): the decode's disk a token 16.0 MiB (Q4_K) and
189.5 (Q8_0) against the old binary's 15.9 and 189.6; the old binary against itself 15.9-16.3 and 189.7: a pass's
read ahead drops by the disk's timing (#285), so the counts move within that spread on both models (LESSONS #316).

**The race** (tools/machines.sh, weak against weakold = fa9a7e4, 5 rounds, build/q90/race; predictions in
build/q90/predictions.txt):

| the 8 GB machine (weak), code.txt's 321 tokens, then 64 | Q4_K before | after | gain | Q8_0 before | after | gain |
|---|---|---|---|---|---|---|
| decode tok/s | 14.91 | **15.59** | **1.046** | 2.33 | **2.40** | **1.030** |
| prompt tok/s | 45.29 | 45.32 | 1.001 | 24.51 | 24.51 | 1.000 |
| disk MiB a decode token | 15.8 | 15.7 | | 189.7 | 189.6 | |

Medians of 3 (stopped steady after round 3 of 5, every arm within 0.4-0.9%), the same tokens in all 10 runs.

- The predictions: Q4_K 1.05-1.08, **out** (1.046, just under); Q8_0 1.02-1.04, in; the prompt +-2%, in. The engine
  kept 55% of the model's Q4_K gain (0.046 of 0.084) and 79% of Q8_0's (0.030 of 0.038).
- **The load was not low at the race's end** (the Docker VM's bursts, up to 5.8 logical processors between rounds):
  the race's own verdict is *structure, not results*; its arms rotate round by round, so the ratio stands as
  structure until the confirmation race (owed, with the machines whose experts fit in RAM, avg and pc, predicted
  unchanged, and `--ab waves` in one process).
- **The rounds drifted** (build/q90/race/q4k/ab_modes.txt): the profiled runs, first, 16.03 against 14.73 tok/s
  (1.088); round 0 15.88 against 14.56 (1.091); rounds 1-3, as the Docker VM's bursts rose, 15.55-15.68 against
  14.80-14.94 (1.046). Q8_0 steady (1.026-1.030). The quiet machine's ratio is the model's (1.084): the gain may
  depend on the background load (the arrival order hands work between three threads a layer with a miss), to
  measure in one process (`--ab arrive`, no drift between arms) and under `tools/busy_machine.sh`.
- **The decode's genome, the profiled runs** (Q4_K, ms a token): disk waits (weight_read) 33.43 -> 28.75, the
  rest 34.42 -> 33.60; the disk's own time for its 15.8 MiB at 500 MB/s is 33.1 ms. The 4.7 ms hidden are the
  present experts' stages (gate, up, down 1.31 ms a layer, 7 of 8 of them) in the ~4 layers a token with a
  miss: everything the arrival order can hide, hidden. The model's layer counted 1.55 ms after the router
  (0.75 of 2.07), the engine's experts take 1.33: the model's 1.084 holds 17% more compute than there is.
- Where Q4_K's other half went is the next measurement, one piece at a time: the waves' extra pool regions (three a
  wave), the I/O thread's wake once a layer with a miss (`queue_wake`), and the model's even split of a layer's
  compute among its units.

**The confirmation** (10-05 night, Chrome closed, the load 1.6-2.9 logical processors; build/q90/race2, race3,
ab_*; predictions in build/q90/predictions.txt, every one of race3's in):

| the 8 GB machine | Q4_K before | after | gain | Q8_0 before | after | gain |
|---|---|---|---|---|---|---|
| race3 decode tok/s (6 rounds, median of 3; each round) | 14.98 | **16.08** | **1.073** (1.061-1.073) | 2.33 | **2.40** | **1.030** (each 1.030) |
| race3 prompt tok/s | 45.47 | 45.48 | 1.000 | 24.51 | 24.56 | 1.002 |
| race2 decode (the load 2-3.5) | 14.55 | 15.74 | 1.082 | 2.33 | 2.39 | 1.026 |
| one process, `--ab arrive`, B (the calling thread) / A | | | 1.110 +- 0.0005 (1.109 before) | | | 0.977 +- 0.0004 (0.979) |
| one process, `--ab waves`, B (two waves) / A | | | 1.036 +- 0.001 | | | 0.961 +- 0.003 |

- **Kept** (the default each late unit as it lands): three races faster on both models, the same bytes and bits.
- **The two instruments disagree on Q8_0** (LESSONS #318): in one process the arrival arm's every compute zone
  slows (qkv_proj +8% Q4_K, +22% Q8_0), and Q8_0's disk waits grow 6.8 ms a token; the races' arms run whole
  processes of one mode. Two waves beat each unit as it lands on Q8_0 by 4-5.5% in one process, lose on Q4_K
  by 3.5% (#319; the model 0.5-0.9%): a wave's handoff costs ~1 ms there, its own compute 0.6.
- **The binary's own share, not found**: on the machines whose experts fit in RAM the new binary's Q4_K read
  1.023 (pc) and 1.063 (avg) where its new code never runs, Q8_0 0.994 and 1.002 (race2); this PC's A/A (pc
  against aa, the same binary) spread 7-70% (another window's test gates beside it, #321): inconclusive.
- Next, one handoff at a time (#318): each take's wake from its read's end, the I/O thread's gap between
  requests, waves a token; then a wait that spins before it blocks, and the waves by the numbers.

**The handoffs measured, whole processes** (question 91, 10-05 05:36-06:12, the load ~2.5 processors; one
binary, `TR_ARRIVE` read at session start, `tools/ab_env.sh`, 5 rounds, code-edit.txt's 411 tokens then 200
(Q4_K) or 80 (Q8_0); build/q91/env_*; predictions in build/q90/predictions.txt, every one in):

| mode | Q4_K decode tok/s | against sync | Q8_0 | against sync |
|---|---|---|---|---|
| sync (`TR_ARRIVE=0`, the calling thread) | 12.08 | 1 | 1.99 | 1 |
| two waves (`TR_ARRIVE=2`) | 12.96 | 1.073 | 2.02 | 1.015 |
| **each late unit as it lands** (the default) | **13.04** | **1.079** | **2.04** | **1.025** |
| the same again (A/A) | 13.03 | 1.079 | 2.04 | 1.025 |

- Spreads 0.0-1.2%; the prompt the same in every mode; the misses the same to the unit (Q4_K) and within 4 of
  3770 (Q8_0). Each unit as it lands is the best mode on both models: the default stays.
- **The handoffs a run** (the store's new counters, ab_modes.txt): Q4_K 1444 blocked takes woken in 16 ms in
  all (11 us a take), the I/O thread woken 18 ms, between its reads 1.3 ms; Q8_0 2933 takes, 38 ms (13 us),
  19 ms, 7 ms; the emulated disk's sleeps woke 31-60 ms late a run in every mode, sync's too. Together 0.16%
  of a Q4_K run: the arrival order's machinery is cheap.
- **So the in-process A/B's verdict on this switch is its own** (LESSONS #318): the arrival arm slows only when
  its passes alternate with the calling thread's; whole processes of one binary agree with the races. An
  in-process A/B is not used for a switch that changes which threads run.

## The router's read ahead (question 92, 2026-10-05)

R1 phase 3, the 8 GB machine, on top of the arrival order. **Built**: at layer L's router, once L's misses are
queued, layer L+1's router on L's FFN input (`pred_in`) names its top k a token (`olmoe_router_ahead`, passes of
at most 8 tokens naming at most half the experts); the absent ones are queued in that order behind L's late units
(`tr_experts_prefetch_ids`), each read alone in pieces of 512 KiB. At L+1's router the queued guesses it does not
name are dropped unread and the one in flight stops at its next piece, absent at once (`tr_experts_prefetch_cancel`,
`stats.stopped`); a guess it names still in flight is a late unit of the arrival order (`stats.ahead_late`, taken
first: its read was queued first), and **from then on is read a part a request** (`experts_job.named`). k is the
next layer's units in 27 MiB (`OLMOE_AHEAD_BYTES`: 8 for Q4_K, 4 for Q8_0); `TR_AHEAD_K`, `TR_AHEAD_CHUNK_KIB`,
`TR_AHEAD_DRY` (the guess made, nothing queued) and `--ab ahead` for measurements. The same bits (test_stream ahead:
the resident's logits on, off and by default).

**The references** (read 10-05, ORIGINS row 7): colibri's PILOT (off by default, `PILOT=1`) reads ahead in decode
passes of at most 8 tokens: layer L+1's router on the state after L's attention under **L+1's own norm**
("IMPROVEMENT B"), the scores **averaged with the last token's** (SMOOTH 0.3), the top k (`WIDE` 1, so
`CONF_LIMIT` never cuts), queued to a worker that reads whole experts, the stale queue flushed every token
(ref/colibri/c/olmoe.c:10-18, 976-989, 1110-1124); ds4 and llama.cpp read nothing ahead in decode.

**Replayed first** (predictions in build/q92/predictions.txt; traces build/q92, code.txt 321 + 256 or + 64 tokens,
recorded in Docker: byte for byte the native ones):

| what was replayed (Q4_K 615 slots, arrival order, head 0.05) | result |
|---|---|
| top-8 recall of layer L+1's choice: ours / colibri's norm / colibri's default (norm + SMOOTH 0.3) | 81.8% / 81.8% / **77.5%** |
| k 8 against the arrival order alone: whole units / pieces of 2304, 1152 (a part), 512, 256 KiB | **0.958** / 1.001, 1.031, **1.042**, 1.042 |
| Q8_0 (290 slots) k 4-8: whole / a part / 512 KiB | 0.932-0.990 / 0.986-1.002 / 1.006 |
| layer L+2 from L's FFN input: recall at top 4 / 8 / 12 / 16 | 46.5% / **74.4%** / 85.9% / 91.2% |
| two layers ahead: the true calls / L+1 top 8 + L+2's top 1-12 | **1.140** (L+1 alone 1.087) / 1.040-1.047 |
| no guess while the disk is busy past G ms (G 0-7) | Q4_K 1.034 -> 1.025-1.028, Q8_0 1.002 -> 1.004 (issued 28.9 -> 2.3 a token) |

- **Pieces, not units**: a wrong unit read whole holds the one disk when the next layer's demand needs it (#324).
- **One layer ahead, ours**: colibri's norm predicts as ours; its average loses 4.3 points (#323). Two layers
  ahead are predicted at 74% and still do not pay: the disk is the scarce resource, and the doubled window's
  wrong bytes cost what it gains (#325).
- **The model and the engine agree on the same tokens** (code.txt + 64, MACHINES_COUNTS=1, build/q92/counts_*;
  #329): a token's guesses stopped 2.94 (model) / 2.95 (engine), dropped unread 0.78 / 0.78, used 1.95 / 2.08,
  the disk 15.4 -> 18.0 / 15.7 -> 18.8 MiB; Q8_0 189.6 -> 195.7 MiB. The model's gain there: 1.034 and 1.002.

**The races** (one binary, `tools/ab_env.sh`, whole processes, the 8 GB machine emulated, code-edit.txt's 411
tokens then 200 (Q4_K) or 80 (Q8_0), 5 rounds, medians, the load 1.2-2.9 processors; build/q92/env_*, dry_*,
prom_*):

| decode tok/s, against off | Q4_K | Q8_0 |
|---|---|---|
| read in pieces to the end, k 8 (the A/A 1.001 / 1.000) | 1.024 (13.33 / 13.02) | **0.991** (2.02 / 2.04) |
| k 12 / a part a request | 1.006 / 1.018 | |
| the guess alone (`TR_AHEAD_DRY=1`) | 1.002 | 1.000 |
| **a named guess read a part a request**, k 8 | **1.031** (13.37 / 12.97) | 1.000 |
| the same, k 4 / k 10 | | **1.005** (2.04 / 2.03) / k 10 on Q4_K 1.023 |

The prompt the same in every arm (+-0.3%), every token the same. **machines.sh**, the default (k by bytes) against
`weakna` (the same binary, `TR_AHEAD_K=0`), code.txt's 321 tokens then 64, median of 3, stopped steady
(build/q92/race_m): the decode **Q4_K 15.92 -> 16.26 tok/s (1.021), Q8_0 2.40 -> 2.41 (1.004)**, the prompt the
same, the disk a token 15.8 -> 18.8 and 189.6 -> 190.8 MiB, the same tokens in all 10 runs of each model.

- **The guess is free; its pieces were not** (#330): the guess alone costs 0.08-0.14 ms a token (the router's
  zone), nothing in the totals; reading on in pieces after the router named a guess tripled Q8_0's requests
  (9 033 -> 28 200 a run) and the emulated disk's late wakes (13 -> 251 ms), its whole loss. Read a part a request
  once named: 7 641 requests for the same bytes.
- **The best k is a byte budget** (#331): 8 x 3.4 MiB (Q4_K) and 4 x 6.4 MiB (Q8_0), the same 27 MiB.
- **Where the time goes** (Q4_K, ms a token, off -> k 8 promoted): disk waits 41.51 -> 38.68, the other zones 35.57 -> 35.96
  (qkv_proj +0.14, gate_up +0.07, the guess +0.08): the I/O thread's extra reads beside the four cores.
- The predictions (build/q92/predictions.txt): decode Q4_K +2.5..+4.5%, **in** (+3.1%); Q8_0 -1..+1.5%, in (0 and
  +0.5%); the prompt +-1%, in; the disk a token Q4_K 17.0-18.5 MiB, **out** (18.8: the first 64 tokens' store, #329),
  Q8_0 191-196, in (195.7); the demand misses -45..-60%, **out** (-36%: the same model on 64 tokens says -43%).
- **Owed**: the race against the old binary (Smart App Control held 12147b2's builds all session, #332): the
  resident machines unchanged, the code's layout; the average machine's Q8_0 (2 GB/s: the guess on a faster disk).

## A draft from the bytes the machine holds (question 94, 2026-10-05)

Phase 1 of build/prompt-draft.md, nothing built: a draft that reads only what the machine already holds (the
experts in the store on a disk-bound machine, those a pass already streams on a RAM-bound one), verified by the
exact model k rows a pass. Counts and models only (no stopwatch); data and predictions in build/q94.

**The bound first** (`tools/spec_replay.py`: the store's `heat` replayed with the decode in passes of r rows, every
row kept, a perfect draft; model, not measured): the disk units a token against the one-row decode, on five Q8_0 traces (the
race's code prompt, English and Italian prose, C, Python):

| slots (of 1024) | rows 2 | rows 4 | rows 8 |
|---|---|---|---|
| 128 | 0.88-0.90 | 0.76-0.78 | 0.61-0.64 |
| 200 | 0.88-0.93 | 0.78-0.82 | 0.63-0.70 |
| **290** (the 8 GB machine's Q8_0) | 0.90-0.97 | **0.80-0.88** | 0.67-0.76 |
| 400 | 0.92-1.00 | 0.83-0.94 | 0.71-0.82 |
| 610 (the 8 GB machine's Q4_K share) | 0.90-1.07 | 0.85-1.08 | 0.82-1.00 |
| the RAM's units (question 54's union) | 0.77 | 0.57 | 0.40 |

- **The store already holds what the union saves** (#335): from RAM a pass of 4 rows reads 0.57 of four tokens'
  experts, from disk 0.80-0.88 of their misses at 290 slots, and no fewer at 610: consecutive tokens' shared
  experts are hits already. A perfect draft's pass of 8 rows bounds the 8 GB machine's Q8_0 at 1.21-1.31x after the
  draft's compute (build/q94/model.txt).

**The agreement, measured** (`tools/resident_probe.c`, `make resident-probe`, container): the store's set before
every row from `spec_replay --dump-resident` on the same run's route trace; at each of 256 greedy rows after the
prompt, each draft's chain from the exact state for up to 8 tokens, then the row exactly. Checked: the probe's exact
routing of every row equals the trace's (`spec_replay --routes`: 0 sets apart in all 9 runs, 48 against another
model's trace), its exact token pass 1's. `sub`: the router's best 8 among the resident; `skip`: the exact routing,
the absent experts' weights 0; `norm`: skip, the present weights scaled to the 8's sum.

| model, slots | text | resident share of a row's routing | first token sub / skip / norm | mean chain of 8, skip / norm | norm where the exact margin < 1 / >= 1 nat |
|---|---|---|---|---|---|
| Q8_0, 290 | code | 0.747 | 0.887 / 0.898 / **0.906** | 5.86 / 6.12 | 0.58 / 0.94 |
| Q8_0, 290 | English prose | 0.556 | 0.594 / 0.578 / 0.500 | 1.23 / 0.88 | 0.33 / 0.57 |
| Q8_0, 290 | Italian | 0.663 | 0.699 / 0.691 / 0.621 | 1.86 / 1.43 | 0.36 / 0.73 |
| Q8_0, 128 | code | 0.509 | - / 0.805 / 0.785 | 4.41 / 3.95 | 0.42 / 0.82 |
| Q8_0, 128 | English prose | 0.314 | - / 0.297 / 0.160 | 0.39 / 0.20 | 0.12 / 0.18 |
| Q8_0, 128 | Italian | 0.441 | - / 0.543 / 0.395 | 1.10 / 0.58 | 0.07 / 0.53 |
| Q4_K, 610 | code | 0.953 | **0.984** / 0.984 / 0.980 | **7.29** / 7.24 | 0.86 / 0.99 |
| Q4_K, 610 | English prose (129 rows) | 0.826 | 0.845 / 0.853 / 0.791 | 3.64 / 3.35 | 0.63 / 0.89 |
| Q4_K, 610 | Italian | 0.903 | 0.867 / 0.879 / 0.844 | 3.78 / 3.31 | 0.62 / 0.94 |

- **The text decides more than the slots** (#337): at the same 290 slots the store holds 75% of a code row's routing
  and 56% of an English one's; code's chains run 6 of 8, prose's 1. `sub` is not run at 128 slots (a layer always
  holds fewer than 8); rescaling the present weights (`norm`) helps code and hurts prose. Prediction: 70-85% at 290
  slots, 92-97% at 610: code above, prose below.

**The verify passes with these chains** (`spec_replay --chain`; the store halved every 16 kept tokens and heated by
every row named, the best of four policies on code at k 7: 1.20 against 1.23-1.33): the disk a token at k 1 and k 7.

| model, slots | code | English prose | Italian |
|---|---|---|---|
| Q8_0, 290 | 1.01-1.02 / 1.20-1.30 | 1.10-1.13 / 2.1-2.5 | 1.06-1.10 / 1.9-2.3 |
| Q8_0, 128 | 1.02 / 1.48-1.50 | 1.30-1.47 / 3.3-3.8 | 1.08-1.24 / 2.3-3.0 |
| Q4_K, 610 | 0.98 / 0.99 (7.3 tokens a pass) | 0.97-1.00 / 1.24-1.37 | 1.00-1.01 / 1.24-1.26 |

- **A real chain's pass reads more than one row a token** (#336): a pass of r rows names union(r) x 128 units, 298
  at 4 rows and 427 at 8 against 290 slots; the rows a pass rejects are read and evicted before the next pass reads
  them again. The perfect draft's 0.80-0.88 becomes 1.01-2.5.

**The model per machine** (build/q94/model_chains.txt; model, not measured, on the measured chains): the draft's compute is the
dense share plus the resident share of the experts of weakr's token (23.6 ms Q8_0, 36 ms Q4_K) or of a 21 GB/s RAM's;
the verify pass its disk units at 400 ms a Q8_0 token and 39 ms of a Q4_K token's waits, Q4_K's extra rows at 0.3-0.8
of a token; on the RAM-bound machines the draft reads the dense bytes and its experts, the pass its union. The best
k, always 1:

| machine | code | English prose | Italian |
|---|---|---|---|
| weak Q8_0, 290 slots (emulated RAM / 21 GB/s) | 0.98 / 0.94 | 0.90 / 0.87 | 0.94 / 0.90 |
| weak Q4_K, 610 slots (rows at 0.3 / 0.8 of a token) | 0.94 / 0.84 | 0.93 / 0.83 | 0.92 / 0.82 |
| avg and pc, the union draft (Q8_0 / Q4_K) | 0.86 / 0.86 | 0.74 / 0.81 | 0.78 / 0.82 |
| larger than RAM (128 slots, Q8_0's shape) | 0.98 | 0.77 | 0.92 |

- **Closed: a draft from the bytes the machine holds loses on every machine** (the rule: build at >= 1.2x, never
  below 1.0). Disk-bound: the store already gives the union's saving and a chain's rejected rows cost more than its
  accepted ones save. RAM-bound: the draft re-reads the dense 37-40% of a token for each token it drafts (question 79
  closed by the same model). Weak Q4_K: the draft costs a token's compute.
- **What survives** (question 95): on weak Q4_K the cost is the draft's compute alone, and the resident chains on
  code are long (7.3 of 8): a draft that reads nothing, with these chains, gives code 1.07-1.40x by the model, prose
  1.03-1.23x, Italian 1.02-1.18x (rows at 0.8-0.3 of a token): the engine's lookup and question 81's gate cost
  nothing but agree less, and this machine's row price is not measured.

**The fast inverse square root's lens** (Marcello, 10-05: a number's bits hold a coarse copy of another quantity;
guess cheaply, then correct, the correction proven): each candidate's share first, on weak Q4_K's profile
(build/q92/env_q4k.log, 75 ms a token) and the GGUFs' bytes.
- (b) **The output head by a bound** (`tools/head_bound.py`, question 73): 104.4 MiB of a Q8_0 token's 1284.6
  (8.1%), 55.3 of a Q4_K one's 723.5 (7.6%; the head is Q4_K, predicted Q6_K at 11%); 2.65 ms of weak Q4_K's 75.
  First closed here by its share (#340: the 1.2x bar was the draft's, not the lens's); then measured. Every row's
  upper bound from its scales and its codes' top b bits (the low bits at their worst against h), exact only the rows
  whose bound reaches the best exact logit; h solved from the engine's logits (residual <= 1e-5, every argmax kept),
  every bound checked against its own row; the three prompts' every position (321, 583, 941):

| head | top bits kept | rows left: mean (median, p99) code / English / Italian | the head's bytes read | a RAM-bound token, by bytes |
|---|---|---|---|---|
| Q8_0 | 3 of 8 | 26% / 67% / 79% | 0.56-0.88 | - |
| Q8_0 | **4 of 8** | **0.2% (3, 2372) / 0.6% (69, 3111) / 3.8% (781, 12590)** | **0.53-0.55** | **1.039x** |
| Q8_0 | 5 of 8 | 0.01% / 0.02% / 0.07% | 0.65 | 1.029x |
| Q4_K | 2 of 4 | 85% / 99% / 99% | 0.93-1.0 | - |
| Q4_K | **3 of 4** | **0.2% (3, 3253) / 0.6% (82, 2741) / 3.7% (764, 12882)** | **0.78** | **1.017x** |

  The high nibble decides almost every row: a Q8_0 head read at 0.53, a RAM-bound Q8_0 token 1.039x (model, not
  measured), Q4_K 1.017x; greedy and the verify passes only (sampling needs every logit). Under question 94's bar,
  above the pieces built at 1.005-1.03: question 73 stays open for the build and the race on avg and pc.
- (b) Attention positions by a bound: closed 2026-09-24 (§Skipping cached positions exactly: the oracle skips
  0.2-0.37% of the bytes; QK-norm keeps OLMoE's attention flat). Not reopened.
- (a) A draft by bit tricks: a draft's cost is its matmuls; the nonlinearities and the router are under 1% of a token
  (expert_act 0.17 ms, router 0.11): nothing to take.
- (c) RMSNorm's 1/sqrt from a seed and Newton: the norms 0.127 ms of 75 (0.17%). Closed by its share.

**The references** (`docs/ORIGINS.md` row 8): none drafts with the main MoE's own resident experts. colibri's DSpark
head drafts with 128 experts of its own (3 a token, a fifth of a main token) and verifies the drafts' union once
(`v4_moe_batch_union`); ds4's DSpark and MTP heads are trained, not byte-identical. The literature (SS-MoE, S2-MoE,
DraftExpert 1.45x with trained resident draft experts at 84-87% acceptance, AcceptMoE, cache-conditional routing,
the last two not exact): the GPU's verify cheap, the bus the limit; none counts the store's own reuse of consecutive
tokens.

## The head's argmax by a bound (question 73, 2026-10-05)

**Asked** (build/prompt-head.md, phase 1: studied to the bottom and measured, nothing in the engine): at every greedy
token the engine scores the 50304 rows of the output head and keeps the argmax. A row's score is a sum over blocks of
scale x codes . h; read each code's top bits only and take the low bits at their worst against h: every row gets a
provable upper bound, and only the rows whose bound reaches the best exact score can win. Predictions first
(build/q73/predictions.txt, their outcomes beside them); everything in build/q73.

**The references** (ORIGINS row 15, read 10-05): all four compute every row. colibri one `matmul_qt` over the
vocabulary (h int8 once, `matmul_q_idot`), a serial argmax (`c/sample.h` 20-24); ds4 `matvec_q8_0` over every row
(h Q8_0), a serial 8x-unrolled argmax; its one pruned head is the Qwen4 MTP draft's (an id prefix or a ranked subset,
`DS4_QWEN4_MTP_DRAFT_VOCAB`: not exact, its verify rows keep the whole head); llama.cpp and ik_llama.cpp the full
`output` matmul after `ggml_get_rows` of the outputs (h Q8_0 / Q8_K), the head repacked like any weight (`repack.cpp`,
`-rtr`), a serial greedy scan. **None prunes rows for an exact argmax.** The literature (known, not re-read): exact
maximum inner product search with bounds, LEMP (Teflioudi et al. 2015) and FEXIPRO (Li et al. 2017: an SVD rotation,
Cauchy-Schwarz on the tail, integer-scaled bounds); the threshold algorithm (Fagin et al. 2001); top-k from bit slices,
most significant first (bit-sliced index arithmetic, Rinfret, O'Neil and O'Neil 2001): the closest to ours. Not exact,
not rivals: SVD-softmax (Shim et al. 2017), learning to screen (L2S, Chen et al. 2019).

**The share, measured** (two profiled decode runs an arm, `MACHINES_COUNTS=1`, build/q73/prof1-2): the head reads at
this PC's RAM ceiling on avg and pc; on weak its Q4_K is the cores' (the exact integers on AVX2), its Q8_0 a token the
disk's.

| machine | Q8_0 lm_head (GB/s), share of a token | Q4_K lm_head (GB/s), share |
|---|---|---|
| weak (4, AVX2) | 2.27-2.31 ms (47-48), 0.5-0.6% of 415 ms | 2.56-2.59 ms (22-23), 4.2% of 61 ms |
| avg (8) | 1.92-2.03 ms (54-57), 7.8% | 1.02-1.06 ms (55-57), 7.2% |
| pc (16) | 1.92-1.95 ms (56-57), 7.8-7.9% | 1.02-1.03 ms (56-57), 7.4-7.5% |

**The rows left, on more data** (`tools/head_bound.py --study`, then `--report`: 13 texts a model in the container,
build/q73/data.sh: the three prompts, 6 new-code files, 2 chats, each with the engine's own 256 greedy tokens after
it, and 2 long contexts of 3800 tokens + 256 (code from this repo, English prose); 13,950 positions a model, h solved
from the engine's logits, every bound checked against its own row). h as the bound pass sees it: two int8 digits a
block of 32 (a 16-bit code; one int8 digit leaves 2.0-2.1x the rows on Q8_0, 3.1-3.3x on Q4_K, its rounding error
counted at the codes' largest), the mean with the threshold the best exact score of the four best bounds' rows (1.02-1.1x
the least possible, whose median, p99 and max the table gives; the best bound's row alone 1.2-2.7x; the previous token's row 38000-50000 rows: never):

| positions | Q8_0, top 4 of 8 bits: mean (median, p99, max) | Q4_K, top 3 of 4 bits |
|---|---|---|
| greedy (the engine's own tokens) | 0.28% (4, 3281, 6497) | 0.36% (4, 3554, 21106) |
| chat, greedy | 0.85% (27, 6492, 15043) | 0.69% (16, 5120, 7197) |
| the prompts' positions | 1.7% (48, 9574, 19874) | 1.6% (49, 8626, 22527) |
| long context, positions 0-3800 | 1.6-2.5% (p99 6900-9900) | 1.6-2.3% (p99 6300-8900) |
| long context, greedy at 3800+ | 1.3% (34, 6824, 12830) | 0.9% (21, 4609, 6549) |

The head's bytes: Q8_0 top 4 bits 0.531-0.541 (5 bits 0.647: 5-27 rows left; 3 bits 0.58-0.86), Q4_K top 3 bits
0.779-0.783 (2 bits 0.94-0.99). A cascade (Q8_0's 4 bits for every row, the 5th for the rows left, then exact) saves
0.001-0.008 of the head: closed by its share. FEXIPRO's bound (the head's SVD, 16 / 64 / 256 coordinates kept as f16,
Cauchy-Schwarz on the rest) leaves 99.4% of the rows: h's energy is not in the head's top directions. Closed.

**The time, piece by piece** (`tests/bench_head_bound.c`, `sh tools/bench_native.sh bench_head_bound`: a head of the
real shape, random codes, two rows aligned with h, one a copy of the other, for the check; the bounds against the
engine's own value of every row, the kernels against the bound's formula in double, the whole argmax against the
engine's (the lowest row on a tie); every timed run reads its own of 4 copies of the heads, #344; medians of 21 in a
quiet window, 1.3-1.4 logical processors busy). The plane kernels: Q8_0's high nibbles (1152 B a row), Q4_K's top 3
bits as a two-bit and a one-bit plane (896 B a row), against h's two digits by `maddubs` (AVX2) or `vpdpbusd`
(AVX-512 VNNI); the margin of the float sums (2^-16 of the terms' magnitudes, above both the engine's rounding and the
bound's) folded into each block's prepared terms at no cost. In cache, one core: AVX2 Q8_0 52 ns a row (22 GB/s of
plane), Q4_K 89 (10 GB/s; 164 before #341); AVX-512 50.5 and 72 ns. The whole argmax (the bounds and each worker's
largest; the best bound's row exactly, four in the engine, a few us more; one more region collecting the rows whose
bound reaches that score and computing them from the two planes; the rows left forced by the threshold):

| threads (tier) | Q8_0: today's head | the argmax by the bound, rows left 100 / 1000 / 3000 / 12000 | Q4_K: today's | by the bound |
|---|---|---|---|---|
| 4 (AVX2, weak) | 1993 us | 1224 / 1345 / 1499 / 2517 | 2963 | 1615 / 1680 / 1962 / 2766 (AVX2) |
| 8 (avg) | 2014 | 1072 / 1212 / 1317 / 1877 (AVX2 kernel) | 1085 | 1015 / 1070 / 1139 / 1445 (AVX-512) |
| 16 (pc) | 2102 | 1162 / 1182 / 1278 / 1620 (AVX2 kernel) | 1141 | 928 / 1007 / 1048 / 1299 (AVX-512) |

A region woken from sleep costs 40-110 us, a hot one 1-5 us; the 50304 bounds scanned by one thread 19-29 us (the
pass scans them in its second region). The Q4_K plane kernel stays below the RAM's speed at 8 threads with AVX2
(37-42 GB/s): AVX-512's bits to bytes (`vpmovm2b`) bring it to 48-50.

**The model per machine** (build/q73/model.py: the bench's argmax at every position's rows left, averaged, scaled to
the engine by today's head; a verify pass of k rows reads the planes once and computes k bounds a row; model.txt):

| a token, greedy | weak | avg | pc |
|---|---|---|---|
| Q8_0 | head 2.29 -> 1.40-1.52 ms: **1.002x** (the disk's token) | 1.98 -> 1.09-1.16: **1.034-1.036x** | 1.94 -> 1.06-1.10: **1.035-1.037x** |
| Q4_K | 2.57 -> 1.43-1.49: **1.018-1.019x** | 1.04 -> 1.02-1.03: **1.0005-1.0012x** | 1.03 -> 0.88-0.89: **1.010-1.011x** |

Verify passes: Q8_0 k 2 1.022-1.026x, k 4 1.009-1.017x; Q4_K on pc 1.003-1.008x, on avg at k 4 0.975x with k bounds
computed apart (a k-row kernel decoding the plane once would not lose: not measured). On the real machines' RAM (the
model's bytes): avg at 34.6 GB/s reads today's Q8_0 head in 3.2 ms and its planes in 1.7, Q4_K 1.7 and 1.3; the 8 GB
machine at 14.3 GB/s 7.6 and 4.0 ms, 4.0 and 3.2: every one gains.

**Decided: build (phase 2)** for greedy tokens, Q8_0 and Q4_K: four machines-and-formats gain 1-3.7%, avg's Q4_K is
level, none loses; the verify passes only with a k-row bound kernel (or off where the model loses). The head repacked
at load into its two planes (the same bytes: no RAM), the exact rows and `logits` from the planes, bit for bit. Model,
not measured: the race in whole processes decides.

### Phase 2: built in the engine, raced (2026-10-05)

**Built** (build/q73b; predictions.txt written first, 17:02, outcomes beside them): the head held at load as its two
planes (`read_head`: 256 rows read at a time and split, the same bytes, no RAM), `src/kernels/head_bound.c`: the
token's prep (scalar, every tier's), the plane kernels (scalar definition, AVX2 + F16C, AVX-512 VNNI), the rebuild of
a row from its planes, the argmax in two regions (the bounds and each worker's 4 largest; those 4 rows exactly, the
best score the threshold; the rows whose bound reaches it, exactly) and the logits from the planes (every row,
`tr_matmul_s`'s bits). The float order is fixed once, so every tier gives the scalar's bits: four chains by block mod
4, the side terms in lane block mod 8, a fixed fold, a multiply then an add (no FMA: kernels.h's contract). Q4_K's prep
counts the engine's own rounding of h (2^-30 of a super-block's max|h| an element, on g and on mu: the test's tight
rows go red without either). Greedy only: `tr_session_set_greedy` (generate, run, chat) makes a one-row pass find its
token by the bound inside the pass, its logits computed from the planes when first asked; `logits`, verify passes and
every other session compute every row from the planes. `TR_HEAD_BOUND=0`: the original rows, today's road. The
profile's lm_head zone counts the high plane and the rows computed (`rows/call`).

**Proved exact**: `tests/test_head_bound.c` (3 tiers: 24000 bounds against the engine's rows on random, adversarial
and tight heads, the tight rows' least gap 5.9e-5 of sum|w h|; 2160 argmax against the scan, 1170 pruned, 702 by the
full road; 5400 logits calls bit for bit), `tests/test_head_model.c` (the engine on synthetic Q8_0 and Q4_K heads),
`test_cli` (`--check-argmax`), `tools/mutate_head.sh` (20 mutations, every one red), and the real models against
phase 1's binary (build/q73b/real, in the container): 13 texts a model (13,821 positions), generate's 256 tokens the
same, every position's logits the same bytes, every position's token by the bound the scan's (`logits
--check-argmax`, 0 differ), Q8_0 and Q4_K.

**Counted** (build/q73b/counts.txt; `MACHINES_COUNTS=1`, code.txt and 64 tokens, one profiled run an arm, a busy
machine: the rows exact, the times indicative): **10.4-10.5 rows computed a token** of 50304 (the 4 threshold rows
included; phase 1's study: a mean of 0.28-0.36% on greedy positions, a median of 4). The lm_head zone a token:

| arm | Q8_0 off -> on (us; MiB a token) | Q4_K off -> on |
|---|---|---|
| weak (4, AVX2) | 2240 -> 1307 (104.4 -> 55.3) | 2631 -> 1381 (55.3 -> 43.0) |
| avg (8) | 2025 -> 1195 | 1165 -> 918 |
| pc (16) | 2041 -> 1170 | 1113 -> 875 |

Predicted (17:02): Q8_0 pc 1.05-1.25 ms, avg 1.10-1.30, weak 1.4-1.6; Q4_K pc 0.88-0.98, avg 0.98-1.06, weak 1.45-1.75.
Outcome: every one at or below its range; avg's Q4_K, level by phase 1's model (the bench: 1015-1139 us against
1085), gains 21% of the zone here: one run on a busy machine, the race decides.

**Raced** (`tools/ab_env.sh`, whole processes, on / off / on again rotating, code-edit.txt and 200 tokens, the same
tokens in every run; build/q73b/race-*, each under the native guards with its load declared):

| machine, format | decode ms a token off -> on | off/on (the A/A on2/on) | lm_head us off -> on |
|---|---|---|---|
| pc (16), Q8_0 | 28.18 -> 27.03 | **1.046 +- 0.008** (1.010 +- 0.007) | 2078 -> 1133 |
| avg (8), Q8_0 | 26.57 -> 25.51 | **1.041 +- 0.006** (1.000 +- 0.007) | 2011 -> 1079 |
| pc (16), Q4_K | 16.40 -> 16.05 | **1.022 +- 0.012** (1.014 +- 0.016) | 1117 -> 887 |
| avg (8), Q4_K | 15.50 -> 15.45 | **1.009 +- 0.008** (1.004 +- 0.008) | 1072 -> 844 |
| weak (4, AVX2, 8 GB), Q4_K | 74.71 -> 73.61 | **1.015 +- 0.001** (0.998 +- 0.001) | 2605 -> 1352 |

Predicted (17:02): pc Q8_0 1.025-1.037, avg Q8_0 1.022-1.036, pc Q4_K 1.000-1.011, avg Q4_K 0.995-1.005, weak Q4_K
1.008-1.019. Outcome: every one at or above its range, none loses; the Q8_0 head 1.83-1.86x faster, Q4_K's 1.26-1.28x
on avg and pc, 1.93x on the weak machine (AVX2's integers there). The first avg Q4_K race ended NOT FREE (Windows
Defender scanning: 1.010 +- 0.010, structure only) and was raced again (above). weak's Q8_0 not raced: its token is
the disk's (the head 2.24 -> 1.31 ms of a 415 ms token, 0.2%). **Decided: on by default on every machine**
(`TR_HEAD_BOUND=0` the switch back). Open: the verify passes (k rows) still compute every row, from the planes; a
k-row bound kernel next (phase 1's model: avg's Q4_K loses at k 4 with k separate bounds).

## The race for the README after the head by a bound (2026-10-05)

`78a82d6` (question 73 phase 2: the head's argmax by a bound, on by default) moves the README's Q8_0 decode, so the
README's table was raced again on HEAD `4b947bc` (build/linux-gcc, the same sources): `RACE_OUT=build/race_llama_1005
sh tools/race_llama.sh 5` and the Q4_K line (`RACE_MODEL=<Q4_K> RACE_PROMPTS=512 RACE_THREADS=4`), the same llama.cpp
build. Predictions first (build/race_llama_1005/predictions.txt, the outcome beside them).

**Every Q8_0 race ended NOT FREE at its after-declaration only** (three races: build/race_llama_1005, 1005b, 1005c):
before the first series 2.40 / 1.24 / 2.08 logical processors busy, after the last 2.85 / 3.29 / 2.97, each time with
Windows' Memory Compression at 0.52-0.59 and Defender at 0.21-0.34; the guard found the machine still before all 24
series. The race's own aftermath: the container lets go of the Q8_0's 7 GB and Windows compresses (0.49 after the
09-26 race too; the Q4_K races 0.20-0.30). The three agree within 1-3% a cell, so the table is the median of their
30 runs an engine (LESSONS #347). The first Q4_K race was NOT FREE before its first series (Defender 0.30) and its
llama.cpp prompt moved 207 / 180 between its series: raced again, free (2.39 before, 1.63 after): build/race_q4k_llama_1005b.

| tok/s, median of 30 (Q4_K: 10) | llama.cpp | Trochilus | ratio | the three races' ratios | 09-26/27 |
|---|---|---|---|---|---|
| Q4_K, prompt 512, 4 threads | 204.7 (205.8 / 203.5) | 218.2 (210.0 / 224.4) | 1.07 | | 1.00 |
| Q4_K, generation after it, 4 threads | 50.6 (50.7 / 50.6) | 52.9 (51.6 / 53.4) | 1.04 | | 1.03 |
| Q8_0, prompt 512, 16 / 8 threads | 374.9 / 235.4 | 515.3 / 338.6 | **1.37 / 1.44** | 1.41 1.38 1.36 / 1.49 1.42 1.42 | 1.33 / 1.40 |
| Q8_0, prompt 2048, 16 / 8 threads | 349.8 / 216.3 | 483.9 / 319.9 | **1.38 / 1.48** | 1.40 1.37 1.37 / 1.47 1.44 1.48 | 1.42 / 1.45 |
| Q8_0, generation at 512, 16 / 8 threads | 33.8 / 33.3 | 35.1 / 35.7 | 1.04 / **1.07** | 1.04 1.02 1.06 / 1.08 1.07 1.07 | 0.97 / 1.00 |
| Q8_0, generation at 2048, 16 / 8 threads | 29.2 / 27.6 | 27.3 / 27.7 | 0.94 / 1.01 | 0.94 0.95 0.95 / 1.00 1.00 1.03 | 0.95 / 0.96 |

Outcome against the predictions: the Q8_0 decode at 8 threads faster than predicted (512: 35.7 against 34.0-34.6,
+8% on 09-27's 33.1: the head's 0.9 ms and what came since; 2048: 27.7 against 26.9-27.4, level with llama.cpp as
predicted); the prompts at the edge of their range (1.37 against 1.33 +- 0.03 at 512, 1.38 against 1.42 at 2048: llama.cpp
341 -> 350 there); the Q4_K right (52.9 and 218). llama.cpp's Q4_K prompt 220.6 -> 204.7 today (its own A/A 205.8 /
203.5), so the Q4_K prompt's 1.07 is theirs moving more than ours. What is left to llama.cpp: the decode at 2048 with
16 threads (0.94, the KV's bytes, §The attention against llama.cpp's).

## The head's bound proved (2026-10-05)

The head by a bound (question 73, on by default since `78a82d6`) was *tested* to give the full scan's token (13 821
positions of two models; random, adversarial and tight heads; 20 mutations red). Its exactness rests on one claim:
every row's bound U(r) is at or above the engine's own score s(r), with every float rounding of both counted. A test
shows that claim on the inputs it tries. Here it is proved for every input, on the C code itself. The attempt found two
places where the code did not hold it (#349, #350), both now fixed, and an adversarial review of the written part (a
pensatore-opus agent) found one false display in it (#353), now corrected.

### The theorem

**Hypotheses.**
- A Q8_0 or Q4_K head of n columns, n a multiple of 256 and at most TR_HB_COLS_MAX = 16384.
- Any codes, every scale finite (Q8_0's d; Q4_K's d and dmin; sc and m are 6-bit integers).
- A token h that `tr_hb_prep_build` accepts: every |h_k| at most 2^64 (no NaN, no infinity), and every block of 32
  either all zero or of max|h| at least 2^-64. Anything else goes to the full head, which is the scan itself.
- The default floating-point environment, which the engine never changes: IEEE-754 binary32 and binary64, round to
  nearest even, no excess precision, gradual underflow (no flush to zero), and a correct `nextafterf`.
- For T2, a call from outside a pool body, as the engine makes it (the greedy pass's token). From inside a body the
  pool hands the caller's worker id on, past the scratch's slots (LESSONS #63's open rule).

s(r) is the engine's score as the scalar definitions compute it: Q8_0 `k_dot_row_q8_0` (sixteen lanes,
`tr_lane_combine`), Q4_K `k_q4x_dot2` (`q4x_prep`'s fixed point, exact integers, then doubles and one float rounding).
U(r) is `hb_bounds_q8_0` / `hb_bounds_q4_k`.

**T1.** For every row r: U(r) >= s(r).

**T2.** `tr_hb_argmax` returns the scan's row, `tr_argmax_f32` over the rows' s (the first of the largest).

**Not by proof.**
- Every SIMD tier of `hb_bounds`, `dot_row` and `q4x_dot2` gives the scalar definition's bits by test
  (tests/test_head_bound.c branch tiers; the kernels' tests), not by proof.
- The engine's own head (`tr_matmul_s`) gives the same bits by test.
- A row with a scale that is not finite has U = +infinity or NaN, by reading: |d| g is +infinity on a nonzero block; 0
  times infinity (a zero block, sc or m 0) is NaN; Q4_K's mu term adds +infinity. Its s is +-infinity or NaN, so lemma
  C's premise still holds; the test is branch full.
- CBMC's model of C and of IEEE-754 is trusted, and so is the standard model of rounding (lemma B).

### The proof: three lemmas, and who checks each

| Piece | Claim | Checked by |
|---|---|---|
| A1 | the planes rebuild every row byte for byte; the bound reads each code's top part: Q8_0 q = 16 (u - 8) + lo, lo in [0, 15]; Q4_K q = 2 hv + bit | CBMC, every byte of two blocks and of a super-block: `planes_q8_0`, `planes_q4_k`; more blocks by reading (one loop, one offset a block) |
| A2 | each lane's int32 P = sum u (256 X + Xl) over its own four elements and their own digits; every P below 2^21, so (float)P exact | CBMC, every code, any element, lane by lane: `bound_q8_0`, `bound_q4_k`; P's range enumerated; `hb_bounds_*` around the lanes (block offsets, chains, side lanes, fold) by reading |
| A3 | the digits: q = 256 X + Xl, both int8, for every q in [-32639, 32639] | CBMC: `digits` |
| A4 | per element: abs(q Q - (16 u - 120.5) Q) <= 7.5 abs(Q) (Q8_0), abs(q Q - (2 hv + 1/2) Q) <= abs(Q) / 2 (Q4_K) | every case enumerated: `proof_enum`, 17 755 888 cases |
| A5 | the block's terms taken exactly sit above its exact contribution by the margin | written below, from A1-A4 |
| B1 | `hb_up(v)` is the least float at or above v | CBMC, every double: `up`; `nextafterf`'s model against its definition: `next_float` |
| B2 | the margin covers the most roundings on any term's path, for every n: hb_margin(n) >= (D + 16) 2^-24 | CBMC, every n: `margin`; that its path counts are the code's, by reading (the table in lemma B) |
| B3 | each float sum within about (D + 1) 2^-24 of its terms' magnitudes of its exact value, underflow included | written below (the standard model) |
| C | the argmax: the scan's row, from any threshold that is some row's exact score | written below for every n; CBMC on the real `tr_hb_argmax`, any scores and bounds: one worker on 6 rows and any prefix, two on 5, a worker's two ranges in reverse order: `argmax_*`; `tr_argmax_f32`'s rule by reading and by test |

`sh tools/proof.sh` (`make proof`, image trochilus-proof) runs them all. **The run of 2026-10-06, 00:02-00:21** (CBMC
5.95.1 with MiniSat 2.2.1 in the Docker VM, 15 GB, this PC; the lemmas two at a time, the argmax one at a time): every
harness proved, **1120 s in all**.

| Harness | Seconds | Harness | Seconds |
|---|---|---|---|
| planes_q8_0 | 1 | margin | 1 |
| planes_q4_k | 3 | next_float | 1 |
| bound_q8_0 | 9 | argmax_q8_0 (every float, 6 rows) | 200 |
| bound_q4_k | 23 | argmax_q8_0_pool | 316 |
| digits | 1 | argmax_q8_0_steal | 334 |
| up | 2 | argmax_q4_k_pool | 243 |

`proof_enum`: 17 755 888 cases, every one holds, under a second. **Seen red** (`tools/mutate_head.sh` in the same
image, 10-06 00:45-01:08): each of 10 mutations turns its harness red (the margin halved, the engine's path one
rounding short, hb_up not rounding up, a plane's bit dropped in each type, a lane's digits misread in each type, the
high digit one off, the threshold strict, the workers' tie to the higher row), the unmutated proof green; the 21 test
mutations red as before, the new one (a block of h below 2^-64 kept) included. At 19 minutes `make proof` stays out of
`make check`; `tools/lint.py` fails the gate while `head_bound.c` or a harness differs from the hashes of the last proof
(tests/proof/proved.sha256, written by `proof.sh` only when every harness passed).

### Lemma A: the bound's terms, taken exactly, sit above the row by the margin

Notation for one block of 32 (Q8_0's block, Q4_K's sub-block):
- h_k its elements; delta > 0 the prep's step; Q_k its integer code, |Q_k| <= 32639. The argument never uses how they
  were found: any delta and any Q work.
- eps_k = h_k - delta Q_k, and E >= sum |eps_k|: the prep's e, each |h - delta Q| in double (delta Q exact there, 24 +
  16 bits; one rounding), 32 additions, then x (1 + 1e-12), far above the 2^-47 of its roundings.
- ha = sum |h_k|; I = sum u_k Q_k (Q4_K: hv_k); sx = sum Q_k; sax = sum |Q_k|.
- M the margin (`hb_margin`), u = 2^-24.

**Q8_0**, a block of scale d. By A1, q_k = 16 u_k - 120.5 + (lo_k - 7.5). By A4, on both sides (d of either sign), and
|q| <= 128:

    c = d sum q_k h_k = d delta sum q_k Q_k + d sum q_k eps_k
      <= d (16 delta) I + d (-120.5 delta sx) + |d| (7.5 delta sax + 128 E)

The prep's floats:
- a = 16 delta exactly.
- be = fl(-120.5 delta sx): delta sx is exact in double (24 + 20 bits), and so is its product with 120.5; one rounding
  to float, so |d (-120.5 delta sx) - d be| <= u 120.5 |d delta sx|.
- g = hb_up(G) >= G (B1). G is the double computed for 7.5 delta sax + 128 E + M (mag + 120.5 |delta sx|), with mag =
  240 delta sax + 120.5 |delta sx| + g_0 + 128 ha and g_0 = 7.5 delta sax + 128 E. Its roundings are a few, and ha's
  31 enter multiplied by M, so G >= its exact value (1 - 2^-47).

Hence, with T = d a I + d be + |d| g the block's terms taken exactly:

    c <= T - |d| (M - u - 2^-47) mag.

**Q4_K**, sub-block j of super-block s. d, dmin, sc_j and m_j come from `tr_q4_k_sc_m` on the same 16 bytes the engine
reads (A1: the planes keep them).

The engine sees h through its fixed point: h~_k = X~_k 2^-sh, X~_k the nearest integer to h_k 2^sh, sh =
`tr_q4x_shift`(ms), ms the super-block's max|h|. That function's sh is m 2^(31 - e) in [2^30, 2^31), lowered once if
above TR_Q4X_XMAX, so ms 2^sh >= XMAX / 2 = 2^29.958. So |h~_k - h_k| <= 2^-sh-1 <= err_el / 1.9427, with err_el = ms
2^-30, and |h~_k| <= |h_k| + err_el / 1.9427.

The row's exact value is sum_j (d sc_j sum q_k h~_k - dmin m_j sum h~_k): `q4x_block`'s T and M are exact integers, and
the base-64591 digits recombine exactly. By A1, A4 and |q| <= 15:

    d sc sum q h~ <= d sc (2 delta I + 0.5 delta sx) + |d| sc (0.5 delta sax + 15 E + 248 err_el)
    -dmin m sum h~ <= -dmin m H + |dmin| m (|hs - H| + |hs - sum h| + 16.5 err_el)

hs is the prep's double sum of the h_k, |hs - sum h| <= 2^-48 ha, and H = fl(hs). The prep:
- a = 2 delta exactly; be = fl(0.5 delta sx), off by at most u 0.5 |delta sx|.
- g = hb_up(...) >= (0.5 delta sax + 15 E + 480 err_el + M mag)(1 - 2^-47), with mag = 14 delta sax + 0.5 |delta sx|
  + g_0 + 15 ha.
- mu = hb_up(4 M ha + |hs - H| + 32 err_el) (|hs - H| exact in double).

So with T_j = d sc a I + d sc be + |d| sc g - dmin m H + |dmin| m mu:

    c_j <= T_j - |d| sc ((M - u - 2^-47) mag + 232 err_el) - |dmin| m ((4 M - 2^-47) ha + 15.5 err_el).

### Lemma B: the float sums stay inside the margin

The standard model (IEEE-754, round to nearest; Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §2.2,
§4.2):
- fl(x op y) = (x op y)(1 + theta), |theta| <= u, without underflow.
- An addition is exact when its result is subnormal.
- A product that underflows errs by at most 2^-150.

A sum evaluated along a fixed tree, term i met by D_i roundings on its way (its own products included), differs from
the exact sum by at most sum gamma(D_i) |t_i|, with gamma(D) = D u / (1 - D u), plus the underflows. The paths, counted
on the code (`margin` restates each and checks M against them for every n):

| | path of a term | roundings |
|---|---|---|
| Q8_0 engine | w = d q exact (11 by 8 bits, \|w\| >= 2^-24); w x; n / 16 terms a lane, the first added to 0 exactly; the lanes' tree of depth 4 | n / 16 + 4 |
| Q8_0 bound, chains | coef = d a; P coef; nb / 4 terms a chain; (c0 + c2), + (c1 + c3), + side, the eight lanes' tree of depth 3 | n / 128 + 7 |
| Q8_0 bound, side | d be or \|d\| g; nb / 4 terms a lane; + into its lane, the tree | n / 128 + 4 |
| Q4_K engine | integers exact; d T, dmin M, their difference, x 2^-sh (exact), n / 256 additions, all in double; one float rounding | 1, plus (n / 256 + 3) 2^-53 |
| Q4_K bound, chains | d sc (exact: 11 by 6 bits, counted anyway), coef, P coef; 2 ns terms a chain; the fold | n / 128 + 8 |
| Q4_K bound, side | (d sc) be, (\|d\| sc) g, (dmin m) H, (\|dmin\| m) mu, two roundings each; 4 ns terms a lane; the fold | n / 64 + 5 |

D, a type's longest path:
- Q8_0: n / 16 + 4 (1028 at n = 16384, 132 at 2048).
- Q4_K: the larger of n / 64 + 5 and n / 128 + 8 (the chains win at n = 256).
- M >= (D + 16) u (B2), and M <= 2^-13 for every n the head takes.

The terms' magnitudes:
- The engine's: |w x| sums to at most sum |d| 128 ha. Q4_K: |d| sc 15 sum |h~| and |dmin| m sum |h~|, with sum |h~| <=
  ha + 16.5 err_el.
- The bound's: |d| (240 delta sax + 120.5 |delta sx| (1 + u) + g), with g <= (g_0 + 2 M mag)(1 + 2u) (hb_up can land
  two units of rounding above). Q4_K's min terms: |dmin| m (|H| + mu), with mu up to 4 M ha + 33 err_el.
- Each mag holds both. Q8_0: mag >= 128 ha plus the bound's part. Q4_K: mag >= 14 (ha - E) + 15 E + 15 ha >= 29 ha, and
  the engine's coefficient is u, not gamma(D).
- gamma(D)(1 + 2M + 3u) <= (D + 1) u for D <= 1028.

So, with R = sum T the bound's terms taken exactly and S the row's exact value:

    |U - R| + |s - S| <= (D + 1) u sum |d| mag  +  (D + 1) u |dmin| m (2 ha + 50 err_el)  +  underflows

(Q4_K: |d| sc in place of |d|, and the err_el parts of the d terms inside the 232 err_el of lemma A).

**Underflows.**
- Q8_0: a block with d = 0 or h = 0 has every product exactly 0 (and g = hb_up(0) = 0). Otherwise |d| >= 2^-24 (f16's
  least) and max|h| >= 2^-64 (the prep sends a block between 0 and 2^-64 to the full head: #349). So mag >= 128 2^-64,
  and the margin's spare 13 u |d| mag >= 2^-104, while the block's at most 44 products (32 engine, 12 bound) err by at
  most 44 2^-150 together.
- Q4_K: the products that can underflow are the engine's final (float)y, once a row, and the bound's, (dmin m) H among
  them (H can be subnormal: hs of a few terms that cancel). A sub-block of zeros in a nonzero super-block still has g
  and mu above zero, from err_el >= 2^-94. The spare that pays: the d term's 13 u |d| sc mag when d sc is not 0, else
  the min term's (4 M - 2 (D + 1) u) |dmin| m ha or 15 err_el |dmin| m, each at least 2^-118 against a few 2^-150.

**No overflow**: |h| <= 2^64, |d| <= 65504 and n <= 2^14 keep every partial sum below 2^110.

With lemma A:

    U >= R - |U - R| >= S + sum |d| (M - u - 2^-47) mag - |U - R| >= S + |s - S| >= s

since M - (D + 2) u - 2^-47 >= 13 u covers the underflows. The min terms work the same way: 4 M against 2 (D + 1) u,
and 15.5 err_el against (D + 1) u 50 err_el. **T1 holds.**

### Lemma C: the argmax

Let r* be the scan's row: the first r with s(r) = max s.
- The threshold th is the largest exact score among the rows of the TR_HB_TOP largest bounds (or -infinity if none).
  It is some row's score, so th <= s(r*).
- Every row with U(r) >= th is computed exactly, with the same function as the scan (`hb_exact2`: `dot_row` or
  `q4x_dot2` on the rebuilt row, A1).
- A row left out has s(r) <= U(r) < th <= s(r*) (T1): it is below the maximum, so it can be neither r* nor a tie of it.
- r*, and every row tying it, has U >= s = s(r*) >= th, so each is computed.
- Each worker keeps its best (the lowest row on a tie) over every range it is given, in any order. The workers merge
  with the same rule, so the result is the first of the largest: r*.
- A NaN bound or score sends the token to the full head, the scan.

Nothing here depends on which rows the top list held: any threshold that is some row's exact score works. **T2 holds.**

CBMC checks it on the real code, past TR_HB_TOP = 4: any prefix, any finite scores, and any bounds at or above them or
NaN. One worker on six rows and any prefix (Q8_0); two workers on five rows, each region cut at any row, a side each
(Q8_0, and Q4_K's road, two rows a call); and worker 0 taking both sides, the later first (a worker's several ranges in
any order, as the balanced pool's stolen blocks). The one-worker harness takes every float; the two-worker ones take
scores and bounds as the integers 0 to 10 (or a NaN bound). That is enough because the code never computes with a
score or a bound, it only compares them (>, >=, ==, x != x; against -infinity and 0 only behind an index guard), so its
result depends on their order alone, and eleven values realize every order of ten floats, -0 == +0 as one value.
Bigger models of the pool (six rows, three chunks, every chunk to either worker in either order, every float) ran
the Docker VM's 15 GB out of memory or past 30 minutes (#351). `tr_argmax_f32`'s own rule (kernels.c, each chunk's lanes
keep the first of their largest, merged by the lowest row) is by reading, and tests/test_head_bound.c compares the two
on every head it builds.

### What the proof found

1. **A subnormal h dropped the scan's row** (#349).
   - The case: h's block 0 at -2^-149, the rest 0. One row has d = 0.5 - 2^-11 and every code -128; four rows have
     d = 0.5 with one code -126.
   - The engine gives the first row 2048 2^-149 (each of its 32 products, 63.94, rounds up to 64) and the others 2047.
   - The first row's bound is one product, 2046.5, rounded to even: 2046. That is below 2047, so the threshold
     excluded it: the bound path returned row 0, the scan row 4, on every tier (tests/test_head_bound.c tiny_case:
     red before the fix, green after).
   - The cause: the old prep folded a block below 2^-64 into h's rounding (X = 0), where the margin is relative to
     magnitudes that underflow.
   - Fixed: such a block, if not zero, goes to the full head (`HB_HMIN`). No real hidden state has one.
2. **The margin fixed at 2^-16 covered paths of at most 240 roundings** (#350).
   - Q8_0's engine has n / 16 + 4, so the proof closed only up to n = 3776, while the head accepts 16384 (D = 1028: a
     margin 4.1x short). Q4_K's side lanes were covered up to n = 14848 (at 16384, D = 261).
   - Not shown to fail on a row: the worst case wants one large lane and a thousand terms, each rounding the same way.
     The proof did not reach these sizes.
   - Fixed: `hb_margin` grows with n, as (D + 16) 2^-24, never below 2^-16. Every head up to 3776 columns, OLMoE's 2048
     among them, keeps the same bits.
3. **The written proof's own error** (#353).
   - The first draft's lemma B left the err_el inside mu out of the Q4_K min terms' magnitudes. The review built a row
     on the real code where that display fails: n = 256, d = 0, dmin = 65504, h_0 = 2^64, h_32 = 1.5 2^-64.
   - T1 held there through a spare the text did not state (err_el is at least 1.94 times the engine's rounding of h).
     The text now states it, and with it the floating-point environment and the worker hypothesis.
