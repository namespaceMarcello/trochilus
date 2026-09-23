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
alternating runs stays question 19 | — | — |
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
| 28 | ~~Il divario di prefill che resta con llama.cpp dov'è, se non è nell'int8?~~ Era nell'int8 (domanda 21). La strada esatta «più token nei registri» è misurata e non rende: una riga contro 8 token dà 1.13× a n=1024, **0.79×** a 2048 e 0.93× a 4096 (§Revisione) | — | — |

| 29 | ~~Quanto vale una variante SIMD per **F16**?~~ **49×** sul kernel (`make bench`, n=2048: 466 M el/s lo scalare, 22.8 G el/s AVX2 con F16C, 22.7 AVX-512; prima ogni tier faceva i 466-824 dello scalare). La conversione half→float è esatta, quindi resta bit-identico (`tests/test_kernels.c`, anche con subnormali, infiniti e NaN). Scritta il 2026-09-19 perché il controllo nuovo, `tests/test_tier_used.c`, pretende che **ogni tipo** di peso passi per i kernel del tier (LESSONS #78) | — | — |
| 30 | ~~La parte seriale del prefill (copia per esperto, scrittura della KV, norme, RoPE) si può dividere per token?~~ **Sì** (§Prefill su prompt lunghi): valeva l'8% del prefill a 512 e il 5-6% a 4000, contando la scelta del router (57 ms su 512 token) che stava dentro la zona `router`. Divisa sul pool per token (almeno 8 a pezzo) passa da 121 a 48 ms a 512 e da 880 a 353 a 4000: norme e RoPE si dividono per 6, le copie in memoria (righe per gli esperti, scrittura della KV: 27 → 19 ms) solo per 2 e per 1.1-1.4, perché lì il limite è scrivere in RAM. Con l'int8 (domanda 21) il peso di ciò che resta raddoppia | — | — |
| 34 | **Chiusa come «no» il 2026-09-19 (Marcello)**: 2.1% stimato a contesto 4000, sotto la soglia delle misure; si riapre se un modello con contesti oltre 4000 la porta sopra. ~~Larghezza per zona~~: l'attenzione del decode su tutto il pool e il resto sulla larghezza del decode (esatto: il contratto del pool). Potenziale misurato per zona (§Decode a contesto lungo): a contesto 4000 l'attenzione fa 21.6 ms su 16 thread e 22.6 su 8, mentre esperti e proiezioni su 16 perdono 1.3 ms; tenendo il meglio dei due si toglie **1.0 ms su 48.1 (2.1%)**, a 2048 0.24 ms su 36.4 (0.7%) | `tr_pool_set_active` attorno alla zona `attention` nelle passate corte; serve una soglia più bassa del 3.8% di questa notte (più giri, o 200 token generati invece di 48) | cresce col contesto: oltre 4000 l'attenzione supera metà del token. Sotto la soglia oggi, quindi non scritta |
| 35 | Cosa resta sotto il tetto nel decode, dal lato esatto? `attn_out_proj` legge a 43-52 GB/s dove le altre moltiplicazioni stanno a 52-55 (0.2-0.3 ms per token); le zone senza byte (norme, RoPE, copia per esperto, attivazione, somma, scelta del token) valgono **1.1 ms per token**, il 4.4% a contesto corto; l'attenzione legge a 47 GB/s dove la sola lettura degli stessi byte fa 52-56 (il calcolo di prodotto, softmax e somma pesata vale il 12% della zona) | profilo per zona con le zone piccole divise; `bench_mem kv` con il kernel a pezzi | in tutto il 6-11% che separa i 48-51 GB/s del decode dai 54 della RAM |
| 36 | **KV a 16 bit** (non esatta, decide Marcello): di quanto si spostano i logit? Guadagno stimato dai byte (§Decode a contesto lungo): +5% a contesto 512, +16-19% a 2048, +26-31% a 4000, e metà memoria della KV | modo dichiarato e spento di default; contro il modo esatto sul modello vero: KL media e primo token uguale su ≥ 1000 posizioni di codice fino a contesto 4000 (`trochilus logits` nei due modi, `tools/compare_llamacpp.py` calcola già la KL: llama.cpp sta a 9e-3), token greedy uguali su 1000 generati (soglia della domanda 17: ≥ 99%), e nessun valore di K o V oltre il massimo dei 16 bit (65504) | è la sola leva grande che resta al decode a contesto lungo: dopo la KV per testa l'attenzione legge già alla banda della RAM |
| 37 | ~~**`expf` nostro**: il softmax (attenzione, router) e l'attivazione degli esperti chiamavano `expf` della libreria C, 30 ns a chiamata con MinGW e altri bit con glibc~~ **Scritto lo scalare il 2026-09-19** (`tr_expf`, §`tr_expf`): arrotondato correttamente su tutti i 4 278 190 082 float per prova esaustiva in `make check` (gcc e clang), 3.5 ns a chiamata, nessuna chiamata alla libreria dentro; su Windows gli stessi bit di prima (logit identici al byte), su Linux KL 3.9e-13 e 1000 token su 1000 uguali, e gli stessi byte del build di emulazione; **Windows e Linux ora danno gli stessi logit al byte**. Prefill **1.03-1.08× a 512, 1.20-1.23× a 2048, 1.29-1.31× a 4000**, decode 1.02-1.10× (stimati 1.12 / 1.23 / 1.27× e +5-10%). Resta il SIMD: domanda 39 | — | — |
| 38 | L'attenzione a gruppi con **GQA** e oltre 4096 token (Qwen3-Coder: 32 teste di query su 4 di chiavi): un gruppo oggi sono 16 token di **una** testa di query; le 8 teste che condividono chiavi e valori potrebbero stare nello stesso gruppo e leggerli una volta sola. E a 16-32 mila token chiavi e valori di una testa (16-32 MB) non stanno in nessuna cache: lì la lettura ripetuta, che a 4000 va dall'1% al 21-39% della zona secondo la run, diventa il grosso | `make bench-attn` con le forme del modello nuovo e `--heads`/`--group`; prefill a 8, 16, 32 mila token quando il modello c'è | il coding usa contesti lunghi; a 4000 il gruppo da 4, 16 o 64 va uguale, più su non è detto |
| 39 | **Chiusa come «no» il 2026-09-19 (Marcello)**: 1.01-1.04× stimato sta alla soglia del rumore, per circa 200 righe col pezzo più delicato del progetto; lo scalare ha già preso quasi tutto. ~~`tr_expf` in SIMD, sì o no~~ (numeri in §`tr_expf` punto 7). Dopo lo scalare restano, nel prefill a 512 / 2048 / 4000, softmax dell'attenzione per circa 9 / 130 / 610 ms e `expert_act` per 25 / 108 / 218 ms. Stima con un esponenziale AVX-512 a 1.0-1.3 ns per elemento: prefill **1.01× / 1.03× / 1.04×**, decode +1-2%; costo circa 200 righe, il pezzo delicato (gather dalla tabella, maschera per le corsie che non passano il test, identità al bit con lo scalare provata su tutti i float per ogni tier) | uno schizzo nel banco (`tests/bench_expf.c`) darebbe il costo vero per elemento prima di decidere; poi voce `exp` nella tabella dei kernel, `make bench-expf` per tier, `prefill_context.sh change` | lo scalare ha preso quasi tutto: a 4000 token il guadagno stimato è vicino alla soglia di una buona sessione (3%) |
| 40 | ~~Le **tabelle RoPE** di Windows e di Linux sono le stesse voce per voce?~~ **Sì**, misurato il 2026-09-19 (`sh tools/platform_bits.sh`, §`tr_expf` punto 5): su 4096 posizioni × 64 coppie i `cos` e i `sin` **in double** delle due librerie differiscono nello 0.83-0.85% delle voci (2175 e 2216 su 262 144: glibc e MinGW arrotondano diversamente l'ultima cifra del double), `pow` mai (64 su 64 arrotondati correttamente su tutte e due), e **nessuna** delle 524 288 voci in float differisce: l'arrotondamento a float assorbe la differenza. Con `tr_expf` il motore dà quindi **gli stessi byte sulle due piattaforme** fino al contesto di addestramento di OLMoE | — | — |
| 41 | **Perché il disco dà 1.5 GB/s, un terzo della sua scheda (Micron 2400, 4.5 GB/s)?** Tre ipotesi: la cifratura del volume di Windows (si legge con `manage-bde -status C:` da amministratore), il QLC senza DRAM su letture sparse, la richiesta sincrona da 2 MiB (provare richieste più grandi e sovrapposte). Vale il triplo dei token al secondo di M1 con la cache al 50%. **2026-09-20: il volume è cifrato** (`manage-bde`: BitLocker ON, XTS-AES 128, quindi in software). Non è provato che sia la causa, né la sola: durante le letture il processo System passa da 0.79 a 0.94 processori (`tools/background_load.ps1`, `build/disk/run3.txt`), nessun core è saturo, quindi se il limite è lì è di latenza e non di calcolo. La prova vera chiede un volume non cifrato sullo stesso disco, e **la cifratura non si toglie per una misura**: i PC con Windows 11 escono cifrati di fabbrica, quindi ~1.5 GB/s **è** il bersaglio, e il piano automatico di M1 misura il disco che trova | `bench_disk --block` a 16 e 64 MiB; più richieste in volo per lettore; lo stesso banco su un disco esterno o una partizione non cifrata, se capita | dopo il primo M1 che gira: prima il percorso esatto, poi la banda |
| 43 | **Da che velocità di disco in su il precaricamento rende, e con quanti candidati?** Il modello a tempo dice: a 1.5 GB/s no (k=8 neutro, k=12 −40%), a 4.3 GB/s k=8 dà l'11-15% (§M1 punto 6) — modello, non misura: un disco, una lettura per volta  — e il decode ha ormai poco da nascondere: lontano dal prompt sbaglia 0.03-0.14 unità per token (§M1 misurato), quindi se il precaricamento serve è nel **prompt** | con M1 che gira: `--expert-budget` al 50%, precaricamento spento e con k=8, su questo disco e su uno veloce (o sul file nella cache del sistema, che rende 12-26 GB/s); soglia nel piano automatico | dopo il primo M1 |
| 42 | **Come si prevede il primo layer?** La previsione col router del layer dopo prende il 92-95% ovunque tranne che al layer 0 (75-80%, domanda 13) | dalla traccia: gli esperti del layer 0 scelti per token uguale (dipendono quasi solo dall'embedding?); in alternativa il layer 0 sta sempre in RAM (64 esperti = 408 MiB, il 6% del modello) | progetto di M1 |
| 44 | ~~Il comportamento del modello sul codice è un grafo piccolo e deterministico?~~ **No, in tutti e quattro i sensi** (Marcello, 2026-09-20; chiusa il 2026-09-20 notte con i cinque testi della prova funzionale). Dalla traccia che c'è (OLMoE-1B-7B, `code-1000`, 1204 token; calcolo una tantum sulla traccia, non ancora nel report): **piccolo no**: usate 1012 unità su 1024, l'89% già dopo 100 token; il 25% più usato copre il 68% delle attivazioni, il 50% l'88%, il 75% il 97%; entropia d'uso 5.1 bit su 6 per layer. **Statico no**: un grafo di co-occorrenze fra layer imparato sui primi 900 token indovina il 53.7% degli esperti del layer dopo sui 300 seguenti (frequenza sola: 40.3%; il router sullo stato vivo: 82-86%); nessun percorso intero si ripete (0 su 1204), il singolo insieme di 8 sì (32%); 3.6 esperti su 8 in comune col token prima (caso: 1.0), che è ciò che l'LRU sfrutta. Coerente col pin dall'uso che perde contro l'LRU (domanda 14) e con la loss di bilanciamento con cui i MoE si addestrano. Limiti: un modello generalista, un prompt, un linguaggio | mancano, con le soglie scritte **prima**: (1) id dei token nella traccia → stesso token, stessi esperti? (la versione «tabella» dell'ipotesi, plausibile al layer 0); (2) 4-6 tracce (file e linguaggi diversi, e prosa di controllo) → la sovrapposizione delle unità calde codice-codice supera quella codice-prosa?; (3) margine fra l'8° e il 9° esperto (probabilità nella traccia); (4) la prova che decide, funzionale e non di routing: **mascherare** gli esperti fuori dal X% più usato e misurare token uguali e KL contro il modello intero su codice mai visto (modo di sola misura) **Soglie scritte prima di misurare (2026-09-20, sì di Marcello)**: *piccolo* = il 25% delle unità copre ≥ 99% delle attivazioni su codice; *grafo del codice* = la sovrapposizione (Jaccard) del 25% più caldo fra due tracce di codice supera di ≥ 0.20 quella fra codice e prosa; *tabella* = stesso id di token → stesso insieme di esperti al layer 0 in ≥ 95% delle ripetizioni (gli altri layer si riportano); *funzionale* = con il 50% delle unità mascherate (le meno usate su un **altro** file di codice) il token greedy coincide in ≥ 99% delle posizioni e la KL media è ≤ 1e-2 su codice mai visto (il metro del progetto per un modo non esatto: llama.cpp sta a 9e-3). Una soglia mancata falsifica quella parte dell'ipotesi per OLMoE-1B-7B; per dirlo «dei modelli» serve almeno un secondo modello | **Risposta (§Il comportamento sul codice…)**: piccolo no, tabella no, statico no, funzionale no (già sul testo della maschera: 93.9% dei token col 50% spento); una **regione del codice** sì (Jaccard 0.68-0.77 fra C, Python e shell, 0.07-0.09 con la prosa inglese), e l'uso ordina gli esperti 13-50 volte meglio del caso **dentro il codice** e per niente fuori (sulla prosa inglese la maschera per uso fa come quella a caso). Prova funzionale su cinque testi su cinque: col 50% spento il token coincide nel 93.9 / 92.8 / 86.0 / 81.4% (testo della maschera, altro C, Python, shell). Resta un secondo modello |
| 45 | **Il costo del primo prompt si può togliere, o solo nascondere?** Sotto budget ogni avvio rilegge il modello intero: 4.7 s misurati a 2048 token, e sono il divario che resta col modello residente (12.41 contro 7.47 s). **Attenzione al meccanismo**: ricaricare all'avvio un elenco delle unità che erano in RAM non toglie niente — rileggerle dal disco *è* quel tempo, e con la lettura diretta il sistema non ha copie da regalarci. Si toglie in un modo solo, **tenendo i byte in RAM fra un avvio e l'altro** (un processo che resta acceso, domanda 49); altrimenti si **nasconde** dietro il tempo dell'uomo (domanda 48) | il secondo avvio con l'archivio vivo contro uno a freddo, stesso prompt: prefill e mancati dei primi 100 token | è il divario che resta fra budget parziale e modello residente |
| 46 | ~~Perché il decode sotto budget migliora con la lunghezza della generazione?~~ **È l'archivio, non un warm-up** (2026-09-21, §M1 misurato): col modello residente il tempo per token non cambia da 200 a 1000 (34.64 → 33.48, dentro lo spread), sotto budget sì (26.53 → 29.84 al 25%). Un costo fisso di ~0.35 s al 50% e ~0.84 s al 25%, che i mancati in più dei primi token spiegano solo per ~115 ms. **Resta**: dove va il resto | profilo per zone dei primi 64 token contro token lontani dal prompt, stesso budget | dice se c'è una leva nella LRU dopo il prompt, e quale numero promettere con metà modello in RAM |
| 47 | ~~Il prefill sotto budget legge gli esperti una volta per prompt?~~ **Adesso sì** (ordine per layer, 2026-09-21: 6 273 MiB e 12.41 s a 2048, **1.89×**). Prima: una volta per passata (2026-09-21, §Il prefill legge il modello una volta per passata). Il prompt si elabora a blocchi di 512 token (`OLMOE_DEFAULT_BATCH`) e ogni passata percorre tutti i layer, quindi sotto budget rilegge la tabella intera: a 2048 token **22 880 MiB invece di 6 528**, 3.65×, e cresce col prompt. Con una passata sola (`-b 2048`) il prefill fa 11.27 s invece di 23.51 (**2.09×**) e l'ultima riga di logit è identica al byte | il tempo dell'ordine per layer quando ci sarà, e la stessa misura a 4000 token (otto passate) | la leva più grossa del prefill sotto budget, e non costa precisione |
| 48 | **Quanto del primo prompt si nasconde dietro il tempo che l'utente impiega a scrivere?** Una fase di preparazione dichiarata (idea di Marcello, 2026-09-21): appena la sessione si apre, e prima che il prompt arrivi, il motore comincia a leggere gli esperti e lo dice («leggo il modello, 6.5 GiB»). Non elimina i 4.7 s, li mette dove non danno fastidio; costa nessun bit di precisione e non serve un processo nuovo. Da decidere: cosa leggere per primo quando ancora non si sa il prompt (l'ordine dei layer è quello giusto, il layer 0 serve per primo) | tempo fra l'apertura e il primo token, con e senza la lettura anticipata, su un prompt che arriva dopo 5, 15 e 30 secondi | è la leva più economica: nessuna struttura nuova |
| 49 | **Un processo che resta acceso fra le sessioni regge il costo che ha?** L'unico modo di togliere davvero i 4.7 s: l'archivio degli esperti vive più della sessione. Da decidere: un demone che parla coi comandi, o memoria condivisa che i processi mappano. Costo: un protocollo, il ciclo di vita, e la RAM tenuta occupata quando nessuno la usa | tempo del primo prompt della seconda sessione, archivio vivo contro archivio nuovo | toglie il divario col modello residente, non lo sposta |

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

Letture:
- AVX-512 va quasi quanto AVX2: il limite è la **catena delle somme**. Ogni corsia somma un
  prodotto dopo l'altro, e una somma aspetta la precedente (~3-4 cicli di clock): 16 corsie danno
  al massimo 16 elementi ogni 3-4 cicli, con registri da 256 o da 512 bit. Più corsie romperebbero
  la definizione scalare (numeri diversi): da valutare solo se un thread singolo torna il limite.
- Sulla carta 16 thread da 20 000 M el/s supererebbero la banda della RAM (~80 GB/s); sul modello
  on real model: traffic stops at 22 GB/s because threads don't pay (section below).
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

Container, volume `trochilus-models`, OLMoE-1B-7B Q8_0 intero, `run -f <prompt> -n 200`,
`--spec 0` e `--spec k` **alternati run per run** (`tools/ab_spec.sh`, LESSONS #46), mediana di 3
giri dopo uno di riscaldamento. Il testo generato è identico in ogni giro: lo script si ferma se
non lo è. Nessun pin dei thread e misure in container: rifatte native e col pin in §Bozza adattiva,
dove il costo di una riga in più risulta circa 3 volte quello scritto qui sotto (13.7-17.6 ms
misurati per zona contro 5.2; LESSONS #59).

| Prompt | Bozza | Decode tok/s (min-max) | Bozze accettate | Contro `--spec 0` |
|---|---|---|---|---|
| `bench/prompts/code-edit.txt` (il file è già nel prompt, va riscritto con una modifica) | 0 | 29.1 (27.1-30.7) | — | — |
| | 2 | 32.5 (31.4-33.5) | 120/160 = 75% | **1.11×** |
| | 8 | **41.3** (41.0-43.2) | 170/264 = 64% | **1.42×** |
| `bench/prompts/code.txt` (codice nuovo, il modello non ricopia niente) | 0 | 29.6 (28.5-31.8) | — | — |
| | 8 | 18.4 (18.4-19.5) | 57/435 = 13% | **0.62×** |
| stesso prompt, 4 thread | 0 | 29.1 (28.7-29.4) | — | — |
| | 8 | 16.3 (16.2-16.4) | 57/435 = 13% | **0.56×** |

Letture:
- **Sul lavoro di codice vero la leva c'è**: quando il testo da produrre è già nel prompt (riscrivere
  un file con una modifica, il caso per cui la leva è stata scelta), la bozza da 8 dà **1.42×** con
  il 64% di token accettati, e i token sono gli stessi.
- **Quando il modello inventa, si perde**: 13% accettate e 0.62×. Una passata da 1 + k posizioni
  costa ~34 ms di pesi più ~5.2 ms per riga in più (34 ms a una riga, 76 ms a nove): il pareggio è
  intorno al **15% di bozze accettate**, e sotto quella soglia si paga.
- Il costo per riga in più (~5.2 ms) è lo stesso lavoro per elemento del prefill, quindi il pin dei
  thread ai core fisici lo abbassa: dopo quel passo la soglia di pareggio scende e la bozza lunga
  conviene più spesso. Da rimisurare lì.
- Conseguenza: `--spec` resta **spento di default** finché la bozza non si accorcia da sola dopo un
  rifiuto (llama.cpp lo fa). Con la bozza adattiva il caso peggiore diventa «come senza».

## Adaptive draft: what one more row really costs (2026-09-18)

Windows **nativo**, macchina ferma, pin al core (il default), 16 thread, `run -f <prompt> -n 200`,
`--spec 0` e `--spec 8` **alternati run per run** (`tools/ab_spec.sh`), mediana di 3 giri dopo uno
di riscaldamento, testo identico in ogni giro. `--spec-fixed` sceglie la bozza fissa. Le due righe
«con pausa» sono la **rimisura a 8 giri con controllo A/A** (`tools/ab_modes.sh`, §Revisione): i 3
giri di prima davano 1.15× (28.63 → 32.98) e 1.01× (27.49 → 27.76), e il secondo era sbagliato.

| Prompt | Politica della bozza | `--spec 0` | `--spec 8` | | Accettate | Bozza media |
|---|---|---|---|---|---|---|
| `code-edit.txt` (il file è già nel prompt) | fissa 8 | 27.99 | **37.61** | **1.34×** | 64.4% | 8.00 |
| | adattiva, solo si accorcia | 26.96 | 32.81 | 1.22× | 70.9% | 2.86 |
| | adattiva con pausa (oggi), 8 giri con A/A | 31.87 | 37.45 | **1.175×** | 69.4% | 2.34 |
| `code.txt` (codice nuovo) | fissa 8 | 29.65 | 17.67 | **0.60×** | 13.1% | 3.04 |
| | adattiva, solo si accorcia | 29.61 | 26.22 | 0.89× | 41.3% | 0.70 |
| | adattiva con pausa (oggi), 8 giri con A/A | 32.67 | 31.15 | **0.953×** | 42.4% | 0.38 |

Il costo di una riga in più, ricavato dai passaggi stampati (ms per passata = 1000 · token / tok·s⁻¹
/ passaggi, meno i ~35 ms di una passata da un token, diviso la bozza media): **15 ms** con bozza
fissa 8, **20 ms** con bozza media 2.3, **27 ms** con bozza media 0.4. Misurato poi per zona
(§Revisione): **17.6 ms** se è la sola riga in più, **13.7 ms** l'una se sono otto, contro i 31.3
di una passata.

Letture:
- **Su un MoE una riga in più non è quasi gratis, come sarebbe su un modello denso.** Il decode
  costa perché legge i pesi degli 8 esperti scelti in ogni strato; il token in bozza ne sceglie
  altri, e la passata legge anche quelli. Una riga in più costa così mezza passata o più, e il
  pareggio sta al **44-56% di bozze accettate** (costo della riga / costo della passata, misurati
  per zona in §Revisione; la stima dai tok/s diceva 60-75%), non al 15% che diceva la misura fatta
  in container (LESSONS #59). È anche il motivo per cui la bozza lunga rende solo dove il testo è
  già nel prompt: lì gli esperti scelti sono gli stessi, perché i token sono gli stessi.
- **Accorciare la bozza non bastava**: con il 41% di accettate si perdeva ancora l'11%. Fermarla del
  tutto dopo una bozza tutta sbagliata (pausa 1, 3, 7, 15 passi, capped 16) porta il caso peggiore a
  **0.95×** (rimisura con A/A: −4.7%, sopra la soglia, 8 giri su 8; i 3 giri di prima dicevano
  1.01×) e lascia il caso buono a **1.175×**. «Come senza» **non è raggiunto**: accendere `--spec`
  di default vuol dire accettare −5% dove il modello inventa per +17.5% dove ricopia. La domanda
  25 resta chiusa (la pausa è misurata), la scelta del default è di Marcello.
- La bozza fissa resta la più veloce dove il modello ricopia (1.34×): chi sa che sta riscrivendo un
  file può ancora chiedere `--spec 8 --spec-fixed`.

## Kernel: one weight row vs 4 tokens (2026-09-17)

`make bench` (`tests/bench_kernels.c`, container, median of 5 × 100 ms, one thread), million
elements per second **per input row**:

| Kernel | n=1024 | n=2048 | n=4096 |
|---|---|---|---|
| dot_row q8_0 (una riga, un token) | 20 525 | 21 154 | 21 517 |
| dot_row q8_0 x4 (una riga, 4 token) | 41 467 | 44 169 | 36 898 |
| dot_f32 (riferimento in float) | 31 421 | 31 653 | 29 153 |

`tr_matmul` su una matrice 1024×2048 q8_0 (forma di una proiezione di esperto), ms per token:

| Thread | 1 token | 64 token |
|---|---|---|
| 1 | 0.100 | 0.059 |
| 4 | 0.027 | 0.016 |
| 8 | 0.014 | 0.008 |
| 16 | 0.014 | 0.009 |

Letture:
- Il kernel a 4 token vale **2.1×** sul prodotto scalare e batte anche `dot_f32`: il peso q8_0 si
  legge e si converte una volta per quattro prodotti, e occupa un quarto dei byte di un float.
- La matrice intera guadagna meno del kernel (1.7× a 1 thread): lì pesano anche i byte letti.
- Da 1 a 16 thread la stessa matrice rende 6.5×, non 16: a 16 thread la macchina è al limite di
  potenza (domanda 20).

## Speed levers: what research says (2026-09-17)

Numbers from sources, on other machines: point the direction, don't promise. Exact = same
numbers; declared = different numbers, difference to measure on real model before adopting.

| # | Leva | Tipo | Guadagno riportato | Per noi |
|---|---|---|---|---|
| 1 | SIMD sul prodotto riga × vettore | esatto | — | fatto: 10× sul kernel |
| 2 | Attivazioni in int8 + VNNI (`VPDPBUSD`), come llama.cpp per Q8_0 | dichiarato | llama.cpp usa questa strada per Q8_0 | alto: meno byte e meno lavoro per elemento |
| 3 | Prefill a blocchi (più token nella stessa moltiplicazione) | esatto | prefill lineare nel lotto ([discussione #18030](https://github.com/ggml-org/llama.cpp/discussions/18030)) | fatto con la 4: prefill 4.3× a 16 thread, decode invariato |
| 4 | Token raggruppati per esperto nel prefill | esatto | nessun numero isolato | fatto con la 3 |
| 5 | Pesi riordinati a blocchi di righe (repack 8×8) | esatto | +53% prefill, ~0% decode su Zen 5 ([PR #9532](https://github.com/ggml-org/llama.cpp/pull/9532)) | dopo 3-4 |
| 6 | Cache blocking (Goto/BLIS) | esatto | base di 3-5 ([Goto 2008](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)) | solo con più token insieme |
| 7 | tinyBLAS di llamafile | esatto | prefill 1.3-4×, decode poco ([PR #6414](https://github.com/ggml-org/llama.cpp/pull/6414)) | idee per 3-5 |
| 8 | Decodifica speculativa | esatto (greedy) | 2-3× ([arXiv 2211.17192](https://arxiv.org/abs/2211.17192)) | serve un modello bozza; per il coding si può usare il testo del prompt |
| 9 | Salto dell'esperto debole | dichiarato | 1.2-1.3×, -3 punti di qualità ([arXiv 2402.14800](https://arxiv.org/abs/2402.14800)) | solo opzionale, con misura di qualità |
| 10 | Sparsità delle attivazioni (PowerInfer) | dichiarato | 2.9× ([arXiv 2312.12456](https://arxiv.org/abs/2312.12456)) | serve un predittore addestrato per modello |
| — | Tabelle (T-MAC, bitnet.cpp), AMX, huge pages, NUMA | — | — | non per noi: pesi a 1-4 bit, solo Intel, carico di I/O, un solo socket |

Cache del 7940HX: L1d 32 KB e L2 1 MB per core, L3 32 MB per ciascuno dei due gruppi da 8 core.
Una riga di esperto (2 KB) e il vettore (8 KB) stanno già in L1: nel decode un peso si legge una
volta per token, e non c'è niente da tenere in cache. Il blocking conta quando più token
riusano le stesse righe (leve 3-6).

## Adversarial review (2026-09-18)

Re-read of entire repository by another model (Fable 5.1), instructed to bring evidence not
opinions. New errors and checks: LESSONS #60-#68. Here are the numbers. Container
`trochilus-dev`, other containers on: so comparisons are only those with effects well above
spread (kernel on one core, alternating runs) or **no stopwatch** (counters, replay). None of
what follows is a product speed measurement except the last section (**Native A/A remeasure**),
which is native and machine idle.

**Invarianti: reggono.** `trochilus logits` su 40-70 token, 3 tier (`TR_CPU_MAX` scalar, avx2,
avx512) × 3 numeri di thread (1, 5, 16) × 3 passate (`-b` 1, 7, 512): 27 file per modello,
identici al byte sulle righe confrontabili, per i tre modelli minuscoli (f32, f16, q8_0) e per
OLMoE vero tagliato a 2 layer (201 216 byte per riga). I test del modello passano anche sotto
`scalar` e `avx2`. Ora è un controllo: `make tier-check`. Tokenizer: 90 000 stringhe costruite
per rompere (tempeste di segni combinanti, jamo, ogni spazio Unicode, contrazioni, token aggiunti
incollati a segni e spazi) × 2 modi contro HF `tokenizers`: 0 differenze, e 12 000 decodifiche
uguali.

**Attivazioni int8, seconda misura** (domanda 21, LESSONS #65). `make bench`, sezione «one row,
by»: i quattro kernel girano alternati sulla stessa riga, 9 run da 40 ms, un core. Nanosecondi
per riga di input e rapporto col nostro kernel; il float x8 è verificato identico al bit al x4,
gli int8 contro il loro dot in C puro.

| kernel | n=1024 | n=2048 | n=4096 |
|---|---|---|---|
| `dot_row q8_0 x4` float, AVX-512 (il nostro) | 25.6 ns | 47.6 ns | 114.1 ns |
| int8 VNNI x4, 256 bit, trucco del segno di ggml | 1.64× | 1.62× | 1.96× |
| int8 VNNI x4, 512 bit, due blocchi per istruzione | **1.83×** | **1.79×** | **2.26×** |
| float x8, esatto (la strada «blocchi più larghi») | 1.13× | **0.79×** | 0.93× |

Spread 3-8% per riga; con clang 1.78× / 1.84× / 2.13× e 1.15× / 0.79× / 0.91×. Letture:
- Il prefill è al 90% moltiplicazioni e llama.cpp è avanti 1.57× a un thread: è l'int8, quasi
  per intero. Il float x8 a n=2048 (la larghezza di OLMoE) **perde**: 8 righe di attivazioni
  float sono 64 KB e la L1 ne tiene 32; in int8 sono 16 KB.
- Limite della misura: è un kernel su un core con tutto in cache; sul prefill intero il guadagno
  sarà più basso (resta il 10% non matmul, e la quantizzazione delle attivazioni). E l'int8
  **non è esatto**: i logit del prefill non sarebbero più quelli del token per token. Può
  esistere solo come modo dichiarato; la decisione è di Marcello.

**Quanto legge una riga di bozza** (domanda 12, LESSONS #67). Contatore `weight_bytes` del
profiler diviso per le passate: deterministico, modello vero, 200 token, `generate --spec k`.

| prompt | bozza | righe/passata | MiB di pesi/passata | MiB per riga in più | esperti nuovi per layer (su 8) |
|---|---|---|---|---|---|
| tutti | 0 | 1.00 | 1200.4 | — | — |
| `code-edit` | fissa 1 | 2.00 | 1675.8 | 475 | 4.7 |
| `code-edit` | fissa 8 | 9.00 | 3221.3 | 253 | 2.5 |
| `code-edit` | fissa 15 | 16.00 | 3891.0 | 179 | 1.8 |
| `code-edit` | adattiva | 3.34 | 2001.0 | 342 | 3.4 |
| `code` | fissa 1 | 1.46 | 1404.9 | 445 | 4.4 |
| `code` | fissa 8 | 4.04 | 1916.7 | 236 | 2.3 |
| `code` | adattiva | 1.38 | 1337.6 | 361 | 3.5 |

Un esperto sono 6.375 MiB (tre matrici Q8_0 da 2048×1024). A 36 GB/s, 253 MiB sono 7 ms dei 15
di una riga in più: la lettura degli esperti nuovi spiega **metà** del costo, non tutto. Nello
stesso profilo (tempi in container, solo indicativi) una riga in più costa 4-5.6 ms anche nelle
zone dense (`qkv_proj`, `attn_out_proj`, `lm_head`), dove non c'è nessun peso nuovo da leggere.
La parte a tempo è misurata nativa più sotto (**Quanto costa una riga di bozza, a zone**): negli
esperti sta il 61-91% del costo, e le zone dense pesano solo con le bozze lunghe.

**Le costanti della pausa, senza cronometro** (LESSONS #60, #67). Che una bozza venga accettata
dipende solo dai token, e i token sono gli stessi con ogni politica: ogni politica si può
rigiocare sulla continuazione registrata. Il replay riproduce alla riga i contatori del motore
(77 passate, 180 bozze, 125 accettate su `code-edit`; 172, 65, 28 su `code`). Tempo = passate ×
35 ms + righe di bozza × C; guadagno su `--spec 0` per C = 15 e 20 ms:

| politica (bozza 8) | `code-edit` passate / bozze | C=15 | C=20 | `code` passate / bozze | C=15 | C=20 |
|---|---|---|---|---|---|---|
| fissa | 33 / 264 | 1.37× | 1.09× | 143 / 435 | 0.61× | 0.51× |
| adattiva, solo si accorcia | 66 / 189 | 1.36× | 1.15× | 155 / 109 | 0.99× | 0.92× |
| pausa 1, 3, 7, 15, 16 (oggi) | 77 / 180 | 1.30× | 1.11× | 172 / 66 | 1.00× | 0.95× |
| pausa 1, 3, 7, 15, 31, 16 (l'errore #60) | 77 / 180 | 1.30× | 1.11× | 172 / 65 | 1.00× | 0.96× |
| pausa 1, 2, 4, 8, 16 | 71 / 183 | 1.34× | 1.14× | 165 / 83 | 1.00× | 0.94× |
| pausa 2, 5, 11, 16 | 78 / 169 | 1.33× | 1.15× | 168 / 65 | 1.02× | 0.97× |
| pausa sempre 4 | 77 / 168 | 1.34× | 1.16× | 173 / 68 | 0.99× | 0.94× |

Letture: le costanti spostano il 2-3%, meno di quanto una misura a tempo su questa macchina
possa vedere; restano quelle. Il caso peggiore «1.01×» misurato nativo su 3 giri corrisponde a
C ≈ 15 ms; con C = 20-22 (il valore che la stessa sezione ricava per le bozze corte) il replay
dà 0.94-0.96×. La domanda «`--spec` acceso di default» va decisa su una misura con più giri e
un controllo A/A (`tools/ab_modes.sh`, LESSONS #66): fatta, è il blocco **Rimisura nativa** più
sotto, e dà **0.953×**.

**Il pool con più di un pool** (LESSONS #61-#63). Programma di prova, Linux: `H1` chiamante
prima `[0-31]`, dopo due pool creati e distrutti nell'ordine di nascita `[0,1]`; `H2` processo
ristretto a `[0,2]`, pool da 4: worker 2 e 3 su `[0]`; `H3` worker 1 di due pool vivi entrambi
su `[2,3]`; `H4` pool interno da 2, id del worker arrivato al body: 3. H1 e H2 corretti e sotto
test; H3 e H4 restano un debito dichiarato in `threads.h`.

**Rimisura nativa con controllo A/A** (LESSONS #66 e #67; `sh tools/remeasure.sh`, ogni run in
`build/remeasure/`). Windows nativo, macchina ferma: lo script ferma i 9 container degli altri
progetti e li riavvia alla fine. Binario del commit 87297b4, OLMoE-1B-7B Q8_0, 16 thread dove non
è detto altro. `tools/ab_modes.sh`: 8 giri dopo uno di riscaldamento, il primo modo del giro
ruota, un modo dato due volte fa da controllo A/A. Mediana (min-max), rapporto col primo modo.

| misura | modo | prefill tok/s | | decode tok/s | |
|---|---|---|---|---|---|
| 1. `--spec`, caso peggiore: `run -f code.txt -n 200` | `--spec 0` | 215.1 (210.1-221.9) | — | 32.67 (31.93-33.00) | — |
| | `--spec 0`, copia A/A | 217.6 (212.2-227.4) | 1.012× | 32.39 (31.72-33.26) | 0.991× |
| | `--spec 8` | 219.2 (211.6-227.6) | 1.019× | 31.15 (30.68-31.44) | **0.953×** |
| 2. `--spec`, caso buono: `run -f code-edit.txt -n 200` | `--spec 0` | 229.2 (221.2-232.5) | — | 31.87 (31.06-32.22) | — |
| | `--spec 8` | 229.4 (222.8-232.8) | 1.001× | 37.45 (36.98-38.09) | **1.175×** |
| 3. pin: `generate -p 512 -n 24` | al core (2, il default) | 242.9 (234.7-246.7) | — | 31.26 (30.54-32.02) | — |
| | al core, copia A/A | 241.0 (235.3-246.1) | 0.992× | 31.11 (30.24-31.58) | 0.995× |
| | al processore (1) | 223.3 (185.3-240.4) | **0.919×** | 30.74 (29.63-31.35) | 0.984× |
| | nessuno (0) | 190.8 (187.7-193.1) | **0.785×** | 31.82 (30.98-32.38) | 1.018× |
| 4. thread: `generate -p 512 -n 48` | 16 | 242.5 (236.3-247.0) | — | 31.70 (30.37-31.89) | — |
| | 16, copia A/A | 240.9 (236.9-243.2) | 0.994× | 31.00 (30.39-32.10) | 0.978× |
| | 8 | 170.9 (165.5-172.6) | **0.705×** | 34.67 (33.94-35.04) | **1.094×** |

**Il rumore A/A della sessione** (due modi identici, differenza fra le mediane): decode 0.9%,
0.5%, 2.2%; prefill 1.2%, 0.8%, 0.6%. Lo stesso decode a 16 thread dà 0.5% nel blocco 3 e 2.2%
nel blocco 4: una coppia A/A sola è essa stessa una misura rumorosa. La soglia usata qui è il
**peggiore A/A della sessione per quella fase** (decode 2.2%, prefill 1.2%), e una differenza
conta solo se la supera contro **tutte e due** le copie del controllo. Sotto: «non distinguibile».

Letture:
- **Caso peggiore di `--spec`: 0.95×, non 1.01×.** −4.7% contro `--spec 0` e −3.8% contro la sua
  copia (soglia 2.2%), intervalli disgiunti, `--spec 8` sotto tutte e due in 8 giri su 8.
  Contatori: 172 passate, 28/66 bozze accettate (42.4%), bozza media 0.38. La pausa porta la
  perdita da 0.60× a 0.95×, non la toglie: la condizione «caso peggiore come senza» **non è
  soddisfatta**. Sul prefill +1.9% e +0.8% contro le due copie: non distinguibile.
- **Caso buono: 1.175×** (era 1.15× su 3 giri), intervalli disgiunti; 77 passate, 125/180 bozze
  accettate (69.4%), bozza media 2.34. Prefill 1.001×: non distinguibile.
- **Pin al core contro nessun pin: prefill 1.27×** (1.26× dalla copia), intervalli disgiunti.
  Decode −1.8% e −2.2% dalle due copie, a cavallo della soglia: **non distinguibile**.
- **Pin al core contro pin al processore: la differenza è nel prefill, non nel decode.** Il
  prefill del pin al processore sta a −8.1% e −7.3% (soglia 1.2%), sotto il pin al core in 8 giri
  su 8, ed è instabile: spread 24.7% (185-240) contro 5.0%. Decode −1.6% e −1.2%: **non
  distinguibile**. Il «−17-18% sul decode» del pin al processore (3 giri, §Il pin dei thread)
  **non si riproduce**: contro nessun pin è −3.4% (sopra la soglia, 7 giri su 8, intervalli
  sovrapposti). Il percorso a un pool e 16 thread è lo stesso nei due binari (il diff di
  `threads.c` in 87297b4 tocca il chiamante con più pool e i worker senza slot): la differenza
  non viene dal codice. La causa non è stata cercata; quella misura aveva 3 giri, ordine fisso e
  nessun A/A.
- **Il decode vuole 8 thread: 1.09× contro 16 e 1.12× contro la sua copia** (soglia 2.2%),
  intervalli disgiunti (33.9-35.0 contro 30.4-32.1), 8 giri su 8. Il prefill vuole 16: 1.42×
  contro 8. Conferma la premessa della domanda 26, che prima stava dentro lo spread.

**Quanto costa una riga di bozza, a zone** (domanda 12, la metà a tempo; LESSONS #67). Stessa
sessione: `generate --tokens <code-edit> -n 200 --profile-json`, tre modi alternati per 3 giri
(`--spec 0`, `--spec 1 --spec-fixed`, `--spec 8 --spec-fixed`). Millisecondi per passata, mediana
(min-max). «Dense» sono `qkv_proj`, `attn_out_proj`, `lm_head`; «esperti» sono `expert_gate_up`
ed `expert_down`. Il costo di una riga in più è la differenza con la passata da una riga dello
stesso giro, divisa per le righe in più. Senza A/A, ma i tre giri di ogni modo stanno entro lo 0.9%.

| righe per passata | ms per passata | dense | esperti | MiB di pesi |
|---|---|---|---|---|
| 1 | 31.27 (31.20-31.34) | 9.23 | 16.37 | 1200.4 |
| 2 | 48.84 (48.55-48.89) | 9.16 | 32.35 | 1675.8 |
| 9 | 141.06 (140.10-141.33) | 42.25 | 83.42 | 3221.3 |

| una riga in più costa | totale | esperti | dense | attenzione | MiB di esperti nuovi |
|---|---|---|---|---|---|
| se è la sola (2 righe) | **17.6 ms** (17.4-17.6) | 16.0 (91%) | −0.1 | 1.1 | 475 |
| se sono otto (9 righe) | **13.7 ms** (13.6-13.8) | 8.4 (61%) | 4.1 (30%) | 0.5 | 253 |

Letture:
- Una riga di bozza costa **più di mezza passata** quando è la sola (17.6 ms su 31.3) e 13.7 ms
  l'una quando sono otto. Prima era ricavato dai tok/s (15-27 ms, §Bozza adattiva); ora è misurato
  per zona. Il pareggio è costo della riga / costo della passata: **56% di bozze accettate** per le
  bozze corte, **44%** per quelle da otto. Torna coi due prompt: 42.4% accettate perde (0.953×),
  69.4% guadagna (1.175×).
- **Il costo sta negli esperti**: il 91% con una riga in più, il 61% con otto. Ai due punti
  misurati il tempo degli esperti per riga segue i MiB di esperti nuovi contati sopra (475 MiB in
  16.0 ms, 253 in 8.4: 29.8 e 30.2 MiB/ms). Il costo di una bozza si attacca dal lato dei pesi
  (quali esperti si leggono), non del calcolo: chiude la domanda 12. Limite: per MiB gli esperti
  nuovi costano 1.65 volte quelli della passata da una riga (816 MiB in 16.4 ms, 49.8 MiB/ms), e
  il perché non è misurato.
- **Le moltiplicazioni dense sono gratis per la prima riga in più** (−0.1 ms: il kernel a 4 token
  legge la riga di pesi una volta sola) e costano 4.1 ms per riga a nove righe: i 4-5.6 ms visti
  in container valgono per le bozze lunghe, non per le corte.
- **Il replay regge alla prova del cronometro**: passate × 31.3 ms + righe di bozza × 17.6 ms dà
  0.95-0.96× per la politica di oggi su `code` (172 passate, 66 righe, contro 199-200 passate
  senza bozza); la misura a tempo del blocco 1 dà 0.953× e 0.962× contro le due copie. Con gli
  stessi costi nessuna delle politiche della tabella del replay arriva a 1.00× su `code` (la
  migliore, pausa 2, 5, 11, 16, dà 0.97-0.98×): le costanti della pausa non bastano, la leva è il
  costo della riga.

## Threads per phase (2026-09-18)

Question 26. Native Windows, machine idle (`tools/threads_phase.sh` stops 9 containers of
other projects and restarts at end), OLMoE-1B-7B Q8_0, pinned to cores, `tools/ab_modes.sh`:
8 rounds after one warmup, first mode of round rotates, A/A control. Median (min-max), ratio
to first mode. Each run in `build/threads_phase/`.

**1. Quanti thread vuole ogni fase** (`sh tools/threads_phase.sh sweep`: un solo `-t` per tutte e
due le fasi, binario del commit b231b28, `generate -p <contesto> -n 48`).

| contesto | thread | prefill tok/s | | decode tok/s | |
|---|---|---|---|---|---|
| 512 | 16 | 231.2 (228.7-237.4) | — | 30.77 (29.31-31.67) | — |
| | 16, copia A/A | 233.4 (228.5-236.3) | 1.009× | 30.70 (30.39-31.12) | 0.998× |
| | 12 | 210.5 (204.5-212.2) | 0.910× | 31.50 (30.34-31.79) | 1.024× |
| | 8 | 168.4 (167.5-170.6) | 0.728× | 33.79 (33.36-34.67) | **1.098×** |
| | 4 | 92.1 (91.5-92.8) | 0.398× | 34.39 (33.34-34.97) | **1.118×** |
| 2048 | 16 | 186.5 (183.2-188.0) | — | 24.06 (22.19-24.45) | — |
| | 16, copia A/A | 187.2 (184.9-189.8) | 1.004× | 23.91 (23.42-24.51) | 0.994× |
| | 12 | 166.4 (162.3-170.2) | 0.892× | 23.20 (21.72-24.05) | 0.964× |
| | 8 | 134.9 (133.2-137.2) | 0.723× | 24.65 (24.17-25.12) | **1.025×** |
| | 4 | 74.3 (73.8-74.4) | 0.398× | 23.77 (23.32-24.27) | 0.988× |

Soglia, il peggiore A/A della sessione: decode 0.6%, prefill 0.9%.

- **Il prefill vuole tutti i thread**, a ogni contesto: 16 contro 8 fa 1.37× a 512 e 1.38× a 2048.
- **Il decode a contesto 512**: 8 e 4 thread battono 16 del 9.8% e dell'11.8% (10.1% e 12.0% contro
  la copia), con intervalli disgiunti da quello dei 16. Fra loro 4 è avanti dell'1.8%, con gli
  intervalli uno dentro l'altro (33.3-35.0 e 33.4-34.7). 12 rende il 2.4%.
- **A contesto 2048** 8 è il solo numero sopra la soglia contro tutte e due le copie (+2.5%, +3.1%).
  4 torna indietro (−1.2% e −0.6%: non distinguibile da 16, e −3.6% da 8) e 12 è il peggiore
  (−3.6%, −3.0%).
- **Scelta: n = 8**, il solo numero sopra la soglia a tutti e due i contesti. 4 vince di poco dove
  il contesto è corto e perde dove è lungo, e nel coding il contesto è lungo (LESSONS #70).
- Perché: il decode legge 1.2 GB di pesi per token e 4 thread riempiono già il bus (34 tok/s sono
  41 GB/s); oltre quel punto ogni thread in più aggiunge solo attesa alla barriera di ogni
  `parallel_for`. Col contesto cresce l'attenzione, che è calcolo e si divide per testa: lì i
  thread in più tornano a servire, e l'ottimo si sposta da 4-6 verso 8. (Corretto il 2026-09-19,
  §Decode a contesto lungo: il bus si riempie a ~54 GB/s, non a 41, e l'attenzione non era calcolo
  ma lettura della KV a salti; con la KV per testa a contesto 4000 la sessione sceglie 16 thread in
  13 run su 16.)

**2. Il decode a larghezza forzata** (motore nuovo, `--decode-threads n`: prompt sempre su 16
thread, passate corte sui primi n slot; contesto 512; sessione con la regola al 2%, che conta solo
per l'ultima riga). Il prefill sta fra 233.6 e 236.7 tok/s in tutti i modi (A/A 1.3%): la
larghezza del decode non lo tocca.

| decode su | decode tok/s | |
|---|---|---|
| 16 | 30.88 (29.75-31.60) | — |
| 16, copia A/A | 31.13 (30.56-31.75) | 1.008× |
| 12 | 31.63 (30.16-32.63) | 1.025× |
| 8 | 34.07 (33.54-34.81) | **1.103×** |
| 6 | 34.63 (31.41-35.19) | **1.122×** |
| 4 | 33.84 (33.29-34.67) | **1.096×** |
| misurata dalla sessione (8 in 7 run, 4 in una) | 33.82 (32.04-34.15) | **1.095×** |

Fra 4 e 8 thread la curva è piatta (33.8-34.6, differenze sotto la soglia della sessione, 1.4%) e
cade sopra gli 8. `-t 16 --decode-threads 8` va come `-t 8` (34.07 contro 33.79 dello sweep): i
worker lasciati fuori dormono e non costano.

**3. La regola del default è una misura, non un numero.** Quanti thread riempiono il bus è un
fatto della macchina (canali di memoria, banda per core), e da una macchina sola non si ricava
una formula che valga sulle altre: `min(core, 8)` e `core/2` danno tutti e due 8 qui, e sbagliano
in direzioni opposte su un portatile a 4 core e su una macchina a molti canali. Quindi ogni
sessione misura la sua: le prime 9 passate da un token girano a turno su tutto il pool, su metà e
su un quarto (16, 8, 4), tre volte ciascuna; tiene il tempo migliore di ogni larghezza e sceglie
la più ampia entro l'1% dalla più veloce; rimisura ogni 1024 token, perché l'ottimo si sposta col
contesto. Una passata è «corta» fino a 4 righe (quelle che il kernel a 4 token copre con una sola
lettura della riga di pesi); le altre usano tutto il pool. `--decode-threads n` forza la larghezza
e non misura niente. Token identici al bit per costruzione (il contratto del pool) e per prova:
`tests/test_phase.c`, `tests/test_base.c`, `make tier-check`.

Il margine è stato misurato tre volte, contando quale larghezza sceglie ogni run (colonna `width`
di `ab_modes.sh`, 16 run per contesto):

| margine | scelte a 512 | scelte a 2048 | decode dopo/prima a 512 | a 2048 |
|---|---|---|---|---|
| 3% | 8 in 15, 4 in 1 | **16 in 10**, 8 in 6 | 1.096× e 1.091× | 0.996× e 1.012× |
| 2% | 8 in 15, 4 in 1 | 16 in 5, 8 in 11 | 1.081× e 1.092× | 1.014× e 1.026× |
| **1%** (adottato) | 8 in 11, 4 in 5 | 16 in 2, **8 in 14** | 1.085× e 1.088× | **1.027× e 1.019×** |

A 2048 le run che scelgono 8 fanno 24.0-25.1 tok/s e quelle che scelgono 16 fanno 23.4-23.8: ogni
scelta sbagliata costa il 4%, e succede perché il tempo migliore di tre passate ha un rumore del
2% circa, quanto la distanza fra 16 e 8 a quel contesto (LESSONS #71). Con l'1% a 512 la scelta
cade su 4 una volta su tre: lì 4 e 8 vanno uguale, e la rimisura ogni 1024 token la riporta su 8
quando il contesto cresce. Uno stimatore più robusto è la domanda 31.

**4. `--spec 8` col motore nuovo** (`run -f <prompt> -n 200 --spec 8 -t 16`; «16 righe» è
`TR_DECODE_ROWS=16`: ogni passata di verifica sulla larghezza del decode, non solo quelle fino a 4
righe). Regola al 2%.

| prompt | modo | decode tok/s | |
|---|---|---|---|
| `code-edit` (la bozza viene accettata) | prima | 37.11 (36.63-37.51) | — |
| | prima, copia A/A | 37.03 (36.73-37.79) | 0.998× |
| | dopo | 37.91 (36.60-38.29) | 1.021× |
| | dopo, 16 righe | 37.75 (36.54-38.49) | 1.017× |
| `code` (il modello inventa) | prima | 31.01 (30.55-31.58) | — |
| | prima, copia A/A | 30.98 (30.13-31.41) | 0.999× |
| | dopo | 33.00 (32.55-33.41) | **1.064×** |
| | dopo, 16 righe | 33.05 (32.45-33.51) | **1.066×** |

- Dove la bozza viene accettata le passate hanno quasi tutte più di 4 righe e girano come prima:
  +2.1%, a cavallo della soglia. Dove il modello inventa le passate sono corte e prendono il
  guadagno del decode: **+6.4%**, intervalli disgiunti. In quel caso la sessione sceglie 4 thread
  in 8 run su 8.
- **Il confine a 4 o a 16 righe non si distingue** (−0.4% e +0.2%): resta a 4, che lascia le
  passate lunghe di verifica esattamente come erano (nessun rischio dove il calcolo conta di più).
- Il rapporto fra `--spec 8` e `--spec 0` sul caso peggiore non è stato rimisurato: tutti e due
  ora prendono il guadagno del decode stretto, il primo il 6.4% e il secondo il 9% circa.

**5. Prima e dopo** (`sh tools/threads_phase.sh change build/trochilus-before.exe`: il binario di
b231b28 contro il motore nuovo con la regola all'1%, tutti e due a `-t 16`, ognuno con la sua
copia A/A, più il motore nuovo con `--decode-threads 8`; `generate -p <contesto> -n 48`).

| contesto | binario | prefill tok/s | | decode tok/s | |
|---|---|---|---|---|---|
| 512 | prima | 235.3 (227.3-238.6) | — | 30.82 (30.03-31.29) | — |
| | prima, copia A/A | 233.8 (229.5-236.7) | 0.994× | 30.71 (29.76-31.64) | 0.996× |
| | dopo | 233.6 (227.0-236.1) | 0.993× | 33.45 (32.46-34.19) | **1.085×** |
| | dopo, copia A/A | 235.7 (230.7-239.1) | 1.002× | 33.53 (32.85-34.13) | **1.088×** |
| | dopo, `--decode-threads 8` | 236.3 (233.4-237.7) | 1.004× | 33.50 (31.91-34.41) | **1.087×** |
| 2048 | prima | 193.4 (189.7-194.0) | — | 23.79 (23.45-24.43) | — |
| | prima, copia A/A | 192.5 (191.3-195.8) | 0.995× | 23.55 (23.29-24.47) | 0.990× |
| | dopo | 192.6 (189.3-193.7) | 0.996× | 24.43 (24.09-24.79) | **1.027×** |
| | dopo, copia A/A | 192.8 (190.8-194.2) | 0.997× | 24.24 (23.34-25.04) | **1.019×** |
| | dopo, `--decode-threads 8` | 193.3 (192.2-194.1) | 0.999× | 24.98 (24.14-25.21) | **1.050×** |

Soglia della sessione: decode 1.0%, prefill 0.6%.

- **Decode a contesto 512: 1.085× e 1.088×** (1.089× e 1.092× contro la copia), intervalli
  disgiunti (32.5-34.2 contro 29.8-31.6). Il default misurato va come la larghezza forzata.
- **Decode a contesto 2048: 1.027× e 1.019×** (1.037× e 1.029× contro la copia), sopra la soglia
  contro tutte e due le copie; con 8 forzato **1.050×**. La metà che manca al default sta nelle 2
  run su 16 che scelgono 16 e nelle 9 passate di misura (3 su 16 thread e 3 su 4, tutte e due più
  lente di 8 a questo contesto) su 47.
- **Prefill: non distinguibile** a nessuno dei due contesti (da −0.7% a +0.4%).
- Su una risposta di 48 token la misura pesa; su una di 200 sono 9 passate su 200, e poi 9 ogni 1024.

## Decode at long context and RAM bandwidth (2026-09-19)

Questions 4 and 18, point 6 of next steps. Native Windows, machine idle, OLMoE-1B-7B Q8_0,
pinned to cores, all in **one window** (`sh tools/decode_context.sh change
build/trochilus-before.exe`, 105 minutes, each run in `build/decode_context/`): `tools/ab_modes.sh`,
8 rounds after one warmup, `generate -p <context> -n 48 -t 16`. Four contexts in **same session**,
16 modes in order putting each context after each other once (de Bruijn sequence): one run on
machine warmed by 4000-token prompt isn't always same context, and one mode and its A/A copy never
follow same context. «Before» = binary at commit 6734910; «After» = KV with one head's positions
in a row (below). Tables from `tools/decode_context_report.py speed | model | zones`.

**1. La banda della RAM** (`make bench-mem`, `tests/bench_mem.c`: 2 GiB letti dai thread del pool,
pinnati come nel motore; mediana di 7; due run nella stessa notte, a due ore di distanza, concordi
entro il 5-10%). GB/s:

| lettura | 1 thread | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| in fila | 24.8 | 47.8 | 57.6 | 52.7 | 53.0 | 52.5 | 53.0 |
| sparsa, blocchi da 2 MiB (una matrice di un esperto) | 24.6 | 49.1 | 57.0 | 56.2 | 54.7 | 53.2 | 52.0 |
| sparsa, blocchi da 256 KiB | 24.0 | 37.3 | 54.4 | 55.3 | 53.7 | 52.0 | 49.9 |
| sparsa, pagine da 4 KiB | 7.2 | 14.0 | 29.1 | 33.5 | 39.1 | 36.6 | 34.2 |
| il matmul del motore, Q8_0, 8 matrici da esperto a caso per chiamata | 19.1 | 32.9 | 58.1 | 53.0 | 46.3 | 50.4 | 50.0 |

- **Il tetto è ~54 GB/s** (52-58) e si raggiunge con 4 thread; un thread ne tira 22-25. Non sono
  gli 83 GB/s teorici della DDR5-5200 a due canali, e non sono i «41 GB/s» scritti finora: quelli
  erano la velocità del motore del 17/09 (senza pin, senza thread per fase), non un limite della RAM.
- **Leggere sparso non costa**, finché i pezzi sono grandi: blocchi da 2 MiB o da 256 KiB vanno come
  la lettura in fila. Gli esperti scelti a caso non pagano niente per essere sparsi. Costa leggere a
  pagine sparse (7 GB/s a un thread, 34-41 da 6 in su): è il caso della KV di prima, sotto.
- Il matmul del motore a un thread è limitato dal calcolo (19 GB/s); da 4 thread in su tira tutta
  la banda (46-58, la riga più rumorosa: spread fino al 30%).

**2. Un token di attenzione sui due modi di tenere la KV** (`bench_mem kv <posizioni>`: cache con la
forma di OLMoE, 16 layer × 16 teste × 128, f32, K e V, 1 GiB; fra un layer e l'altro 64 MiB di altra
memoria passano nelle cache, come i pesi nel motore; kernel vero `tr_attention_head`). «Righe» è il
modo di prima, `[posizione][testa]`: una testa legge 512 byte ogni 8 KiB, una pagina nuova a ogni
posizione. «Teste» è `[testa][posizione]`: una testa legge le sue posizioni in fila. 8 thread, ms per
token (GB/s di KV):

| posizioni | righe | teste | | teste, 16 thread | sola lettura degli stessi byte: righe / teste |
|---|---|---|---|---|---|
| 512 | 4.17 (32.1) | 2.90 (46.3) | **1.44×** | 2.69 (49.9) | 21.5 / 52.9 GB/s |
| 2048 | 15.93 (33.7) | 11.52 (46.6) | **1.38×** | 10.61 (50.6) | 20.0 / 54.1 GB/s |
| 4000 | 29.54 (35.5) | 22.01 (47.6) | **1.34×** | 20.97 (50.0) | 19.6 / 53.7 GB/s |

Gli stessi byte, letti a salti, passano a 20-22 GB/s; in fila a 53-54, cioè al tetto. Col kernel vero
la differenza è 32-35 contro 46-48 GB/s (nella prima run della notte a 4000 le righe facevano 28.0).

**3. Il decode ai quattro contesti, prima e dopo** (tok/s, mediana (min-max) di 8, poi dopo/prima
contro le due copie).

Con `--decode-threads 8` (`speed-d8`; soglia della sessione, il peggiore A/A: decode 3.8%, prefill 2.8%):

| contesto | prima | prima, copia A/A | dopo | dopo, copia A/A | dopo / prima |
|---|---|---|---|---|---|
| 32 | 38.00 (37.31-38.69) | 37.95 (35.65-39.01) | 38.30 (35.64-39.58) | 39.55 (37.20-40.09) | 1.008-1.042×: non distinguibile (A/A 3.3%) |
| 512 | 32.30 (31.15-34.16) | 33.52 (32.58-34.14) | 35.48 (34.25-36.49) | 35.47 (33.15-36.33) | **1.058-1.098×** |
| 2048 | 24.18 (23.68-24.61) | 24.30 (22.92-25.03) | 27.32 (25.34-27.88) | 27.62 (26.73-27.95) | **1.124-1.142×** |
| 4000 | 17.91 (15.66-18.50) | 18.16 (16.87-18.42) | 20.84 (19.83-21.33) | 21.04 (20.46-21.32) | **1.147-1.175×** |

Col default, la larghezza misurata dalla sessione (`speed-auto`; soglia decode 3.6%, prefill 4.8%;
sessione più rumorosa dell'altra, spread 9-29% contro 4-16%: l'altra finestra di Claude lavorava, senza
container; le mediane concordano con la tabella sopra):

| contesto | prima | prima, copia A/A | dopo | dopo, copia A/A | dopo / prima | larghezze scelte (dopo, 16 run) |
|---|---|---|---|---|---|---|
| 32 | 38.05 | 37.55 | 38.83 | 38.63 | 1.015-1.034×: non distinguibile | 4 in 13, 8 in 3 |
| 512 | 32.16 | 32.77 | 33.13 | 32.47 | 0.991-1.030×: non distinguibile | 4 in 8, 8 in 4, 16 in 4 |
| 2048 | 22.27 | 22.75 | 25.93 | 26.20 | **1.140-1.176×** | 8 in 10, 16 in 6 |
| 4000 | 17.09 | 16.86 | 19.44 | 20.13 | **1.137-1.195×** | 16 in 13, 8 in 3 |

Il prefill, negli stessi run (tok/s, `speed-d8`): a 512 **non distinguibile** (234.9 e 234.9 prima,
232.9 e 235.5 dopo); a 2048 **1.096-1.109×** (185.7 → 204.8); a 4000 **1.392-1.430×** (119.4 → 166.1).
Anche l'attenzione del prompt legge una testa alla volta: la sua zona passa da 17.8 a 8.5 s su 4000
token (dal 52% al 34% del prefill) e da 3.15 a 1.86 s su 2048. La scrittura della KV, ora testa per
testa, costa 25.7 ms su un prompt da 512 invece di 15.7 (domanda 30).

**4. L'ipotesi di STATO: regge la forma, non i numeri.** L'ipotesi era `tok/s = banda / (pesi + KV
per token × contesto)` con 41 GB/s, 1.2 GB di pesi e 262 KB di KV per token di contesto. Il decode è
davvero byte diviso banda, ma le bande **erano due**, e nessuna era 41:

| contesto (a metà risposta) | misurato prima | formula a 41 GB/s | byte al secondo, prima | misurato dopo | formula a 48.2 GB/s | byte al secondo, dopo |
|---|---|---|---|---|---|---|
| 56 | 37.97 | 33.11 (−13%) | 47.0 GB/s | 38.91 | 38.91 | 48.2 GB/s |
| 536 | 33.00 | 30.06 (−9%) | 45.0 | 35.48 | 35.32 (−0.5%) | 48.4 |
| 2072 | 24.19 | 23.21 (−4%) | 42.7 | 27.50 | 27.27 (−0.8%) | 48.6 |
| 4024 | 18.04 | 17.99 (−0.3%) | 41.1 | 20.91 | 21.15 (+1.2%) | 47.7 |

(`--decode-threads 8`, un modo e la sua copia insieme: 16 run per punto.)

- **Prima**: la retta per i quattro punti dice 26.2 ms a contesto zero e 7.29 ms ogni 1000 token di
  contesto, cioè i pesi letti a **47 GB/s** e la KV a **36 GB/s** (31.6 col default). La KV si
  leggeva più piano dei pesi, il contrario della prima ipotesi di STATO («la KV si legge a banda più
  alta»). La formula a 41 GB/s tornava a 4000 **per caso**: troppo bassa sui pesi, troppo alta sulla
  KV, i due errori si compensano solo lì.
- **Dopo**: 25.2 ms e 5.57 ms ogni 1000 token, pesi a 48.6 e KV a **47.1 GB/s**: una banda sola.
  `tok/s = 48.2 / (1.2236 + 0.000262 × contesto)` (GB) sbaglia al più dell'1.2% a ogni contesto.
- Su un modello con GQA la KV per token è più piccola e il calo col contesto si accorcia in
  proporzione; la formula resta quella.

**5. Dove va il token, per zona** (`tools/profile_suite.py` su `bench/scenarios-decode-context.json`,
mediana di 5, `--decode-threads 8`; il «prima» è il gemello del binario di prima compilato coi byte
per zona, stesso layout; i due profili non sono alternati, quindi fra prima e dopo contano solo le
differenze grandi). ms per token, e GB/s letti dentro la zona:

| zona | MiB per token | prima, a 32 / 512 / 2048 / 4000 | dopo, a 32 / 512 / 2048 / 4000 |
|---|---|---|---|
| `attention` | 14 / 134 / 518 / 1006 | 0.64 / 4.36 / 15.79 / **30.53** ms (22.9 / 32.3 / 34.4 / 34.6 GB/s) | 0.48 / 2.90 / 11.52 / **22.64** ms (30.5 / 48.4 / 47.1 / 46.6 GB/s) |
| `expert_gate_up` | 544 | 10.6-10.9 ms (52.2-53.6) | 10.5-10.8 ms (52.6-54.5) |
| `expert_down` | 272 | 5.4-5.6 ms (51.2-52.8) | 5.3-5.6 ms (51.3-53.3) |
| `qkv_proj` | 204 | 4.1-4.2 ms (50.6-52.3) | 4.0-4.2 ms (50.4-53.4) |
| `lm_head` | 104 | 2.0-2.1 ms (52.7-53.7) | 2.0-2.1 ms (53.2-54.8) |
| `attn_out_proj` | 68 | 2.0-2.2 ms (32.5-36.2) | 1.4-1.6 ms (43.2-52.0) |
| `router` | 8 | 0.24-0.26 ms | 0.23-0.24 ms |
| zone senza byte (norme, RoPE, scrittura KV, copia per esperto, attivazione, somma, scelta) | — | 1.06-1.10 ms | 1.09-1.15 ms |
| **token** | 1214 / 1334 / 1718 / 2206 | 26.8 / 29.6 / 41.4 / **56.3** ms (47.5 / 47.3 / 43.5 / 41.1 GB/s) | 25.0 / 27.5 / 36.4 / **48.1** ms (50.9 / 50.9 / 49.5 / 48.1 GB/s) |

- **Le moltiplicazioni sui pesi stavano già al tetto** (50-55 GB/s su 54): lì non c'è niente da
  prendere senza leggere meno byte. Sotto il tetto c'era **solo l'attenzione** (32-35 GB/s), e con
  lei `attn_out_proj`, la zona che viene subito dopo (32-36 GB/s: le letture a salti le lasciavano
  TLB e cache da rifare).
- **Dopo, l'attenzione legge a 47-48 GB/s**: a 4000 token −7.9 ms su 56.3.
- **Quanto manca al tetto della RAM**: il decode muove **48-51 GB/s su ~54**, il 6-11%. Quel che
  resta dal lato esatto è piccolo e sparso (domanda 35): 1.1 ms di zone senza byte, l'attenzione a 47
  invece di 52-56 (il calcolo vale il 12% della zona), `attn_out_proj`. Da qui in poi si va più
  forte solo leggendo **meno byte**.
- **8 thread o 16 per l'attenzione** (dopo, contesto 4000): su 16 la zona fa 21.6 ms invece di 22.6,
  ma esperti e proiezioni perdono 1.3 ms e il token va uguale (48.4 contro 48.1). Una larghezza per
  zona prenderebbe 1.0 ms su 48.1 (2.1%; a 2048 lo 0.7%): sotto la soglia di questa notte, non
  scritta (domanda 34).

**6. La leva esatta: le posizioni di una testa in fila** (`src/kv/kv.h`). La KV era
`[layer][posizione][testa]`: a ogni token generato ogni testa leggeva 512 byte ogni 8 KiB, una pagina
nuova a ogni posizione, e il prefetcher non aveva niente da seguire. Ora è
`[layer][testa][posizione]`: due flussi in fila per testa, le chiavi e poi i valori. Cambia dove sta
un numero, non il numero: stesse chiamate a `dot_f32` e `axpy_f32` sugli stessi float nello stesso
ordine. Prova: `tests/test_kv.c` (il layout e le scritture; 5 mutazioni su 5 viste dai test, tre da
`test_kv`, tutte dall'oracolo), `make check`, e sul modello vero lo stadio `exact` di
`tools/decode_context.sh`: logit identici al byte al binario di prima su 600 posizioni un token per
passata (120 MB) e a passate da 64 su 8 thread, token identici dopo un prompt da 4000.

**7. Le leve non esatte: i numeri per decidere.** Dopo la KV per testa l'attenzione legge alla banda
della RAM: per andare più forte a contesto lungo restano solo meno byte. Stime dai byte, con la zona
`attention` di oggi (2.90 / 11.52 / 22.64 ms a 512 / 2048 / 4000) e il 12% della zona che è calcolo e
non si dimezza (per gli 8 bit, fra il 12% e il doppio: c'è la decodifica):

| leva | byte di KV | decode a 512 | a 2048 | a 4000 | memoria della KV a 4096 |
|---|---|---|---|---|---|
| oggi, f32 (esatta) | 262 KB per token di contesto | 36.4 tok/s | 27.5 | 20.8 | 1.07 GB |
| KV a 16 bit | metà | ~38 (+5%) | ~32 (+16-19%) | ~26-27 (+26-31%) | 0.54 GB |
| KV a 8 bit (blocchi tipo Q8_0) | 27% | ~39 (+6-7%) | ~33-34 (+20-25%) | ~28-30 (+33-43%) | 0.28 GB |

Come si misurerebbe la qualità è scritto nella domanda 36 (KL e primo token contro il modo esatto sul
modello vero, token greedy uguali, nessun valore oltre il massimo dei 16 bit). Riferimento: llama.cpp,
che tiene la KV a 16 bit **e** le attivazioni a 8, sta a KL 9e-3 da noi. Non implementate: decide
Marcello. Il confronto col decode di llama.cpp a contesto lungo (domanda 19) va rifatto adesso: il suo
vantaggio lì era anche questo.

## Prefill on long prompts (2026-09-19)

Punto 4 dei prossimi passi, domande 7 e 30. Windows nativo, macchina ferma, OLMoE-1B-7B Q8_0, 16
thread, pin al core. Tre finestre di `tools/prefill_context.sh`: `measure` alle 05:27 (il binario del
commit 06f8e30 da solo: microbenchmark, velocità con A/A, profilo; 21 minuti), `bench` alle 07:55
(solo il microbenchmark, col kernel nuovo) e `change build/trochilus-before.exe` alle 10:13 (prima
contro dopo; 40 minuti). Ogni run sta in `build/prefill_context/`; le
tabelle escono da `tools/prefill_context_report.py attn | zones` e da
`tools/decode_context_report.py speed`. I prompt da 512, 2048 e 4000 stanno nella **stessa
sessione** di `tools/ab_modes.sh` (8 giri dopo uno di riscaldamento, `generate -p <prompt> -n 48 -t
16`), in un ordine che mette ogni lunghezza dopo ogni altra; un modo e la sua copia A/A non seguono
mai la stessa lunghezza.

**1. L'attenzione di un prompt, smontata** (`make bench-attn`, `tests/bench_attn.c`: un layer, 16
teste da 128, KV `[testa][posizione]`, passate da 512 token come nel motore, 64 MiB di altra memoria
letti fra una passata e l'altra; mediana di 5). «Una query alla volta» è il kernel del motore
(`tr_attention_head`): per ogni token i prodotti con tutte le sue chiavi, il softmax della riga, la
somma pesata dei valori. «A gruppi» è lo stesso calcolo con 16 query contro un blocco di 64 chiavi
(e poi di valori) mentre il blocco è in cache; «x4» aggiunge una query contro 4 chiavi nei registri.
Ogni variante a gruppi stampa l'hash di tutte le uscite del prompt: **gli stessi bit** del kernel del
motore, a ogni lunghezza. 16 thread e 16 teste, ms per layer (ns di thread per coppia
query-posizione). Tre run nella stessa giornata (05:27, 07:55, 10:43), le ultime due anche col kernel
scritto nel motore (`tr_attention_group`): a 512 e a 2048 concordano entro il 4-13%; a 4000 la prima
dice una cosa e le altre due un'altra (431.9 e 428.2 ms), e in tabella ci sono la prima e la terza:

| | prompt 512 | 2048 | 4000, run delle 05:27 | 4000, run delle 10:43 |
|---|---|---|---|---|
| una query alla volta, tutto | 5.95 (45.3) | 104.1 (49.6) | 518.2 (64.8) | 428.2 (53.5) |
| — senza il softmax | 2.56 | 44.2 | 342.0 | 184.5 |
| — solo i prodotti (legge K) | 1.27 (9.7) | 20.2 (9.6) | 79.1 (9.9) | 85.4 (10.7) |
| — solo il softmax | 4.27 (32.6) | 69.6 (33.2) | 266.7 (33.3) | 278.8 (34.8) |
| — solo la somma pesata (legge V) | 1.36 (10.4) | 19.3 (9.2) | 140.0 (17.5) | 82.6 (10.3) |
| a gruppi, tutto | 6.32 | 109.9 | 382.7 | 396.9 |
| a gruppi, senza il softmax | 2.60 | 35.6 | 141.9 | 156.7 |
| a gruppi + x4 (prototipo nel banco), tutto | 5.49 | 95.1 | 355.2 | 370.6 |
| a gruppi + x4, senza il softmax | 1.53 | 22.8 | 101.2 | 92.1 |
| **il kernel del motore, `tr_attention_group`**, contro «una query alla volta» della stessa run (07:55 / 10:43) | 1.06× / 0.95×: non distinguibile | **1.12× / 1.16×** | — | **1.15× / 1.15×** (376.4 e 373.3 ms; 1.38× sulla run delle 05:27) |

Per 16 layer il kernel di prima fa 95, 1665 e 6850-8291 ms: la zona `attention` del profilo ne
misura 116, 1883 e 7420. Un thread solo con una testa sola (le cache tutte per sé) fa **42.7, 43.0
e 42.6 ns per coppia**: a un thread la lunghezza non conta, e il tempo è 73% softmax, 14% prodotti,
12% somma pesata; lì il kernel del motore vale 1.05-1.09× (è il solo x4). Gruppo da 4, 16 o 64 query
e blocco da 16, 64 o 256 posizioni a 4000: 352-387 ms, tutti dentro lo spread (2-15%): si tiene 16 × 64.

**2. L'ipotesi: vera a metà.** L'ipotesi era: «a 4000 token l'attenzione del prompt vale il 34% e
legge 2 TB in 8.5 s, quasi tutto da cache: ogni token rilegge le stesse chiavi e gli stessi valori».
Il 34% c'è (31-35% in due profili) e i 2 TB anche (283 GB/s nella zona: cache, non RAM). Ma la
lettura ripetuta **non è il grosso**:

| della zona `attention`, 16 thread | prompt 512 | 2048 | 4000, run delle 05:27 | 4000, run delle 07:55 e delle 10:43 |
|---|---|---|---|---|
| softmax (`expf`, massimo, somma, divisione) | 72% | 61-70% | 51% | 63-65% |
| prodotti e somma pesata, con chiavi e valori in cache | 28% | 30-34% | 27% | 36-37% |
| chiavi e valori riletti a ogni token | niente | 0-8% | **21-39%** | **0-6%** |

(Il softmax è la sua riga; la lettura ripetuta è ciò che resta tolti softmax e «a gruppi senza
softmax», oppure «una query alla volta» meno «a gruppi» senza softmax: le fasi misurate da sole non
si sommano esattamente, lo spread a 16 thread arriva al 20%.) Fino a 2048 token chiavi e valori di
una testa (2 MB) stanno nelle cache e rileggerli non costa. A 4000 sono 4 MB per testa: 64 MB per i
16 thread contro 64 MB di L3, **proprio sul bordo**, e quanto si paga dipende da cos'altro sta in L3
in quel momento: un quarto della zona nella prima run, quasi niente nelle altre due (il motore, che
fra un'attenzione e l'altra fa passare i pesi, sta in mezzo: 464 ms per layer). A gruppi il tempo
**non dipende più da questo**: 355, 349 e 371 ms nelle tre run. Oltre i 4000 token il bordo è
superato per tutti (domanda 38).

Il grosso è il **softmax**, cioè `expf` della libreria C: **30 ns a chiamata** con MinGW
(`tests/bench_expf.c`; istruzioni x87), contro **2.3 ns** di glibc nel container, sulla stessa
macchina. Lo stesso `expf` è quasi tutta la zona `expert_act` (il 5-7% del prefill, 39 ns per
elemento). È la sola funzione della zona calda che non è nostra: vedi il punto 6.

**3. Il prefill prima, e la sua parte seriale** (binario del commit 06f8e30; tok/s, mediana
(min-max) di 8; zone dal profilo, mediana di 5, `bench/scenarios-prefill-context.json`):

| | prompt 512 | 2048 | 4000 |
|---|---|---|---|
| prefill, tok/s | 231.7 (230.2-235.7) | 203.9 (200.7-206.9) | 169.5 (147.2-173.0) |
| la sua copia A/A | 233.3 (0.7%) | 202.9 (0.5%) | 168.1 (0.8%) |
| tempo del prompt, ms | 2175 | 10157 | 23639 |
| `attention` | 116 (5.3%) | 1883 (18.5%) | 7420 (31.4%) |
| moltiplicazioni (esperti, q/k/v, uscita, `lm_head`) | 1682 (77%) | 6767 (67%) | 13291 (56%) |
| `expert_act` (`expf`) | 161 (7.3%) | 656 (6.5%) | 1295 (5.4%) |
| `router` | 80 (3.6%) | 324 (3.2%) | 627 (2.6%) |
| **lavoro di un token solo, su un thread**: norme, RoPE, scrittura della KV, righe per gli esperti, embedding | **121 (5.6%)** | **455 (4.5%)** | **880 (3.7%)** |
| — di cui scrittura della KV testa per testa | 27.4 | 85.7 | 160.5 |
| — di cui righe copiate per gli esperti | 38.4 | 142.4 | 281.5 |
| — di cui norme di q e k / norme dei layer / RoPE | 21.4 / 20.5 / 12.7 | 87.9 / 83.2 / 52.1 | 172.9 / 156.4 / 102.3 |

La parte seriale non è solo quella: dentro `router` la scelta degli esperti token per token gira
su un thread, e lo scenario a **un thread** lo dice (prompt 512: `router` 415 ms a un thread, 80 a
16: circa 57 ms non si dividono). E l'altra metà della zona era un kernel mancante: la matrice del
router è **F32**, e le righe F32 giravano sul ciclo scalare in ogni tier (0.11 GB/s nella zona;
LESSONS #78). In tutto il lavoro su un thread solo valeva l'**8%** del prefill a 512 e il 5-6% a 4000.

**4. Le tre leve esatte.** Ognuna cambia dove e quando si fa un calcolo, mai il calcolo: stesse
chiamate di kernel sugli stessi float nello stesso ordine per ogni uscita.

- **Attenzione a gruppi** (`tr_attention_group`, `src/kernels/kernels.c`): 16 token consecutivi di una
  testa sono un gruppo; un blocco di 64 posizioni incontra tutte le query del gruppo mentre sta in
  cache, 4 posizioni per carico della query (`dot_f32_x4`) e 4 valori per carico dell'uscita
  (`axpy_f32_x4`); poi il softmax di ogni riga; poi i valori, allo stesso modo. Chiavi e valori si
  leggono una volta per gruppo: il profiler conta **31 MiB di KV per token a 4000 invece di 500**
  (a 2048: 16 invece di 256). Il decode è un gruppo da una query: un solo percorso.
- **Il lavoro di un token solo sul pool**: embedding, norme, norme di q e k, RoPE, scrittura della
  KV, somma del residuo, scelta del router (uno scratch per worker), righe per gli esperti; almeno 8
  token a pezzo, quindi una passata corta resta sul thread che chiama, come prima.
- **Le righe F32 col kernel del tier**: la matrice del router non passa più dal ciclo scalare.

Prova: `tests/test_kernels.c` (x4 contro scalare e contro 4 chiamate, 525 gruppi contro
`tr_attention_head` query per query), `test_prefill` con un prompt da 141 token, `test_hot` sotto
ThreadSanitizer, 20 mutazioni su 20 viste (`tools/mutate_prefill.sh`), `make check`; e sul modello
vero lo stadio `exact`: logit **identici al byte** al binario di prima su 600 posizioni un token per
passata (120 MB, 16 thread), a passate da 64 su 8 thread, su un prompt da 4000 a passate da 512 e da
100 su 16 thread (l'ultima riga uguale nelle due forme), e gli stessi token dopo un prompt da 4000.

**5. Prima e dopo** (`sh tools/prefill_context.sh change build/trochilus-before.exe`; tok/s, mediana
(min-max) di 8; soglia della sessione, il peggiore A/A: **prefill 2.1%, decode 2.4%**; «dopo» è la
seconda copia del binario nuovo, LESSONS #81, hash in `build/prefill_context/binaries.sha256`):

| prefill | prima | prima, copia | dopo | dopo, copia | dopo / prima |
|---|---|---|---|---|---|
| prompt 512 | 228.7 (222.5-236.1) | 230.2 | 240.9 (222.2-247.2) | 246.1 | **1.047-1.076×** |
| 2048 | 197.8 (184.5-201.2) | 197.6 | 211.3 (202.1-219.4) | 213.3 | **1.069-1.079×** |
| 4000 | 160.1 (156.9-164.1) | 161.2 | 181.7 (178.8-182.5) | 179.6 | **1.114-1.135×** |

Il decode dopo quei prompt **non è distinguibile**: 33.0 contro 33.1 tok/s a 512 (1.000-1.013×),
25.7 contro 25.6 a 2048 (0.973-1.005×), 19.9 contro 19.9 a 4000 (0.972-1.001×). (In questa sessione
il binario di prima fa 160 tok/s a 4000 dove al mattino ne faceva 169: la macchina era appena uscita
da due ore di standby, LESSONS #82. Conta il confronto dentro la sessione.)

Per zona (profilo del mattino contro profilo del binario nuovo, ms del prompt):

| | 512: prima | dopo | 2048: prima | dopo | 4000: prima | dopo |
|---|---|---|---|---|---|---|
| `attention` | 115.9 | 106.2 | 1883 | 1688 | 7420 | 6568 (**1.13×**) |
| `router` | 79.7 | **6.2** | 323.5 | **21.8** | 626.9 | **45.2** |
| lavoro di un token solo | 121.2 | 48.2 | 454.6 | 174.2 | 880.1 | 353.1 |
| — norme di q e k, norme dei layer, RoPE | 54.6 | 8.9 | 223.2 | 34.1 | 431.6 | 70.2 |
| — righe per gli esperti | 38.4 | 19.5 | 142.4 | 67.4 | 281.5 | 133.2 |
| — scrittura della KV | 27.4 | 19.3 | 85.7 | 71.8 | 160.5 | 147.4 |
| tutto il prompt | 2175 | 2071 | 10157 | 9503 | 23639 | 22044 |

Le norme e la RoPE, che sono calcolo, si dividono per 6; le righe per gli esperti e la scrittura
della KV, che sono copie in memoria, solo per 2 e per 1.1-1.4: lì il limite è scrivere in RAM, non
il thread. Il `router` va 13 volte più veloce: 57 ms erano la scelta su un thread, il resto il
kernel scalare. Le moltiplicazioni non sono state toccate (nel profilo del dopo valgono il 2-3% in
più: un'altra sessione, non un effetto).

Dopo, a 4000 token l'attenzione è il **30%** del prefill e dentro c'è quasi solo il softmax (4.3 s
su 6.6): dal lato esatto nell'attenzione resta poco. Il prossimo pezzo è `expf` (punto 6).

**6. Le leve non esatte: i numeri per decidere.** Non implementate.

- **`expf` nostro** (domanda 37). Oggi `expf` è della libreria C: 30.2 ns a chiamata su Windows
  (MinGW), 2.3 ns nel container (glibc). `tests/bench_expf.c` lo confronta, su **tutti i 4 278 190 082
  float** che non sono NaN, con l'esponenziale in doppia precisione arrotondato a float, che è il
  valore arrotondato correttamente (salvo una manciata di casi su 4 miliardi):

  | libreria | costo di `expf` | risultati diversi dall'arrotondamento corretto | nel campo del softmax (−104…0) |
  |---|---|---|---|
  | MinGW-w64 (Windows nativo) | 30.2 ns | **0** | **0** |
  | glibc 2.39 (container) | 2.3 ns | 170 648 (0.004%), sempre di un'unità sull'ultima cifra | 97 052 (0.009%) |

  Due conseguenze. (a) **Oggi Windows e Linux non danno gli stessi bit**: su un prompt da 4000 il
  softmax chiama `expf` due miliardi di volte, e glibc ne arrotonda male circa una su diecimila. (b) Un `expf`
  scritto da noi che arrotonda correttamente (C scalare come definizione, varianti AVX identiche al
  bit, come ogni altro kernel) darebbe **su Windows gli stessi bit di oggi**, per prova esaustiva
  (lo stesso `bench_expf` col nostro al posto del riferimento: 0 differenze su 2^32), e su Linux
  cambierebbe lo 0.004% delle chiamate di un'unità sull'ultima cifra, rendendo le due piattaforme
  identiche. «Cambia i bit» vale quindi solo per Linux, e lì in meglio. Guadagno stimato col softmax
  a 2-3 ns per elemento invece di 32 e `expert_act` a un decimo:

  | | prompt 512 | 2048 | 4000 |
  |---|---|---|---|
  | softmax dell'attenzione, ms del prefill | 83 | 1262 | 3821 |
  | `expert_act`, ms | 161 | 656 | 1295 |
  | prefill stimato, sul binario di prima | **1.11×** | **1.21×** | **1.25×** |
  | prefill stimato, sul binario di dopo (2071 / 9503 / 22044 ms) | **1.12×** | **1.23×** | **1.27×** |
  | decode stimato (softmax + `expert_act` per token: 1.4 / 2.8 / 4.5 ms su 27.9 / 36.4 / 47.5) | +5% | +8% | +10% |

  Come si misurerebbe la qualità: `bench_expf` sulle due piattaforme (0 differenze su Windows è la
  condizione per chiamarla esatta); stadio `exact` di `tools/prefill_context.sh` prima contro dopo
  su Windows (atteso: identico al byte); su Linux KL e primo token contro il binario di prima su
  1000 posizioni e oracoli invariati. Il rischio è nello scriverlo (un `expf` arrotondato
  correttamente vuole una strada lenta per i casi vicini a metà fra due float), non nel misurarlo.
- **KV a 16 bit**: al prefill non serve più. Con l'attenzione a gruppi chiavi e valori si leggono una
  volta ogni 16 token, e la lettura ripetuta era il solo pezzo della zona legato ai byte. Resta una
  leva del **decode** a contesto lungo (domanda 36: +16-31% a 2048-4000).
- **int8/VNNI nelle moltiplicazioni** (domande 21 e 33): le moltiplicazioni sono il 77% del prefill
  a 512 e il 56% a 4000; col kernel a 1.8-2.3× (§Attivazioni int8) il prefill andrebbe **1.5-1.8×** a
  512 e **1.3-1.5×** a 4000. Non è esatto: la decisione resta quella del punto 3 di STATO.

**7. `expf` nostro: preparato, non scritto** (domanda 37; decide Marcello). **Poi scritto, lo stesso
giorno: lo scalare, coi numeri misurati, sta in §`tr_expf`.** I tre numeri chiesti allora:

- **Il test sui 2^32 float è pronto**, e ha già girato su un prototipo. `make bench-expf` confronta
  un candidato con l'`expf` della libreria e col riferimento su tutti i float; il candidato si dà
  alla compilazione (`EXTRA_CFLAGS=-DTR_EXPF_CANDIDATE=tr_expf` quando il motore ne avrà uno). Senza,
  prova uno schizzo che sta **nel banco, non nel motore**: 25 righe di C scalare, tabella di 64
  valori di 2^(j/64), polinomio di grado 5 in double, e un test di arrotondamento (se un errore di
  2^-50 non può spostare il float, il risultato è certo; altrimenti strada lenta).

  | | costo a chiamata | diversi dalla libreria | diversi dal riferimento | strada lenta |
  |---|---|---|---|---|
  | Windows (MinGW): libreria / schizzo | 30.2 ns / **3.7 ns** | **0 su 4 278 190 082** | 0 | 8 argomenti |
  | Linux (glibc): libreria / schizzo | 2.5 ns / 3.8 ns | 170 648 (dove glibc sbaglia) | 0 | 8 argomenti |

  Il riferimento stesso è provato: `bench_expf --hard` scrive i 369 argomenti il cui esponenziale
  in double cade entro 2^-45 da un confine di arrotondamento (gli stessi sulle due piattaforme), e
  `tools/expf_hard_cases.py` li ricalcola a 200 bit: **0 errori**. Quindi l'`expf` di MinGW è
  arrotondato correttamente su tutti i float, e «zero differenze dalla libreria» è una prova, non
  un campione. Lo schizzo scalare è già 8 volte più veloce della libreria su Windows; la versione
  AVX-512 (8 double per registro) è stimata a 0.8-1 ns per elemento.
- **La KL su Linux contro il binario di prima** (`tools/expf_quality.sh` nel container: il motore
  compilato con ogni `expf` sostituito dal valore arrotondato correttamente, per emulazione, contro
  il binario normale; modello vero, prompt `code-edit.txt` più 1000 token generati, 1411 posizioni a
  un token per passata): nessuna riga di logit identica al byte, differenza massima fra due logit
  **2.3e-5** (su un campo di 65.8), **KL media 3.9e-13, massima 1.5e-11**, **0 posizioni** che
  scelgono un altro token, **1000 token su 1000** uguali nella generazione greedy. (llama.cpp sta a
  9e-3 da noi: dieci ordini di grandezza sopra.) Su Windows non c'è niente da misurare: gli stessi bit.
- **Il costo di scriverlo**: circa **450 righe di C e 40 di Python**, una sessione di lavoro.

  | pezzo | righe | difficoltà |
  |---|---|---|
  | `tr_expf` scalare, la definizione: casi speciali, riduzione esatta (ln2/64 in due pezzi), polinomio, scala, test di arrotondamento | 70 | media: è lo schizzo, con la tabella da costanti e non da `exp2` |
  | tabella di 64 double arrotondati correttamente, generata con mpmath | 64 + 40 di Python | bassa |
  | strada lenta: gli argomenti che non passano il test sono **8 su 4 miliardi**, quindi una tabella di eccezioni verificata a 200 bit (o 40 righe di double-double) | 20 | bassa, ma va riprovata per esaustione a ogni modifica |
  | varianti AVX2 (4 double) e AVX-512 (8 double) identiche al bit allo scalare: gather dalla tabella, scala con interi, maschera per la strada lenta, code | 140 | medio-alta: è il pezzo delicato |
  | voce `exp` nella tabella dei kernel, softmax e SwiGLU che la usano (uno scratch sullo stack per l'attivazione) | 60 | media |
  | test: tier contro scalare su ogni lunghezza e sui valori speciali, variante sbagliata che il test deve vedere, `bench-expf` col candidato dentro `make check` (15 s: la prova esaustiva a ogni cancello) | 110 | bassa |

  Il rischio non è la qualità (la prova è esaustiva e dura 15 secondi) ma il tempo del pezzo SIMD. Cosa
  si guadagna è al punto 6: prefill 1.12× / 1.23× / 1.27×, decode +5-10%, e Windows e Linux con gli
  stessi bit.

**8. Come si rifà.** Ogni lunghezza è uno scenario di `bench/scenarios-prefill-context.json` (512,
2048 e 4000 a 16 thread, 512 a un thread) e ogni misura è un comando:

| misura | comando | dura |
|---|---|---|
| l'attenzione smontata (punti 1-2) | `make bench-attn`, o a macchina ferma `sh tools/prefill_context.sh bench` | 3 minuti |
| `expf`: libreria, candidato, tutti i 2^32 float (punti 6-7) | `make bench-expf`, nativo e nel container; `build/tests/bench_expf.exe --hard <file>` e `tools/expf_hard_cases.py <file>` per i casi al confine | 15 s |
| cosa cambierebbe un `expf` corretto su Linux (punto 7) | `tools/expf_quality.sh` nel container, col volume dei modelli | 6 minuti |
| velocità con A/A e zone di un binario (punto 3) | `sh tools/prefill_context.sh measure` | 21 minuti |
| logit al byte, prima contro dopo con A/A, banco, zone (punto 5) | `sh tools/prefill_context.sh change <binario prima>` | 40 minuti |
| solo le zone | `make profile SCENARIOS=bench/scenarios-prefill-context.json` | 6 minuti |
| le tabelle | `tools/prefill_context_report.py attn` e `zones`, `tools/decode_context_report.py speed` | — |
| i test vedono gli errori? | `tools/mutate_prefill.sh` nel container | 45 minuti |


## Clean machine remeasure (2026-09-19)

Quattro processi `yes`, residui di una prova del 17/09, sono rimasti al 100% di un core l'uno sotto
**ogni misura nativa del 18 e del 19/09** (LESSONS #84). Rimisurato il 19/09 dalle 16:09 alle 17:52,
senza di loro, ciò da cui era stato scelto il default del motore: thread per fase e decode a
contesto lungo. Binario con `tr_expf` (hash in `build/prefill_context/binaries.sha256`), 8 giri,
A/A, ogni run in `build/threads_phase/` e `build/decode_context/`; le sessioni coi `yes` restano in
`build/threads_phase-with-yes/` e `build/decode_context-with-yes/`. Ogni sessione dichiara nel log
il **carico di fondo** (LESSONS #85): il processo `System` del kernel tiene da solo 0.7-0.9 core,
sempre; è un fatto di questa macchina e sta sotto ogni numero di questo documento.

| sessione | ora | carico di fondo prima → dopo (processori logici occupati, di cui `System`) | peggiore A/A |
|---|---|---|---|
| larghezze forzate a 512 (`threads_phase.sh widths`) | 16:09 | 3.82 (0.84; `svchost` 1.23, indicizzatore di Windows) → 2.14 (0.72) | decode 2.6% |
| decode a 16 contro 8 thread, quattro contesti (`decode_context.sh widths`) | 16:19 | 2.01 (0.78) → 1.17 (0.71) | decode 4.6% |
| automatico contro 8, banda della RAM, zone (`decode_context.sh measure`) | 16:51 | 1.25 (0.75) → 0.98 (0.73) | decode 2.5% |
| sweep dei thread (`threads_phase.sh sweep`) | 17:30 | 0.99 (0.71) → 1.02 (0.74) | decode 2.7%, prefill 4.1% |

**Il rumore del giorno è un fatto, non un difetto da nascondere**: con Marcello alla macchina, un'altra
finestra al lavoro e l'indicizzatore di Windows, le prime sessioni hanno spread del 15-30% fra minimo
e massimo e non distinguono differenze sotto il 5%; la sera, a macchina lasciata sola, l'A/A torna
all'1-3%. Le sessioni strette del 18/09 erano strette perché notturne, non grazie ai `yes`. Le
misure che decidono si fanno di notte, e ora si possono lasciare sole (orfani rifiutati, guardia
sulla CPU prima di ogni run, carico dichiarato).

**1. La banda della RAM** (`bench_mem ram`, lettura in fila; domanda 4):

| thread | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| coi `yes` (19/09 mattina), GB/s | 22.4 | 42.7 | 54.0 | 54.0 | 52.3 | 51.0 | 51.5 |
| macchina pulita, GB/s | 26.2 | 52.2 | **57.3** | **57.6** | 55.7 | 54.4 | 49.5 |

Il tetto è **~57 GB/s, non 54**, si tocca con 4-6 thread e oltre **cala**: il decode, che è lettura
di memoria, con più lettori non ha niente da guadagnare. È il perché di tutto quello che segue.

**2. Il decode a larghezza forzata, 16 contro 8 thread** (tok/s, mediana di 8; ogni modo con la sua
copia; rapporto di 8 su 16 contro tutte e due le copie):

| contesto | 16 thread | 8 thread | 8 / 16 | A/A | coi `yes` |
|---|---|---|---|---|---|
| 32 | 38.5 / 37.7 | 40.2 / 39.3 | 1.02-1.07× | 1.9% / 2.4% | — |
| 512 | 34.6 / 34.3 | 34.7 / 35.0 | **1.00-1.02×** | 0.9% / 1.0% | **1.10×** |
| 2048 | 27.1 / 25.9 | 28.1 / 26.9 | 0.99-1.09× | 4.6% / 4.3% | 1.03-1.05× |
| 4000 | 21.1 / 21.5 | 21.6 / 22.2 | 1.01-1.06× | 1.9% / 2.9% | — |

Con la soglia della sessione (4.6%) 8 e 16 **non sono distinguibili a nessun contesto**; 8 è avanti
dello 0-7% nominale e non è mai dietro. Il «+10% a 8 thread» del 18/09 era l'effetto dei quattro
core presi: 16 thread fissati ai core pagavano il thread più lento, 8 no.

**3. L'automatico contro 8 thread, e cosa sceglie** (la sessione più quieta; soglia 2.5%). «Scelte»:
quante volte su 8 run la sessione ha scelto ogni larghezza, e fra parentesi quanti cambi da una run
alla successiva (`tools/decode_context_report.py speed`), per il modo e per la sua copia:

| contesto | automatico | 8 forzati | 8 / automatico | scelte, macchina pulita | scelte, coi `yes` (19/09 mattina) |
|---|---|---|---|---|---|
| 32 | 40.7 / 41.1 | 40.6 / 40.6 | 0.99-1.00× | 7×4 1×8 (1) · 7×4 1×8 (1) | 7×4 1×8 (1) · 6×4 2×8 (3) |
| 512 | **36.9** / 36.1 | 34.8 / 34.6 | **0.94-0.96×** | 7×4 1×8 (1) · 6×4 1×8 1×16 (3) | 6×4 2×8 (4) · 2×4 2×8 4×16 (4) |
| 2048 | 28.5 / 27.7 | 28.1 / 28.6 | 0.99-1.03× | 8×8 (0) · 7×8 1×16 (2) | 6×8 2×16 (3) · 4×8 4×16 (5) |
| 4000 | 22.7 / 22.1 | 22.4 / 22.6 | 0.99-1.02× | 7×8 1×16 (2) · 5×8 3×16 (3) | 1×8 7×16 (2) · 2×8 6×16 (2) |

A 512 l'automatico, che sceglie quasi sempre **4 thread**, batte 8 del 4-6%, sopra la soglia contro
tutte e due le copie: a contesto corto il decode vuole 4 thread, a contesto lungo 8, e 16 non vince
mai. Il verso della conclusione del 18/09 («pochi thread») era giusto; la misura era gonfiata.
Le larghezze forzate a 512 della prima sessione (rumorosa) dicono lo stesso: 16 thread 33.1 (copia
32.3), 12 31.1, 8 31.1, 6 33.1, 4 33.1, automatico 33.6 (scelte 4×4 1×8 3×16, 3 cambi); coi `yes`
erano 30.9, 31.6, 34.1, 34.6, 33.8 e 33.8.

**Lo stimatore tira a sorte** (LESSONS #88, domanda 31). Le scelte cambiano da una run all'altra: poco a
macchina quieta (una run su otto), molto appena c'è rumore (4×4 1×8 3×16), e succedeva già coi `yes`,
dove a 2048 cambiava 3-5 volte su 8: la velocità mediana non lo mostrava, perché dove le larghezze si
equivalgono una scelta instabile non costa. Il margine (la più ampia entro l'**1%** dalla più veloce,
su tre passate da un token) è sotto il rumore di quelle passate (2-5%), ed era stato stretto dal 3%
all'1% **sulla macchina coi core presi**, dove 16 thread perdevano davvero. Va riscritto lo stimatore,
non il default: margine dal rumore misurato, più giri se non decide, isteresi nelle rimisure, e a
parità la larghezza **stretta**, che su macchina pulita non perde mai e sotto il carico di altri vale
il 10%.

**4. Lo sweep dei thread** (`-t n` per tutte e due le fasi; il decode sceglie da sé la larghezza
dentro il pool, scritta fra parentesi; tok/s):

| `-t` | 512: prefill | decode | 2048: prefill | decode |
|---|---|---|---|---|
| 16 | 316.8 (copia 303.7) | 37.4 (4) | 308.5 (copia 305.8) | 29.5 (8) |
| 12 | 287.0 | 37.3 (6) | 265.7 | 29.3 (6) |
| 8 | 213.1 | 36.7 (4) | 196.7 | 28.0 (8) |
| 4 | 112.5 | 32.3 | 104.5 | 24.6 |

Il prefill vuole tutto il pool (da 8 a 16 thread 1.49× a 512 e 1.57× a 2048); il decode va uguale
con ogni pool purché la larghezza scelta sia 4-8. Con `-t 4` lo stimatore prova anche 2 e 1 thread:
in una run da 48 token quelle sei passate lente pesano il 14-17%, quindi quel numero è il costo del
sondaggio, non il regime (un altro argomento per la domanda 31: non sondare larghezze sotto 4).

**5. Il modello del decode** (`decode_context_report.py model`, automatico): **24.5 ms a contesto zero
più 5.0 ms ogni 1000 token di contesto** (la KV letta a 52 GB/s); 41.0 / 36.6 / 28.4 / 22.5 tok/s a
contesto 32 / 512 / 2048 / 4000, cioè il 3-8% in più della mattina coi `yes` (38.9 / 35.5 / 27.5 /
20.9 a 8 thread). Il decode muove 50-51 GB/s su 57: dal lato esatto resta il 10-12%.

**6. Come si rifà**, a macchina lasciata sola: `sh tools/threads_phase.sh widths`, `sh
tools/decode_context.sh widths` (16 contro 8 ai quattro contesti, 40 minuti), `sh
tools/decode_context.sh measure`, `sh tools/threads_phase.sh sweep`; le tabelle con
`tools/decode_context_report.py speed <file>` (scelte e cambi compresi) e `model`.


## `tr_expf`: our exponential, scalar (2026-09-19)

Domanda 37, primo tempo: **solo lo scalare**, che è la definizione. Il SIMD è una decisione a parte
(punto 7). Velocità misurate sulla macchina pulita della sezione sopra, dalle 17:58 alle 18:39:
carico di fondo 1.10 → 0.98 processori occupati (0.68-0.71 del `System`), A/A peggiore 3.0% sul
prefill e 1.9% sul decode; ogni run in `build/prefill_context/`, hash in `binaries.sha256` («prima» è
il binario del commit ef3cb99). Le due sessioni del giorno fatte prima, una coi `yes` accesi
(spread 24-57%) e una interrotta, restano in `build/prefill_context-expf-noisy/` e `-interrupted/`.

**1. La funzione** (`src/kernels/expf.c`). exp(x) = 2^(k/64) · exp(r), k = round(x · 64/ln2), r = x −
k · ln2/64 con ln2/64 in due pezzi (il primo esatto per ogni k), tutto in double: 64 valori di
2^(j/64), un polinomio di Taylor di grado 5, e un **test di arrotondamento**: se y·(1 − 2^-50) e
y·(1 + 2^-50) arrotondano allo stesso float, exp(x) arrotonda lì. Gli argomenti che lo falliscono sono
**8 su 4 278 190 082** e stanno in una tabella di eccezioni con l'esponenziale calcolato a 200 bit (e
a 400: uguale). Un argomento che fallisse il test senza essere in tabella darebbe un NaN: un
arrotondamento che nessuno ha provato non esce dalla funzione come numero. Dentro non c'è nessuna
chiamata alla libreria C, e nessuna costante viene da una libreria: le scrive `tools/gen_expf_table.py`
con mpmath (`src/kernels/expf_table.h`, confrontato col suo generatore da `tools/lint.py`, LESSONS
#83). Le 8 eccezioni le trovano uguali tre strade: il C su Windows (MinGW gcc 15.2), il C su Linux
(gcc 13.3), e un mirror numpy bit a bit della strada veloce su tutti i float in campo
(`gen_expf_table.py --scan`, 4 minuti). Softmax (attenzione e router) e SwiGLU la usano;
`tools/lint.py` rifiuta `expf(` e `exp(` nella zona calda.

**2. La prova è esaustiva** (`make bench-expf`: tutti i 2^32 float contro il riferimento, che è
provato a 200 bit sui 369 casi al confine; in `make check` con gcc e con clang):

| | costo a chiamata, libreria / `tr_expf` | diversi dalla libreria | diversi dal riferimento | non provati | tempo |
|---|---|---|---|---|---|
| Windows (MinGW-w64) | 29.9 ns / **3.5 ns** | **0 su 4 278 190 082** | **0** | 0 | 18 s |
| Linux (glibc), gcc | 2.4 ns / 3.8 ns | 170 648 (dove glibc sbaglia) | **0** | 0 | 7 s |
| Linux (glibc), clang | 4.0 ns / 4.1 ns | 170 648 | **0** | 0 | 6 s |

Rosso prima: con la tabella delle eccezioni vuota, 8 NaN e `check: FAILED`. `tools/mutate_expf.sh`:
**17 mutazioni su 17 viste** da almeno un controllo (test rapido `tests/test_expf.c`, prova esaustiva,
confronto col generatore, lint della zona calda). Tre cose che le mutazioni insegnano. (a) Tre costanti
spostate di un'unità sull'ultima cifra (2^(1/64), 1/6, il pezzo basso di ln2/64) **non cambiano nessun
risultato** su 4 miliardi di float: le vede solo il confronto dell'header col generatore. (b) Col
margine del test a 2^-60 (che in double vuol dire nessun margine) i risultati restano **tutti giusti**:
il double della strada veloce arrotonda già al float giusto ovunque. Test e tabella non correggono
numeri: rendono la prova strutturale, perché un cambio che sposta un risultato lascia argomenti non
provati o voci di tabella morte (8, in quella mutazione), che il controllo conta, invece di un
arrotondamento sbagliato in silenzio. (c) Softmax o SiLU rimessi sull'`expf` della libreria: su Linux
li vede `test_expf` (il softmax e il SiLU sono la loro definizione con `tr_expf`, bit a bit), ovunque
il lint; su Windows i bit sono gli stessi e resta il lint.

**3. Su Windows gli stessi bit di prima.** Stadio `exact` di `tools/prefill_context.sh`, contro il
binario del commit precedente, passato tre volte nel giorno: logit **identici al byte** su 600
posizioni un token per passata (120 MB, 16 thread), a passate da 64 su 8 thread, su un prompt da 4000
a passate da 512 e da 100, e gli stessi token dopo un prompt da 4000.

**4. Su Linux cambia quanto previsto, e solo l'esponenziale** (`tools/expf_quality.sh` nel container,
modello vero, prompt `code-edit.txt` più 1000 token generati, 1411 posizioni). Contro il binario di
prima (`build/linux-before`, sull'`expf` di glibc): nessuna riga identica al byte, differenza massima
fra due logit **2.3e-5** su un campo di 65.8, **KL media 3.9e-13, massima 1.5e-11, 0 posizioni** che
scelgono un altro token, **1000 token su 1000** uguali. Contro il build di emulazione (il commit di
prima con ogni `expf` sostituito da `(float)exp((double)x)`, `tools/cr_expf_emul.h`): **1411 righe su
1411 identiche al byte** e gli stessi token, ed è un controllo (lo script esce 1 se no): nel motore è
cambiato l'esponenziale e nient'altro.

**5. Windows e Linux danno gli stessi byte** (`sh tools/platform_bits.sh`): logit identici al byte
fra il binario nativo e quello del container sulle fixture minuscole F32, F16 e Q8_0 (120 posizioni) e
sul **modello vero a 2 layer su 1000 posizioni (201 MB)**. Il sospetto sulla tabella RoPE (`pow`, `cos`
e `sin` della libreria C, in double, arrotondati a float: il solo posto rimasto dove i numeri vengono
da una libreria) è misurato voce per voce (`tests/dump_rope.c`, `tools/rope_table_compare.py`; la
seconda copia dello strumento, LESSONS #81, perché Smart App Control bloccava la prima): su 4096
posizioni × 64 coppie i **double** di `cos` e `sin` differiscono fra le due librerie nello 0.8% delle
voci (2175 coseni, 2216 seni su 262 144), `pow` mai, e i **float** della tabella del motore **mai**:
l'arrotondamento a float assorbe l'ultima cifra del double. Domanda 40 chiusa: fino al contesto di
addestramento di OLMoE le due piattaforme calcolano sugli stessi numeri.

**6. Prima e dopo** (tok/s, mediana (min-max) di 8; decode forzato a 8 thread nei due binari, perché
la larghezza scelta dalla sessione oscilla, domanda 31; soglia: prefill 3.0%, decode 1.9%):

| | prima | prima, copia | dopo | dopo, copia | dopo / prima | stima (§Prefill su prompt lunghi, punto 6) |
|---|---|---|---|---|---|---|
| prefill, prompt 512 | 296.3 (271.6-313.7) | 290.8 | 305.5 (298.7-349.7) | 314.6 | 1.03-1.08× | 1.12× |
| 2048 | 249.4 (239.4-261.5) | 252.7 | 303.2 (297.8-321.9) | 307.8 | **1.20-1.23×** | 1.23× |
| 4000 | 210.9 (204.2-218.5) | 214.0 | 276.1 (267.8-281.3) | 276.5 | **1.29-1.31×** | 1.27× |
| decode dopo 512 | 35.1 (34.8-36.1) | 35.1 | 36.0 (35.2-36.5) | 35.9 | **1.02-1.03×** | +5% |
| dopo 2048 | 28.0 (27.6-28.2) | 27.5 | 29.3 (28.6-30.0) | 29.4 | **1.05-1.07×** | +8% |
| dopo 4000 | 21.4 (21.1-21.8) | 21.2 | 23.2 (22.9-23.4) | 23.3 | **1.08-1.10×** | +10% |

A 512 il prefill è sopra la soglia contro tre copie su quattro (1.031× contro la quarta): «fra 1.03 e
1.08×». Per zona (profilo dei due binari nella stessa sessione, ms del prompt):

| | 512: prima | dopo | 2048: prima | dopo | 4000: prima | dopo |
|---|---|---|---|---|---|---|
| `attention` | 92.3 | 40.5 | 1537 | 591 | 5960 | 2331 (**2.56×**) |
| `expert_act` | 138.2 | 24.6 | 594.9 | 108.0 | 1201 | 218.4 (**5.5×**) |
| `router` | 5.0 | 4.0 | 19.6 | 16.4 | 41.8 | 34.8 |
| tutto il prompt | 1802 | 1644 (1.10×) | 8236 | 6825 (1.21×) | 19087 | 14693 (1.30×) |

Il softmax passa da 32-35 a **4.4-6.6 ns per elemento** (`make bench-attn`) e da 61-73% a 22-31%
dell'attenzione. A un thread il prefill a 512 va 1.15× (25.2 → 28.9 tok/s).

**7. Il SIMD: i numeri per decidere** (non scritto). Dopo lo scalare, nel prefill a 512 / 2048 / 4000
restano: softmax dell'attenzione circa 9 / 130 / 610 ms (il 22-27% di una zona che vale il 2.5 / 8.7 /
15.9% del prompt) e `expert_act` 24.6 / 108 / 218 ms (l'1.5%). Con un esponenziale AVX-512 stimato a
1.0-1.3 ns per elemento (8 double per registro; su Zen 4 i 512 bit sono due passate da 256 e il gather
costa) e il resto del softmax vettoriale, softmax e attivazione scenderebbero del 65-70%: prefill
stimato **1.01× / 1.03× / 1.04×**, decode +1-2%. Costo: circa 200 righe (varianti AVX2 e AVX-512
identiche al bit allo scalare, gather dalla tabella, maschera per le corsie che non passano il test e
code; voce `exp` nella tabella dei kernel con uno scratch per l'attivazione; la prova esaustiva per ogni
tier, 7 s l'una nel cancello). Lo scalare ha preso quasi tutto: a 4000 token il guadagno che resta è
sotto la soglia di una buona sessione. Decide Marcello (domanda 39).

**8. Come si rifà.**

| misura | comando | dura |
|---|---|---|
| la prova su tutti i float, e il costo a chiamata | `make bench-expf`, nativo e nel container | 7-18 s |
| le costanti dal generatore; le eccezioni ritrovate da zero | `tools/gen_expf_table.py [--check]`; `--scan` | 1 s; 4 minuti |
| i controlli vedono gli errori? | `sh tools/mutate_expf.sh` nel container | 8 minuti |
| Linux: KL e token contro il binario di prima, byte contro l'emulazione | `sh tools/expf_quality.sh` nel container, col volume dei modelli | 8 minuti |
| Windows e Linux, gli stessi byte? | `sh tools/platform_bits.sh` | 3 minuti |
| logit al byte, prima e dopo con A/A, banco, zone | `GEN_EXTRA="--decode-threads 8" PROF_BEFORE=<prima> sh tools/prefill_context.sh change <prima>` | 45 minuti |


## M1, before writing code: routing and disk (2026-09-19/20)

Domande 13-16. Una traccia del routing sul modello vero e un banco del disco; nessun cronometro sul
motore, quindi nessuna dipendenza dallo stimatore della larghezza non ancora validato.

**La traccia.** `trochilus run --route-trace <file>` registra, per ogni token e ogni layer, gli 8
esperti scelti e due previsioni dei 16 più probabili del layer dopo, col router del layer dopo:
`pred_in` sullo stato che entra nel FFN del layer corrente (noto **prima** che girino i suoi
esperti), `pred_out` sull'uscita del layer passata per la `ffn_norm` del layer dopo (noto solo a
layer finito). Spenta non costa niente e non cambia un byte. `tests/test_route.c` la prova esatta su
modelli sintetici (con l'uscita dell'attenzione a zero `pred_out` **è** la scelta del layer dopo; con
il router del layer 0 generato come quello del layer 1 `pred_in` **è** la scelta del layer 0), cinque
mutazioni viste rosse (`tools/mutate_route.sh`). Run: OLMoE-1B-7B Q8_0, `bench/prompts/code-1000.txt`
(904 token di codice nostro), 300 token generati, nel container (si contano esperti, non secondi).
Tabelle da `tools/route_trace_report.py build/route/code-1000.bin`.

Domanda 13, quota degli esperti scelti dal layer L+1 che stavano fra i primi K previsti:

| | `pred_in` 8 | 12 | 16 | `pred_out` 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| tutti i token | 82.4% | 92.4% | 95.8% | 86.1% | 94.9% | 97.2% |
| solo prompt | 81.4% | 91.7% | 95.3% | 85.5% | 94.5% | 97.0% |
| solo generati | 85.2% | 94.5% | 97.1% | 88.0% | 96.0% | 97.8% |
| layer peggiore (L=0) | 63.2% | 74.6% | 82.0% | 67.7% | 79.7% | 85.7% |

La soglia era «almeno l'80% prevedendone al massimo 12»: **passata** (92-96%), e già con la
previsione presa **prima** degli esperti, che lascia al disco il tempo di un layer intero; aspettare
l'uscita del layer dà 2-3 punti in più e metà del tempo. Il primo layer è l'eccezione (75-80%): chi
lo prevede è l'embedding, non un layer.

Domanda 14, cache di unità (layer, esperto) simulata su tutta la traccia (modello superato dalla misura: §M1 misurato), contata sui soli token
generati; un esperto di un layer sono 6.375 MiB (tre matrici Q8_0), il modello ne ha 1024:

| cache | LRU: mancati per token | LRU: MiB per token | pin dall'uso nel prompt: mancati | MiB |
|---|---|---|---|---|
| 25% (256) | 54.6 | 347.9 | 58.3 | 371.7 |
| 50% (512) | 22.4 | 143.0 | 35.0 | 222.9 |
| 75% (768) | 5.0 | 32.2 | 14.7 | 93.8 |

Un token usa 128 unità (8 × 16): con metà modello in RAM ne mancano 22. **Il pin statico dall'uso
perde contro l'LRU a ogni capacità** (a 75% legge il triplo): il piano di M1 diceva «pin dall'uso»,
e su questa traccia è la scelta peggiore delle due.

Domanda 15, streaming per layer interi (ogni token tocca ogni layer): 5106 / 3404 / 1702 MiB per
token con il 25 / 50 / 75% dei layer in RAM, **15-53 volte** lo streaming per esperti. Chiusa: per
esperti.

**La lettura diretta, sul modello vero in nativo** (2026-09-20, una run per modo, non una misura:
serve solo a dire che la catena regge fuori dal container e fuori dai modelli minuscoli). Budget
3264 MiB, prompt 8, 2 token: `experts: 511 of 1024 units in RAM (3264 MiB, direct), 109 hits, 584
misses, 3723 MiB read in 2.96 s`, e con `TR_EXPERT_DIRECT=0` gli stessi colpiti, mancati e MiB in
3.41 s. Due cose: **3723 MiB in 2.96 s sono 1258 MiB/s**, cioè la banda del disco misurata da
`make bench-disk` (~1.4 GB/s a blocchi da 2 MiB) e non quella della RAM, quindi si sta leggendo
davvero dal disco anche su un volume cifrato; e gli slot sono **511 contro 512**, perché in diretta
ognuno porta un settore di margine per parte (3 × 4 KiB su 6.4 MiB: lo 0.2%).

**Il costo di una chiamata di lettura** (LESSONS #94, misurato il 2026-09-20 in nativo,
`bench_event_overhead` in `tests/bench_disk.c`): su Windows `tr_file_pread` crea e chiude un evento
a ogni chiamata. Su un blocco da 4 KiB già in cache, 20 000 giri: **2.54-2.61 µs** contro
**2.11-2.20 µs** con un evento tenuto vivo, cioè 0.41-0.44 µs a chiamata su tre run. Una lettura da
2 MiB su questo disco è 1.4 ms: lo 0.03%. Non si cambia.

**Il disco** (domanda 16). `make bench-disk` (`tests/bench_disk.c`): blocchi grandi quanto una
matrice di un esperto (2.125 MiB) e quanto tre (6.375 MiB), a posizioni casuali allineate del file
del modello (6.85 GiB), aperto senza la cache del sistema (`FILE_FLAG_NO_BUFFERING`; `O_DIRECT` su
Linux), con 1-16 lettori; prima di ogni numero il lettore diretto e `tr_file_pread` devono dare gli
stessi byte. Nativo, macchina ferma (`tools/machine_still.sh`), due sessioni, mediana di 5 run
ciascuna (`build/disk/run1.txt`, `run2.txt`); NVMe Micron 2400 da 1 TB (QLC, senza DRAM).

| blocco | 1 lettore | 4 | 8 | 16 | spread |
|---|---|---|---|---|---|
| 2.125 MiB | 1396 / 1454 MB/s | 1489 / 1471 | 1504 / 1514 | 1373 | 8-16% |
| 6.375 MiB | 1641 / 1718 | 1749 / 1696 | 1666 / 1723 | 1614 | 7-20% |

Attraverso la cache del sistema (`tr_file_pread`, 8 lettori): la prima lettura 1.66-1.93 GB/s, la
seconda degli stessi blocchi **12-26 GB/s** (è una copia dalla RAM).

1. **Il disco dà ~1.5 GB/s e i lettori non contano**: uno solo prende quanto otto, sedici perdono.
   Il «pool di I/O» di M1 non serve per la banda: bastano 1-2 thread, e servono solo a non fermare
   il calcolo. Tre matrici in fila rendono il 15% in più di tre letture separate.
2. **1.5 GB/s è un terzo della scheda del disco** (4.5 GB/s in lettura sequenziale). Non si sa
   perché: cifratura del volume, QLC senza DRAM su letture sparse, o la richiesta sincrona da 2 MiB.
   Domanda 41.
3. **Token al secondo attesi** (decode a 8 thread ~28-36 ms di calcolo per token; 1 MB = 10^6 byte,
   1.5 GB/s = 1430 MiB/s): con la cache al 50% si leggono 143 MiB = 100 ms per token, cioè **7-8
   tok/s** se lettura e calcolo si sommano e **10 tok/s** se si sovrappongono del tutto (la
   previsione della domanda 13 serve a questo); al 75% 22 ms, **17-20 tok/s** sommati e il solo calcolo
   (28-36) se sovrapposti; al 25% 243 ms, **3.6-4.1 tok/s**. Il criterio era «per esperti se almeno 5 tok/s al 50%»: **passato**.
4. La cache del sistema rende 12-26 GB/s su ciò che ha già letto, ma non si governa (né quanto
   tiene né cosa butta): il budget di RAM di M1 resta nostro, e la lettura va fatta senza passare
   due volte dalla RAM (o diretta, o nostra: da decidere nel progetto).
5. Limiti della misura: una sola traccia (codice, 904 + 300 token) e un solo modello, con esperti
   piccoli (8 attivi su 64); il prompt è contato token per token, mentre un prefill a blocchi tocca
   quasi tutti gli esperti di ogni layer in una passata (per il prompt lo streaming per esperti vale
   quanto leggere tutto il modello una volta).
6. **Con questo disco il precaricamento non rende, e con 12 candidati fa danno.** Il report simula
   la stessa LRU col precaricamento dei primi k di `pred_in` (per token generato, cache al 50%):
   gli stalli scendono da 22.4 a 7.2 (k=8) e 5.5 (k=12), ma le letture salgono da 22.4 a 29.9 e
   49.8, perché 5 e 21 sono di esperti che poi non servono. Il disco è il collo, quindi conta il
   totale letto, non gli stalli. **Modello a tempo** (modello, non misura: un disco, una lettura per
   volta a 4.46 ms per unità, 33 ms di calcolo per token; `route_trace_report.py`, provato a mano
   in `--check` e con quattro mutazioni rosse):

   | cache | senza | k=8 | k=12 | k=16 | | disco 3× (4.3 GB/s): senza | k=8 | k=12 |
   |---|---|---|---|---|---|---|---|---|
   | 25% | 3.6 tok/s | 3.5 | 2.2 | 1.6 | | 8.8 | 10.0 | 6.4 |
   | 50% | 7.5 | 7.0 | 4.5 | 3.0 | | 15.1 | 17.4 | 12.7 |
   | 75% | 18.0 | 18.1 | 13.0 | 9.0 | | 24.7 | 26.9 | 24.6 |

   La soglia della domanda 13 («80% con 12 candidati») misurava la cosa sbagliata: la previsione è
   buona, ma ogni candidato in più è una lettura, e la lettura è ciò che manca. Il primo M1 è
   **LRU e letture a richiesta, senza precaricamento**; il precaricamento (k = gli 8 più probabili,
   mai di più) resta un'opzione che il piano automatico accende solo su un disco veloce, dove vale
   l'11-15%. Domanda 43.

## M1 measured: what experts really cost from disk (2026-09-20 night)

Il motore vero, Windows nativo, macchina ferma (carico di fondo 1.0-1.4 processori su 16, quasi
tutto il processo System), `sh tools/experts_budget.sh measure | misses | long | direct`. Prompt di
512 token, larghezza del decode **forzata a 8** (lo stimatore non è validato, punto 0), lettura
diretta verificata prima di ogni sessione, ordine a rotazione e una copia A/A per ogni modo
(`tools/ab_modes.sh`). Il modello ha 1024 unità (layer, esperto) da 6.375 MiB: 6528 MiB in tutto.

**1. Velocità ai quattro budget** (mediana di 6, 64 token generati; fra parentesi la copia A/A):

| budget | decode tok/s | vs 100% | prefill tok/s | miss della run | MiB letti |
|---|---|---|---|---|---|
| 100% (residente) | 35.45 (35.66) | 1.00× | 306.9 (319.4) | — | — |
| 75% | 29.56 (29.32) | 0.83× | 80.5 (80.9) | 1009 | 6432 |
| 50% | 24.27 (24.25) | 0.68× | 82.3 (82.1) | 1106 | 7051 |
| 25% | 20.56 (20.64) | 0.58× | 84.0 (83.5) | 1204 | 7676 |

**Il dirupo è fra residente e non residente, non fra i budget.** Il prefill fa 306.9 tok/s se il
modello è in RAM e 80-84 con qualunque budget parziale — il 75% non aiuta più del 25%, perché il
prompt legge comunque tutto il modello: 6432-7676 MiB contro i 6528 minimi (ogni unità una volta).
Sono ~4.3 s di disco a 1.5 GB/s, pagati **una volta per sessione**. Il budget decide il decode.

**2. Quanto costa un token generato** (differenza fra due lunghezze di generazione, così il prompt
esce dal conto; i contatori non si muovono da run a run, spread 0.0%):

| budget | 72 contro 8 token: miss/token | MiB/token | 1000 contro 200: miss/token | MiB/token |
|---|---|---|---|---|
| 75% | 0.1 | 0.5 | — | — |
| 50% | 0.3 | 1.8 | 0.03 | 0.2 |
| 25% | 0.6 | 3.7 | 0.14 | 0.9 |

**La simulazione della domanda 14 (modello superato dalla misura) diceva 22.4 miss e 143 MiB per token al 50%: il motore ne fa
0.3 subito dopo il prompt e 0.03 lontano dal prompt**, cioè 70-700 volte meno. Non è un errore di
conto: quella simulazione (modello superato dalla misura) contava i miss su una generazione lunga di 1000 token partendo da una cache
qualunque, mentre il motore arriva al decode con la LRU appena riempita dal prompt — che ha toccato
quasi ogni unità. Le ultime 512 (o 256) unità toccate sono esattamente quelle che servono, e più la
generazione va avanti **meno** sbaglia, non di più (LESSONS #98).

**3. Il decode migliora con la lunghezza della generazione**, e il colpevole è l'archivio:

| budget | 64 token | 200 token | 1000 token | ms/token in più a 200 | costo fisso |
|---|---|---|---|---|---|
| 100% (residente) | 35.45 | 34.64 | 33.48 | **−1.0** | **nessuno** |
| 50% | 24.27 | 29.21 | 30.79 | +1.75 | ~0.35 s |
| 25% | 20.56 | 26.53 | 29.84 | +4.18 | ~0.84 s |

La sessione che decide è quella del 2026-09-21 notte, con dentro il **budget pieno**: col modello
residente il tempo per token non migliora affatto passando da 200 a 1000 (anzi peggiora di 1 ms,
dentro uno spread del 4-7%). Quindi il costo fisso **non** è scaldamento dei kernel, né il primo
token, né il nostro metro: compare solo quando si legge dal disco e cresce quanto più il budget è
stretto. È l'archivio che si riassesta dopo il prompt, e una fase di preparazione all'avvio non lo
può nascondere, perché succede *dopo* il prompt.

I mancati in più dei primi token ne spiegano una parte (al 25%: 27 unità in più nei primi 200
token, 172 MiB, ~115 ms su ~840). Il resto è da trovare con un profilo per zone dei primi 64 token
contro quelli lontani dal prompt: domanda 46, ristretta a questo.

**4. La lettura diretta contro la cache del sistema** (budget 50%, mediana di 6, stessi miss e
stessi MiB nei due modi: cambia solo da dove arrivano i byte):

| | prefill tok/s | decode tok/s |
|---|---|---|
| `direct` (`FILE_FLAG_NO_BUFFERING`) | 81.4 | 24.2 |
| `buffered` (`TR_EXPERT_DIRECT=0`) | 197.0 | 32.6 |
| | **2.42×** | **1.35×** |

Con 7 GiB letti e 31 GiB di RAM la cache del sistema tiene tutto il modello: «buffered» misura la
RAM, non il disco. Per questo `experts_budget.sh` si rifiuta di partire se la riga `experts:` non
dice `direct` — senza quella guardia ogni numero di M1 sarebbe gonfiato di 2.4× sul prompt.

## Prefill reads model once per pass (2026-09-21)

Domanda 47, nata dalla domanda della sovrapposizione I/O-calcolo: quanto del prefill è attesa del
disco e quanto calcolo. Il profilo separa le due cose (zona `weight_read`), e il conto ha trovato
dell'altro. `sh tools/prefill_overlap.sh 5` (mediana di 5 giri più uno di riscaldamento, ordine a
rotazione, macchina ferma a 1.07-1.19 processori su 16, binario `build/native2` perché Smart App
Control bloccava il primo, LESSONS #81):

| modo | prefill s | disco s | calcolo s | MiB letti | GB/s | tetto sovrapposizione |
|---|---|---|---|---|---|---|
| prompt 512, residente | 1.65 | 0.00 | 1.65 | 0 | — | — |
| prompt 512, budget 50% | 6.11 | 4.44 | 1.68 | 6 031 | 1.33 | 1.38× |
| prompt 2048, residente | 7.11 | 0.00 | 7.11 | 0 | — | — |
| prompt 2048, budget 50% | 23.51 | 16.26 | 7.24 | **22 880** | 1.37 | 1.45× |
| prompt 2048, budget 50%, **una passata** (`-b 2048`) | **11.27** | 4.60 | 6.67 | **6 273** | 1.33 | 1.69× |

Spread 0.7-8.7%. Il «tetto della sovrapposizione» è `totale / max(disco, calcolo)`: **modello, non
misura**, è il limite di un lettore che non esiste ancora.

**Il controllo torna al centesimo**: a 512 il prefill sotto budget meno il residente fa
`6.11 − 1.68 = 4.43` contro i 4.44 s che la zona dichiara, e sui modi residenti il disco è 0.00.
La contabilità del profilo è esatta, e il disco va a 1.33-1.37 GB/s come dice `make bench-disk`.

**Il difetto**: 22 880 MiB sono **3.5 volte** la tabella degli esperti (6 528 MiB). Il prompt si
elabora a blocchi di 512 token (`OLMOE_DEFAULT_BATCH` in `src/models/olmoe.c`) e **ogni passata
percorre tutti i layer**: sotto budget la LRU non può tenere il modello fra una passata e l'altra,
quindi ogni passata rilegge tutto. 2048 / 512 = 4 passate, 3.65× i byte. A 4000 token sono otto.

**Esatto, e già dimostrabile senza toccare codice**: con `-b 2048` i byte scendono a 6 273 MiB
(una volta sola, più il margine dei settori) e il prefill passa da 23.51 a 11.27 s. `logits -b 512`
e `logits -b 2048` sullo stesso prompt danno l'**ultima riga identica al byte** (201 216 byte,
`cmp`), che è quello che le due forme hanno in comune: la passata non cambia i numeri, cambia
quante volte si legge il disco.

**Il calcolo non peggiora con la passata grande**: 6.67 s contro 7.24 a quattro passate (e 7.11 il
residente). Quindi il motivo per non alzare `-b` e basta **non sono i kernel, è la memoria**: le
attivazioni, la KV e lo scratch di una passata crescono col numero di token, e a 4000 o 32000 una
passata unica non sta in RAM.

**La cura è l'ordine per layer, scritta il 2026-09-21** (`src/models/olmoe.c`: il corpo di un
layer estratto in `forward_layer`, e `forward_prompt_layer_major` che per ogni layer percorre tutte
le passate del prompt; lo stato nascosto dell'intero prompt sta in `s->x_all`, allocato alla
creazione della sessione e solo quando l'archivio è parziale — nella zona calda non si alloca).
Ogni esperto si legge una volta per prompt tenendo i blocchi piccoli; a budget pieno il percorso è
quello di prima, invariato.

| prompt 2048, budget 50% | prima | dopo | |
|---|---|---|---|
| MiB letti | 22 880 | **6 273** | 3.65× meno |
| prefill | 23.51 s | **12.41 s** | **1.89×** |
| calcolo | 7.24 s | 7.73 s | non distinguibile |

Mediana di 5 giri più uno di riscaldamento, spread 11.4%, macchina ferma (0.94-1.37 processori su
16). Nella stessa sessione `-b 2048` (una passata) dà 12.03 s e gli stessi 6 273 MiB: le due forme
coincidono entro il 3%, cioè l'ordine per layer arriva al tetto dell'esperimento senza far crescere
le attivazioni. A prompt 512 (una passata sola) niente cambia: 6.27 s contro 6.11, dentro lo spread.

Il controllo che chiude la lezione #99 è `tests/test_stream.c` §`once_per_prompt`: 36 token in
passate da 12 sull'archivio minimo, le unità lette devono stare entro `n_units + n_slots`. Visto
rosso prima del fix (33 lette, 16 in tabella) e verde dopo. **Con `--route-trace` attivo si resta
sull'ordine di prima**: la traccia numera le righe da `n_tokens`, che avanza solo dopo l'ultimo
layer, quindi in ordine per layer i blocchi si sovrascriverebbero; è un modo di sola misura e non
vale un secondo modo di contare.

**Le leve del prefill sotto budget, in ordine di resa** (le prime due si moltiplicano):

| leva | guadagno | costo |
|---|---|---|
| esperti letti una volta per prompt (ordine per layer) | **2.09×** a 2048, di più sui prompt lunghi | esatto, nessuno |
| archivio che sopravvive alla sessione (domanda 45) | toglie il disco dalla seconda sessione in poi | un demone |
| sovrapporre disco e calcolo | 1.38-1.69×, tetto (modello, non misura) | un lettore, un thread |
| riordino del file per co-attivazione (mbolt, MIT) | ≤ 1.15× qui | un formato nostro |

Su mbolt (`github.com/doramirdor/mbolt`): esiste, MIT, e il suo README dichiara 2.23× **sulle
letture** e 1.55× end-to-end su Qwen3-Next-80B, 512 esperti per layer, con il modello più grande
della RAM. Dice anche dove smette di rendere: «~4 MB slices cap gains at 1.3×» e «greatest benefit
at deep offload (≳ 2.5× RAM ratio)». I nostri esperti sono 6.375 MiB e il modello sta in 7 GiB su
31 di RAM: siamo fuori dal suo campo, e il nostro `bench-disk` (1.4-1.5 GB/s a blocchi da 2.125
MiB contro 1.6-1.75 da 6.375) mette il tetto a ~1.15×. Non misurato da noi: è lettura, non misura.

## Is code behavior a small, deterministic graph? (2026-09-20)

Domanda 44 (Marcello), con le soglie scritte prima di misurare (riga 44 di §Da misurare). Sei
tracce del routing su OLMoE-1B-7B Q8_0, 300 token generati ciascuna, nel container (si contano
esperti, non secondi): quattro di codice nostro (`code-1000` C, `trace-c2` C, `trace-py` Python,
`trace-sh` shell; 842-904 token di prompt) e due di prosa di controllo (`trace-prose-it`, un pezzo
di questo progetto in italiano, 941 token; `trace-prose-en`, un testo inglese sui colibrì, 583
token e 265 generati). Traccia versione 2: anche l'id di ogni token e il margine del router
(probabilità dell'ultimo esperto scelto e del migliore escluso), esatti per prova
(`tests/test_route.c`: al layer 0 l'escluso di un modello con 2 esperti per token **è** l'ultimo
scelto dello stesso modello con 3). Tabelle da `tools/route_graph_report.py` (`--check` calcolato
a mano, mutazioni in `tools/mutate_reports.py`); file in `build/route/`.

| | code-1000 | c2 | py | sh | prosa it | prosa en |
|---|---|---|---|---|---|---|
| unità usate, di 1024 | 1012 | 998 | 1006 | 1011 | 1013 | 1018 |
| il 25% più usato copre | 68.1% | 73.2% | 72.5% | 70.0% | 60.9% | 51.7% |
| il 50% più usato copre | 87.7% | 92.5% | 91.2% | 87.7% | 83.6% | 79.2% |
| unità per il 99% delle attivazioni | 84.9% | 75.6% | 81.8% | 84.0% | 85.6% | 89.5% |
| grafo statico di co-occorrenze, primi 8 (il router vivo: 82-86%) | 53.7% | 64.2% | 64.8% | 54.3% | 59.3% | 51.3% |
| sola frequenza, primi 8 (caso: 12.5%) | 40.2% | 51.9% | 54.2% | 40.0% | 37.7% | 20.7% |
| stesso id di token → stesso insieme, layer 0 / media / ultimo | 14% / 11% / 12% | 16 / 14 / 16 | 13 / 11 / 14 | 9 / 10 / 14 | 6 / 9 / 27 | 3 / 6 / 8 |
| idem, Jaccard medio, layer 0 / media | 0.51 / 0.60 | 0.51 / 0.64 | 0.46 / 0.62 | 0.42 / 0.62 | 0.40 / 0.56 | 0.33 / 0.51 |
| esperti in comune col token prima, di 8 (caso 1.0) | 3.6 | 3.9 | 3.9 | 4.0 | 3.3 | 3.0 |
| percorsi interi ripetuti | 0 | 0 | 0 | 0 | 0 | 0 |
| margine del router, mediana (quota della probabilità dell'ultimo scelto) | 5.1% | | | | | 6.3% |

Sovrapposizione del 25% più caldo fra due tracce (Jaccard; fra parentesi la quota delle
attivazioni della seconda che cade nell'insieme caldo della prima):

| | c2 | py | sh | prosa it | prosa en |
|---|---|---|---|---|---|
| code-1000 | 0.75 (71%) | 0.73 (70%) | 0.68 (68%) | 0.34 (45%) | 0.09 (19%) |
| c2 | | 0.77 (71%) | 0.72 (68%) | 0.33 (45%) | 0.07 (17%) |
| py | | | 0.74 (68%) | 0.37 (48%) | 0.08 (17%) |
| prosa it | | | | | 0.16 (24%) |

Col 50% più caldo: codice-codice 0.72-0.77 (85-90% delle attivazioni dentro), codice-prosa inglese
0.28 (46%).

1. **Piccolo: no.** Soglia «il 25% delle unità copre il 99%»: copre il 68-73% sul codice, e per il
   99% serve il 76-85% del modello. Ogni traccia tocca il 97-99% delle unità in mille token.
2. **Tabella: no, nemmeno al layer 0.** Soglia «stesso id → stesso insieme nel 95% dei casi»: 9-16%
   sul codice (Jaccard 0.42-0.51). Lo stesso token prende esperti diversi secondo il contesto già
   al primo layer, dove fra l'embedding e il router c'è solo un'attenzione.
3. **Statico: no.** Un grafo di co-occorrenze senza stato indovina il 54-65% del layer dopo, il
   router sullo stato vivo l'82-86%; nessun percorso intero si ripete, in nessuna traccia. E le
   scelte sono strette: metà sono decise da meno del 5% della probabilità dell'ultimo esperto
   scelto (mediana 5.1%), quindi il grafo discreto è anche fragile a piccole perturbazioni.
4. **Una regione del codice: sì, netta.** Soglia «codice-codice supera codice-prosa di almeno 0.20»:
   0.68-0.77 contro 0.07-0.09 con la prosa inglese, e 0.31-0.37 con la prosa tecnica italiana
   (che di codice parla). C, Python e shell scaldano le stesse unità; il testo sui colibrì quasi
   nessuna di quelle. Il codice è anche più concentrato della prosa (25% → 68-73% contro 52-61%) e
   più prevedibile dalla sola frequenza (40-54% contro 21%). È specializzazione per dominio, non
   un grafo piccolo: l'insieme caldo di un file copre solo il 68-71% delle attivazioni di un altro.
5. Per il motore: l'LRU resta la scelta giusta dentro una sessione (3.6-4.0 esperti su 8 in
   comune col token prima); la regione del codice dice che una cache **calda fra sessioni di
   codice** parte già col 68-71% di ciò che serve (domanda 45), e non dice niente su quanto del
   modello si possa togliere: quello lo dice solo la prova funzionale qui sotto.
6. Limiti: un modello generalista da 64 esperti, sei testi, 300 token generati a traccia, prompt
   fatti di codice nostro; «deterministico» qui vuol dire prevedibile senza lo stato, non
   ripetibile (ripetibile lo è per costruzione).

**La prova funzionale** (`sh tools/mask_quality.sh`: esperti spenti con `--expert-mask`, modo di sola
misura; maschera = le unità fuori dal 75 / 50 / 25% più usato in `code-1000`, e per controllo lo
stesso numero estratto a caso; logit di ogni posizione contro il modello intero). **Cinque testi su
cinque** (2026-09-20 notte, `build/mask/quality.txt`): `code-1000` è il caso **più favorevole** (è
quello da cui la maschera è ricavata, 904 posizioni), `trace-c2` un file C mai visto (860),
`trace-py` Python (842), `trace-sh` shell (875), `trace-prose-en` prosa inglese (583), cioè il
controllo fuori dominio:

| unità tenute | | per uso: KL media | token diverso | a caso: KL media | token diverso |
|---|---|---|---|---|---|
| 75% (256 spente) | code-1000 | 0.0154 | 29 (3.2%) | 0.598 | 178 (19.7%) |
| | trace-c2 | 0.0198 | 30 (3.5%) | 0.702 | 209 (24.3%) |
| | trace-py | 0.0539 | 63 (7.5%) | 0.877 | 246 (29.2%) |
| | trace-sh | 0.084 | 83 (9.5%) | 1.26 | 319 (36.5%) |
| | trace-prose-en | 1.01 | 212 (36.4%) | 0.865 | 226 (38.8%) |
| 50% (512 spente) | code-1000 | 0.0902 | 55 (6.1%) | 4.58 | 663 (73.3%) |
| | trace-c2 | 0.103 | 62 (7.2%) | 4.70 | 644 (74.9%) |
| | trace-py | 0.276 | 118 (14.0%) | 5.02 | 713 (84.7%) |
| | trace-sh | 0.384 | 163 (18.6%) | 5.05 | 759 (86.7%) |
| | trace-prose-en | 2.41 | 350 (60.0%) | 4.43 | 486 (83.4%) |
| 25% (755 spente) | code-1000 | 0.309 | 121 (13.4%) | 8.27 | 880 (97.3%) |
| | trace-c2 | 0.334 | 117 (13.6%) | 8.39 | 848 (98.6%) |
| | trace-py | 0.691 | 199 (23.6%) | 8.84 | 822 (97.6%) |
| | trace-sh | 0.976 | 264 (30.2%) | 8.47 | 867 (99.1%) |
| | trace-prose-en | 6.37 | 550 (94.3%) | 6.35 | 549 (94.2%) |

7. **Funzionale: no.** Soglia «col 50% spento, stesso token nel 99% delle posizioni e KL ≤ 1e-2»:
   93.9% e KL 0.090 sul testo stesso della maschera, 92.8% e 0.103 su un file C mai visto, e sugli
   altri due linguaggi si scende (86.0% e 0.276 in Python, 81.4% e 0.384 in shell); nemmeno tenendo
   il 75% si passa (96.8 / 96.5 / 92.5 / 90.5%, KL 0.015-0.084). Il grafo piccolo non c'è nemmeno
   nel senso debole: per tenere il comportamento servono quasi tutti gli esperti.
8. **La maschera regge sul C mai visto, si consuma sugli altri linguaggi.** Col 50% spento il token
   coincide nel 93.9% (testo della maschera), 92.8% (altro C), 86.0% (Python), 81.4% (shell): la
   regione del punto 4 esiste ma ha un centro, e più ci si allontana dal C più costa. Non è memoria
   del testo (fra i due C c'è un punto percentuale), è distanza dal linguaggio da cui la maschera
   viene.
9. **Ma l'uso dice moltissimo su quali esperti servono**: a parità di unità spente, la maschera per
   uso sposta l'uscita 13-50 volte meno di quella a caso (KL 0.09-0.38 contro 4.6-5.1 col 50%
   spento, su tutti e tre i linguaggi). Non è un grafo piccolo, è una graduatoria ripida: il modello
   regge male la perdita degli esperti giusti e quasi per niente quella degli esperti sbagliati.
10. **Fuori dal dominio la graduatoria non vale più niente**, ed è il controllo che serviva: sulla
    prosa inglese la maschera per uso fa come quella a caso (col 75% tenuto 36.4% contro 38.8% di
    token diversi; col 25%, 94.3% contro 94.2%; KL 6.37 contro 6.35). Gli esperti caldi sul codice
    non sono «i migliori esperti», sono quelli del codice: una cache scaldata su un dominio è un
    guadagno per quel dominio e zero per un altro (domanda 45).

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

## Generated mutations (2026-09-22, 2026-09-23)

`tools/mutate_auto.py` swaps operators, drops `± 1`, swaps `return 0` / `return -1`, drops `(!`,
one at a time, rebuilds, runs the given tests and, with `--cmd`, the oracles; a mutant whose object
file is byte-identical to the original (a branch this platform does not compile, or folded by the
compiler) is set aside. A check that fails is run again on the same mutant, and only a second
failure kills it (LESSONS #115: under memory pressure a load refused had counted as a kill); a
check out of time runs again with three times the budget, and a mutant that times out twice is
listed (#117); a failure that is the machine's memory (the engine's guard, the OOM killer) is no
verdict, and that mutant is judged again alone at the end (#119: a refusal that outlasted the
repeat had killed an equivalent mutant). The rows before `tokenizer.c` were counted before #117
and #119, and may hide a few survivors until they run again. Linux
container, `--asan` where a mutant can read out of bounds without crashing; the checks and the jobs
of each file are in `tools/mutate_files.sh` (jobs sized to the VM's 15 GB, not to its processors).
Background: 2026-09-23, the other windows' containers idle (OpenEMR's stack up, not measuring).

| file | checks | mutants | killed | survived first | survived now | same object |
|---|---|---|---|---|---|---|
| `src/format/gguf.c` (ASan) | `test_gguf` | 221 | 163 (+9 timed out) | 89 | 43 | 6 |
| `src/memory/experts.c` (ASan) | `test_experts`, `test_stream` | 100 | 87 | 27 (`test_experts` alone) | 9 | 4 |
| `src/base/threads.c` | `test_base`, `test_hot` | 75 | 35 (+9 timed out) | 33 | 23 | 7 |
| `src/base/platform.c` | `test_base` | 170 | 9 (+35 timed out) | 77 | 22 | 104 (Windows branches) |
| `src/base/prof.c` (ASan) | `test_prof`, `test_model_prof` | 46 | 36 | 42 | 10 | 0 |
| `src/models/olmoe.c` (ASan) | the 10 model tests, the tiny oracle, the same under the smallest store, the options oracle | 400 | 319 (+1 timed out) | 92 (9 tests, 2 oracles) | 70 | 10 |
| `src/kernels/kernels.c` (ASan) | `test_kernels`, `test_expf`, `test_tier_used` (also under `TR_CPU_MAX=scalar`), `test_prefill`, both tiny oracles | 110 | 76 (+5 timed out) | 37 | 26 | 3 |
| `src/kernels/kernels_x86.c` (ASan) | `test_kernels`, `test_tier_used` (also scalar) | 39 | 22 (1 did not build) | 20 | 16 | 0 |
| `src/kernels/expf.c` | `test_expf` (the proof on every float is `bench_expf --check`, in `make check`) | 7 | 5 (1 did not build) | 1 | 1 | 0 |
| `src/tokenizer/chat.c` (ASan) | `test_tokenizer`, `test_cli`, the tokenizer oracle | 31 | 24 (+1 timed out) | 4 (with kills made by the OOM killer) | 6 | 0 |
| `src/tokenizer/unicode.c` (ASan) | `test_unicode`, `test_tokenizer`, the tokenizer oracle without its sweep | 120 | 80 (+11 timed out) | 27 | 17 | 2 |
| `src/tokenizer/tokenizer.c` (ASan) | `test_tokenizer`, `test_cli`, the tokenizer oracle without its sweep | 327 | 271 (+11 timed out, each a real loop; +1 refused for memory) | 75 | 43 (all read) | 1 |
| `src/app/main.c` (ASan) | `test_cli`, the tiny oracle, the tokenizer oracle without its sweep | 646 | 306 | 334 | 334 (not yet read) | 6 |

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
- `unicode.c`: 3 more (a Hangul syllable's end, U+0080's encoding length, the ASCII class table's end) die only under the tokenizer oracle's full Unicode sweep, which `make check` runs and a run per mutant cannot afford (4 GB, `tools/mutate_files.sh unicode-sweep`); 7 died to new boundary cases in `test_unicode` (each UTF-8 length at both edges, U+10FFFF, U+DFFF, the jamo just outside the composing ranges). The 17 left: out of memory and the size overflow checks of `tr_nfc` (10, a text is at most 1 GiB); `TR_CP_INVALID_BASE + 0`, 2 (byte 0 is valid ASCII, never an invalid byte); the equal case of two binary searches, 2 (returned before); the NFC fast path's test, 3 (U+00C0, the threshold, is NFC-stable alone, and slower is not different).
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
- `main.c` (**open**, 334): the command line's branches no test runs — `inspect` and `cpu` (18),
  the option parsing and usage errors of every command, `logits`, `tokenize --pieces/--decode`,
  `run --route-trace`, the chat's `/reset` and its context-full turn, the speed lines.

## Attempts

| Data | Cosa | Prima | Dopo | Spread | Esito |
|---|---|---|---|---|---|
| 2026-09-17 | `dot_row q8_0` AVX-512, bit-identico | 1 957 M el/s | 19 837 M el/s | 15% / 22% | tenuto: attivo di default sulle CPU AVX-512 |
| 2026-09-17 | `dot_row q8_0` AVX2, bit-identico | 1 957 M el/s | 19 532 M el/s | 15% / 5% | tenuto: attivo sulle CPU AVX2 senza AVX-512 |
| 2026-09-17 | OLMoE-1B-7B Q8_0 decode, kernel AVX-512 | 6.87 tok/s (16 thread) | 21.08 tok/s (8 thread) | una run | tenuto; 16 thread 17.72: il default dei thread va rivisto |
| 2026-09-17 | pool di thread: attesa attiva 2 ms, uno slot per thread (Linux/Docker) | matmul esperto 16 thread 0.376 ms | 0.014 ms | 9% / 16% | tenuto; da misurare su Windows nativo e sul modello vero (mediana di 5) |
| 2026-09-17 | OLMoE-1B-7B Q8_0 decode, pool nuovo (Windows nativo, mediana di 5) | 21.08 tok/s (8 thread, una run) | 27.02 tok/s (8 thread) | 2.4% | tenuto; 16 thread = 8 thread (26.90) |
| 2026-09-17 | rope da tabella, attivazioni in parallelo, attenzione per testa + `axpy_f32` AVX-512 (logit identici al bit) | 26.34 tok/s (16 thread) | 32.78 tok/s (16 thread) | 1.4% / 2.8% | tenuto; 4 thread = 16 thread (32.93): limite della memoria |
| 2026-09-17 | prefill a blocchi in C esatto: passate da 512 token, token per esperto, matrici a blocchi di 16 token, logit solo dell'ultimo (logit identici al bit) | prefill 30.1 tok/s (16 thread, prompt 512) | 196.9 tok/s col kernel a 4 token | 14% / 27% | tenuto; decode invariato; scala 1.16× da 8 a 16 thread (domanda 20) |
| 2026-09-17 | riga q8_0 decompressa una volta per blocco di token in uno scratch per worker, poi `dot_f32` (esatto per contratto) | prefill 144 tok/s (16 thread) | 140 tok/s | 6% / 6% | **scartato**: nessun guadagno (legge 4 byte per elemento invece di 1, e la misura non era alternata); l'idea buona è tenere il peso compresso e dividerlo fra più token (riga sotto) |
| 2026-09-17 | kernel `dot_row_x4`: una riga di pesi contro 4 token nei registri (scalare, AVX2, AVX-512), risultati identici al bit a `dot_row` | prefill 124 tok/s (16 thread, run alternate) | 180 tok/s | 12% / 16% | tenuto; da solo il kernel vale 2.1× (44 contro 21 G elementi/s) |
| 2026-09-17 | accumulatori del kernel a 4 token in un array `__m512 acc[4]` invece che in registri con nome | — | — | — | **scartato**: il compilatore li tiene sullo stack e il guadagno sparisce (LESSONS #45) |
| 2026-09-17 | blocco di token di `tr_matmul` a 32, 64, 128 invece di 16 | prefill 180.5 tok/s (16 thread, tile 16) | 176.9 / 183.3 / 172.5 | 10-13% | scartato: tutto dentro il rumore, resta 16 |
| 2026-09-18 | una riga di pesi contro **8** token (float, identico al bit a x4), solo nel banco | x4: 25.6 / 47.6 / 114.1 ns per riga (n = 1024 / 2048 / 4096) | 1.13× / 0.79× / 0.93× | 3-8% | **scartato**: a n=2048 le attivazioni di 8 righe (64 KB) escono dalla L1 (§Revisione) |
| 2026-09-18 | dot int8 VNNI con la struttura del x4 (256 bit col trucco del segno; 512 bit a due blocchi), solo nel banco, non esatto | x4 float come sopra | 256 bit 1.64× / 1.62× / 1.96×; 512 bit **1.83× / 1.79× / 2.26×** | 3-8% | misurato, non adottato: l'int8 non è bit-identico, può essere solo un modo dichiarato. Decisione aperta (§Revisione, LESSONS #65) |
| 2026-09-18 | tetto della pausa della bozza adattiva corretto (31 → 16) | replay esatto: 172 passate, 65 righe di bozza (`code`) | 172 passate, 66 righe | — | tenuto: è una correzione, non una leva; le costanti spostano ±2-3% (§Revisione) |
| 2026-09-18 | thread per fase: prompt su tutto il pool, passate corte (fino a 4 righe) sui primi n slot, n misurato dalla sessione (16/8/4, la più ampia entro l'1% dalla più veloce, di nuovo ogni 1024 token), `--decode-threads` lo forza; token identici al bit | decode 30.82 tok/s a contesto 512, 23.79 a 2048 (16 thread) | 33.45 (**1.085×**) e 24.43 (**1.027×**); con 8 forzato 33.50 e 24.98 (1.050×); prefill invariato; `--spec 8` caso peggiore 1.064× | A/A 1.0% | **tenuto**, acceso di default. Margine del 3% e del 2% misurati e scartati (a 2048 tenevano 16 thread in 10 e in 5 run su 16); confine a 16 righe invece di 4: non distinguibile (§Thread per fase) |
| 2026-09-19 | KV con le posizioni di una testa in fila (`[layer][testa][posizione]`, `src/kv/`), logit identici al byte | decode a contesto 512 / 2048 / 4000: 32.30 / 24.18 / 17.91 tok/s (8 thread forzati) | 35.48 (**1.06-1.10×**) / 27.32 (**1.12-1.14×**) / 20.84 (**1.15-1.18×**); prefill a 2048 **1.10×**, a 4000 **1.39-1.43×**; a contesto 32 non distinguibile | A/A 3.8% | **tenuto**: la zona `attention` passa da 32-35 a 47-48 GB/s letti, su 54 della RAM (§Decode a contesto lungo) |
| 2026-09-19 | larghezza per zona: l'attenzione del decode su 16 thread e il resto su 8 (esatto), stimata dalle zone di due profili | token a contesto 4000: 48.1 ms | 47.1 ms stimati (−2.1%); a 2048 −0.7% | A/A 3.8% | **non scritta**: sotto la soglia della sessione; domanda 34 |
| 2026-09-19 | attenzione del prompt a gruppi di 16 token per testa, blocchi di 64 posizioni, `dot_f32_x4` e `axpy_f32_x4` (logit identici al byte) | zona `attention` a 512 / 2048 / 4000: 116 / 1883 / 7420 ms | 106 / 1688 / 6568 (1.09× / 1.12× / **1.13×**) | A/A 2.1% | **tenuto**; nel banco 1.15× a 4000, 1.38× quando la L3 è in affanno (§Prefill su prompt lunghi) |
| 2026-09-19 | lavoro di un token solo sul pool (norme, RoPE, KV, router, righe per gli esperti) e righe F32 col kernel del tier | zone per token 121 / 455 / 880 ms, `router` 80 / 324 / 627 | 48 / 174 / 353 e 6 / 22 / 45 | A/A 2.1% | **tenuto**; con la riga sopra il prefill fa **1.05-1.08× / 1.07-1.08× / 1.11-1.14×** a 512 / 2048 / 4000, decode non distinguibile (A/A 2.4%) |
| 2026-09-19 | righe di pesi **F16** con i kernel del tier (`vcvtph2ps`, conversione esatta), bit-identiche allo scalare; nate dal controllo `test_tier_used` (LESSONS #78) | `dot_row f16` 466 M el/s in ogni tier (n=2048) | 22.8 G el/s AVX2, 22.7 AVX-512 (**49×**) | 3-7% | **tenuto**; nessun modello F16 negli scenari: il numero è del kernel, non di un prefill |
| 2026-09-19 | gruppo da 4 o 64 query, blocco da 16 o 256 posizioni invece di 16 × 64 (solo nel banco) | 355 ms per layer a 4000 | 352-387 | 2-15% | scartato: tutto dentro lo spread, resta 16 × 64 |
| 2026-09-19 | **`tr_expf` scalare**, arrotondato correttamente su tutti i 2^32 float (prova esaustiva in `make check`), al posto dell'`expf` della libreria in softmax e SwiGLU; Windows: logit identici al byte; Linux: KL 3.9e-13, stessi byte dell'emulazione | prefill 296 / 249 / 211 tok/s a 512 / 2048 / 4000; decode a 8 thread 35.1 / 28.0 / 21.4; `attention` 92 / 1537 / 5960 ms, `expert_act` 138 / 595 / 1201 | 306-315 / 303-308 / 276; 36.0 / 29.3 / 23.2; 41 / 591 / 2331 e 25 / 108 / 218 | A/A 3.0% prefill, 1.9% decode | **tenuto**: prefill 1.03-1.08× / 1.20-1.23× / 1.29-1.31×, decode 1.02-1.10×; Windows e Linux ora danno gli stessi byte (§`tr_expf`) |
| 2026-09-19 | `tr_expf` in SIMD (non scritto: i numeri per decidere) | dopo lo scalare, softmax 9 / 130 / 610 ms e `expert_act` 25 / 108 / 218 ms del prefill | stima: prefill 1.01× / 1.03× / 1.04×, decode +1-2%, circa 200 righe | — | **decide Marcello** (domanda 39): lo scalare ha preso quasi tutto |
| 2026-09-19 | rimisura a macchina pulita di thread per fase e decode a contesto lungo (quattro `yes` dimenticati sotto le misure del 18-19/09, LESSONS #84) | decode a 512, 8 contro 16 thread: 1.10×; RAM ~54 GB/s | 1.00-1.02× (non distinguibile a nessun contesto); RAM ~57 GB/s con 4-6 thread; a 512 il decode vuole 4 thread | A/A 2.5-4.6% | conclusione corretta: pochi thread sì, il «+10%» no; lo stimatore della larghezza va riscritto (domanda 31, §Rimisura a macchina pulita) |
| 2026-09-19 | stimatore della larghezza del decode riscritto (domanda 31, LESSONS #88): margine dal rumore misurato nelle passate stesse (non più un tetto fisso), scelta a coppie (mai il rumore di una terza larghezza), estensione fino a `TR_DECODE_TUNE_ROUNDS_MAX` sui due contendenti, nuova misura ad ogni classe di contesto e dopo `TR_DECODE_TUNE_REMEASURE_KEPT` passate mantenute con un cambio in sospeso, isteresi a due voti prima di cambiare scelta | — | — | — | test C verdi (`tests/test_phase.c`, `tools/mutate_tune.sh`); **da validare sul modello vero**, di notte a macchina tranquilla: nessun numero ancora |
