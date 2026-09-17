# Origini del codice

Registro di ogni file portato da colibri o ds4. Serve a due cose: rispettare le licenze
(Apache-2.0 chiede di dire cosa è cambiato) e sapere da dove riportare i miglioramenti futuri.
Si aggiorna nello stesso commit che porta il codice.

## Riferimenti fissati

| Progetto | Licenza | Commit | Worktree locale | Clone d'origine |
|---|---|---|---|---|
| colibri (`JustVugg/colibri`, branch `dev`) | Apache-2.0 | `a90bed9` (2026-09-17) | `ref/colibri` | `Desktop\colibri` |
| ds4 (`antirez/ds4`, branch `main`) | MIT (contiene codice ggml, MIT) | `8db1d1d` (2026-09-16) | `ref/ds4` | `Desktop\ds4` |
| llama.cpp (`ggml-org/llama.cpp`, `master`) — riferimento di confronto e fonte di idee, nessun codice portato | MIT | `b49650a` (2026-09-17) | `ref/llama.cpp` (clone superficiale; build in `build-trochilus/` con `tools/build_llamacpp.sh`) | — |

Per spostare un riferimento: `git -C <clone> fetch`, poi `git -C ref/<progetto> checkout --detach <commit>`,
e si aggiorna la tabella. I file già portati restano legati al commit scritto nella loro riga.

## Intestazione obbligatoria di un file derivato

```c
/* <nome file> — <cosa fa>.
 * Derived from <colibri|ds4> <commit> <percorso> (<licenza>), modified: <cosa è cambiato, una riga>. */
```

## File portati

| File Trochilus | Da | Commit | Percorso d'origine | Cosa è cambiato |
|---|---|---|---|---|
| `src/format/gguf.c` (solo la tabella dei tipi) | ds4 | `8db1d1d` | `ds4.c` `gguf_types[]` | aggiunti tq1_0/tq2_0, iq4_nl corretto a blocco da 32 × 18 byte; il parser è nuovo |
| `tools/make_tiny_olmoe.py` | colibri | `a90bed9` | `c/tools/make_olmoe_tiny.py` | GQA 4/2, 16 token, logit di riferimento per ogni posizione |

## Idee prese senza codice

Non richiedono attribuzione, si registrano per sapere dove guardare quando si migliora quel pezzo.

| Idea | Da | Dove l'origine | In Trochilus |
|---|---|---|---|
| pool di thread persistente, niente OpenMP | ds4 | `ds4.c` `ds4_parallel_for` | `src/base/threads.c` |
| thread sui core fisici, non logici | colibri | `c/omp_tune.h` | `tr_pool_create(0)` |
| ogni thread del pool fissato a un processore logico suo | llama.cpp (le due chiamate di sistema, e l'idea di una maschera per thread di `--cpu-strict`) | `ggml-cpu.c` `ggml_thread_apply_affinity`, `ggml_thread_cpumask_next` | `src/base/platform.c` `tr_thread_pin`, ordine degli slot in `src/base/cpu.c`, pin nel pool in `src/base/threads.c` |
| pesi con `pread` invece di `mmap` | colibri | `c/st.h` (commento sul bug RSS) | `src/base/platform.h` |
| oracolo su modelli minuscoli generati | colibri | `c/tools/make_*_tiny.py`, job CI | `tools/`, `make oracle` |
| tokenizer letto dai metadati GGUF, merge "a b" come chiave del rango | ds4 | `ds4.c` `vocab_load`, `bpe_rank` | `src/tokenizer/tokenizer.c` (il merge diventa coppia di id → rango e risultato) |
| pretokenizer che rigioca la regex in C sui codepoint; classi Unicode da tabelle generate | colibri | `c/tok.h` `pretok_chunk`, `c/tok_unicode.h` | `src/tokenizer/tokenizer.c` `split_gpt2`, `tools/gen_unicode_tables.py` (tabelle sondate su HF, non su Python) |
| prompt in passate da al più 512 token | llama.cpp | `common/common.h` `n_ubatch` | `olmoe.c` `OLMOE_DEFAULT_BATCH`, `-b` |
| coppie (token, esperto) ordinate per esperto con un counting sort, un lavoro per esperto | ds4 (llama.cpp con una mappa di indici) | `ds4.c` `layer_routed_moe_batch`; `ggml-cpu.c` `ggml_compute_forward_mul_mat_id` | `olmoe.c` `forward_pass`, `tr_matmul_grouped` |
| una riga di pesi contro un blocco di token, invece di tutta la matrice per ogni token | llama.cpp (blocchi 16 × 16), ds4 e colibri (riga × tutti i token) | `ggml-cpu.c` `..._mul_mat_id_one_chunk`; `ds4.c` `matmul_q8_0_batch_worker`; `c/olmoe.c` `matmul` | `kernels.c` `matmul_tiled` (blocchi di `TR_MATMUL_TILE` token) |
| una riga di pesi contro più token nei registri (2 in ds4, 4 in Trochilus): un solo carico e una sola conversione per tutti | ds4 | `ds4.c` `dot_q8_0_row_2` | `kernels.c` `k_dot_row_x4_q8_0`, `kernels_x86.c` `avx512_dot_row_x4_q8_0` |
| K e V del lotto scritti tutti, poi attenzione in parallelo per (testa, token) | colibri (ds4 in opzione) | `c/olmoe.c` `attention`; `ds4.c` `layer_attention_prefix_batch` | `olmoe.c` `attn_body` |
| logit solo dell'ultimo token del prompt | tutte e tre | `c/olmoe.c` `step`; `ds4.c` `output_logits_one` | `olmoe_eval` |
| bozza dal testo già nel prompt: cerca all'indietro l'ultima occorrenza dell'n-gramma di coda e propone ciò che la seguiva | colibri (scansione esatta, senza cache), llama.cpp (cache di n-grammi con conteggi) | `c/deepseek_v4.c` `v4_ngram_draft`; `common/ngram-cache.cpp` `common_ngram_cache_draft` | `src/gen/lookup.c` `tr_lookup_draft` (codice nuovo) |
| verifica di 1 + k posizioni in una passata sola, accetta finché il token proposto è quello che il modello avrebbe scelto | tutte e tre | `examples/lookup/lookup.cpp`; `ds4.c` `metal_graph_verify_decode2_exact`; `c/deepseek_v4.c` (ciclo di accettazione) | `src/gen/greedy.c` `tr_greedy_step` |
| annullare la bozza rifiutata troncando l'indice della cache | llama.cpp (ds4 e colibri non possono: attenzione ricorrente, devono ripristinare un'istantanea) | `llama_memory_seq_rm` | `tr_session_rewind` (`s->pos = n`) |

### Prefill a blocchi: le tre fonti (2026-09-17)

| | llama.cpp `b49650a` | ds4 `8db1d1d` | colibri `a90bed9` |
|---|---|---|---|
| spezzatura del prompt | `n_ubatch` 512 | nessuna su CPU: tutto il prompt; FFN a sotto-lotti da 128 (`DS4_PREFILL_BATCH`) | tutto il prompt in `step()` |
| proiezioni | sgemm, attivazioni Q8_0 | riga di pesi × token a coppie, attivazioni Q8_0 | riga di pesi × token, float |
| MoE | righe per esperto con mappa di indici, blocchi 16 × 16, dot Q8 × Q8 | counting sort per esperto; gate e up con una sola quantizzazione; down e somma per riga, esperti in ordine di id | nessun raggruppamento: token per token |
| attenzione | flash attention o matrice con maschera | scrittura KV seriale, poi attenzione per token (in opzione per testa e token) | K e V tutti, poi (testa, token) in parallelo |
| esattezza contro un token per passata | no (attivazioni int8) | no, nessun test | non dichiarata né provata |

Non preso: le attivazioni int8 (non esatte: leva 2, opzione a parte) e la somma di ds4 per riga su
tutti gli esperti, che a ogni riga di pesi rilegge le attivazioni intermedie di tutto il lotto (con 512
token ~16 MB per riga): Trochilus tiene l'ordine per esperto a blocchi di token, e somma per token.

### Collocamento dei thread: le tre fonti (2026-09-17)

| | llama.cpp | colibri | ds4 |
|---|---|---|---|
| quanti thread | `logici / 2`, o i core fisici dove il conteggio Windows è compilato (escluso su MinGW-w64, UPSTREAM #4) | core fisici contati davvero (`GetLogicalProcessorInformationEx`, `thread_siblings_list`, `hw.perflevel0`), e mai indovinati | `omp_get_max_threads()` |
| dove vanno | in nessun posto: senza `--cpu-mask` non chiama l'affinità | `OMP_PROC_BIND=close` passato a libgomp, solo Linux, con un re-exec del processo | niente |
| chi decide la maschera | l'utente, a mano (`--cpu-mask`, `--cpu-strict` = un processore per thread) | libgomp | — |
| oltre 64 processori | non ci arriva (`SetThreadAffinityMask`, UPSTREAM #3) | dipende da libgomp | — |

Preso: le due chiamate di sistema (Windows e Linux) e l'idea di `--cpu-strict`, cioè una maschera
da un processore solo per ogni thread invece di una maschera comune. Non preso: la maschera scritta
a mano dall'utente (qui la topologia la conta il motore), `OMP_PROC_BIND` con il re-exec (nessun
OpenMP, e le variabili d'ambiente lette dentro il motore sono un anti-pattern di colibri), e
`SetThreadAffinityMask` (sostituito da `SetThreadGroupAffinity`, che arriva a ogni gruppo).
Nuovo qui: l'**ordine** degli slot (un thread per core fisico prima di ogni fratello SMT, e i core
presi a giro sulle cache di ultimo livello, perché 4+4 sui due chiplet batte 8 sullo stesso), il
rispetto di una maschera già imposta al processo (`taskset`, `start /affinity`), il ripristino
dell'affinità del chiamante quando il pool muore, e un test che **chiede al sistema operativo** su
quale processore ha girato ogni chunk (`tests/test_base.c`): nessuna delle tre fonti lo verifica.

### Speculazione sul prompt: le tre fonti (2026-09-17)

| | llama.cpp | ds4 | colibri |
|---|---|---|---|
| da dove viene la bozza | tre cache di n-grammi (contesto, sessione precedente, corpus), n da 4 a 1, voto `conteggio × conteggio nel corpus` | nessun prompt lookup: DSpark e MTP, cioè teste addestrate (`docs/SPECULATIVE_DECODING.md`) | `v4_ngram_draft`: n-gramma 3 poi 2, scansione all'indietro, occorrenza più recente |
| verifica | una `llama_decode` sul lotto della bozza, logit a ogni posizione | 2-3 righe per passata, `row0_top == draft1` | ciclo `predictions[i] == drafts[i]` |
| annullare un rifiuto | troncamento della cache per indice | scambio di buffer con un'istantanea (stato ricorrente) | `spec_attention_restore` e riesecuzione del prefisso (misurata così cara da disattivare MTP) |
| token identici alla generazione senza speculazione | algebricamente sì, bit a bit **non dichiarato** (il lotto può sommare in un altro ordine) | dichiarato **non** identico al bit sulle continuazioni lunghe | non dichiarato |

Preso: l'idea della bozza dal contesto (colibri, la più semplice: nessuno stato) e la verifica in una
passata sola con i logit di ogni posizione (tutte e tre). Non preso: le cache con conteggi di
llama.cpp (allocazioni e stato, da valutare solo se le misure lo chiedono), le teste addestrate
(DSpark, MTP, EAGLE3: sono altri modelli), il campionamento «opportunistico» di ds4. Il ripristino per
istantanea non serve: la nostra cache è indicizzata per posizione, si torna indietro con `s->pos = n`.
In più, qui i token sono **identici al bit**, perché ogni riga del lotto è lo stesso calcolo della
passata da un token (`tests/test_prefill.c`, `tests/test_spec.c`): nessuna delle tre fonti lo prova.

