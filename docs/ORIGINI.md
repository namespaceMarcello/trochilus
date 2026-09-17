# Origini del codice

Registro di ogni file portato da colibri o ds4. Serve a due cose: rispettare le licenze
(Apache-2.0 chiede di dire cosa è cambiato) e sapere da dove riportare i miglioramenti futuri.
Si aggiorna nello stesso commit che porta il codice.

## Riferimenti fissati

| Progetto | Licenza | Commit | Worktree locale | Clone d'origine |
|---|---|---|---|---|
| colibri (`JustVugg/colibri`, branch `dev`) | Apache-2.0 | `a90bed9` (2026-09-17) | `ref/colibri` | `Desktop\colibri` |
| ds4 (`antirez/ds4`, branch `main`) | MIT (contiene codice ggml, MIT) | `8db1d1d` (2026-09-16) | `ref/ds4` | `Desktop\ds4` |
| llama.cpp (`ggml-org/llama.cpp`, `master`) — solo riferimento di confronto, nessun codice portato | MIT | `b49650a` (2026-09-17) | `ref/llama.cpp` (clone superficiale; build in `build-trochilus/` con `tools/build_llamacpp.sh`) | — |

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
| pesi con `pread` invece di `mmap` | colibri | `c/st.h` (commento sul bug RSS) | `src/base/platform.h` |
| oracolo su modelli minuscoli generati | colibri | `c/tools/make_*_tiny.py`, job CI | `tools/`, `make oracle` |
| tokenizer letto dai metadati GGUF, merge "a b" come chiave del rango | ds4 | `ds4.c` `vocab_load`, `bpe_rank` | `src/tokenizer/tokenizer.c` (il merge diventa coppia di id → rango e risultato) |
| pretokenizer che rigioca la regex in C sui codepoint; classi Unicode da tabelle generate | colibri | `c/tok.h` `pretok_chunk`, `c/tok_unicode.h` | `src/tokenizer/tokenizer.c` `split_gpt2`, `tools/gen_unicode_tables.py` (tabelle sondate su HF, non su Python) |
