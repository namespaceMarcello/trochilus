# Fatto

Archivio, in coda. Una voce per commit o per passo chiuso: `### <data> — <titolo>`,
cosa è stato implementato e come si prova.

### 2026-09-17 — Nascita del repo
Repo locale `Desktop\trochilus` (branch `main`, LF). Licenza Apache-2.0 con `NOTICE` che
attribuisce colibri (Apache-2.0) e ds4 (MIT, testo in `licenses/`). Sorgenti di riferimento
fissati come worktree in `ref/` (ignorati da git): colibri `a90bed9`, ds4 `8db1d1d`.
Hook che rifiuta i commit di codice senza voce qui.

### 2026-09-17 — M0 passo 1: base, GGUF, convertitore
`src/base/` (file con pread, memoria, tempo, log; pool di thread senza OpenMP; rilevamento CPU a
runtime con controllo XGETBV), `src/format/gguf.c` (lettore GGUF v2/v3 che valida ogni misura
contro il file), `tools/hf_to_gguf.py` e `tools/check_gguf.py` (gguf-py 0.19.0, OLMoE), comandi
`trochilus cpu` e `trochilus inspect`. Prova: `make WERROR=1 test`, poi
`build/trochilus cpu` e `build/trochilus inspect fixtures/tiny-olmoe/model-q8_0.gguf`.

### 2026-09-17 — Test di sicurezza del lettore GGUF
`tests/test_gguf.c`: file valido letto campo per campo; lo stesso file troncato a ogni byte
(mai aperto); 22 corruzioni mirate (magic, versione, conteggi enormi, stringhe con NUL, tipi e
dimensioni false, offset disallineati, fuori file o che traboccano, nomi e chiavi duplicati), tutte
rifiutate con messaggio; 3000 file mutati a caso, quelli aperti hanno tutti i tensori leggibili
nei limiti. Pulito sotto AddressSanitizer + UBSan + leak check (gcc, Linux). Prova: `make test`.

### 2026-09-17 — Microbenchmark dei kernel e prima misura
`tests/bench_kernels.c`: per ogni tier disponibile e ogni kernel (dot_f32, dot_row f16/q8_0) su
righe da 64 a 4096 elementi, mediana di N run e spread (rumore); più `tr_matmul` di una matrice
da esperto OLMoE con 1 thread e con i core fisici. Base scalare registrata in `docs/MISURE.md`.
Prova: `gcc -std=c11 -O2 -ffp-contract=off tests/bench_kernels.c src/base/*.c src/format/*.c src/kernels/*.c -o build/bench/bench_kernels.exe && build/bench/bench_kernels.exe`.

### 2026-09-17 — Profiler del motore
`src/base/prof.h/.c`: zone fisse per ogni fase del passaggio (embedding, proiezioni, norme, rope,
scrittura KV, attenzione, router, esperti, lm_head, campionamento, letture dal disco, attese del
pool), prefill e decode separati, byte di pesi toccati per token e banda di memoria effettiva.
Vive nella sessione, costa un salto quando è spento, conta con RDTSC (TSC invariante, calibrato
sul clock del sistema). Report a tabella e JSON. Prova: `make test` (`tests/test_prof.c`).
Non ancora collegato al grafo OLMoE (lo sta scrivendo l'agente del passo 2).

### 2026-09-17 — Ciclo di controllo
`CLAUDE.md` §ciclo di controllo (7 punti: prima, errori, ogni errore diventa un controllo, ogni
scenario entra nei test, `make check` prima di "fatto", verifica dopo un agente, documenti);
`docs/LEZIONI.md` con le prime 13 lezioni (causa, prevenzione, stato, trovato da); hook di commit
che rifiuta `src/` cambiato senza test, oracolo o scenario (salvo `no-test: <motivo>`);
`test_parallel_varying_chunks` in `tests/test_base.c` per le due race del pool (LEZIONI #4).
Prova: hook provato su 5 casi (2 rifiuti, 3 accettati); `make test`.

### 2026-09-17 — Gradino 0: OLMoE minuscolo esatto, e il cancello `make check`
Kernel scalari (`src/kernels/kernels.c`, contratto a 16 corsie), grafo OLMoE (`src/models/olmoe.c`),
registro architetture e guardia della memoria (`src/models/model.c`), comandi `generate` e `logits`,
`tools/oracle.py`. Oracolo contro transformers: f32 16/16 token (scarto logit 2.1e-7), f16 16/16
(3.2e-4), q8_0 16/16 (8.2e-3). `make check`: lint (`tools/lint.py`: caratteri di controllo nei
documenti, tabella dei tipi contro ggml, `tmpfile` nei test, tetti dei documenti, tabella lezioni),
build Windows con 0 warning, poi in Docker `trochilus-dev` test gcc e clang, test sotto ASan+UBSan,
oracolo. Il lint ha trovato iq1_s sbagliato nella tabella di ds4 (corretto). Prova: `make check`.

### 2026-09-17 — Profiler collegato a OLMoE, suite di scenari
`tr_session_prof(tr_session*)` (unica aggiunta a `model.h`, con la voce `prof` in `tr_arch_vtable`):
la sessione possiede un `tr_prof`, spento di default. `src/models/olmoe.c` cronometra ogni zona
del passaggio in avanti (embedding, norme, proiezioni QKV, rope, scrittura KV, attenzione, router,
i quattro passi dell'esperto, norma finale, lm_head) e conta i byte di pesi toccati per ogni matmul
ed embedding; non ancora collegati POOL_WAIT e WEIGHT_READ. `trochilus generate` ha `--profile`
(tabella su stderr), `--profile-json <file>` e `-p <n>` (prompt sintetico deterministico, niente
tokenizer); prefill e decode separati, campionamento nella zona `sample`. `tools/profile_suite.py`
(`make profile`) lancia `bench/scenarios.json` (modello minuscolo f32/q8_0) più volte, mediana e
spread per fase e per zona, confronto con la misura precedente sulla stessa macchina (migliorato /
peggiorato / rumore); `--smoke` (in `make check-linux`, dopo l'oracolo) verifica che `--profile`
non cambi i token generati. `tests/test_model_prof.c`: un OLMoE minuscolo costruito a mano (niente
fixture esterna) prova che il profiler è spento di default e che accenderlo non cambia i logit
(bit a bit) rispetto a una sessione gemella senza profiler. Prova: `make check`, poi `make profile`.

### 2026-09-17 — Flag -c, contesto pieno come errore, OLMoE-1B-7B scaricato
`trochilus generate -c <token>` limita la cache KV (per profilare un modello vero con poca memoria);
un contesto che si riempie prima della fine ora esce con codice 3 invece di 0 (trovato dal test
nuovo del cancello). Hook contro le barre rovesciate nelle scritture da shell. Scaricato
`models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf` (AllenAI, 7.36 GB, sha256 verificato). Prova:
`make check`; `build/trochilus inspect models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf`.

### 2026-09-17 — Kernel AVX2 e AVX-512 per dot_f32 e dot_row q8_0
`src/kernels/kernels_x86.c`: varianti AVX2 e AVX-512 dei due kernel caldi, scelte all'avvio dalla
CPU (AVX-512, poi AVX2, poi scalare; `TR_CPU_MAX` le limita), bit-identiche allo scalare. Parti comuni
in `kernels_internal.h`; `tr_rmsnorm` usa il tier attivo. `test_kernels` confronta ogni tier con lo
scalare su tutte le code e su valori speciali, e verifica di accorgersi di una variante sbagliata
apposta. Kernel q8_0 10× più veloce su un thread; OLMoE-1B-7B Q8_0 da 6.9 a 21.1 token/s (8 thread),
stessi token dello scalare (`docs/MISURE.md`). `Makefile`: con gcc le istruzioni AVX allineate diventano
non allineate (crash sotto ASan e su MinGW), e un flag cambiato ricompila tutto. Prova: `make check`,
`make bench`, `build/trochilus generate -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -p 32 -n 16 -c 128 -t 8 --profile`.

### 2026-09-17 — Zona calda, pool con attesa attiva, misure ripetute
Zona calda marcata nel codice (`/* hot: begin */`): `tools/lint.py` rifiuta allocazioni, stringhe,
I/O e matematica per elemento, con un autotest del controllo; `tests/test_hot.c` conta le allocazioni
mentre il modello genera (zero) e verifica logit identici con 1/2/3/8 thread. Pool di thread nuovo:
uno slot per thread, attesa attiva 2 ms poi sonno (dispatch a 16 thread da 53 a 1.5 µs su Windows).
Il cancello aggiunge ThreadSanitizer, `test_hot` 20 volte e la compilazione del benchmark.
`profile_suite.py`: campo `context`, minimo e massimo, token identici fra run e fra thread;
scenari del modello vero in `bench/scenarios-olmoe-1b-7b.json`. Prova: `make check`, `make bench`,
`make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json`.

### 2026-09-17 — Parte a thread singolo: rope da tabella, attivazioni e attenzione in parallelo
Rope: tabella cos/sin per posizione costruita alla creazione della sessione (`tr_rope_table`), il
forward non fa più trigonometria. Esperti a stadi (gate+up di tutti, poi le attivazioni di tutti come
un solo lavoro parallelo `tr_swiglu(pool, ...)`, poi down, poi somma pesata). Attenzione: una testa
per volta in `tr_attention_head` (somma pesata con il kernel nuovo `axpy_f32`, scalare + AVX2 +
AVX-512), teste in parallelo con una riga di punteggi per thread. Logit del modello vero identici al
bit a prima. Prova: `make check`; `trochilus logits` sugli stessi token con il binario di prima e di
dopo, `cmp` dei file; `make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json`.
