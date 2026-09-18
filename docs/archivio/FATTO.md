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

### 2026-09-17 — Tokenizer BPE dai metadati GGUF, esatto contro transformers; `trochilus run` con testo
`src/tokenizer/`: token, tipi e merge letti dal GGUF (il merge diventa coppia di id → rango e
risultato), added token cercati leftmost-longest per primo byte, NFC e classi `\p{L}` `\p{N}` `\s`
da tabelle generate sondando HF `tokenizers` 0.22.2 (`tools/gen_unicode_tables.py`), regole GPT-2
rigiocate sui codepoint, BPE con coda di priorità (n log n). Famiglie ammesse solo con oracolo: oggi
`olmo`. Comandi `trochilus tokenize` (anche a lotti, pezzi, decodifica) e `trochilus run` (testo in
entrata, generazione greedy in uscita). Convertitore: vocabolario come `convert_hf_to_gguf.py` di
llama.cpp e modalità `--vocab-only`. Windows: argomenti in UTF-8 (`wmain`), stdout binario.
Lint: niente variabili statiche mutabili senza `global-ok`. Test con timeout.
Prova: `make check` (test C `test_tokenizer` e `test_unicode`, `make oracle-tokenizer`: 20 745 testi
con sweep di tutti i codepoint, 0 differenze su id, pezzi, testo normalizzato e decodifica; metadati
del GGUF vero identici); `trochilus run -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -f prompt.txt -n 120`.

### 2026-09-17 — `trochilus chat`: conversazione con il template del modello
`src/tokenizer/chat.c`: il template di chat del GGUF si riconosce dai suoi byte esatti (lunghezza e
FNV-1a) e si rende in C; un template sconosciuto si rifiuta (oggi: OLMoE-0125-Instruct).
`tr_session_rewind` nel modello: a ogni turno la chat rende tutta la conversazione, tiene i token
già in cache uguali e calcola solo il resto. Console Windows letta in UTF-16 (`tr_stdin_line`),
caratteri spezzati fra due token stampati interi. Comandi `trochilus chat` e `chat-template`
(per l'oracolo). Prova: `make check` (`test_session`: logit identici al bit dopo il rewind;
`make oracle-tokenizer`: 600 conversazioni uguali ad `apply_chat_template`, testo e token;
`make chat-check`: sul modello vero la seconda risposta della chat è identica a `run` sulla
conversazione intera); a mano: `build/trochilus chat -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf`.
Senza `-c`, se la RAM non basta per 4096 token la chat dimezza il contesto e lo scrive (LEZIONI #37).

### 2026-09-17 — Confronto con llama.cpp sul modello vero (strumenti)
`ref/llama.cpp` (commit `b49650a`, solo riferimento) compilato nel container da
`tools/build_llamacpp.sh`, con `tools/llamacpp_logits.c` (logit per posizione nello stesso formato di
`trochilus logits`, generazione greedy). `tools/compare_llamacpp.py` confronta tokenizzazione, greedy
e logit per posizione (parola migliore, KL, margini). Risultati in `docs/MISURE.md`. Prova:
`sh tools/build_llamacpp.sh` e `tools/compare_llamacpp.py ... --prompt bench/prompts/dante.txt` nel
container. `make check` su Windows ora svuota alla fine la cache della VM di Docker.

### 2026-09-17 — Oracolo transformers sul modello vero tagliato a 2 layer
`tools/make_olmoe_2layer_gguf.py` copia i primi 2 layer del GGUF vero (tensori byte per byte, solo
`block_count` cambia); `tools/make_olmoe_2layer_ref.py` costruisce OLMoE in transformers con la
configurazione del GGUF e i pesi dequantizzati, e scrive token greedy e logit per posizione di due
prompt (27 e 1024 token). `tools/oracle.py` legge anche il formato a più prompt (`--logit-tol`).
`make oracle-real` entra in `make check` (saltato senza il modello): token identici, logit entro
1e-3 (misurato 2.4e-4). Prova: `make check`, oppure nel container
`make BUILD=build/linux-gcc CC=gcc oracle-real`. Numeri in `docs/MISURE.md`.

### 2026-09-17 — Confronto di velocità con llama.cpp e colibri
`tools/speed_compare.py`: stessi thread e stesse lunghezze per Trochilus (`generate`), llama.cpp
(`llama-bench`, decode alla stessa profondità di contesto) e colibri (`olmoe` sulla sua conversione
int8), mediana di N run; con `--tok-file` anche la velocità dei tokenizer (Trochilus, `llama-tokenize`,
HF `tokenizers`). `tools/build_llamacpp.sh` compila anche `llama-bench`. `trochilus generate` ora
stampa token e valutazioni e divide per le valutazioni; `make check` lo verifica. Numeri in
`docs/MISURE.md`. Prova, nel container con i modelli nel volume `trochilus-models`:
`tools/speed_compare.py --model /models/<gguf> --trochilus build/linux-gcc/trochilus --llama-bench
ref/llama.cpp/build-trochilus/bin/llama-bench --threads 16,8 --prompt 32 --gen 32`.

### 2026-09-17 — Prefill a blocchi in C esatto
`tr_session_eval` corre in passate da al più `n_batch` token (default 512; `tr_session_create` prende
`n_batch`, `-b` in `generate`/`run`/`chat`/`logits`). In una passata: K/V di tutti i token, attenzione
per (testa, token); router per token e counting sort delle coppie (token, esperto) per esperto;
`tr_matmul_grouped` (nuovo, `kernels.c`) per gate/up/down, e `tr_matmul` che visita i token a blocchi
di 16 contro ogni riga di pesi; somma degli esperti per token in ordine di id; logit solo dell'ultimo
token. Il decode è la passata da un token. Prompt 512 a 16 thread: 30 → 197 tok/s (col kernel qui
sotto), decode invariato,
logit identici al bit al binario di prima. Prova: `make check` (`tests/test_prefill.c`: 90 casi f32/Q8_0,
n_batch, chiamate, thread, rewind; `test_kernels`: gruppi vuoti e righe spezzate fra thread;
`tools/oracle.py`: `logits -b 3/64/tutto` al byte su tiny e OLMoE a 2 layer); velocità con
`tools/speed_compare.py ... --prompt 512`.

### 2026-09-17 — Profilo del prefill, kernel a 4 token, confronti alternati
`tools/profile_suite.py` riporta le zone anche in prefill e accetta `batch` in uno scenario;
`bench/scenarios-olmoe-1b-7b.json` ha gli scenari `prefill512-t16/8/4/1`. Il profilo dice che il
prefill sta al 90% nelle moltiplicazioni (attenzione 2%, seriale 8%), quindi kernel `dot_row_x4`
(scalare, AVX2, AVX-512): una riga di pesi contro 4 token nei registri, un carico e una conversione
per quattro prodotti, ognuno identico al bit al suo `dot_row`. `TR_MATMUL_TILE` si può cambiare da
riga di compilazione. `tests/bench_kernels.c` misura `dot_row q8_0 x4` e `tr_matmul` con 64 token;
`tests/test_kernels.c` confronta x4 con `dot_row` su ogni livello SIMD. Nuovo `tools/ab_speed.sh`:
due binari alternati run per run, perché il portatile che si scalda falsa i confronti in sequenza
(LEZIONI #46). Prova: `make check`, `make bench`, e
`sh tools/ab_speed.sh models/<gguf> build/base/b/trochilus build/linux-gcc/trochilus` nel container.

### 2026-09-17 — Perché il prefill non scalava: dove Windows mette i thread
Misurato sul nativo (in Docker la topologia non è quella vera, LEZIONI #47): il clock sotto carico
dal contatore di Windows e il prefill con l'affinità del processo fissata, casi alternati a ogni giro.
Il prefill a 16 thread non era limitato dalla potenza (il clock scende del 6-8% da 1 a 16 thread) né
dal CCD (8 thread divisi 4+4 sui due chiplet vanno il 7-9% meglio di 8 su uno solo): lo scheduler
appoggiava due dei 16 thread sullo stesso core fisico. Un thread per core fisico vale +30%
(134 → 174 tok/s a 2048 token di prompt) e da 8 a 16 core il prefill rende 2.02×. Chiuse le domande
20 e 2 di `docs/MISURE.md`, metà della 3; aperta la 22 (Linux, decode, macchina occupata).
Prova: `docs/MISURE.md` §Dove vanno i thread ha tabelle, metodo e numeri.

### 2026-09-17 — Decodifica speculativa dal prompt, esatta al bit
`tr_session_eval_rows` tiene i logit delle ultime n posizioni di una passata invece che solo
dell'ultima (`tr_session_logits_back`), e ogni riga è identica al bit ai logit che quel token dà da
solo. Sopra ci stanno `src/gen/lookup.c` (l'n-gramma di coda cercato all'indietro nel contesto,
proposta la continuazione della sua ultima occorrenza; da 4 a 2 token, niente stato, niente
allocazioni) e `src/gen/greedy.c` (`tr_greedy_step`: emette il token già scelto, verifica 1 + k
posizioni in una passata, tiene i token che il modello avrebbe scelto comunque e torna indietro con
`tr_session_rewind` su quelli rifiutati). `generate` e `run` hanno `--spec <bozza>` e stampano
quante bozze sono state accettate. Le fonti e cosa si è preso da ognuna: `docs/ORIGINI.md`
§Speculazione sul prompt. Prova: `make check` (nuovo `tests/test_spec.c`: stessi token con bozza
1..15, su due vocabolari perché con quello del modello nulla verrebbe mai rifiutato, LEZIONI #50;
nuovo `make spec-check` sul modello vero tagliato a 2 layer: stesso testo con `--spec 0/1/4/8/15`),
e `sh tools/ab_spec.sh <gguf> <binario> bench/prompts/code.txt 8` per la velocità a run alternate.
Misurata sul modello intero (`docs/MISURE.md` §Speculazione dal prompt): 1.42× riscrivendo un file
già nel prompt (64% di bozze accettate), 0.62× scrivendo codice nuovo (13%), pareggio intorno al 15%.
`--spec` resta spento di default fino alla bozza adattiva. `make` rifiuta ora di mescolare oggetti di
due piattaforme nella stessa cartella (LEZIONI #52).

### 2026-09-18 — I thread del pool fissati ai core fisici
`src/base/cpu.c` costruisce la lista dei posti (`tr_cpu_info.slot`, `n_slots`): un processore logico
per core fisico prima di ogni fratello SMT, e i core presi a giro sulle cache di ultimo livello,
perché 4+4 sui due chiplet batte 8 sullo stesso. Su Windows la topologia viene da
`GetLogicalProcessorInformationEx` (core e cache L3), su Linux da `topology/` e `cache/index3/id`,
su macOS non c'è (un thread non si può fissare: `n_slots` resta 0 e non si pinna niente). La lista
rispetta una maschera già imposta al processo (`taskset`, `start /affinity`), e in quel caso anche il
conteggio dei core scende a quelli usabili, così il pool non si dimensiona su core dove non girerà
mai. `src/base/platform.c` aggiunge `tr_thread_pin` (Windows `SetThreadGroupAffinity`, che arriva
oltre il primo gruppo di processori — llama.cpp no, UPSTREAM #3; Linux `sched_setaffinity`) e
`tr_thread_affinity_restore`. In `src/base/threads.c` ogni worker si fissa al suo posto appena parte,
il thread chiamante prende il posto 0 e torna dov'era quando il pool muore (un processo che crea pool
di dimensioni diverse — i benchmark — non deve restare inchiodato al primo core). `TR_POOL_PIN`
sceglie il modo: 0 niente, 1 un processore logico per thread, **2 il core fisico intero (default)**.
Il modo conta: legare un thread a un processore costa il 17% sul decode, legarlo al suo core no
(LEZIONI #58). Prova: `make check`, dove `tests/test_base.c` chiede al sistema
operativo (`GetCurrentProcessorNumber`, `sched_getcpu`) su quale processore ha girato ogni chunk e
pretende che sia il posto assegnato — con `TR_POOL_PIN=0` quel test è rosso. Velocità:
`sh tools/ab_speed.sh <gguf> build/pin/on.sh build/pin/off.sh` e `docs/MISURE.md` §Il pin dei thread.
Fonti e cosa si è preso: `docs/ORIGINI.md` §Collocamento dei thread.

### 2026-09-18 — Bozza adattiva per `--spec`
`src/gen/greedy.c` tiene `k_cur`: parte da `n_draft`, dopo un rifiuto parziale scende a quanto è
stato davvero accettato, dopo una bozza accettata per intero risale di uno fino a `n_draft`, e dopo
una bozza **tutta** sbagliata la speculazione si ferma per qualche passo, con pausa che raddoppia
finché la sonda da un token continua a sbagliare (1, 3, 7, 15, al massimo 16). La pausa è la metà
importante: su un MoE una riga in più costa 15-27 ms contro i ~35 di una passata, perché il token in
bozza sceglie altri esperti e la passata legge anche i loro pesi, quindi il pareggio è al 60-75% di
bozze accettate (`docs/MISURE.md` §Bozza adattiva, LEZIONI #59). Risultato: 1.01× quando il modello
inventa (era 0.60× con bozza fissa) e 1.15× quando ricopia un file già nel prompt (1.34× con bozza
fissa, che resta disponibile con `--spec-fixed`). Una bozza vuota (il lookup non ha trovato nulla da
proporre) non cambia niente. La politica si sceglie
(`tr_draft_policy`): adattiva di default, `--spec-fixed` torna alla bozza fissa per le misure. La
riga di statistiche stampa anche la bozza media per passata. I token restano identici a quelli senza
speculazione, con ogni politica: è l'invariante di `tests/test_spec.c` e di `make spec-check`, che
ora gira sul modello intero e pretende bozze accettate (LEZIONI #54). Nuovi test: `k_cur` scende sotto
`n_draft` a ogni rifiuto vero e, partendo da 1, risale fino a `n_draft` su un contesto che si ripete
(LEZIONI #55), e dopo una bozza tutta sbagliata il passo successivo non deve proporre niente —
controllato a ogni passo, e la suite fallisce se in nessun caso una bozza è mai stata rifiutata del
tutto, così il test non può passare per il motivo sbagliato. Prova: `make check`, e
`sh tools/ab_spec.sh <gguf> <binario> bench/prompts/code.txt 8` per la velocità; numeri in
`docs/MISURE.md` §Bozza adattiva.

### 2026-09-18 — Quanto darebbe la leva 2 (attivazioni int8 con VNNI): niente
Domanda 21 chiusa con una misura, senza scrivere il kernel. `tests/bench_kernels.c` ha ora un
candidato int8 × int8 con `_mm256_dpbusd_epi32` (stessa struttura di ggml) accanto ai nostri dot:
sulla stessa riga fa 11.4 G elementi/s contro gli 11.3 del nostro dot float, mentre il nostro
`dot_row_x4` (una riga di pesi contro 4 token) ne fa 30.6. Il candidato resta nel benchmark, non
diventa un kernel: le attivazioni a 8 bit non sono lo stesso numero. Prova: `make bench`, righe
`dot q8_0xq8_0` e `quantize x q8_0`; numeri e letture in `docs/MISURE.md` §Attivazioni int8 con VNNI.

### 2026-09-17 — Una misura che non misura si ferma
`tools/ab_speed.sh` e `tools/ab_spec.sh` si fermano alla prima run che non produce la riga dei tok/s
e stampano le ultime righe di quella run: prima stampavano la tabella delle mediane su un file vuoto,
e una misura muta sembrava una misura riuscita (LEZIONI #56).

### 2026-09-18 — Revisione avversariale di tutto il repository
Rilettura completa da parte di un altro modello (Fable 5.1): kernel, prefill, speculazione,
tokenizer, pool, GGUF, piattaforma, riga di comando, strumenti e documenti. Esito: gli invarianti
reggono; tre errori corretti, ognuno con un test rosso prima e verde dopo; una conclusione di misura
rovesciata; due controlli e uno strumento nuovi (LEZIONI #60-#68, `docs/MISURE.md` §Revisione).
- `src/gen/greedy.c`: il tetto della pausa della bozza adattiva valeva sul valore vecchio, quindi
  15 raddoppiava a 31. Prova: `build/.../tests/test_spec` (`test_adaptive_pause_is_capped`).
- `src/base/threads.c`: l'affinità di prima del chiamante sta nel thread con un contatore dei pool
  vivi (due pool distrutti nell'ordine di nascita lo lasciavano sul core 0), e un worker senza slot
  riprende quell'affinità invece di ereditare il pin (Linux). Prova: `tests/test_base`
  (`test_pool_caller_affinity_any_order`, `test_pool_oversubscribed`), anche sotto TSan.
- `src/base/cpu.c`: un `TR_CPU_MAX` che non è un tier è un avviso sul log, non più un silenzio.
- `make tier-check` (`tools/tier_check.sh`, dentro `check-linux`): il motore sotto `scalar` e
  `avx2`, e i logit dei modelli minuscoli identici al byte fra tier, thread e `-b`.
- `tests/bench_kernels.c`, sezione «one row, by»: int8 VNNI con la struttura del kernel a 4 token
  (1.8-2.3× il nostro float) e float a 8 token (0.79× a n=2048), alternati. Prova: `make bench`.
- `tools/ab_modes.sh`: confronto fra modi dello stesso binario con rotazione dell'ordine e
  controllo A/A. Prova: `sh tools/ab_modes.sh 3 "a=<comando>" "a2=<stesso comando>"`.
- Commenti corretti dove dicevano il contrario del codice: `lookup.h`, `threads.h`, `threads.c`.
