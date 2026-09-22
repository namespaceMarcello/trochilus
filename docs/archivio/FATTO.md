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

### 2026-09-18 — Rimisura nativa con controllo A/A
Nessun codice cambiato: `sh tools/remeasure.sh` (14 minuti, macchina ferma, 8 giri, ordine a
rotazione, A/A) sulle tre conclusioni che stavano dentro lo spread, più le zone del profiler per la
domanda 12. Caso peggiore di `--spec` **0.953×** (non 1.01×) e caso buono 1.175×; il −17% sul decode
del pin al processore non si riproduce (la differenza fra i due pin è nel prefill, +9%); decode a 8
thread 1.09-1.12× su 16. Una riga di bozza costa 13.7-17.6 ms su 31.3, per il 61-91% negli esperti.
Numeri in `docs/MISURE.md` §Revisione, decisioni in `docs/STATO.md`, LEZIONI #58, #59, #66, #67.
Prova: `sh tools/remeasure.sh`, poi `build/remeasure/` (ogni run e le mediane).

### 2026-09-18 — Thread per fase: il prompt su tutto il pool, il decode sulla larghezza che la sessione misura
Domanda 26 chiusa. Misura (nativo, macchina ferma, 8 giri, A/A, contesti 512 e 2048): il prefill
vuole 16 thread, il decode 8 (4 vince di poco a 512 e perde a 2048, 12 non rende). Il numero non è
scritto nel motore: ogni sessione lo misura. Dopo contro prima: decode **1.085×** a 512,
**1.02-1.03×** a 2048, prefill invariato, `--spec 8` caso peggiore 1.064×; token identici al bit.
Numeri in `docs/MISURE.md` §Thread per fase, decisione in `docs/STATO.md`, LEZIONI #69-#72.
- `src/base/threads.{h,c}`: `tr_pool_set_active(p, n)` e `tr_pool_active`: i `parallel_for`
  seguenti usano i primi n thread (i primi n slot: core distinti sui due chiplet), gli altri
  dormono. Prova: `tests/test_base` (`test_pool_active`: 5000 cambi di larghezza, anche sotto TSan).
- `src/models/model.{h,c}`: ogni eval passa da `session_eval` (zona calda, ora sotto `tools/lint.py`):
  passata lunga su tutto il pool; passata corta (fino a `TR_DECODE_ROWS` = 4 righe) sulla larghezza
  del decode, che la sessione misura sulle prime 9 passate da un token (tutto il pool, metà, un
  quarto; la più ampia entro l'1% dalla più veloce; di nuovo ogni 1024) o che
  `tr_model_set_decode_threads` forza. `TR_DECODE_ROWS` nell'ambiente sposta il confine per le
  misure (0: il motore di prima). Prova: `tests/test_phase` (le larghezze e la scelta su tempi
  finti, la sequenza delle larghezze su una sessione vera, larghezza forzata 1..8 e oltre, ogni
  logit identico a un thread solo, f32 e Q8_0; rosso prima, verde dopo, rimisura provata per
  mutazione).
- `src/app/main.c`: `--decode-threads <n>` in `generate`, `run`, `chat`, `logits`; dopo le righe di
  velocità `threads: 16 prompt, 8 decode (measured)` (o `forced`); `decode_threads` nel JSON del
  profilo. Prova: `build/trochilus generate -m <gguf> -p 64 -n 24` e lo stesso con
  `--decode-threads 8`.
- `tools/tier_check.sh`: `test_phase` sotto `scalar` e `avx2`, e i logit identici al byte anche con
  la larghezza misurata (5 thread) e con `--decode-threads 2`. Prova: `make tier-check`.
- `tools/threads_phase.sh` (`sweep`, `change <binario prima>`, `after <binario prima>`): aspetta un
  exe bloccato da Smart App Control invece di ricompilare, ferma e riavvia i container, aspetta 12
  GiB liberi. `tools/ab_modes.sh` raccoglie la colonna `width` (la larghezza che ogni run si è
  misurata). Prova: `sh tools/threads_phase.sh change build/trochilus-before.exe`.
- Ogni `tools/*.sh` ha il corpo dentro `main()` chiamata dall'ultima riga: uno script modificato
  mentre gira non si rompe più (LEZIONI #69); `tools/lint.py` lo pretende. `tools/remeasure.sh`
  aspetta anche lui la memoria libera (LEZIONI #72). Prova: `make lint`.

### 2026-09-19 — Decode a contesto lungo: la banda della RAM, e la KV con le posizioni di una testa in fila

Punto 6 dei prossimi passi e domanda 4. Misurato prima (RAM ~54 GB/s; il decode perdeva col contesto
perché la KV si leggeva a 32-36 GB/s, a salti), poi la leva esatta: decode 1.06-1.10× a contesto
512, 1.12-1.14× a 2048, 1.15-1.18× a 4000, prefill 1.39-1.43× a 4000, logit identici al byte.
Numeri in `docs/MISURE.md` §Decode a contesto lungo.

- `src/kv/kv.{h,c}` (strato nuovo): `tr_kv`, cache `[layer][testa KV][posizione][head_dim]`, K e V in
  due blocchi; `tr_kv_bytes` per la guardia di memoria, `tr_kv_init`/`tr_kv_free`, `tr_kv_keys` e
  `tr_kv_values` (posizione 0 di una testa; la posizione t sta `t * head_dim` float più avanti),
  `tr_kv_write` (una passata di un layer, testa per testa; riscrive dopo un rewind). Zona calda sotto
  `tools/lint.py`. Prova: `tests/test_kv.c` (layout, scritture in più passate, riscrittura, ultimo
  elemento; nessun altro float toccato), rosso prima (non c'era) e verde dopo, e per mutazione
  (`tools/mutate_kv.sh` nel container: 5 indici sbagliati su 5 visti, tre da `test_kv`, tutti
  dall'oracolo).
- `src/models/olmoe.c`: la sessione tiene un `tr_kv`; l'attenzione chiama `tr_attention_head` con
  passo `head_dim` e scostamento 0 sulle chiavi e sui valori della sua testa KV (`h / group`): stesse
  chiamate sugli stessi float. Prova: `make check` (test_prefill, test_spec, test_session, test_phase,
  oracoli, `tier-check`, `spec-check`), e sul modello vero lo stadio `exact` qui sotto.
- `src/base/prof.{h,c}`: byte letti **per zona** (`tr_prof_count(p, zona, pesi, disco)`,
  `tr_prof_count_kv` per la KV letta dall'attenzione: ogni posizione, una volta per testa e per token),
  `kv_bytes_read` per fase; la tabella stampa MiB per token e GB/s della zona, il JSON `bytes` per
  zona e `kv_bytes` per fase. Gli esperti contano `gate_up` e `down` separati. Prova:
  `tests/test_prof.c`, `tests/test_model_prof.c` (KV di una passata da 3 token e di un token a
  posizione 3, pesi di `qkv_proj`, somma delle zone = totali della fase);
  `build/trochilus generate -m <gguf> -p 512 -n 48 --profile`.
- `tests/bench_mem.c`, `make bench-mem` (compilato con 0 warning in `make check`, nativo e Linux):
  `ram` (letture in fila e sparse a blocchi da 2 MiB, 256 KiB, 4 KiB, 1-16 thread), `weights` (il
  matmul del motore su matrici da esperto a caso), `kv <posizioni>` (un token di attenzione sui due
  layout, kernel vero e sola lettura). Stampa data e ora di compilazione (LEZIONI #24). Ogni gruppo
  sta sotto i 60 s.
- `tools/profile_suite.py`: chiave `decode_threads` negli scenari; per zona ms per token, MiB per
  token e GB/s; KV letta per token e traffico pesi + KV. `bench/scenarios-decode-context.json`:
  contesto 32, 512, 2048, 4000, larghezza misurata e 8 forzati (16 forzati ai due contesti lunghi).
  Prova: `make profile SCENARIOS=bench/scenarios-decode-context.json`.
- `tools/decode_context.sh` (`measure`, `change <binario prima>`; `PROF_BEFORE=<binario>` profila
  anche quello): ferma e riavvia i container, aspetta un exe bloccato e 12 GiB liberi; stadio `exact`
  (logit al byte prima contro dopo su 600 posizioni, un token per passata e a passate da 64, e token
  dopo un prompt da 4000: se differiscono non misura); i quattro contesti in **una** sessione di
  `ab_modes.sh`, 16 modi in ordine di de Bruijn, ogni modo con la sua copia A/A; poi `bench_mem` e i
  profili. `tools/decode_context_report.py speed | model | zones` fa le tabelle di MISURE dai file
  delle run. Prova: `sh tools/decode_context.sh change build/trochilus-before.exe` (105 minuti).
- `tools/ab_modes.sh`: `AB_GUARD`, un comando eseguito prima di ogni run; se fallisce la misura si
  ferma (uscita 3). `decode_context.sh`, `threads_phase.sh` e `remeasure.sh` lo impostano a «nessun
  container acceso» (LEZIONI #73). Prova: `AB_GUARD=false sh tools/ab_modes.sh 1 "a=true" "b=true"`.

### 2026-09-19 — Prefill su prompt lunghi: attenzione a gruppi, lavoro per token sul pool, righe F32 col kernel del tier

Punto 4 dei prossimi passi, domande 7 e 30. Misurato prima (il softmax, cioè `expf` della libreria
C, è il 51-72% dell'attenzione del prompt; la lettura ripetuta di chiavi e valori fra l'1% e il
21-39% a 4000 token, secondo la run, e niente sotto; l'8% del prefill girava su un thread solo), poi
le tre leve esatte. Numeri in
`docs/MISURE.md` §Prefill su prompt lunghi. Prefill **1.05-1.08× a 512, 1.07-1.08× a 2048,
1.11-1.14× a 4000** (A/A 2.1%), decode non distinguibile (A/A 2.4%), logit identici al byte sul
modello vero (600 posizioni un token per passata, passate da 64, prompt da 4000 a passate da 512 e
da 100). `expf` nostro, la leva grande che resta, non è scritto: lo decide Marcello (domanda 37).

- `src/kernels/kernels.{h,c}`, `kernels_x86.c`: `tr_attention_group` (l'attenzione di un gruppo di
  token consecutivi di una testa: blocchi di `TR_ATTN_BLOCK` = 64 posizioni contro tutte le query
  del gruppo, poi il softmax di ogni riga, poi i valori a blocchi; ogni uscita identica al bit a
  `tr_attention_head`, che resta come definizione) e due kernel nuovi nella tabella, `dot_f32_x4`
  (una query contro 4 chiavi, un accumulatore per chiave in un registro con nome) e `axpy_f32_x4` (4
  valori sommati a un'uscita, in ordine, un carico e una scrittura dell'uscita per 4): scalare (la
  definizione: 4 chiamate), AVX2, AVX-512 con le code a maschera. Le righe di pesi **F32** (il
  router) usano i kernel del tier (`dot_row` = `dot_f32`, `dot_row_x4` = `dot_f32_x4`) invece del
  ciclo scalare. Prova: `tests/test_kernels.c` (x4 contro scalare e contro 4 chiamate del proprio
  tier, ogni lunghezza fino a 200 e code; 525 gruppi contro `tr_attention_head` query per query,
  punteggi compresi, con sentinelle oltre l'uscita e oltre la riga di punteggi; ogni tier deve avere
  kernel suoi), sotto ogni tier in `make tier-check`.
- `src/models/olmoe.c`: l'attenzione corre a gruppi di `OLMOE_ATTN_QUERIES` = 16 token per testa (il
  decode è un gruppo da uno: un solo percorso), con 16 righe di punteggi per worker a passo non
  multiplo di 4 KiB; il lavoro di un token solo (embedding, norme, norme di q e k, RoPE, scrittura
  della KV, somma del residuo, scelta del router con uno scratch per worker, righe copiate per gli
  esperti, somma degli esperti) si divide sul pool per token, almeno `OLMOE_TOKENS_PER_CHUNK` = 8 a
  pezzo: una passata corta resta sul thread che chiama. Il profiler conta i byte di KV per gruppo (le
  posizioni che vede l'ultimo token del gruppo), non più per token. Prova: `tests/test_prefill.c`
  (prompt da 141 token, contesto 160, passate fino a 141: 108 casi identici a un token per passata),
  `tests/test_hot.c` (passate da 1, 4 e 20 token, anche sotto ThreadSanitizer), `tests/test_model_prof.c`
  (byte di una passata da 3, da 20 e da 5 a posizione 20), `make check`.
- `tools/mutate_prefill.sh` (nel container; con un argomento solo le mutazioni che lo contengono):
  20 errori plausibili (17 nelle tre leve, 3 sul tier usato), 20 visti dai test; le due corse critiche le vede solo
  ThreadSanitizer. Una mutazione non era vista (un punteggio calcolato oltre il contesto della query):
  il test ora mette sentinelle dopo la riga (LEZIONI #80).
- `tests/bench_attn.c`, `make bench-attn` (0 warning in `make check`, nativo e Linux): l'attenzione di
  un prompt intero su un layer, una query alla volta (tutto, senza softmax, solo prodotti, solo
  softmax, solo somma pesata), a gruppi con i kernel di oggi, a gruppi con x4, e il kernel del motore
  (`grp full`); hash di tutte le uscite, i gruppi devono dare i bit di «una query alla volta»;
  `--threads`, `--heads`, `--group`, `--block`, `--only`. `tests/bench_mem.c kv` chiama
  `tr_attention_group` come il motore. `tests/bench_expf.c`, `make bench-expf` (nativo e nel
  container: le due librerie C non sono lo stesso codice): costo di `expf` e confronto con
  l'arrotondamento corretto su tutti i 2^32 float, 13 s a 16 thread.
- `tools/prefill_context.sh` (`bench`, `measure`, `change <binario prima>`; `TROCHILUS=<binario>`
  misura un motore che non è `build/trochilus.exe`, LEZIONI #81): come `decode_context.sh`, per il
  prefill a 512, 2048 e 4000 in **una** sessione (ordine che mette ogni lunghezza dopo ogni altra,
  copie A/A), stadio `exact` esteso a un prompt da 4000 a passate da 512 e da 100 su 16 thread,
  `bench_attn`, profilo su `bench/scenarios-prefill-context.json` (512, 2048, 4000 a 16 thread e 512
  a un thread: quanto di una zona non si divide). `tools/prefill_context_report.py attn | zones` fa le
  tabelle; `tools/decode_context_report.py speed` legge anche le etichette `p512`. Prova: `sh
  tools/prefill_context.sh change build/trochilus-before.exe`.
- **Il tier viene usato** (LEZIONI #78, quarta volta di un verde che non vede il ramo):
  `tests/test_tier_used.c`, in `make test` e sotto `TR_CPU_MAX=scalar` e `avx2` in `make tier-check`.
  La tabella: ogni voce della zona calda di ogni tier è una funzione sua, per ogni tipo di peso che
  il motore accetta. Il motore: un modello per tipo (F32, F16, Q8_0, router F32 accanto) gira con la
  tabella attiva avvolta in contatori (`tr_kernels_set_active`, solo per i test, in
  `kernels_internal.h`) e i prodotti contati per tipo sono esattamente righe × token. Rosso prima
  (F16 scalare in ogni tier), verde dopo i **kernel F16** di AVX2 (F16C) e AVX-512 (`dot_row`,
  `dot_row_x4`: conversione esatta, bit-identici, 49× sullo scalare in `make bench`);
  `tests/synth_olmoe.h` scrive anche modelli F16 e col router F32 (`synth_f32_router`). Tre
  mutazioni nuove in `tools/mutate_prefill.sh`: due le vede solo questo test. Regola generale in
  `CLAUDE.md`. Prova: `make test`, `make tier-check`, `make bench`.
- **`expf` nostro: preparato, non scritto** (decide Marcello; MISURE §Prefill su prompt lunghi, punto
  7). `tests/bench_expf.c` prova un candidato dato alla compilazione (`-DTR_EXPF_CANDIDATE=<funzione>`)
  su tutti i 2^32 float contro la libreria e contro il riferimento; senza candidato prova uno schizzo
  scalare che vive solo nel banco (0 differenze da MinGW, 3.7 ns contro 30, 8 argomenti sulla strada
  lenta); `--hard <file>` scrive i casi al confine e `tools/expf_hard_cases.py` li ricalcola a 200
  bit con mpmath (369 casi, 0 errori del riferimento, sulle due piattaforme).
  `tools/expf_quality.sh` (container, volume dei modelli) compila un motore di sola misura con ogni
  `expf` arrotondato correttamente (`tools/cr_expf_emul.h`) e lo confronta col binario normale sul
  modello vero (`tools/expf_quality.py`): KL media 3.9e-13, 0 token diversi su 1000. Il motore non
  è stato toccato. Prova: `make bench-expf`; `sh tools/expf_quality.sh` nel container.
- `tools/measure_guard.lib` e `tools/stay_awake.ps1`, usati da `prefill_context.sh`,
  `decode_context.sh`, `threads_phase.sh` e `remeasure.sh` (LEZIONI #82): una misura alla volta
  (`build/.measuring.lock` col pid; una seconda esce con 5, il lock di uno script morto si riprende)
  e la macchina tenuta sveglia finché il lock esiste (una richiesta di alimentazione, nessuna
  impostazione cambiata). Prova: due `sh tools/prefill_context.sh bench` insieme, il secondo si
  rifiuta.

### 2026-09-19 — `tr_expf` scalare, la pulizia negli script, la rimisura a macchina pulita

- **`tr_expf`** (`src/kernels/expf.c`, `src/kernels/expf_table.h` generato da
  `tools/gen_expf_table.py` con mpmath): exp(x) arrotondato correttamente a float su ogni float,
  scalare, nessuna chiamata alla libreria C dentro. Tabella di 64 valori di 2^(j/64), ln2/64 in due
  pezzi, polinomio di grado 5, test di arrotondamento a 2^-50, 8 eccezioni calcolate a 200 e a 400
  bit (le ritrova `gen_expf_table.py --scan`, mirror numpy della strada veloce), NaN per un
  arrotondamento non provato. `tr_softmax` e `tr_swiglu` la usano; `tr_expf_path` e
  `tr_expf_exception` dicono ai test quale strada prende un argomento. Prova: `make bench-expf`
  (tutti i 2^32 float contro il riferimento; su Windows anche contro l'`expf` di MinGW: 0 e 0),
  `make test` (`tests/test_expf.c`: bordi, strada veloce a campione, ogni eccezione, softmax e SiLU
  contro la definizione), `sh tools/mutate_expf.sh` nel container (17 mutazioni su 17 viste).
- **Nel cancello**: `make check` gira `make bench-expf` con gcc e con clang; `tools/lint.py` rifiuta
  `expf(` e `exp(` nella zona calda, confronta `expf_table.h` col suo generatore, pretende il trap di
  pulizia in ogni `tools/*.sh` e rifiuta le pipe verso `tee` di ciò che può fallire; `clean-machine`
  (`tools/orphans.sh`, `tools/test_cleanup.sh`) prima di tutto.
- **Qualità e piattaforme**: `sh tools/expf_quality.sh` nel container (binario di prima, build di
  emulazione dai sorgenti del commit di prima tirati fuori da git, binario nuovo: KL e token, e gli
  stessi byte dell'emulazione o esce 1); `sh tools/platform_bits.sh` (logit di Windows e Linux al byte
  su fixture e modello vero a 2 layer; le tabelle RoPE con `tests/dump_rope.c` e
  `tools/rope_table_compare.py`, saltate se Smart App Control blocca lo strumento).
- **La pulizia** (LEZIONI #84): `tools/cleanup.lib` in ogni script (discendenti uccisi dal trap EXIT,
  anche l'albero nativo su Windows; INT e TERM diventano `exit 130`; `cleanup_run` per i passi lunghi,
  perché una shell che aspetta un figlio in primo piano non serve i segnali); `tools/orphans.sh` e
  `tools/orphans.ps1`; `tools/busy_machine.sh`; `tools/test_cleanup.sh`. Prova: `sh
  tools/test_cleanup.sh`; un `yes` lasciato acceso e poi `sh tools/orphans.sh` o `make check`.
- **La macchina interrogata** (LEZIONI #85): `tools/machine_still.sh` e `tools/cpu_busy.ps1` (processori
  occupati, prima della sessione e prima di ogni run, in `MEASURE_AB_GUARD`), `tools/background_load.ps1`
  e `measure_declare` (il carico di fondo e la quota di `System` nel log di ogni misura);
  `prefill_context.sh` scrive da sé `binaries.sha256` e accetta `GEN_EXTRA` (decode a larghezza
  forzata nei confronti prima/dopo). Prova: `sh tools/machine_still.sh 3.0 0 5` a macchina quieta e con
  tre `yes` accesi da `sh tools/busy_machine.sh 3 sh tools/machine_still.sh 3.0 0 5`.
- **La rimisura**: `threads_phase.sh widths` e `decode_context.sh widths` (decode forzato a 16 contro 8
  thread ai quattro contesti), e `decode_context_report.py speed` che conta scelte e cambi di
  larghezza. Prova: i comandi di MISURE §Rimisura a macchina pulita, a macchina lasciata sola.

### 2026-09-19 — Lo stimatore della larghezza del decode, riscritto

- **Cosa**: la larghezza delle passate corte non la sceglie più «la più ampia entro l'1%» (tirava a
  sorte: LEZIONI #88, domanda 31). `src/models/model.c`: `tr_decode_tune_widths` dà le larghezze
  dalla più stretta e non scende sotto 4 thread; `tr_decode_tune_stats` dà di ogni larghezza il
  centro (la passata più veloce) e il rumore (il distacco della seconda); `tr_decode_tune_pick`
  tiene la più stretta entro il rumore della coppia (lei e la più veloce) e chiede altre passate
  sulle due, fino a 6, quando decide il margine; `tr_decode_tune_debounce` cambia solo con due
  misure concordi. La sessione rimisura a ogni raddoppio del contesto (da 32, ricalcolato dalla
  posizione: un `rewind` lo abbassa) e dopo 128 passate se un cambio aspetta il secondo voto.
  `tr_session_decode_history` tiene ciò che ogni misura ha deciso, e la riga `threads:` lo stampa:
  `threads: 8 prompt, 4 decode (measured), choices 26:4 38:4(8) 76:8` (posizione:larghezza in uso,
  fra parentesi la scelta che il debounce ha trattenuto). `tools/decode_context.sh long`: 1500 token
  dopo un prompt di 1000, i cambi di ogni run contati da quella storia (`tools/long_switches.awk`).
- **Come si prova**: `tests/test_phase.c` (in `make check`): le funzioni pure su tempi scritti a
  mano, la sessione con un orologio finto (`tr_session_set_tune_clock`), ogni ramo col suo
  contatore. Visto rosso: prima della riscrittura il caso piatto con l'attesa «la stretta»
  (`0 != 2`); dopo, dieci mutazioni, nel container: `MSYS_NO_PATHCONV=1 docker run --rm -v
  "$(pwd -W):/src" -w /src trochilus-dev:local sh tools/mutate_tune.sh` (mezz'ora; `-e ONLY=history`
  per le due della storia). La riga: `build/trochilus generate -m fixtures/tiny-olmoe/model-f32.gguf
  -p 20 -n 100 -t 8`. **Sul modello vero non è ancora misurato**: la validazione è il punto 0 di
  `docs/STATO.md`, di notte a macchina quieta.

### 2026-09-20 — M1, prima di scrivere codice: la traccia del routing e il banco del disco

- **Cosa**: `trochilus run --route-trace <file>` (`src/models/olmoe.c`, `model.h`: `tr_session_route_trace_begin`,
  `tr_session_route_trace`) registra per token e layer gli esperti scelti e due previsioni del
  layer dopo (`pred_in` prima degli esperti del layer, `pred_out` a layer finito); spenta non cambia
  un byte e non tocca la zona calda. `tools/route_trace_report.py` ne ricava le domande 13-15
  (previsione, cache LRU e pin, streaming per layer). `tests/bench_disk.c` / `make bench-disk`:
  letture senza la cache del sistema, a blocchi grandi quanto una matrice di un esperto, 1-16
  lettori (domanda 16). Prompt nuovo `bench/prompts/code-1000.txt` (904 token). Chiuse come «no» le
  domande 5, 34, 39; aperte la 41 (il disco a un terzo della scheda) e la 42 (il primo layer).
- **Come si prova**: `tests/test_route.c` in `make check` (previsioni esatte su due modelli
  sintetici fatti apposta, traccia uguale per ogni forma delle passate, logit uguali con la traccia
  accesa); cinque mutazioni rosse: `MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src
  trochilus-dev:local sh tools/mutate_route.sh`; `tools/.venv/Scripts/python.exe
  tools/route_trace_report.py --check` (in `make lint`). I numeri: nel container `build/linux-gcc/trochilus
  run -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -f bench/prompts/code-1000.txt -n 300 -t 8
  --decode-threads 8 --route-trace build/route/code-1000.bin`, poi il report su quel file; `make
  bench-disk` in nativo a macchina ferma. Risultati in `docs/MISURE.md` §M1, prima di scrivere codice.

### 2026-09-20 — M1, lotti 1 e 2: gli esperti passano per un archivio con un budget di RAM

- **Cosa**: `src/memory/experts.{h,c}`: unità (layer, esperto) in slot allocati al caricamento,
  indice diretto e lista LRU O(1), letture a richiesta sul thread che chiama, dimensioni delle parti
  per layer. `src/models/olmoe.c`: il denso resta in RAM, gli esperti si chiedono all'archivio dopo
  il router (`olmoe_refresh_experts`, fuori dalla zona calda), il GGUF resta aperto per la vita del
  modello, un errore di lettura fa fallire la valutazione e rimette `pos` dov'era. Piano automatico
  (`tr_expert_budget_plan`), `tr_model_load_budget`, `--expert-budget <MiB|min>` e
  `TR_EXPERT_BUDGET_MIB`, riga `experts:` dopo `threads:`. Un solo percorso: col budget pieno
  l'archivio si riempie al caricamento. `tools/lint.py` ora rifiuta i fine riga CRLF (LEZIONI #95).
- **Come si prova**: `tests/test_experts.c` (contenuti al byte, LRU calcolata a mano, un layer
  intero col minimo, residente, errori iniettati, 800 chiamate contro un modello di riferimento) e
  `tests/test_stream.c` (logit identici al byte fra residente, minimo e intermedio, con e senza
  pool; byte dell'archivio contro `tr_gguf_read_range`; errore di lettura a metà prompt; piano su
  numeri finti), in `make check`, dove gli oracoli girano anche sotto `TR_EXPERT_BUDGET_MIB=min`;
  mutazioni `tools/mutate_experts.sh` (10) e `tools/mutate_stream.sh` (5), tutte rosse. Modello vero
  nel container: `trochilus logits` su 32 posizioni, residente contro `--expert-budget min` (72
  unità su 1024): `cmp` identico. Nessuna misura di velocità ancora (lotto 3: letture dirette).

### 2026-09-20 — Domanda 44: il codice è un grafo piccolo e deterministico? Traccia versione 2 e maschera degli esperti

- **Cosa**: la traccia del routing porta anche l'id di ogni token e il margine del router;
  `tr_model_set_expert_mask` / `--expert-mask <file>` spegne esperti (sola misura: l'uscita non è
  più del modello, e la riga `expert mask:` lo dice); `tools/route_graph_report.py` (copertura,
  grafo statico, ripetizioni, tabella per id, margini, `--compare`, `--mask-from` anche a caso);
  `tools/mask_quality.sh` (KL e token contro il modello intero); `tools/mutate_reports.py`; prompt
  `bench/prompts/trace-{c2,py,sh,prose-it,prose-en}.txt`. Risultati in `docs/MISURE.md`.
- **Come si prova**: `tests/test_route.c` (id dei token, margini esatti contro lo stesso modello
  con un esperto in più per token, maschera mai scelta e per layer, vuota e tolta = il modello);
  `tools/.venv/Scripts/python.exe tools/route_graph_report.py --check` e `tools/mutate_reports.py`
  (12 mutazioni rosse). Nel container: `trochilus run ... --route-trace`, poi il report sul file;
  `sh tools/mask_quality.sh` da Git Bash. **Non ancora rifatti dopo queste modifiche**: `make check`
  e `tools/mutate_route.sh` per intero (fermati per memoria il 20/09).

### 2026-09-20 — M1 lotto 3: gli esperti si leggono senza la cache del sistema

- **Cosa**: `tr_file_open_direct` e `tr_file_alignment` in `src/base/platform.{h,c}`; l'archivio
  legge allineato a 4096 senza copie (margine di un settore per parte nello slot, spostamento
  registrato a ogni riempimento) e `tr_experts_stats` dice se è diretta; `olmoe_load` apre il
  secondo handle, prova una lettura allineata di saggio e ricade sulla lettura normale se il file
  system rifiuta; `TR_EXPERT_DIRECT=0` e `TR_MEM_AVAILABLE_MIB` per le misure; la riga `experts:`
  dice `direct` o `buffered`. Nuovo per le misure: `tools/experts_budget.sh`,
  `tools/experts_steady.awk`, i contatori dell'archivio in `tools/ab_modes.sh`.
- **Come si prova**: `tests/test_experts.c` gira ogni suo caso con entrambi gli allineamenti e ha
  il caso dei due esperti con resti diversi nello stesso slot; `tests/test_stream.c` confronta
  diretta e normale al byte (residente e al minimo) e forza il ramo stretto del piano;
  `tests/test_base.c` prova la lettura corta nell'ultimo settore. Quindici mutazioni rosse e la riga di controllo verde:
  `MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src trochilus-dev:local sh
  tools/mutate_experts.sh`. Tutto in `make check`.

### 2026-09-21 — M1 misurato (le quattro sessioni), domanda 44 chiusa, un controllo su MISURE

Le misure di M1 sul modello vero, a macchina ferma: `experts_budget.sh measure | misses | long |
direct` (`build/experts_budget/`) e `mask_quality.sh` sui tre testi che mancavano
(`build/mask/quality.txt`). Numeri e conclusioni in `docs/MISURE.md` §M1 misurato e §Il
comportamento sul codice, decisioni in `docs/STATO.md`. In breve: il costo di M1 è il **prompt**
(ogni sessione rilegge il modello intero, ~4.3 s di disco), un token generato costa 0.03-0.6 unità,
il decode a generazione lunga sta a 0.87-0.90× del modello residente, e la cache del sistema
gonfierebbe il prefill di 2.42×. La simulazione della domanda 14 (22.4 unità per token) rispondeva
a un'altra domanda: LEZIONI #98.

Nuovi: `tools/experts_budget.sh long` (200 contro 1000 token generati, contesto 1600) con
`experts_steady.awk` parametrizzato (`-v short_n= -v gap=`); `tools/check_misure.py`, agganciato a
`lint` quindi a `make check`: una riga di MISURE che dà un numero da simulazione o da modello a
tempo deve portare il tag «modello, non misura» (o «modello superato dalla misura» se è storia), e
una domanda tagliata col primo tag non può essere barrata come chiusa.

- **Come si prova**: `tools/.venv/Scripts/python.exe tools/check_misure.py` (verde; togliere un tag
  a una delle righe taggate di MISURE lo fa fallire, e barrare la domanda 43 lo fa fallire con
  l'altra regola). Le misure si ripetono con gli stessi comandi: servono un'ora o due di macchina
  ferma, e ogni sessione dichiara il carico di fondo nel log.

### 2026-09-21 — Il prefill sotto budget legge il modello una volta per passata (domanda 47)

Nuovi `tools/prefill_overlap.sh` e `tools/prefill_overlap_report.py`: il prefill diviso nelle sue
due metà (attesa del disco, dalla zona `weight_read`, e calcolo) a prompt 512 e 2048, con modello
residente e a budget 50%, più il modo con una passata sola (`-b 2048`). Mediana di 5 giri più uno
di riscaldamento, ordine a rotazione, macchina ferma, guardie delle altre misure native.

Il conto ha trovato dell'altro: a 2048 token l'archivio legge 22 880 MiB, 3.5 volte la tabella
degli esperti, una volta per passata da 512 token. Con una passata sola: 6 273 MiB e 11.27 s
contro 23.51 (2.09×), calcolo invariato, ultima riga di logit identica al byte. Numeri e leve in
`docs/MISURE.md` §Il prefill legge il modello una volta per passata, priorità in `docs/STATO.md`,
lezione #99. Il codice del motore non è stato toccato.

- **Come si prova**: `TROCHILUS=<binario> sh tools/prefill_overlap.sh 5` (~20 minuti, macchina
  ferma; risultati e profili in `build/prefill_overlap/`). L'esattezza fra le due forme:
  `trochilus logits ... -b 512` e `-b 2048` sullo stesso prompt, l'ultima riga dei due file
  confrontata con `cmp`.

### 2026-09-21 — La mappa del progetto, il glossario, e il CLAUDE.md rimesso in misura

`docs/stato.json` tiene lo stato del progetto come dato: le tappe M0-M6, un blocco per pezzo con
stato (fatto / in corso / prossimo / aperto), cosa vuol dire in parole semplici, i numeri misurati
con la data, le domande di MISURE collegate, i file, i comandi e le dipendenze. `tools/stato_html.py`
ne fa una pagina sola (colonne per tappa, frecce fra i blocchi, pannello al clic), pubblicata come
artifact `6mx3NS4KtQrBFPRLkAYupr`. `docs/glossario.json` spiega da zero 38 parole chiave (LRU, KV,
prefill, quantizzare, tier, oracolo, mutazione, KL...): nel pannello ogni parola riconosciuta
diventa cliccabile e apre un fumetto con i termini imparentati.

Il `CLAUDE.md` era fuori dai tetti (169 righe, 12.1 KB): i comandi interi sono passati a
`docs/COMANDI.md` senza cancellarne nessuno, nella mappa restano quelli di ogni giorno, ed è
tornato a 136 righe / 8.5 KB, timbrato con l'impronta delle preferenze. Nella tabella «prima di
ogni commit» c'è ora la riga che tiene viva la mappa: un passo chiuso o una decisione si scrivono
anche in `docs/stato.json`, e l'artifact si rigenera e si ripubblica sullo stesso URL.

- **Come si prova**: `tools/.venv/Scripts/python.exe tools/stato_html.py` scrive
  `build/stato/index.html` (deve dire quanti blocchi sono fatti e quante voci di glossario);
  `node ~/.claude/hooks/misura-claude-md.cjs CLAUDE.md` per i tetti della mappa.

### 2026-09-21 — Ordine per layer nel prefill: 1.89× sotto budget, esperti letti una volta per prompt

Il corpo di un layer estratto in `forward_layer(m, s, L, tokens, n_tok, pos0, x)` e l'embedding e i
logit nei loro `forward_embed` / `forward_logits`; `forward_pass` li chiama nello stesso ordine di
prima, con `x = s->x`. Nuovo `forward_prompt_layer_major`: embedding di ogni blocco nella sua fetta
di `s->x_all`, poi per ogni layer tutte le passate del prompt, poi i logit. `s->x_all` ([n_ctx]
[n_embd]) è allocato alla creazione della sessione e solo quando l'archivio degli esperti è parziale
(glielo chiede `tr_experts_get_stats`), contato nella guardia di memoria: nella zona calda non si
alloca. `olmoe_eval` prende il percorso nuovo con `x_all != NULL && n > n_batch && trace == NULL`;
a budget pieno, e con `--route-trace`, resta quello di prima.

Misura (`sh tools/prefill_overlap.sh 5`, macchina ferma): a prompt 2048 e budget 50%, **6 273 MiB
letti invece di 22 880 e 12.41 s invece di 23.51 (1.89×)**, calcolo non distinguibile. A 512 niente
cambia. Numeri in `docs/MISURE.md` §Il prefill legge il modello una volta per passata.

- **Come si prova**: `make check` (il caso `once_per_prompt` di `tests/test_stream.c` conta le unità
  lette per un prompt di 36 token in passate da 12 e vuole al massimo `n_units + n_slots`; rosso
  prima del fix con 33 su 16). Sul modello vero: `sh tools/prefill_overlap.sh 5`, risultati in
  `build/prefill_overlap/report.txt`.

### 2026-09-21 — Il costo fisso della generazione è l'archivio, non un warm-up (domanda 46)

`sh tools/experts_budget.sh long` con dentro anche il **budget pieno** (200 contro 1000 token
generati, contesto 1600, due giri, macchina ferma dopo aver chiuso una chat rimasta aperta): col
modello residente il tempo per token non migliora da 200 a 1000 (34.64 → 33.48 tok/s, spread
4-7%), sotto budget sì (29.21 → 30.79 al 50%, 26.53 → 29.84 al 25%). Il costo fisso di ~0.35 s e
~0.84 s esiste solo quando si legge dal disco: è la LRU che si riassesta dopo il prompt, non i
kernel che si scaldano né il contatore del decode. Numeri in `docs/MISURE.md` §M1 misurato punto 3,
domanda 46 ristretta a «dove va il resto» (i mancati spiegano ~115 ms su 840).

- **Come si prova**: `sh tools/experts_budget.sh long` (~30 minuti, macchina ferma; prima
  `sh tools/orphans.sh` deve essere pulito). Risultati in `build/experts_budget/steady-long.txt`.

### 2026-09-22 — `README.md` in inglese: dove siamo, cosa manca, di cosa ci vantiamo

Primo documento del repo rivolto a chi non lavora qui dentro (i `docs/` restano in italiano).
L'asse è quello chiesto da Marcello: **cosa abbiamo fatto, cosa faremo, dove arriviamo**, con in
testa lo scopo — *democratizzare l'AI locale*: niente pagina dei requisiti minimi (lo scalare è la
definizione, SIMD e GPU sono acceleratori), niente da installare, la macchina decide da sé, la RAM
non decide che modelli puoi usare, e una macchina più piccola non dà una risposta peggiore (stessi
token cambiando thread, `-b`, tier, budget).

Sezioni: perché; cosa abbiamo fatto (funzionante + numeri misurati + di cosa ci vantiamo, cinque
voci ognuna con la prova accanto); cosa faremo (tappe M0-M6 con lo stato vero e i prossimi passi in
ordine); dove vogliamo arrivare (quattro cose che non abbiamo: dimensione slegata dalla RAM, ogni
GPU o nessuna, pesi piccoli senza perdite silenziose, sempre zero opzioni); cosa manca senza
sconti; build; mappa dei `docs/`; **cosa leggiamo** (colibri, ds4, llama.cpp come fonti primarie
con le idee prese una per una, più transformers/`tokenizers`/OLMoE; due soli file di codice
portato); licenza.

**Fuori dal README, per scelta di Marcello** (2026-09-22): nessun confronto con altri motori e
niente che dia il fianco — via i rapporti col tokenizer di HF e llama.cpp, via il confronto di
velocità vecchio, via le righe su cosa i tre motori non provano e sulla policy AI di llama.cpp.
I difetti trovati e le PR mandate a colibri e ds4 restano fuori: stanno in `docs/UPSTREAM.md`, il
README non li nomina.

I due documenti DeepSeek in `docs/` (report di terzi) entrano in `.gitignore`: non sono nostri e
non devono finire in un repo che può diventare pubblico.

- **Come si prova**: `cat README.md`; i numeri citati stanno in `docs/MISURE.md` (§Prefill su
  prompt lunghi, §Decode a contesto lungo, §M1 misurato, §Velocità — Trochilus contro llama.cpp)
  e in `docs/STATO.md`; `git check-ignore -v docs/DeepSeek_V41_Tech_Report.md` deve rispondere.
