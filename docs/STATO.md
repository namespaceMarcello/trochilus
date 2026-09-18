# Stato

Sostituisci, non appendere. Tetto 40 KB. Lo storico sta in `archivio/FATTO.md`.

## Decisioni

- 2026-09-17 — Progetto nuovo, non un fork: architettura propria, i pezzi migliori di colibri e ds4
  si portano a mano (scelta di Marcello). Nome: Trochilus. Licenza Apache-2.0.
- Primo modello OLMoE (esiste vero da 4-7 GB, gira ovunque); DeepSeek V4 Flash è la M4.
- Formato GGUF v3 standard: i file di Hugging Face si aprono senza conversione, e i kernel CUDA
  di ds4 (ggml) lavorano già su quei tipi.
- Kernel scelti a runtime, non con `-march`; varianti SIMD bit-identiche allo scalare.
- Niente OpenMP: pool di thread proprio (idea di ds4), dimensionato sui core fisici (colibri).
- Pesi con `pread`, non `mmap` (colibri: mmap tiene residente il modello intero).
- Backend GPU come moduli caricati a runtime: il nucleo resta senza dipendenze.
- Da ds4 **non** si porta: hyper-connections, indexer DSA, compressor, engram, DSpark (saldati a
  DeepSeek/GLM, si rifanno alla M4 sulle primitive generiche); il lettore GGUF va ripulito degli
  agganci per architettura.
- Da colibri **non** si porta: i kernel int8 triplicati fra motori, la scansione LRU O(cap),
  le variabili d'ambiente lette dentro i kernel, lo stato globale per modello.
- Riferimento di correttezza: transformers 5.14.1 + torch 2.13.0 CPU, numpy < 2.4 (la 2.5 rompe
  `import torch._dynamo` su Windows).

- 2026-09-17 — **Focus sul coding** (idea di Marcello, confermata): il motore resta generale, ma il
  coding decide l'ordine. Gradino 2 = Qwen3-Coder-30B-A3B; prima le leve del coding: cache dei file già
  letti (riuso del prefisso), decodifica speculativa sul testo già nel prompt, velocità del prefill su
  prompt lunghi, contesti lunghi; immagini e audio non si implementano; qualità misurata anche su un
  banco di esercizi di programmazione.

- 2026-09-17 — **Zona calda** (Marcello: prestazioni fino all'ultima goccia): il codice che gira a
  ogni token ha regole controllate da lint e test (niente allocazioni, stringhe, I/O, matematica per
  elemento; logit identici con ogni numero di thread). Commenti e numero di righe non cambiano il
  codice macchina (verificato): non sono una leva di velocità. Regole in `docs/ARCHITETTURA.md`.
- 2026-09-17 — **Mai un numero da una run sola** (Marcello): mediana di N con minimo, massimo e
  spread; token identici in ogni run; i test con thread girano più volte e sotto ThreadSanitizer.
- 2026-09-17 — Pool di thread con attesa attiva (2 ms di `pause`, poi sonno), uno slot per thread:
  il dispatch da 53 µs a 1.5 µs a 16 thread su Windows (`docs/MISURE.md`).
- 2026-09-17 — Rope da tabella per sessione, esperti a stadi (attivazioni di tutti in un solo lavoro
  parallelo), attenzione per testa: decode 26.3 → 32.8 tok/s a 16 thread, logit identici al bit.
  Ogni ottimizzazione esatta si prova così: `trochilus logits` sugli stessi token col binario di
  prima e di dopo, `cmp` dei file.
- 2026-09-17 — Repository privato su GitHub (richiesta di Marcello), primo commit con tutto il lavoro
  fin qui.
- 2026-09-17 — **Prima la correttezza, poi la velocità** (Marcello): tokenizer, stessi token di
  llama.cpp sul modello vero e confronto di velocità vengono prima di altre ottimizzazioni; poi
  prefill a blocchi, poi memoria e VNNI. Sì al download di llama.cpp (sorgente in `ref/`, build nel
  container). Durante le misure si possono fermare i container Docker degli altri progetti, e si
  riavviano alla fine.
- 2026-09-17 — **Tokenizer**: il riferimento è transformers, non llama.cpp (che non fa NFC). Una famiglia
  di pretokenizer entra nella tabella di `tokenizer.c` solo dopo un oracolo sul suo `tokenizer.json`
  (oggi solo `olmo`); una famiglia sconosciuta si rifiuta. Classi Unicode e NFC sondate da HF
  `tokenizers` 0.22.2 (pin), non da Python. UTF-8 non valido si tiene byte per byte, tranne i byte
  che il vocabolario non ha (si scartano come fa HF). Da ds4 il caricamento dal GGUF, da colibri la
  regex rigiocata in C; codice nuovo (BPE n log n, niente `exit`).
- 2026-09-17 — **`trochilus chat`** (Marcello): nessun motore Jinja; ogni template supportato si
  riconosce dai byte esatti e si scrive in C, confrontato con `apply_chat_template`. A ogni turno si
  rende e tokenizza tutta la conversazione (come transformers) e si riusa la cache fino al primo
  token diverso (`tr_session_rewind`): la risposta è identica a quella calcolata da zero.
- 2026-09-17 — **Confronto colibri/ds4 per componente** (Marcello, LEZIONI #29), con poco usage: una
  cartella per volta (kernel, grafo OLMoE, GGUF, pool; piattaforma solo se serve), mappa delle funzioni
  con Haiku, un Sonnet per cartella su righe mirate (≤ 30 righe di resoconto), verifica mia solo dei
  punti segnalati, stop dopo ogni cartella. Risultato: una riga per cartella in `docs/ORIGINI.md`; le
  correzioni dopo, con test; bug di colibri/ds4 in `docs/UPSTREAM.md` con la prova.
- 2026-09-17 — **Correttezza sul modello vero**: llama.cpp non basta come riferimento esatto (attivazioni
  a 8 bit, KV f16: KL media 9e-3, `docs/MISURE.md`). Si fa l'oracolo con transformers sugli stessi pesi
  reali tagliati a 2 layer; Marcello: non deve fermare gli sviluppi, lo fa un agente Sonnet in
  parallelo (serve ~5 GB di RAM libera: niente chat aperte) e poi entra in `make check`.
- 2026-09-17 — Idee DeepSpeed (offload su SSD) valutate: streaming per **esperti**, non per layer; KV
  su disco come checkpoint dei file già letti, non per i token vecchi con attenzione piena. Si decide
  con le misure 13-17 di `docs/MISURE.md` §Da misurare; ordine proposto: 13-16 dopo i passi 1-2 qui
  sotto, 17 con il lavoro sulla KV (in attesa del sì di Marcello sull'ordine).
- 2026-09-17 — **llama.cpp terza fonte** (Marcello): da llama.cpp/ggml (MIT) si portano solo i pezzi
  dove le misure dicono che vince (prefill a blocchi, attivazioni int8/VNNI come opzione), con il metodo
  di colibri/ds4 (intestazione, `docs/ORIGINI.md`, bit-identità, benchmark). Non si portano grafo ggml,
  allocatore, backend, C++. `sgemm.cpp` di llamafile è C++ con intrinseci, non assembly: per i pesi
  Q8_0 vuole attivazioni Q8_0 (non esatte), e il percorso float usa FMA (non bit-identico allo scalare).

- 2026-09-17 — **Prefill a blocchi in C esatto** (passo 1): mappa delle tre fonti in `docs/ORIGINI.md`.
  Presi: passate da 512 token (llama.cpp), coppie (token, esperto) ordinate per esperto (ds4), attenzione
  del lotto per (testa, token) dopo aver scritto tutti i K/V (colibri), logit solo dell'ultimo token
  (tutte). Scartati: attivazioni int8 nel lotto (llama.cpp, ds4: non esatte) e la somma di ds4 per riga su
  tutti gli esperti (rilegge tutto il lotto a ogni riga). Il decode è una passata da un token: un solo
  percorso. Invariante: logit e cache identici al bit per ogni `n_batch` e ogni divisione in chiamate
  (`tests/test_prefill.c`, oracolo al byte). Nessuna delle fonti lo prova (LEZIONI #42).
  `tr_session_create` prende `n_batch`, la riga di comando `-b`.
- 2026-09-17 — **Kernel `dot_row_x4`** (una riga di pesi contro 4 token nei registri, idea di ds4 che ne
  fa 2): risultati identici al bit a `dot_row`, +45% sul prefill, niente sul decode (che ha un token
  solo). Gli accumulatori vanno in registri con nome, mai in un array (LEZIONI #45).
- 2026-09-17 — **Confronti di velocità solo con run alternate** (`tools/ab_speed.sh`): misurando prima
  tutto A e poi tutto B, la macchina che si scalda sposta i numeri del 10-25% e nasconde il vero
  guadagno (LEZIONI #46: una buona ottimizzazione era stata scartata così).

- 2026-09-17 — **Il prefill non scalava per colpa del collocamento dei thread**, non della potenza né
  del CCD (domanda 20, `docs/MISURE.md` §Dove vanno i thread). Il clock scende del 6-8% da 1 a 16
  thread, e 8 thread divisi 4+4 sui due chiplet vanno meglio di 8 su uno solo: è Windows che appoggia
  due dei 16 thread sullo stesso core fisico. Un thread per core fisico: **+30%** sul prefill, e da 8
  a 16 core **2.02×**. Le prove di collocamento si fanno native: dentro Docker la topologia non è
  quella della macchina (LEZIONI #47). Conseguenza: fissare i thread del pool ai core fisici e
  rimisurare le righe di velocità con quella base.
- 2026-09-17 — **Decodifica speculativa dal prompt** (leva del coding, passo 2): la bozza viene dal
  testo già nel contesto (n-gramma di coda cercato all'indietro, idea di colibri; llama.cpp ha in più
  cache con conteggi, non prese), la verifica è **una sola passata** su 1 + k posizioni, e si tiene
  solo ciò che il modello avrebbe scelto comunque. A differenza di tutte e tre le fonti qui i token
  sono **identici al bit** alla generazione senza speculazione, perché ogni riga del lotto è lo stesso
  calcolo della passata da un token: è un invariante provato (`tests/test_spec.c`, `make spec-check`),
  non una speranza. Il rifiuto costa `s->pos = n` (cache indicizzata per posizione: ds4 e colibri
  devono ripristinare un'istantanea). Misurato (§Speculazione dal prompt): **1.42×** riscrivendo un
  file già nel prompt (64% di bozze accettate), **0.62×** scrivendo codice nuovo (13%); il pareggio
  è intorno al 15%. Perciò `--spec` resta **spento di default** finché la bozza non si accorcia da
  sola dopo un rifiuto: con quella, il caso peggiore diventa «come senza».

- 2026-09-18 — **I thread del pool si fissano al core fisico, non al processore logico** (domande 22,
  27 e 3). Uno per core, i core presi a giro sui due chiplet, i fratelli SMT solo se i thread sono più
  dei core; ogni thread è libero fra i due fratelli del **suo** core. Rimisurato con controllo A/A
  (8 giri, `docs/MISURE.md` §Revisione): a 16 thread prefill **1.27×** su nessun pin e **+9%** sul
  pin al processore, che è anche instabile (spread 25% contro 5%); sul decode il pin al core non è
  distinguibile dagli altri due (soglia 2.2%). Il «−17% sul decode» del pin al processore, da cui
  era nata la scelta, **non si riproduce** (−3.4% contro nessun pin): la scelta resta, per il
  prefill. 32 thread (SMT) affondano: prefill -33%, decode -90%. Il pin regge anche a macchina
  occupata (+19% sul prefill, 3 giri). Su macOS non si pinna (non si può) e una maschera già imposta
  al processo viene rispettata. `TR_POOL_PIN=0/1/2` per i confronti.
- 2026-09-18 — **Su un MoE una riga di bozza in più non è quasi gratis**: misurata per zona, nativa
  (`docs/MISURE.md` §Revisione), costa **17.6 ms** se è la sola e **13.7 ms** l'una se sono otto,
  contro i 31.3 di una passata. Il costo sta negli esperti (91% e 61%): la riga in bozza fa leggere
  2.3-4.7 esperti nuovi per layer su 8 e il tempo segue quei MiB; le moltiplicazioni dense sono
  gratis per la prima riga in più. Il pareggio della speculazione è al **44-56%** di bozze
  accettate, non al 15% misurato in container (LEZIONI #59). Perciò la bozza adattiva non si limita
  ad accorciarsi: dopo una bozza tutta sbagliata **si ferma** per 1, 3, 7, 15 e poi 16 passi
  (LEZIONI #60). Rimisurato con controllo A/A, 8 giri: caso peggiore **0.953×** (i 3 giri di prima
  dicevano 1.01×: sbagliato, LEZIONI #66), caso buono **1.175×** (1.34× con `--spec-fixed`, 3 giri,
  per chi sa di star riscrivendo un file). «Caso peggiore come senza» **non è raggiunto**, e le
  costanti della pausa non ci arrivano (replay coi costi misurati: al massimo 0.97-0.98×): `--spec`
  resta **spento di default**.
- 2026-09-18 — **Quando una differenza è una conclusione**: confronti con `tools/ab_modes.sh` (8
  giri, ordine a rotazione, un modo dato due volte come controllo A/A). Una coppia A/A sola è
  rumorosa (lo stesso decode: 0.5% in un blocco, 2.2% in un altro): la soglia è il peggiore A/A
  della sessione per quella fase, e la differenza deve superarla contro tutte e due le copie. Sotto
  la soglia si scrive «non distinguibile» (LEZIONI #66).
- 2026-09-18 — **La leva 2 (attivazioni int8/VNNI) è la leva del prefill, ed è una decisione aperta**
  (domanda 21, corretta dalla revisione, LEZIONI #65). La prima misura confrontava una riga contro un
  token (+1-8%) e ne era uscito «non si scrive». Con la struttura del nostro kernel a 4 token l'int8
  VNNI a 512 bit è **1.8-2.3×** il nostro x4 float, e il prefill è al 90% moltiplicazioni; la strada
  esatta «8 token nei registri» invece non rende (0.79× a n=2048). Ma l'int8 non è bit-identico:
  romperebbe «prefill a blocchi = token per token» e può esistere solo come modo dichiarato.
- 2026-09-18 — **Thread per fase, e il numero lo misura il motore** (domanda 26, `docs/MISURE.md`
  §Thread per fase). Misurato a 4, 8, 12, 16 thread e a due contesti: il prefill vuole tutto il pool,
  il decode **8**, il solo numero sopra la soglia a contesto 512 (1.10×) e 2048 (1.03×); 4 vince di
  poco a 512 e perde a 2048 (LEZIONI #70). Ma 8 è un fatto di questa macchina (quanti thread
  riempiono il bus della memoria), e una formula sui core ricavata da una macchina sola sbaglia sulle
  altre. Quindi il default è una **misura**, come vuole il principio 0: ogni sessione prova tutto il
  pool, metà e un quarto sulle sue prime 9 passate da un token, tiene la più ampia entro l'1% dalla
  più veloce e rimisura ogni 1024 token; `--decode-threads n` forza. Passata corta = fino a 4 righe
  (16 non si distingue); il resto usa tutto il pool. Esatto per costruzione (il contratto del pool:
  il risultato non dipende dai thread) e per prova. Dopo contro prima, A/A: decode **1.085×** a 512
  e **1.02-1.03×** a 2048 (8 forzato: 1.05×), prefill invariato, `--spec 8` caso peggiore 1.064×.
  Il margine è stato scelto contando le scelte di ogni run (3%, 2%, 1%: LEZIONI #71).

## Problemi noti

- Smart App Control di Windows blocca un .exe appena compilato anche per 20 minuti: correttezza in
  Docker, misure native quando il binario passa (LEZIONI #12).
- Gli hook di progetto (`.claude/settings.json`) valgono solo se Claude Code parte dalla cartella
  `trochilus`: una sessione aperta dal Desktop non li carica, nemmeno dopo `/hooks` (LEZIONI #18).
- RAM disponibile sul PC spesso ~10-14 GB su 31: i container Docker degli altri progetti (OpenEMR,
  colibri-dev, Redis: ~3,6 GB) e la cache della VM dopo prove pesanti (LEZIONI #38). Una chat aperta
  tiene 7 GB: prima di prove sul modello vero va chiusa.
- Tokenizer e template di chat: solo OLMoE (famiglia `olmo`, template OLMoE-0125-Instruct).
  Qwen3-Coder (`qwen2`, gradino 2) chiede un oracolo sul suo `tokenizer.json` e il suo template in
  C prima di entrare. `trochilus run` resta senza template (testo grezzo); `trochilus chat` lo
  applica. La chat è greedy (nessun campionamento) e si ferma al contesto pieno (`/reset`).
- `src/tokenizer/unicode_data.h` generato pesa 157 KB di sorgente: da compattare se diventa un
  problema; si rigenera solo con `tools/gen_unicode_tables.py` quando cambia il pin di `tokenizers`.
- OLMoE in transformers 5.x: esperti salvati fusi (`gate_up_proj` [E, 2I, H], gate prima),
  `q_norm` su tutta la proiezione (hidden), `k_norm` su kv_heads × head_dim, RoPE stile neox
  (`rotate_half`), softmax su tutti gli esperti poi top-k, normalizzazione solo con `norm_topk_prob`.
- Misure di velocità nel container: modelli nel volume Docker `trochilus-models` (GGUF + conversione
  colibri, 14 GB), non dal disco di Windows (LEZIONI #41). I container degli altri progetti li ferma
  e li riavvia da solo `tools/remeasure.sh` (2026-09-18: 9 fermati, 9 ripartiti); un `docker stop`
  lanciato a mano era bloccato dal controllo automatico dei permessi di Claude Code.
- Sul PC non ci sono clang né CUDA toolkit: build Windows con MinGW-w64 gcc 15.2 (scoop), Linux
  nel container `trochilus-dev` (Ubuntu 24.04, gcc + clang; ASan, UBSan e TSan con gcc).
- **Un pool alla volta** (LEZIONI #63): due pool vivi prendono gli stessi slot e si dividono gli
  stessi core, e un `tr_parallel_for` annidato su un altro pool riceve un id di worker fuori dal suo
  intervallo. Oggi il processo ha un solo pool e un solo modello: va risolto (slot da un registro di
  processo, id per pool) prima del secondo modello nello stesso processo.
- Core ibridi P/E (Intel) e più gruppi di processori su Windows: `cpu.c` li legge ma nessuna macchina
  li ha provati. Il lavoro è diviso in parti uguali, quindi su P+E il passo lo dà il core più lento;
  `EfficiencyClass` di Windows non viene letta.

- Le misure di velocità native vogliono la macchina ferma: i container degli altri progetti vanno
  fermati e riavviati dopo (il guard di memoria rifiuta di caricare il modello sotto i ~10 GB liberi),
  e nessun agente deve compilare in Docker nel frattempo (LEZIONI #57). `tools/ab_speed.sh` e
  `ab_spec.sh` ora si fermano se una run non produce un numero, invece di stampare una tabella vuota;
  `threads_phase.sh` e `remeasure.sh` aspettano che il motore veda 12 GiB liberi prima della prima
  run, perché dopo un `make check` Windows si riprende la memoria con minuti di ritardo (LEZIONI #72).

## Prossimi passi

Fatto il 2026-09-17 (`docs/MISURE.md` §Prefill a blocchi + kernel a 4 token): prefill a blocchi in C
esatto, profilo, kernel `dot_row_x4`. Prompt 512 a 16 thread: 30 → 197 tok/s (6.5×), a 1 thread 14 →
28; decode invariato; logit identici al bit al binario di prima e fra passate di ogni dimensione.
llama.cpp avanti 1.97× a 16 thread e 1.57× a 1 (era 11.8×).

Fatto dopo (`docs/MISURE.md` §Dove vanno i thread, `docs/archivio/FATTO.md`): chiusa la domanda 20
(era il collocamento dei thread, non la potenza), e decodifica speculativa dal prompt con `--spec`,
esatta al bit e provata da `tests/test_spec.c` e `make spec-check`.

Fatto il 2026-09-18 (`docs/MISURE.md` §Il pin dei thread, §SMT, §Bozza adattiva, §Attivazioni int8):
pin dei thread al core fisico (prefill +24-30%, decode invariato), bozza adattiva con pausa (caso
peggiore da 0.60× a 0.95×, caso buono 1.175×: numeri della rimisura con A/A), e la leva 2 chiusa con
un microbenchmark invece che con del codice. Chiuse le domande 3, 21, 22 (tranne Linux), 24, 25, 27; aperta la 26. La revisione ha poi
riaperto e richiuso la 21 con la risposta opposta, chiuso la 28 e contato metà della 12.

Fatto il 2026-09-18, revisione avversariale di tutto il repository (`docs/MISURE.md` §Revisione,
LEZIONI #60-#68): gli invarianti reggono (logit identici al byte fra tier, thread e `-b`, anche sul
modello vero a 2 layer; tokenizer 90 000 stringhe contro HF, 0 differenze). Corretti con un test
rosso prima e verde dopo: il tetto della pausa, l'affinità del chiamante con due pool, i thread senza
slot su Linux. Nuovo nel cancello: `make tier-check`. Nuovo per le misure: `tools/ab_modes.sh`.

Fatto il 2026-09-18, rimisura nativa a macchina ferma con controllo A/A (`sh tools/remeasure.sh`, 14
minuti, ogni run in `build/remeasure/`; `docs/MISURE.md` §Revisione, LEZIONI #66-#67). Delle tre
conclusioni che stavano dentro lo spread: una era **sbagliata** (caso peggiore di `--spec` 0.953×,
non 1.01×), una **non si riproduce** (il −17% sul decode del pin al processore: è −3.4%) e una è
**confermata e più netta** (decode a 8 thread 1.09-1.12× su 16, intervalli disgiunti). Chiusa la
domanda 12 (il costo di una riga di bozza sta negli esperti), riscritte la 26 e la 27.

Fatto il 2026-09-18, thread per fase (`docs/MISURE.md` §Thread per fase, LEZIONI #69-#72; ogni run
in `build/threads_phase/`): il prompt gira su tutto il pool, le passate corte sui primi n slot, e n
lo misura ogni sessione (decisione sopra). Decode 1.085× a contesto 512 e 1.02-1.03× a 2048 contro
il binario di prima, prefill invariato, token identici al bit (`tests/test_phase.c`,
`test_pool_active`, `make tier-check`). Chiusa la domanda 26, aperta la 31. Nuovo per le misure:
`tools/threads_phase.sh`, la colonna `width` di `ab_modes.sh`, l'attesa della memoria libera, e gli
script di shell che non si rompono se modificati mentre girano.

1. **Accendere `--spec` di default?** La condizione «caso peggiore come senza» **non è
   soddisfatta**: 0.953× dove il modello inventa, 1.175× dove ricopia (A/A, 8 giri). Cambiare le
   costanti della pausa non basta (replay coi costi misurati: al massimo 0.97-0.98×); la leva è il
   costo della riga di bozza, che per le bozze corte sta al 91% negli esperti (domanda 12). Lo
   decide Marcello: acceso accettando −5% / +17.5%, o spento come oggi. Coi thread per fase i due
   rapporti vanno rimisurati: `--spec 0` prende il 9% dal decode stretto, il caso peggiore di
   `--spec 8` il 6.4%, il caso buono il 2%.
2. **La larghezza misurata, su altre macchine e con uno stimatore migliore** (domanda 31): su questa
   macchina a contesto 2048 il default prende metà del guadagno di `--decode-threads 8` (1.02-1.03×
   contro 1.05×), perché il «migliore di tre passate» ha il 2% di rumore e 16 sta al 2.5-4% da 8.
   Da provare: scartare subito le larghezze lontane e dare più giri alle vicine. E `sh
   tools/threads_phase.sh after <binario>` su una macchina diversa, appena ce n'è una.
3. **Int8/VNNI nel prefill, sì o no** (domande 21 e 28, LEZIONI #65): è dove sta il divario con
   llama.cpp (1.8-2.3× sul kernel), ma non è esatto. Se sì: un modo dichiarato (`--fast-prefill`),
   spento di default, con l'oracolo che misura di quanto si spostano i logit. La strada esatta «8
   token nei registri» è misurata e scartata. Lo decide Marcello.
4. Prefill su prompt lunghi (2048-4096), dove l'attenzione per token cresce col quadrato: a 512 token
   è al 2%, a 2048 il prefill scende da 235 a 193 tok/s (domanda 7).
5. Confronto colibri/ds4 per componente (decisione sopra), poi le correzioni che ne escono.
   Misure 13-16 di `docs/MISURE.md` (routing, cache esperti, SSD): ordine da confermare con Marcello.
6. Decode a contesto lungo: Trochilus perde l'11-13% da 32 a 512 token, llama.cpp il 5-7% (domande
   18-19 di `docs/MISURE.md`); da 512 a 2048 il decode scende da 33.5 a 24.4 tok/s.
7. Memoria: banda reale della RAM (domanda 4 in `docs/MISURE.md`), poi pagine da 2 MB (domanda 5).
   Il decode va uguale da 4 a 8 thread (34 tok/s, 41 GB/s) e scende sopra gli 8: la banda è il
   tetto, e la domanda 4 dice quanto manca.
8. Il pin su Linux: il codice c'è e `make check` lo prova, i numeri no (serve una macchina Linux
   vera, in WSL2 la topologia è sintetica). Resta l'ultima metà della domanda 22.
