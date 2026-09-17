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
  colibri, 14 GB), non dal disco di Windows (LEZIONI #41). Fermare i container degli altri progetti è
  bloccato dal controllo automatico dei permessi di Claude Code: o li ferma Marcello, o una regola nei
  permessi.
- Sul PC non ci sono clang né CUDA toolkit: build Windows con MinGW-w64 gcc 15.2 (scoop), Linux
  nel container `trochilus-dev` (Ubuntu 24.04, gcc + clang; ASan, UBSan e TSan con gcc).

## Prossimi passi

Fatto il 2026-09-17 (`docs/MISURE.md` §Prefill a blocchi + kernel a 4 token): prefill a blocchi in C
esatto, profilo, kernel `dot_row_x4`. Prompt 512 a 16 thread: 30 → 197 tok/s (6.5×), a 1 thread 14 →
28; decode invariato; logit identici al bit al binario di prima e fra passate di ogni dimensione.
llama.cpp avanti 1.97× a 16 thread e 1.57× a 1 (era 11.8×).

Fatto dopo (`docs/MISURE.md` §Dove vanno i thread, `docs/archivio/FATTO.md`): chiusa la domanda 20
(era il collocamento dei thread, non la potenza), e decodifica speculativa dal prompt con `--spec`,
esatta al bit e provata da `tests/test_spec.c` e `make spec-check`. Manca la sua misura di velocità
sul modello vero: è il passo 2 qui sotto.

1. **Thread fissati ai core fisici** nel pool (Windows `SetThreadAffinityMask`, Linux
   `sched_setaffinity`): la misura dice +30% sul prefill e 2.02× da 8 a 16 core. Poi rimisurare
   prefill e decode con quella base, e rispondere alla domanda 22 (Linux, decode, macchina occupata)
   e alla seconda metà della 3 (32 thread SMT contro 16 fissati).
2. **Bozza adattiva** (domanda 25): accorciarla dopo un rifiuto e allungarla dopo un'accettazione,
   così il caso peggiore misurato (0.62× quando il modello inventa) diventa «come senza» e il
   guadagno (1.42× quando ricopia) resta. Solo dopo si può accendere `--spec` di default.
3. Attivazioni int8/VNNI come opzione (leva 2, domanda 21): è quello che resta del divario per
   elemento. Definizione scalare nuova, differenza dichiarata e misurata sul modello vero, attivabile;
   il default lo decide Marcello.
4. Prefill su prompt lunghi (2048-4096), dove l'attenzione per token cresce col quadrato: a 512 token
   è al 2%, a 2048 il prefill è già sceso da 197 a 174 tok/s (domanda 7).
5. Confronto colibri/ds4 per componente (decisione sopra), poi le correzioni che ne escono.
   Misure 13-16 di `docs/MISURE.md` (routing, cache esperti, SSD): ordine da confermare con Marcello.
6. Decode a contesto lungo: Trochilus perde l'11-13% da 32 a 512 token, llama.cpp il 5-7% (domande
   18-19 di `docs/MISURE.md`).
7. Memoria: banda reale della RAM (domanda 4 in `docs/MISURE.md`), poi pagine da 2 MB (domanda 5).
   4 thread vanno come 16 (33 tok/s, 41 GB/s): da rimisurare dopo il pin.
