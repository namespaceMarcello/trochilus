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

## Problemi noti

- Smart App Control di Windows blocca un .exe appena compilato anche per 20 minuti: correttezza in
  Docker, misure native quando il binario passa (LEZIONI #12).
- Gli hook di progetto (`.claude/settings.json`) valgono solo se Claude Code parte dalla cartella
  `trochilus`: una sessione aperta dal Desktop non li carica, nemmeno dopo `/hooks` (LEZIONI #18).
- RAM disponibile sul PC spesso ~14 GB su 31: i container Docker (OpenEMR, Redis) ne occupano molta.
- `main.c` su Windows riceve `argv` nella code page locale, non in UTF-8: un percorso con lettere
  accentate non si apre. Va letto con `GetCommandLineW` e convertito (platform.h promette UTF-8).
- OLMoE in transformers 5.x: esperti salvati fusi (`gate_up_proj` [E, 2I, H], gate prima),
  `q_norm` su tutta la proiezione (hidden), `k_norm` su kv_heads × head_dim, RoPE stile neox
  (`rotate_half`), softmax su tutti gli esperti poi top-k, normalizzazione solo con `norm_topk_prob`.
- Sul PC non ci sono clang né CUDA toolkit: build Windows con MinGW-w64 gcc 15.2 (scoop), Linux
  nel container `trochilus-dev` (Ubuntu 24.04, gcc + clang; ASan, UBSan e TSan con gcc).

## Prossimi passi

1. Memoria: banda reale della RAM (domanda 4 in `docs/MISURE.md`), poi pagine da 2 MB e thread
   fissati ai core (domande 2, 3, 5). Dopo la parte a thread singolo 4 thread vanno come 16
   (33 tok/s, 41 GB/s): il decode è limitato dalla memoria.
2. Hook contro le barre rovesciate anche sullo strumento PowerShell (LEZIONI #26): serve il sì di
   Marcello, tocca la configurazione degli hook.
3. Attivazioni in int8 con VNNI (leva 2 in `docs/MISURE.md`): nuova definizione scalare, differenza
   dichiarata e misurata sul modello vero, attivabile; decisione di Marcello prima di farla default.
   Meno byte per elemento: attacca proprio il limite della memoria.
4. Prefill a blocchi: più token nella stessa moltiplicazione (oggi prefill = decode = 33 tok/s).
5. Correttezza sul modello vero: stessi token di un riferimento su un prompt reale (llama.cpp in
   Docker, o transformers se la RAM lo consente).
6. M0, passo 4: tokenizer BPE dai metadati GGUF, `trochilus run` con testo.
7. M0, passo 5: confronto con llama.cpp e colibri sullo stesso modello e prompt → `docs/MISURE.md`.
