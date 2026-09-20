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
- 2026-09-18, **rimisurato a macchina pulita il 2026-09-19** — **Thread per fase, e il numero lo
  misura il motore** (domanda 26, `docs/MISURE.md` §Thread per fase e §Rimisura a macchina pulita).
  Il prefill vuole tutto il pool (16 contro 8 thread: 1.49-1.57×). Il decode vuole **pochi** thread,
  4 a contesto corto e 8 a contesto lungo, perché la RAM tocca il tetto (57 GB/s) con 4-6 lettori e
  oltre cala; 16 non vince mai ma da 8 **non è distinguibile** (1.00-1.09×, soglia 4.6%). I numeri
  del 18/09 (8 sopra 16 dell'1.10× a 512) avevano sotto quattro processi `yes` dimenticati (LEZIONI
  #84): il verso era giusto, la misura gonfiata. Il default resta una **misura**, come vuole il
  principio 0: ogni sessione prova tutto il pool, metà e un quarto (mai sotto 4 thread) sulle sue
  prime passate da un token; `--decode-threads n` forza. Passata corta = fino a 4 righe; il resto
  usa tutto il pool. Esatto per costruzione (il risultato non dipende dai thread) e per prova. Lo
  stimatore di prima **tirava a sorte** (la più ampia entro un 1% fisso, sotto il rumore di tre
  passate: LEZIONI #71, #88) ed è **riscritto** (2026-09-19): centro = la passata più veloce, rumore
  = distacco della seconda; vince la più stretta entro il rumore **della coppia** (lei e la più
  veloce), con altre passate sulle due, fino a 6, se decide il margine; rimisura a ogni raddoppio
  del contesto (da 32) e cambia solo con due misure concordi (la seconda arriva al più dopo 128
  passate). La riga `threads:` stampa la storia delle scelte. Resta la validazione sul modello vero
  (primo dei prossimi passi).
- 2026-09-19 — **La KV tiene in fila le posizioni di una testa** (`src/kv/`, primo pezzo dello strato
  KV; `docs/MISURE.md` §Decode a contesto lungo). La RAM di questa macchina dà **~57 GB/s** in
  lettura con 4-6 thread e oltre cala (domanda 4, rimisurata a macchina pulita: i «~54» di questa
  sezione avevano sotto quattro core presi, LEZIONI #84; non 41, che era la velocità del motore del
  17/09). Le moltiplicazioni sui pesi
  ci stavano già (50-55 GB/s per zona); sotto c'era solo l'attenzione, a 32-35, perché con la KV
  `[layer][posizione][testa]` ogni testa leggeva 512 byte ogni 8 KiB. Con `[layer][testa][posizione]`
  legge a 47-48: decode **1.06-1.10×** a contesto 512, **1.12-1.14×** a 2048, **1.15-1.18×** a 4000,
  prefill **1.10×** a 2048 e **1.39-1.43×** a 4000, non distinguibile a contesto 32 (A/A 3.8%); logit
  identici al byte sul modello vero. L'ipotesi del calo col contesto regge nella forma e non nei
  numeri: le bande erano due (pesi 47, KV 36 GB/s); ora è una, e `tok/s = 48.2 / (1.2236 + 0.000262 ×
  contesto)` sbaglia al più dell'1.2% da 32 a 4000 token. Il decode muove 48-51 GB/s su 54: dal lato
  esatto resta il 6-11%, sparso (domande 34-35); **da qui il decode a contesto lungo va più forte
  solo leggendo meno byte**, cioè con leve non esatte (KV a 16 o 8 bit, domanda 36), che decide
  Marcello.
- 2026-09-19 — **Prefill su prompt lunghi: tre leve esatte, e il pezzo grosso che resta è `expf`**
  (`docs/MISURE.md` §Prefill su prompt lunghi). L'ipotesi «l'attenzione del prompt è lettura ripetuta
  di chiavi e valori» era vera a metà: smontata nel banco (`make bench-attn`), la zona è per il
  51-72% **softmax**, cioè `expf` della libreria C (30 ns a chiamata con MinGW, 2.3 con glibc), per
  il 27-37% prodotti e somma pesata, e la lettura ripetuta pesa solo a 4000 token, dallo 0 al 39%
  secondo la run (64 MB di chiavi e valori contro 64 MB di L3: sul bordo). Scritte le leve esatte:
  l'attenzione a **gruppi di 16 token** per testa su blocchi di 64 posizioni (`tr_attention_group`,
  `dot_f32_x4`, `axpy_f32_x4`; il decode è un gruppo da uno), il **lavoro di un token solo diviso sul
  pool** (norme, RoPE, KV, router, righe per gli esperti: era l'8% del prefill su un thread), e le
  **righe F32 e F16 col kernel del tier** (il router girava sullo scalare in ogni tier e nessun test
  lo vedeva: ora `tests/test_tier_used.c` pretende che ogni tipo di peso passi per il tier attivo, e
  lo conta nel motore; LEZIONI #78). Prefill **1.05-1.08×
  a 512, 1.07-1.08× a 2048, 1.11-1.14× a 4000** (A/A 2.1%), decode non distinguibile, logit identici
  al byte sul modello vero anche su un prompt da 4000.
- 2026-09-19 — **`tr_expf`: l'esponenziale è nostro, scalare, arrotondato correttamente** (decisione
  di Marcello; `docs/MISURE.md` §`tr_expf`; il SIMD no, in questo passo). `src/kernels/expf.c`: tabella
  di 64 valori da mpmath, polinomio, test di arrotondamento, 8 eccezioni calcolate a 200 bit, nessuna
  chiamata alla libreria C dentro; un arrotondamento non provato uscirebbe come NaN. **La prova è
  esaustiva e sta nel cancello**: tutti i 4 278 190 082 float contro il riferimento, con gcc e con
  clang (`make bench-expf`); su Windows 0 differenze anche dall'`expf` di MinGW. Su Windows i logit
  sono **identici al byte** a quelli di prima; su Linux KL 3.9e-13, 1000 token su 1000 uguali, e gli
  stessi byte del build di emulazione; **Windows e Linux ora danno gli stessi logit al byte** (fixture e
  modello vero a 2 layer, 1000 posizioni; le tabelle RoPE identiche al float fino a 4096 posizioni). Misurato a macchina pulita (A/A 3.0% e 1.9%): prefill
  **1.03-1.08× a 512, 1.20-1.23× a 2048, 1.29-1.31× a 4000**, decode a 8 thread **1.02-1.03× /
  1.05-1.07× / 1.08-1.10×**; `attention` 2.3-2.6× più corta, `expert_act` 5.5×. `tools/lint.py`
  rifiuta `expf(` nella zona calda e confronta le costanti generate col loro generatore.
- 2026-09-19 — **Un tetto si misura con uno strumento suo** (`make bench-mem`), e il profiler dice
  per ogni zona quanti byte legge e a quanti GB/s: una zona al tetto è limitata dalla memoria, una
  sotto da altro (LEZIONI #75-#76). Le misure a più contesti stanno in **una** sessione di
  `ab_modes.sh`, in un ordine che mette ogni contesto dopo ogni altro (`tools/decode_context.sh`).

## Problemi noti

- Smart App Control di Windows blocca un .exe appena compilato anche per 20 minuti, e il 2026-09-19
  per **quattro ore**: il verdetto sull'hash resta in cache e il cloud viene richiamato ogni due ore
  (registro eventi `CodeIntegrity`, evento 3118). Correttezza in Docker, misure native quando il
  binario passa; dopo mezz'ora di attesa si compila una **seconda copia in un'altra cartella**, senza
  toccare la prima, e si misura con quella che parte (`TROCHILUS=<binario>`; LEZIONI #12, #81).
- **Le misure si proteggono da sole, e la macchina si interroga** (LEZIONI #82, #84-#89): gli script
  di misura prendono un lock (una misura alla volta), tengono sveglia la macchina, si rifiutano di
  partire se `tools/orphans.sh` trova qualcosa di nostro rimasto acceso, aspettano che la CPU sia
  ferma prima della sessione e prima di **ogni run** (`tools/machine_still.sh`), dichiarano nel log
  il carico di fondo, e finiscono ciò che hanno lanciato (`tools/cleanup.lib`): una misura si ferma
  uccidendo il pid scritto nel lock, e ora si ferma davvero (una shell che aspetta un figlio in primo
  piano non serviva il segnale: i passi lunghi girano con `cleanup_run`).
- **Il carico di fondo di questa macchina**: a macchina libera restano occupati 1.0-2.5 processori
  logici, e **0.7-0.9 core sono del processo `System` del kernel, sempre** (LEZIONI #85): sta sotto
  ogni misura nativa, si dichiara e non si corregge; chi c'è dentro chiede una sessione da
  amministratore. Di giorno, con Marcello alla macchina e un'altra finestra al lavoro, le sessioni non
  distinguono differenze sotto il 5%: le misure che decidono si fanno di notte.
- **La VM di Docker riparte da sola** e riaccende i container con una politica di riavvio (i sei di
  OpenEMR, il 2026-09-19 alle 15:57 e alle 18:40; LEZIONI #87): Docker Desktop la ferma quando non
  gira nessun container, e il primo comando che ha bisogno del motore la riaccende (`docker ps -q` no).
  La guardia prima di ogni run vede i container e ferma il confronto; dentro una misura non gira
  nessun altro comando Docker.
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
- **Un'altra finestra di Claude può riaccendere i container a metà misura** (2026-09-19: OpenEMR al
  95% di CPU e 6.4 GiB, sette minuti dopo l'inizio; LEZIONI #73). Gli script di misura ora si fermano
  alla run successiva se vedono un container acceso (`AB_GUARD` in `ab_modes.sh`), invece di
  proseguire su una macchina occupata; il lavoro **senza** container dell'altra finestra non si vede,
  e alza lo spread (sessione `speed-auto`: 9-29% contro 4-16%). Le misure lunghe si lanciano quando
  le altre finestre sono ferme.

## Prossimi passi

I passi del 17 e del 18/09 (prefill a blocchi, `--spec`, pin dei thread, bozza adattiva, revisione
avversariale, rimisura con A/A, thread per fase) stanno in `docs/archivio/FATTO.md`, i loro numeri
in `docs/MISURE.md`.

I passi del 19/09 (decode a contesto lungo e la KV per testa, prefill su prompt lunghi,
`tr_expf` scalare, la pulizia negli script e la rimisura a macchina pulita) stanno in
`docs/archivio/FATTO.md`, i loro numeri in `docs/MISURE.md`; chiuse le domande 4, 7, 18, 29, 30,
37, 40, aperte la 34, 35, 36, 38, 39.

Fatto il 2026-09-19 sera e il 20 (`docs/MISURE.md` §M1, prima di scrivere codice; LEZIONI #91-#94):
lo stimatore della larghezza riscritto e la storia delle scelte nella riga `threads:` (decisione
sopra, validazione al punto 0), e il **primo pezzo di M1**, le misure 13-16: la traccia del routing
(`--route-trace`, `tools/route_trace_report.py`, `tests/test_route.c`) e il banco del disco (`make
bench-disk`). Il router del layer dopo prevede il 92-95% degli esperti con 12 candidati (il layer 0
no: 75-80%); con una LRU restano da leggere 348 / 143 / 32 MiB per token con la cache al 25 / 50 /
75% del modello; lo streaming per layer interi costa 15-53 volte di più; il disco dà ~1.5 GB/s con
qualunque numero di lettori. Attesi (modello a tempo, non misura) 7.5 tok/s con metà modello in
RAM, 18 col 75%, 3.6 col 25%. Chiuse le domande 13-16 e, come «no», la 5, la 34 e la 39; aperte la
41, la 42 e la 43.

**M1 in corso** (progetto in `docs/ARCHITETTURA.md` §Esecuzione «Esperti (M1)», dai numeri sopra):
archivio a slot allocati al caricamento, indice e LRU O(1), non il pin dall'uso (LEZIONI #93);
**letture a richiesta sul thread che chiama, senza thread di I/O e senza precaricamento** (con un
disco da 1.5 GB/s il disco è il collo: k=8 neutro, k=12 −40% nel modello a tempo; l'11-15% solo su un
disco 3× più veloce: domanda 43); un solo percorso (budget pieno = motore di prima); budget minimo
un layer più un token; un errore di lettura fa fallire la valutazione, non il processo. Letto come
fa colibri (tre difetti in `docs/UPSTREAM.md` #7-#9; preso: indice esperto → slot, esperto in uno
slot solo). Il volume è cifrato (BitLocker XTS-AES 128 in software, domanda 41) e non si tocca: è
lo stato di fabbrica dei PC bersaglio, M1 si progetta su ~1.5 GB/s.
- **Lotto 1, fatto** (2026-09-20): `src/memory/experts.{h,c}`, `tests/test_experts.c` (contenuti al
  byte, LRU calcolata a mano, un layer intero col minimo, residente, errori iniettati, 800 chiamate
  contro un modello di riferimento), otto mutazioni rosse (`tools/mutate_experts.sh`).
- **Lotto 2, fatto** (2026-09-20, riletto; quattro file riportati da CRLF a LF, LEZIONI #95): dimensioni delle parti per layer nell'archivio; `olmoe.c`
  prende gli esperti dall'archivio (denso residente, GGUF aperto per la vita del modello, piano
  automatico, `--expert-budget <MiB|min>` e `TR_EXPERT_BUDGET_MIB`), riga `experts:`, zona
  `weight_read`; `tests/test_stream.c` (logit al byte col budget minimo, errore di lettura a metà
  prompt, piano su numeri finti), `tools/mutate_stream.sh`, oracoli anche sotto `min`, modello vero
  `cmp` al byte. Se il lavoro è a metà si vede da `git status src/memory src/models tests/test_stream.c`.
- **Lotto 3, fatto** (2026-09-20): letture **senza la cache del sistema**
  (`tr_file_open_direct`: `FILE_FLAG_NO_BUFFERING`, `O_DIRECT`, `F_NOCACHE`), perché altrimenti il
  modello sta in RAM due volte e ogni misura direbbe la banda della RAM invece del disco. Il
  prezzo è l'allineamento a 4096: lo slot tiene un margine di un settore per parte e la lettura
  allineata atterra lì dentro senza copie, con la posizione della parte registrata a ogni
  riempimento (il resto dell'offset dipende da quale esperto è). Se il file system rifiuta si
  torna alla lettura normale, e la riga `experts:` dice quale delle due. `TR_EXPERT_DIRECT=0` la
  forza; `TR_MEM_AVAILABLE_MIB` fa credere al piano di avere meno RAM, così il suo ramo stretto si
  percorre davvero e non solo su numeri finti. 15 mutazioni rosse (più la riga di controllo verde), logit identici al byte fra
  diretta e normale, residente e al minimo.
- **Pronto, mai girato**: `sh tools/experts_budget.sh measure | misses | direct` (tok/s ai quattro
  budget con A/A, il costo di un token generato come differenza fra 72 e 8 token, la cache del
  sistema sì contro no). Si rifiuta di partire se la riga `experts:` non dice `direct`, perché
  misurerebbe la RAM. `tools/ab_modes.sh` raccoglie anche i contatori dell'archivio,
  `tools/experts_steady.awk` fa la sottrazione. **Serve un'ora o due di macchina ferma**: è il
  passo che manca a M1.
- **Poi**: le domande 42 (il layer 0), 43 (da che disco in su il precaricamento rende), 45 (una
  cache calda fra sessioni di codice). Tre matrici in una lettura sola: **no**, in GGUF i tre
  tensori sono separati (docs/ORIGINI.md §L'archivio degli esperti).

- **Domanda 44, fatta quasi tutta** (2026-09-20, `docs/MISURE.md` §Il comportamento sul codice…): sul
  codice OLMoE-1B-7B **non** è un grafo piccolo (il 25% delle unità copre il 68-73%), **non** è una
  tabella per token (9-16% al layer 0), **non** è statico (54-65% contro l'82-86% del router vivo),
  **non** è comprimibile spegnendo esperti (93.9% dei token col 50% spento, sul testo stesso della
  maschera); **c'è** una regione del codice comune a C, Python e shell (Jaccard 0.68-0.77, 0.07-0.09
  con la prosa inglese), e l'uso ordina gli esperti 40-50 volte meglio del caso. Nuovo: traccia
  versione 2 (id dei token, margini), `--expert-mask` (sola misura), `tools/route_graph_report.py`,
  `tools/mutate_reports.py`, `tools/mask_quality.sh`, cinque prompt `bench/prompts/trace-*.txt`.
  `make check` verde il 2026-09-20 con tutto questo dentro, e le 12 mutazioni di
  `tools/mutate_route.sh` tutte rosse (7 nuove: maschera e margini). **Resta `sh
  tools/mask_quality.sh` sui tre testi mancanti** (`trace-py`, `trace-sh`, `trace-prose-en`): la
  run è stata fermata due volte perché la macchina era senza memoria (3-5 GiB liberi su 31, le
  altre finestre al lavoro; ~7 minuti a testo). I due fatti bastano per la risposta, il terzo
  linguaggio e la prosa servono solo a irrobustirla.

**Ordine deciso da Marcello il 2026-09-19 sera**: (a) **M1**, esperti dal disco, a partire dalle
misure 13-16 e con dentro le cartelle GGUF e pool del confronto per componente (punto 5); (b) KV a
16 bit come modo dichiarato (punto 6); (c) int8 nel prefill come modo dichiarato, dopo il banco a 16
bit (punto 3, domanda 33); (d) `--spec` dalla lunghezza del match (punto 1, domanda 32); (e) chiusi
come «no», con la motivazione in MISURE: `tr_expf` in SIMD (domanda 39), pagine da 2 MB su Windows
(domanda 5), larghezza per zona (domanda 34). **La validazione dello stimatore (punto 0) viene dopo
tutto questo**, in una notte sola con le misure di (b) e (c). Fino ad allora **ogni misura di
velocità forza la larghezza** (`--decode-threads`), mai `auto`: nessuna conclusione poggia sullo
stimatore non validato.

0. **Validare lo stimatore riscritto sul modello vero** (domanda 31, LEZIONI #88). Il codice c'è
   dal 2026-09-19 sera (decisione sopra; `tests/test_phase.c` con un orologio finto, dieci mutazioni
   viste rosse con `tools/mutate_tune.sh`, `make check` verde) e **non ha ancora un numero**: nessuna
   run sul modello vero. Validazione **a macchina quieta, di notte**, la lancia Marcello alla fine
   dell'ordine qui sopra: `sh tools/decode_context.sh widths` (16 contro 8 e 4 contro 8 ai
   quattro contesti: da 2048 in su 4 e 8 stanno nel rumore, e un A/A del 4.6% non vede un −4%),
   `sh tools/decode_context.sh measure 16`, `sh tools/decode_context.sh long` (1500 token dopo un
   prompt di 1000: attraversa 1024 e 2048, ~1 minuto a run; stampa i cambi di ogni run dalla storia
   della riga `threads:`), `sh tools/threads_phase.sh widths`. Criterio: l'automatico sceglie ciò che
   la verità forzata indica a ogni contesto, cambi a zero fra run, al più un cambio 4 → 8 nella run
   lunga. I logit non cambiano (il risultato non dipende dai thread). Se il criterio cade, prima di
   toccare una costante si guardano i tempi delle passate di ogni misura (si aggiungono alla storia).
1. **Accendere `--spec` di default?** La condizione «caso peggiore come senza» **non è
   soddisfatta**: 0.953× dove il modello inventa, 1.175× dove ricopia (A/A, 8 giri). Cambiare le
   costanti della pausa non basta (replay coi costi misurati: al massimo 0.97-0.98×); la leva è il
   costo della riga di bozza, che per le bozze corte sta al 91% negli esperti (domanda 12). Lo
   decide Marcello: acceso accettando −5% / +17.5%, o spento come oggi. Coi thread per fase i due
   rapporti vanno rimisurati: `--spec 0` prende il 9% dal decode stretto, il caso peggiore di
   `--spec 8` il 6.4%, il caso buono il 2%. **Prima di decidere** (Marcello, 2026-09-19: «una
   ramificazione che eviti il −5 e tenga il +17»): la pausa è già quella ramificazione, e il −5% è
   il costo di scoprirlo sbagliando; la strada è decidere prima di provare, dalla lunghezza del
   match (domanda 32: si conta senza cronometro, poi replay).
2. **La larghezza misurata, su altre macchine** (domanda 31, dopo il punto 0): `sh
   tools/threads_phase.sh widths` e `sh tools/decode_context.sh widths` su una macchina diversa, appena
   ce n'è una (4-8 core, più canali di memoria, Apple Silicon senza pin, core P/E).
3. **Int8/VNNI nel prefill, sì o no** (domande 21 e 28, LEZIONI #65): è dove sta il divario con
   llama.cpp (1.8-2.3× sul kernel), ma non è esatto. Se sì: un modo dichiarato (`--fast-prefill`),
   spento di default, con l'oracolo che misura di quanto si spostano i logit. La strada esatta «8
   token nei registri» è misurata e scartata. Lo decide Marcello, che il 2026-09-19 ha chiesto se
   esiste altro fra float e int8: si misura nel banco la via di mezzo a 16 bit (domanda 33) prima
   di decidere; la parte seriale del prefill è fatta (punto 4), e dal lato esatto restano `expf`
   (domanda 37: le moltiplicazioni sono il 77% del prefill a 512 e il 56% a 4000, il resto è
   soprattutto `expf`) e, per il decode, i modelli a 4 bit. Ogni modo non esatto (int8, e la KV a 16 bit che il punto 6
   ora chiama in causa) si decide coi numeri di qualità davanti: token uguali e KL sul modello vero
   contro il modo esatto (`tools/compare_llamacpp.py` la calcola già: llama.cpp sta a 9e-3).
4. Prefill su prompt lunghi: **la parte esatta e `tr_expf` scalare sono fatti** (decisioni del
   2026-09-19 sopra: a macchina pulita 306-315 tok/s a 512, 303-308 a 2048, 276 a 4000). Restano:
   - `tr_expf` in SIMD: **no** (domanda 39, chiusa il 2026-09-19: 1.01-1.04× stimato, alla soglia
     del rumore, per circa 200 righe delicate; lo scalare ha preso quasi tutto);
   - i gruppi dell'attenzione con **GQA** e oltre 4096 token (domanda 38), quando arriva
     Qwen3-Coder: 8 teste di query per testa di chiavi possono stare nello stesso gruppo;
   - la KV a 16 bit **non** serve più al prefill (chiavi e valori si leggono una volta per gruppo):
     resta una leva del decode (punto 6).
5. Confronto per componente (decisione sopra), poi le correzioni che ne escono. **Sì di Marcello il
   2026-09-19, su tre fonti**: colibri, ds4 e llama.cpp. È una revisione di ciò che è già costruito
   (kernel, grafo OLMoE, GGUF, pool), una cartella per volta. Nel grafo guardare anche come le tre
   fonti tengono la KV: se è per posizione come era la nostra, la misura del 2026-09-19 (attenzione
   da 32-35 a 47-48 GB/s con le posizioni di una testa in fila) è una segnalazione per `UPSTREAM.md`.
   Le cartelle GGUF e pool si fanno dentro M1; le misure 13-16 (routing, cache esperti, SSD) sono
   il suo primo pezzo (ordine sopra).
6. Decode a contesto lungo: **la parte esatta è fatta** (decisione del 2026-09-19 sopra: 38.9 tok/s a
   contesto 32, 35.5 a 512, 27.5 a 2048, 20.9 a 4000, a 8 thread). Restano, in ordine:
   - **KV a 16 bit, sì o no** (domanda 36, decide Marcello): è la sola leva grande rimasta a contesto
     lungo, stimata dai byte **+5% a 512, +16-19% a 2048, +26-31% a 4000** e metà memoria della KV
     (a 8 bit: +6-7%, +20-25%, +33-43%). Non è esatta: solo come modo dichiarato e spento di
     default, deciso coi numeri di qualità davanti (KL e primo token contro il modo esatto sul
     modello vero, token greedy uguali su 1000 generati; llama.cpp, che fa questo e le attivazioni a
     8 bit, sta a KL 9e-3).
   - il confronto col decode di llama.cpp a run alternate (domanda 19), che ora va rifatto anche a
     contesto lungo: parte del suo vantaggio lì era la nostra KV letta a salti;
   - dal lato esatto, poco e sparso: le zone sotto il tetto (domanda 35: 6-11% in tutto); la
     larghezza per zona è chiusa come «no» (domanda 34: 2.1% a contesto 4000, sotto la soglia).
7. Memoria: la banda è misurata (~57 GB/s, domanda 4) e il decode ne usa l'89-94%. Le pagine da 2 MB
   (domanda 5) sono chiuse come «no» su Windows: chiedono un privilegio che un utente normale non ha.
8. Il pin su Linux: il codice c'è e `make check` lo prova, i numeri no (serve una macchina Linux
   vera, in WSL2 la topologia è sintetica). Resta l'ultima metà della domanda 22.
