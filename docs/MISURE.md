# Misure

Ogni tentativo di ottimizzazione, riuscito o scartato, con il suo numero. Regola
(`docs/ARCHITETTURA.md` §C e assembly): un guadagno conta solo se sposta la mediana più dello
spread misurato sulla stessa riga. Si sostituiscono le tabelle di base quando cambiano; i
tentativi si aggiungono in coda.

**Mai un numero da una run sola** (Marcello, 2026-09-17): ogni misura è la mediana di N run con
minimo, massimo e spread, e i token generati devono essere identici in tutte le run e con ogni
numero di thread (`tools/profile_suite.py` lo verifica ed esce con errore se no). Le righe più
sotto marcate «una run» sono precedenti alla regola e vanno rimisurate.

## Da misurare: domande aperte

Test per scendere nel dettaglio e trovare dove mettere mano. Si aggiunge quando nasce un dubbio,
si sposta nella sezione giusta con il numero quando è misurato.

| # | Domanda | Come si misura | Perché conta |
|---|---|---|---|
| 1 | ~~Quanto costa svegliare i thread su Windows nativo?~~ Misurato: §Pool di thread | — | — |
| 2 | ~~8 thread vincono perché stanno in un solo gruppo di core (CCD, L3 da 32 MB)?~~ No: 4+4 sui due CCD vanno **meglio** di 8 su uno solo. Misurato: §Dove vanno i thread | — | — |
| 3 | ~~Le copie SMT aiutano o peggiorano?~~ **Peggiorano**: 32 thread pinnati (i due fratelli di ogni core) contro 16 (uno per core) danno prefill 110 contro 164 e decode 2.2 contro 22.9. Misurato: §SMT | — | — |
| 4 | Qual è la banda reale della RAM di questa macchina? | microbenchmark di lettura sequenziale e `memcpy` da 1 GB, 1-16 thread | il tetto del decode (1.2 GB per token) si calcola da qui, non dai ~83 GB/s teorici (2×16 GB DDR5-5200); dal 2026-09-17 4 thread vanno come 16 a 41 GB/s |
| 5 | Quanti TLB miss per token, e quanto valgono le pagine da 2 MB? | Linux: pesi con `madvise(MADV_HUGEPAGE)` contro senza, stessi token/s mediani | 300 000 pagine da 4 KB toccate per token |
| 6 | Il portatile rallenta quando si scalda? | 60 s di decode continuo, token/s ogni 5 s | una misura breve può non valere per un uso reale |
| 7 | Quanto costa l'attenzione quando il contesto cresce (128, 1024, 4096 token)? | scenari con prompt lungo, zona `attention` | il coding usa contesti lunghi |
| 8 | Quanto costa il profiler acceso sul modello vero? | stessi scenari con e senza `--profile` | per fidarsi delle percentuali delle zone |
| 9 | Windows nativo e Linux sulla stessa macchina vanno alla stessa velocità? | stesso scenario nativo e in Docker/WSL | in Docker svegliare un thread costava ~20 µs |
| 10 | Più corsie (32 o 64) sbloccano la catena delle somme su un thread? | kernel sperimentale, `make bench` | AVX-512 va come AVX2 per quel limite; cambierebbe i numeri (dichiarato) |
| 11 | Quanto pesa la divisione per riga e la chiamata indiretta in `matmul_body`? | kernel che riceve tutta la matrice contro uno per riga | 1024 chiamate e 2 divisioni a 64 bit per matrice |
| 12 | Gli esperti scelti in token consecutivi si ripetono? **In parte sì** (§Revisione): una riga in più nella stessa passata fa leggere 2.3-4.7 esperti nuovi per layer su 8, cioè il 40-70% dei suoi esperti è già letto dalle altre righe. Resta la parte a tempo: quanto di una riga in più è lettura di pesi e quanto calcolo | zone del profiler native a macchina ferma, `generate --spec 0` contro `--spec k --spec-fixed` | dice se il costo di una bozza si attacca dal lato dei pesi (cache degli esperti) o del calcolo |
| 13 | Il router del layer successivo, applicato allo stato del layer corrente, indovina gli esperti che verranno scelti? | traccia del routing sul modello vero (prompt di codice, ~1000 token): esperti previsti (primi 8, 12, 16) contro scelti | se prevedendone al massimo 12 se ne indovina almeno l'80%, gli esperti si precaricano dal disco prima che servano (M1) |
| 14 | Quanti byte per token restano da leggere dal disco con una cache degli esperti grande il 25, 50, 75% del modello? | simulazione LRU sulla stessa traccia | decide il budget di RAM di default della M1 |
| 15 | Streaming dal disco per layer interi (DeepSpeed) o per esperti: quanti byte per token? | stessa traccia: layer interi contro soli esperti scelti e non in cache | con un solo utente leggere tutto il modello a ogni token dà al massimo ~0.3 tok/s su 10 GB |
| 16 | Quanto legge davvero l'NVMe in blocchi grandi quanto un esperto, a posizioni casuali, senza cache? | microbenchmark `pread` senza cache del sistema (`FILE_FLAG_NO_BUFFERING`, `O_DIRECT`), al massimo 60 s | con 14 e 15 dà i token/s attesi (M1 per esperti se ≥ 5 tok/s al 50%); decide anche la KV vecchia su SSD, attesa no con attenzione piena |
| 17 | Ricaricare dal disco la KV di un file già letto costa meno del prefill? La KV in q8 cambia i token? | checkpoint: logit identici al bit dopo il ricaricamento, tempo almeno 10 volte sotto il prefill; q8: token greedy uguali ≥ 99% su 1000 token di codice | leve del coding (file già letti, contesti lunghi), M5 |
| 18 | Perché il decode perde l'11-13% da 32 a 512 token di contesto, e llama.cpp solo il 5-7%? | profilo per zone a contesto 32, 512, 2048: quota dell'attenzione e della sua parte a thread singolo | nel coding il contesto è lungo: è la differenza che cresce di più dopo il prefill |
| 19 | Decode a 4-16 thread: llama.cpp è davvero avanti dell'8-12%? | `tools/speed_compare.py` con i due motori alternati run per run nella stessa sessione (fra due sessioni la mediana di llama.cpp si è spostata del 3-7%) | dice se c'è una leva nel decode a più thread o solo rumore |
| 20 | ~~Il prefill rende 1.16× da 8 a 16 thread: è il limite di potenza del portatile?~~ No, e nemmeno il CCD: è lo scheduler che mette due thread sullo stesso core fisico. Misurato: §Dove vanno i thread | — | — |
| 22 | Fissare i thread ai core fisici vale anche **sul decode** e **con la macchina occupata**? Misurato (§Il pin dei thread): sul decode **no a 16 thread** (0.82×) e **sì a 4-8** (1.14-1.32×); con la macchina occupata il prefill tiene (1.19×) e il decode perde (0.67×). Resta la sola parte **Linux**: su questa macchina non è misurabile (niente Linux nativo, e in WSL2 la topologia è sintetica, LEZIONI #47); il codice Linux gira ed è provato in `make check`, i numeri no | una macchina Linux vera, stessi scenari | il pin è una decisione di prodotto: sulle altre righe è già presa (acceso) |
| 26 | Quanti thread vuole **ogni fase**? Il prefill scala fino a 16 core (231.9 tok/s), il decode è al massimo a 4-8 thread pinnati (32.7) e a 16 scende a 25.1 | scenari di decode e prefill a 1/4/8/16 thread pinnati, contesto 512 e 2048; poi un pool che nel decode usa meno worker (i primi n slot, che sono già core distinti sui due chiplet) | oggi il numero di thread è uno solo per tutte e due le fasi: il decode lascia circa il 30% sul tavolo |
| 27 | ~~Perché il decode a 16 thread pinnati va il 17% più piano che senza pin?~~ Perché era pinnato a **un processore**: legando ogni thread al suo **core** (i due fratelli SMT) il decode torna alla pari e il prefill resta 1.24×. Non è la latenza di risveglio (spin da 2 a 20 ms: peggio). Misurato: §Il pin dei thread | — | — |
| 23 | ~~Quante bozze vengono accettate su lavoro di codice vero?~~ Misurato: §Speculazione dal prompt. 64% riscrivendo un file già nel prompt (1.42×), 13% scrivendo codice nuovo (0.62×) | — | — |
| 24 | ~~Una bozza sbagliata quanto costa?~~ **Molto più di quanto diceva la misura in container** (5.2 ms): nativo, col pin, una riga in più costa **15-27 ms** contro i ~35 di una passata, perché il token in bozza sceglie altri esperti e la passata legge anche i loro pesi. Il pareggio non è al 15% di bozze accettate ma intorno al **60-75%** (§Speculazione dal prompt, LEZIONI #59) | — | — |
| 25 | ~~Una bozza che si accorcia dopo un rifiuto e si allunga dopo un'accettazione toglie il caso peggiore?~~ Accorciarla non basta (0.89×): serve **fermarla** dopo una bozza tutta sbagliata, con pausa che raddoppia. Con quella: **1.01×** quando il modello inventa e **1.15×** quando ricopia (§Speculazione dal prompt) | — | — |
| 21 | ~~Quanto del divario col prefill di llama.cpp è il dot int8 VNNI contro il nostro float?~~ **Quasi tutto** (§Revisione, LEZIONI #65; la prima risposta, «niente», confrontava una riga contro un token): con la struttura del nostro kernel a 4 token il dot int8 VNNI a 512 bit è **1.8-2.3×** il nostro x4 float, e llama.cpp è avanti 1.57× a un thread. L'int8 non è esatto: scriverlo o no, e come modo esplicito, lo decide Marcello | — | — |
| 28 | ~~Il divario di prefill che resta con llama.cpp dov'è, se non è nell'int8?~~ Era nell'int8 (domanda 21). La strada esatta «più token nei registri» è misurata e non rende: una riga contro 8 token dà 1.13× a n=1024, **0.79×** a 2048 e 0.93× a 4096 (§Revisione) | — | — |

| 29 | Quanto vale una variante SIMD per **F16**? Oggi non c'è: `dot_row f16` fa 824 M el/s in ogni tier (codice scalare) contro 21 044 di `dot_row q8_0` AVX-512, cioè 25× (revisione, `make bench`, n=2048) | kernel con `vcvtph2ps` (F16C: la conversione half→float è esatta, quindi resta bit-identico), stesso confronto di `test_tiers_match_scalar` | un GGUF in f16 gira molto più piano del necessario; è una leva esatta e piccola da scrivere |
| 30 | La parte seriale del prefill (7.6% a 16 thread: copia per esperto, scrittura della KV, norme, RoPE) si può dividere per token? | `tr_parallel_for` sui token in quelle zone (ogni token è indipendente: stessi bit), profilo per zone prima e dopo | a 16 thread è il secondo pezzo dopo le moltiplicazioni; con l'int8 (domanda 21) il suo peso relativo raddoppia |

Macchina di riferimento: Ryzen 9 7940HX (Zen 4, 16 core / 32 thread, AVX-512 VNNI/BF16), 31 GB
RAM (2×16 GB DDR5-5200), NVMe Micron 1 TB, GPU RTX 4070 Laptop 8 GB e Radeon 610M (non usate fino
a M3/M6), Windows 11, MinGW-w64 gcc 15.2 `-O2`. Portatile in corrente, altri programmi aperti
(container Docker attivi): lo spread lo dice.

## Kernel CPU — base scalare (2026-09-17)

`build/bench/bench_kernels.exe --runs 5 --ms 100` (`tests/bench_kernels.c`). Milioni di elementi
al secondo per un prodotto scalare fra una riga di pesi e un vettore f32, un thread.

| Kernel | n=64 | n=1024 | n=2048 | n=4096 | spread |
|---|---|---|---|---|---|
| dot_f32 | 2779 | 3195 | 3201 | 3230 | 1.5-4% |
| dot_row f16 | 699 | 700 | 632 | 639 | 1-14% |
| dot_row q8_0 | 1728 | 1796 | 1909 | 1952 | 4-15% |

`tr_matmul` q8_0 1024×2048 (forma di una proiezione di esperto OLMoE), un token:
1 thread 1.070 ms (spread 1.6%); 16 thread 0.350 ms (spread 74%).

Letture:
- f16 è il più lento: la conversione half → float elemento per elemento costa più del prodotto.
- Con 16 thread una sola matrice va solo 3 volte più veloce, con spread enorme: per un lavoro da
  un millisecondo svegliare i thread pesa quanto il calcolo. I thread vanno usati su blocchi più
  grandi (un intero strato, tutti gli esperti di un token), non per matrice.

## Profilo del motore — modello minuscolo (2026-09-17)

`make profile` (3 scenari su OLMoE minuscolo: hidden 64, 4 strati), kernel scalari, nativo Windows.
Numeri che servono solo a provare la suite: con dimensioni così piccole il costo fisso di ogni
chiamata domina, e le proporzioni non valgono per un modello vero.

| Scenario | Prefill tok/s | Decode tok/s | Zone più pesanti in decode |
|---|---|---|---|
| f32, prompt 6 → 64 token | 13380 (±3.5%) | 10909 (±0.9%) | attenzione 28%, rope 17%, gate_up esperti 13%, proiezioni qkv 13% |
| f32, prompt 96 → 8 token | 9784 (±4.1%) | 7318 (±3.7%) | attenzione 51%, rope 12% |
| q8_0, prompt 6 → 64 token | 10704 (±18%) | 8872 (±2.3%) | attenzione 24%, gate_up 16%, qkv 16%, rope 14% |

Costo del profiler acceso: +0.25% (dentro il rumore). Da verificare sul modello vero: rope al 17%
fa pensare a seno e coseno ricalcolati a ogni chiamata (una tabella precalcolata sarebbe esatta).

## Profilo del motore — OLMoE-1B-7B Q8_0, kernel scalari (2026-09-17)

`build/trochilus generate -m models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf -p 32 -n 16 -c 128 --profile`,
nativo Windows, 16 thread (core fisici), un solo run: base di partenza, non ancora mediana.
RAM libera prima del caricamento 10.8 GiB (container Docker attivi).

| Fase | Token/s | Pesi toccati per token | Traffico di memoria |
|---|---|---|---|
| prefill (un token alla volta) | 6.92 | 1200 MiB | 8.7 GB/s |
| decode | 6.87 | 1200 MiB | 8.7 GB/s |

Dove va il tempo in decode: esperti gate+up 44.9%, esperti down 22.3%, proiezioni q/k/v 14.3%,
proiezione di uscita dell'attenzione 5.0%, lm_head 4.8%, attivazione degli esperti 3.5%, rope 2.7%,
attenzione 1.3%, router 0.9%, tutto il resto sotto lo 0.2%.

Letture:
- **Limitati dal calcolo, non dalla memoria**: 8.7 GB/s contro gli ~80 GB/s teorici della RAM.
  I kernel SIMD sono la leva giusta adesso; la memoria diventerà il limite solo dopo.
- **Il 92% del tempo è in prodotti matrice-vettore q8_0** (esperti 67%, proiezioni 19%, lm_head 5%):
  il primo kernel da vettorizzare, e poi da portare in assembly, è `dot_row q8_0`.
- Il prefill va alla stessa velocità del decode perché elabora un token alla volta: il prefill a
  blocchi (più token nella stessa moltiplicazione) è la seconda leva.
- Rope al 2.7%: sul modello vero non è una priorità (sul minuscolo sembrava 17%).
- I token generati da un prompt sintetico ripetono `64,301`: normale per greedy su input senza
  senso, ma la correttezza sul modello vero non è ancora verificata (serve il confronto con un
  riferimento: llama.cpp o transformers).

## Kernel CPU — AVX2 e AVX-512, bit-identici (2026-09-17)

`make bench` nativo, stessa macchina, milioni di elementi al secondo su un thread (n = 2048):

| Kernel | scalare | AVX2 | AVX-512 |
|---|---|---|---|
| `dot_f32` | 3 310 | 26 356 (8.0×) | 31 807 (9.6×) |
| `dot_row q8_0` | 1 957 | 19 532 (10.0×) | 19 837 (10.1×) |

`tr_matmul` q8_0 1024×2048 (una matrice di esperto, 1 token): 1 thread 0.100 ms, 16 thread 0.146 ms.

Letture:
- AVX-512 va quasi quanto AVX2: il limite è la **catena delle somme**. Ogni corsia somma un
  prodotto dopo l'altro, e una somma aspetta la precedente (~3-4 cicli di clock): 16 corsie danno
  al massimo 16 elementi ogni 3-4 cicli, con registri da 256 o da 512 bit. Più corsie romperebbero
  la definizione scalare (numeri diversi): da valutare solo se un thread singolo torna il limite.
- Sulla carta 16 thread da 20 000 M el/s supererebbero la banda della RAM (~80 GB/s); sul modello
  vero no: il traffico si ferma a 22 GB/s perché i thread non rendono (sezione sotto).
- Su una matrice di esperto 16 thread sono **più lenti** di uno: svegliare il pool costa più del
  lavoro (0.1 ms). Confermato sul modello vero (sezione sotto).

## Profilo del motore — OLMoE-1B-7B Q8_0, kernel AVX-512 (2026-09-17)

`trochilus generate -p 32 -n 16 -c 128 --profile`, una run per riga (non ancora mediana di 5).
Token generati identici in tutte le righe e con `TR_CPU_MAX=scalar`.

| Kernel | Thread | Decode tok/s | Prefill tok/s | Traffico memoria |
|---|---|---|---|---|
| scalare | 16 | 6.87 | 6.92 | 8.7 GB/s |
| scalare | 8 | 7.18 | — | — |
| AVX-512 | 16 | 17.72 | 17.83 | 22.3 GB/s |
| AVX-512 | 8 | **21.08** | — | — |
| AVX-512 | 4 | 20.68 | — | — |
| AVX-512 | 1 | 13.85 | — | — |

Zone del decode a 8 thread: gate+up degli esperti 34.8%, rope 13.0%, q/k/v 11.3%, attivazione degli
esperti 8.5%.

Letture:
- **3× sul modello vero** (6.87 → 21.08), ma 16 thread vanno più piano di 8 e 1 thread fa già
  13.85: i thread rendono solo 1.5×. Il pool si sveglia 464 volte per token per matrici da 0.1 ms.
  Il default (thread = core fisici) oggi è sbagliato per il decode; la cura non è un numero fisso
  ma parallelizzare lavori più grossi (gli 8 esperti di un token, lo strato intero).
- **Rope è salito al 13%**: calcola `pow`, `cos` e `sin` per ogni elemento a ogni token. Una tabella
  per posizione dà gli stessi numeri (stesse operazioni in double) e toglie quasi tutto.
- **Attivazione degli esperti all'8.5%**: `expf` scalare su 1024 elementi per esperto.
- Il traffico (22 GB/s) è ancora sotto la banda della RAM (~80 GB/s): il limite resta il calcolo
  e il coordinamento dei thread, non la memoria.

## Pool di thread: attesa attiva invece del sonno (2026-09-17)

`bench_kernels --runs 7 --ms 60` in Linux (Docker su WSL2, stessa macchina), mediana di 7 run.
Dispatch = una `tr_parallel_for` vuota che dà un pezzo a ogni thread; matmul = una matrice di
esperto q8_0 1024×2048, mediana di 50 chiamate di fila.

| Thread | Dispatch prima | Dispatch dopo | Matmul prima | Matmul dopo |
|---|---|---|---|---|
| 1 | 0.002 µs | 0.002 µs | 0.099 ms | 0.098 ms |
| 2 | 41.0 µs | 0.10 µs | 0.100 ms | 0.049 ms |
| 4 | 90.0 µs | 0.37 µs | 0.126 ms | 0.025 ms |
| 8 | 181.8 µs | 1.62 µs | 0.211 ms | 0.015 ms |
| 16 | 361.9 µs | 2.80 µs | 0.376 ms | 0.014 ms |

Letture:
- Prima ogni thread svegliato dal sistema costava ~20 µs in Docker: più thread, più lento.
- Adesso i thread aspettano svegli fino a 2 ms (istruzione `pause`), poi dormono: la matrice
  scala 2× a 2 thread, 4× a 4, 6.5× a 8; a 16 quasi non guadagna più (0.015 → 0.014 ms): da
  capire se è la banda della RAM o i due gruppi di core (domande 2 e 4).

Windows nativo, stesso benchmark (pool vecchio compilato a parte), mediana di 7 run:

| Thread | Dispatch prima | Dispatch dopo | Matmul prima | Matmul dopo |
|---|---|---|---|---|
| 1 | 0.024 µs | 0.017 µs | 0.123 ms | 0.099 ms |
| 2 | 0.48 µs | 0.05 µs | 0.106 ms | 0.092 ms |
| 4 | 2.2 µs | 0.19 µs | 0.051 ms | 0.047 ms |
| 8 | 5.7 µs | 0.89 µs | 0.052 ms | 0.024 ms |
| 16 | 53.5 µs | 1.48 µs | 0.079 ms | 0.014 ms |

- Windows sveglia i thread molto più in fretta di Docker (53 µs contro 362 a 16 thread), ma il
  guadagno resta grande da 8 thread in su: a 16 la matrice va 5.6× più veloce.
- Spread alti (fino al 1305% sul dispatch a 2 thread): un singolo risveglio dal sonno pesa
  su una misura da 0.05 µs. Le matmul, più lunghe, stanno sotto il 36%.

## Profilo del motore — OLMoE-1B-7B Q8_0, AVX-512 + pool nuovo (2026-09-17)

`make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json`: prompt 32, 32 token generati, contesto
128, Windows nativo, **mediana di 5 run** (più una di riscaldamento). Token identici in tutte le
20 run e con ogni numero di thread.

| Thread | Decode tok/s (min–max) | Prima (pool vecchio, una run) | Traffico memoria |
|---|---|---|---|
| 16 | **26.90** (26.54–27.53) | 17.72 | 33.9 GB/s |
| 8 | **27.02** (26.42–27.06) | 21.08 | 34.0 GB/s |
| 4 | 24.48 (24.37–25.45) | 20.68 | 30.8 GB/s |
| 1 | 13.99 (12.47–14.39) | 13.85 | 17.6 GB/s |

Zone del decode a 16 thread: gate+up esperti 32.9%, down esperti 16.7%, q/k/v 12.3%, **attivazione
esperti 10.7%**, **rope 9.2%**, lm_head 5.9%, **attenzione 5.4%**, uscita attenzione 5.3%.

Letture:
- 16 thread non vanno più piano di 8, ma nemmeno più forte: 27 tok/s per entrambi.
- **Un quarto del token gira su un thread solo**: attivazione degli esperti, rope e attenzione
  (in grassetto sopra) valgono il 25%. Con 16 thread quella parte non accelera (Amdahl): è il
  prossimo lavoro esatto (tabella per rope, esperti in parallelo, attenzione vettorizzata).
- **Gli esperti ora sono limitati dalla memoria**: una moltiplicazione di esperto nel modello vero
  dura ~48 µs a 16 thread, contro 14 µs nel benchmark dove la matrice è già in cache. Ogni matrice
  è 2.2 MB da leggere dalla RAM: 2.2 MB in 48 µs sono ~46 GB/s, vicino alla banda reale di due
  moduli DDR5. Da misurare quella banda (domanda 4) prima di cercare altro sugli esperti.

## Profilo del motore — OLMoE-1B-7B Q8_0, parte a thread singolo parallela (2026-09-17)

Stesso comando del profilo sopra (`make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json`,
Windows nativo, mediana di 5), prima e dopo nella stessa ora: rope da tabella per posizione,
attivazioni di tutti gli esperti come un solo lavoro parallelo, attenzione divisa per testa con la
somma pesata AVX-512 (`axpy_f32`). Token identici fra prima e dopo, fra run e fra thread; logit
del modello vero identici al bit a prima (200 token con 16, 3 e 1 thread, e con `TR_CPU_MAX=scalar`).

| Thread | Decode prima (min–max) | Decode dopo (min–max) | Guadagno | Traffico memoria dopo |
|---|---|---|---|---|
| 16 | 26.34 (26.15–26.51) | **32.78** (32.35–33.27) | +24% | 41.3 GB/s |
| 8 | 26.02 (25.68–26.40) | **33.88** (30.39–34.91) | +30% | 42.6 GB/s |
| 4 | 23.69 (23.60–24.54) | **32.93** (32.06–33.60) | +39% | 41.5 GB/s |
| 1 | 13.11 (12.04–13.32) | 14.47 (14.36–14.61) | +10% | 18.2 GB/s |

Zone del decode a 16 thread, ms su 32 token:

| Zona | Prima | Dopo |
|---|---|---|
| attivazione esperti | 124.2 (10.6%) | 10.0 (1.1%) |
| rope | 107.6 (9.1%) | 0.8 (0.1%) |
| attenzione | 62.2 (5.2%) | 23.6 (2.5%) |
| gate+up esperti | 389.8 (33.1%) | 397.5 (41.9%) |
| down esperti | 196.3 (16.7%) | 199.2 (20.9%) |
| q/k/v | 143.6 (12.2%) | 144.9 (15.3%) |

Letture:
- La parte a thread singolo è passata dal 25% al 4% del token: il guadagno previsto c'è tutto.
- **4 thread vanno come 16** (32.9 contro 32.8 tok/s): quello che resta sono moltiplicazioni di
  matrici che leggono 1.2 GB di pesi per token, e a 41 GB/s il limite è la memoria, non i core.
  La domanda 4 (banda reale della RAM: 2×16 GB DDR5-5200, ~83 GB/s teorici) è ora la prima.
- A un thread +10%: rope e somma pesata vettorizzata, niente parallelismo.
- Spread 13% a 8 thread (una run a 30.4): rumore della macchina, le altre righe stanno sotto il 5%.

## Correttezza sul modello vero — Trochilus contro llama.cpp (2026-09-17)

`tools/compare_llamacpp.py` nel container, OLMoE-1B-7B-0125-Instruct Q8_0, prompt
`bench/prompts/dante.txt` (template di chat), 128 token greedy, llama.cpp `b49650a` su CPU.

| Cosa | Risultato |
|---|---|
| token del prompt (27) | identici a `llama-tokenize` |
| generazione greedy | identica per i primi 19 token, poi le strade si separano |
| stessa parola migliore, 155 posizioni (prompt + generati, teacher forcing) | 149/155; nelle 6 diverse il margine di Trochilus fra le prime due è 0.04–0.18 |
| KL(llama.cpp ‖ Trochilus) | media 8.9e-3, massimo 0.30 |
| massima differenza di un logit | 3.63 |

Non conclusivo: llama.cpp moltiplica i pesi Q8_0 con attivazioni quantizzate a 8 bit e tiene la
cache KV in f16, Trochilus usa attivazioni esatte. La prova che decide è transformers sugli stessi
pesi (sezione sotto). Entrambi i motori inventano i versi di Dante: limite del modello.

## Correttezza sul modello vero — Trochilus contro transformers, 2 layer (2026-09-17)

`make oracle-real`: i primi 2 layer del GGUF Q8_0 vero copiati byte per byte
(`tools/make_olmoe_2layer_gguf.py`); transformers 5.14.1 float32 con gli stessi pesi dequantizzati e
la configurazione letta dal GGUF (`tools/make_olmoe_2layer_ref.py`). 32 token greedy e logit a ogni
posizione (teacher forcing). Container Linux, gcc.

| Prompt | Token | Greedy | Parola migliore | Massima differenza di un logit |
|---|---|---|---|---|
| `bench/prompts/dante.txt` | 27 + 32 | identici | 59/59 | 1.3e-5 (posizione 8) |
| primi 1024 token di `src/models/olmoe.c` | 1024 + 32 | identici | 1056/1056 | 2.4e-4 (posizione 813) |

Soglia del controllo: 1e-3. La differenza residua viene dall'aritmetica, non dal modello: Trochilus
moltiplica i blocchi Q8_0 con attivazioni esatte, torch moltiplica matrici già dequantizzate; cresce
con la posizione (sommatorie dell'attenzione più lunghe). Costo: riferimento 29 s e 6.0 GB di picco
(solo quando manca), confronto 33 s e 1.1 GB a ogni `make check`.

## Velocità — Trochilus contro llama.cpp e colibri, OLMoE-1B-7B (2026-09-17)

`tools/speed_compare.py` nel container Linux (`trochilus-dev`, gcc; la VM di Docker vede 32 CPU e
15 GB), modelli nel volume Docker `trochilus-models` (non dal disco di Windows: LEZIONI #41), un motore
alla volta, **mediana di 5 run** (min–max) più una di riscaldamento; container degli altri progetti
accesi ma fermi. Trochilus e llama.cpp sullo stesso GGUF Q8_0; colibri (`a90bed9`, `ARCH=native`) sulla
sua conversione int8 da Hugging Face, cache di 64 esperti per layer (tutti), processo nuovo a ogni run.
llama.cpp `b49650a` con `GGML_NATIVE`: `llama-bench`, prefill a contesto vuoto e decode dopo un
contesto grande quanto il prompt (`-d`). Decode = valutazioni di un token al secondo (LEZIONI #40).
Token del prompt: codice vero (`src/models/olmoe.c`) per Trochilus e colibri, casuali per llama-bench.

**Prompt 32, 32 generati** (tok/s)

| Thread | Trochilus prefill | Trochilus decode | llama.cpp prefill | llama.cpp decode | colibri prefill | colibri decode |
|---|---|---|---|---|---|---|
| 16 | 30.4 (28.5–33.0) | 31.1 (28.0–31.7) | 232.9 (230.3–245.4) | 34.8 (34.0–35.3) | 10.1 (3.9–10.8) | 12.4 (11.5–15.1) |
| 8 | 34.0 (33.1–34.1) | 34.0 (31.3–34.2) | 200.9 (198.0–212.6) | 36.9 (36.3–38.2) | 12.1 (11.2–13.1) | 13.3 (11.1–14.0) |
| 4 | 33.7 (32.9–34.1) | 33.3 (30.3–34.5) | 126.0 (124.7–129.5) | 36.1 (35.7–36.4) | 12.3 (12.2–12.8) | 12.5 (10.8–13.6) |
| 1 | 15.4 (13.8–15.6) | 15.4 (15.0–15.5) | 38.3 (37.9–38.9) | 22.5 (22.0–22.7) | 8.3 (7.9–8.6) | 6.2 (5.4–6.5) |

**Prompt 512, 128 generati** (tok/s; colibri non misurato)

| Thread | Trochilus prefill | Trochilus decode | llama.cpp prefill | llama.cpp decode |
|---|---|---|---|---|
| 16 | 29.6 (28.7–29.9) | 27.7 (27.5–29.0) | 376.6 (361.3–382.5) | 33.1 (32.5–34.2) |
| 8 | 31.8 (31.1–32.4) | 29.7 (28.2–31.0) | 270.2 (262.7–275.3) | 34.2 (33.9–34.5) |

**Tokenizer** su 2.2 MB di codice e testo (sorgenti di llama.cpp, README di colibri in italiano,
inglese e cinese), tempo senza il caricamento del vocabolario:

| Tokenizer | Secondi (min–max) | MB/s | Token |
|---|---|---|---|
| Trochilus | 0.100 (0.097–0.104) | 22.0 | 815 460 |
| HF `tokenizers` 0.22.2 (in processo) | 1.31 (1.24–1.45) | 1.7 | 815 460 |
| llama.cpp `llama-tokenize` | 3.60 (3.52–3.64) | 0.6 | 814 194 (diversi da HF, non indagato) |

Letture:
- **Prefill: llama.cpp 8× più veloce a 16 thread con prompt 32, 13× con prompt 512** (2.5× a 1
  thread). Moltiplica tutti i token del prompt insieme; in Trochilus prefill = decode. È il primo lavoro.
- **Decode a 4-16 thread: llama.cpp avanti dell'8-12%** a contesto corto, del 15-19% a contesto 512.
  Con prompt 512 Trochilus perde l'11-13%, llama.cpp il 5-7%: il costo del contesto lungo è nostro
  (domanda 18). La prima serie di llama.cpp (decode a contesto vuoto) dava 33.2 / 34.4 / 33.6: fra due
  sessioni la mediana si sposta del 3-7%, il numero preciso chiede run alternate (domanda 19).
- **1 thread: llama.cpp 1.46× nel decode**: attivazioni in int8 con VNNI (leva 2, non esatta).
- 8 thread battono 16 in entrambi i motori: il limite è la banda della memoria.
- colibri a 12-13 tok/s riparte ogni run con la cache degli esperti vuota (hit 90%): misura del suo
  avvio a freddo, non del suo regime.
- **Tokenizer: Trochilus 13× HF e 36× llama.cpp**, stessi token di HF.
- Trochilus nel container va ~5% sotto Windows nativo (31.1 contro 32.8 tok/s a 16 thread).

## Prefill a blocchi in C esatto + kernel a 4 token — OLMoE-1B-7B Q8_0 (2026-09-17)

Container, volume `trochilus-models`, prompt 512 token sintetici, 16 generati, **run alternate**
binario vecchio / binario nuovo a ogni giro (`tools/ab_speed.sh`; LEZIONI #46: misurando prima tutto
A e poi tutto B la macchina che si scalda falsa il confronto), mediana di 5 giri più uno di
riscaldamento. «Prima» = binario di `38d0771` (un token per passata). «Dopo» = passate da 512 token,
coppie (token, esperto) ordinate per esperto, matrici a blocchi di 16 token con una riga di pesi
contro 4 token nei registri, logit solo dell'ultimo token.

| Thread | Prefill prima | Prefill dopo | Guadagno | Decode prima | Decode dopo |
|---|---|---|---|---|---|
| 16 | 30.1 (26.9–31.1) | **196.9** (158.3–211.2) | 6.5× | 27.9 (24.9–30.9) | 29.4 (24.8–30.9) |
| 8 | 31.0 (29.5–31.9) | **170.3** (147.9–185.1) | 5.5× | 29.6 (26.9–31.6) | 29.7 (28.3–29.8) |
| 4 | 31.0 (30.7–32.0) | **100.0** (94.4–101.2) | 3.2× | 29.4 (26.0–31.1) | 27.0 (26.6–30.1) |
| 1 | 14.4 (13.4–14.6) | **28.4** (26.8–29.0) | 2.0× | 13.5 (12.8–13.7) | 13.4 (12.7–13.8) |

Contro llama.cpp, anche qui alternati run per run (stesso GGUF, `llama-bench -p 512 -n 0 -r 1`):

| Thread | Trochilus | llama.cpp | Divario |
|---|---|---|---|
| 16 | 162.8 (121.8–173.8) | 321.0 (184.9–349.7) | 1.97× |
| 8 | 156.7 (154.2–163.2) | 216.8 (180.0–251.6) | 1.38× |
| 4 | 93.6 (90.4–96.8) | 146.3 (136.9–148.3) | 1.56× |
| 1 | 26.1 (24.3–27.0) | 41.0 (40.9–41.2) | 1.57× |

Esattezza sul modello intero (16 layer), 200 token: `logits` del binario di prima e di dopo, un token
per passata, identici al bit (`cmp`), e identici fra 16 e 3 thread; passate da 16, 64 e 200 token
identiche al bit alle stesse posizioni. Sul modello a 2 layer (`make oracle-real`) passate da 3, 64 e
1056 token identiche al bit; `generate` sul prompt da 1024 token dà i token greedy di transformers.

Letture:
- **Prefill 6.5× a 16 thread** (30 → 197 tok/s), 2.0× anche a 1 thread. Il divario con llama.cpp
  scende da 11.8× a **1.97×** (1.4-1.6× con meno thread).
- **Decode invariato** (differenze dentro lo spread): stesso codice con una passata da un token,
  logit identici al bit.
- **Scala ancora male**: da 4 a 16 thread rende 2.0×, da 8 a 16 solo 1.16×; llama.cpp 2.2× da 4 a 16.
  Le due misure di Trochilus a 16 thread di questa sessione (197 e 163) hanno spread ampio: a 16
  thread la macchina è al limite di potenza e la variabilità sale (domanda 20).
- **A 1 thread llama.cpp è 1.57× avanti**: stesso lavoro con dot int8 VNNI (leva 2, domanda 21).

## Profilo del prefill — dove va il tempo (2026-09-17)

`bench/scenarios-olmoe-1b-7b.json` (scenari `prefill512-t*`, `tools/profile_suite.py` ora riporta le
zone anche in prefill), prompt 512, mediana di 5, prima del kernel a 4 token.

| Thread | Prefill tok/s | matmul | attenzione | seriale (norme, RoPE, KV, router, copia per esperto, mix) |
|---|---|---|---|---|
| 16 | 144.3 | 89.8% | 2.4% | 7.6% |
| 8 | 114.9 | 91.1% | 2.3% | 6.4% |
| 4 | 66.2 | 93.6% | 2.2% | 4.0% |
| 1 | 18.2 | 95.5% | 2.1% | 2.3% |

Zona per zona a 16 thread: `expert_gate_up` 43.9%, `expert_down` 23.2%, `qkv_proj` 17.2%,
`attn_out_proj` 5.5%, `attention` 2.4%, `kv_write` 1.6%, `router` 1.5%, `expert_gather` 1.5%.
Byte di pesi distinti 12.5 MiB/token (1.9 GB/s): il prefill non è al limite della banda come il
decode (1.2 GiB/token, 41 GB/s), il tempo sta nelle moltiplicazioni.

## Dove vanno i thread — il prefill a 16 thread (2026-09-17)

Windows **nativo** (nel container la topologia non è quella vera, LEZIONI #47), binario di
`5db8117`, OLMoE-1B-7B Q8_0 dal disco, prompt sintetico di 2048 token, `-n 0`. Affinità del
processo fissata appena parte, prima che il pool lavori; i casi girano alternati a ogni giro
(LEZIONI #46), mediana di 3 giri dopo uno di riscaldamento. Il clock viene dal contatore di
Windows «Informazioni processore(_Total)\Prestazioni processore %» (nominale 2400 MHz),
campionato mentre il prefill gira.

| Collocamento | Thread | Core usati | Prefill tok/s (min-max) | Clock mediano |
|---|---|---|---|---|
| default, nessuna affinità | 16 | li sceglie Windows fra 32 logici | 134.1 (134.1-138.2) | 4.4-4.6 GHz |
| uno per core fisico (`0x55555555`) | 16 | 16 fisici, i due CCD | **173.9** (173.3-183.1) | 4.2-4.5 GHz |
| un solo CCD (`0x0000FFFF`) | 8 | core 0-7, una L3 da 32 MB | 78.6 (77.7-80.8) | 4.4-4.6 GHz |
| l'altro CCD (`0xFFFF0000`) | 8 | core 8-15 | 80.8 (79.1-82.8) | 4.6-4.7 GHz |
| 4+4 sui due CCD (`0x00FF00FF`) | 8 | core 0-3 e 8-11 | **86.1** (84.2-89.0) | 4.7-4.8 GHz |

Clock durante il prefill senza affinità, mediane di 3 giri: 1 thread 4.66-4.80 GHz, 4 thread
4.64-4.78, 8 thread 4.53-4.69, 16 thread 4.46-4.54.

Letture:
- **Non è il limite di potenza** (domanda 20): da 1 a 16 thread il clock scende del 6-8%, non
  del 40% che servirebbe a spiegare 1.16×. In corrente la macchina tiene ~4.5 GHz su tutti i core.
- **Non è il CCD** (domanda 2): 8 thread divisi 4+4 sui due chiplet vanno il **7-9% meglio** di 8
  sullo stesso chiplet. Due L3 da 32 MB e il doppio di L2 battono la località, e il clock sale.
- **È dove finiscono i thread**: con 16 thread su 32 processori logici Windows ne appoggia due
  sullo stesso core fisico e lascia core liberi. Un thread per core fisico dà **+30%**
  (134 → 174 tok/s). Da 8 core (86.1) a 16 (173.9) il prefill rende **2.02×**: il parallelo
  scala, era il collocamento a non scalare (LEZIONI #48).
- A 2048 token di prompt il prefill fa 174 tok/s contro i 197 misurati a 512 senza pin:
  l'attenzione per token cresce col contesto, da misurare a parte (domanda 7).

Prossimo passo che ne discende: fissare i thread del pool ai core fisici, e rimisurare le righe
di velocità con quella base.

## Il pin dei thread ai core fisici (2026-09-18)

Windows **nativo** (LEZIONI #47), macchina ferma: container degli altri progetti fermati e cache
della VM di Docker restituita, 15.9 GB liberi (misurare mentre un agente compila in Docker dà numeri
falsi, LEZIONI #57). Un solo binario e tre modi, scelti con `TR_POOL_PIN`: **0** nessuna affinità,
**1** ogni thread su un processore logico suo, **2** (il default) ogni thread sul suo **core fisico**
intero, cioè libero fra i due fratelli SMT di quel core ma solo di quello. Run **alternate**
(`tools/ab_speed.sh`, LEZIONI #46), mediana dei giri dopo uno di riscaldamento, OLMoE-1B-7B Q8_0 dal
disco di Windows, prompt 512, 24 token generati.

**Il default contro nessun pin** (2 contro 0), 6 giri:

| thread | prefill core | prefill senza | | decode core | decode senza | |
|---|---|---|---|---|---|---|
| 16 | **217.8** (184.9-220.3) | 175.1 (148.4-183.4) | **1.24×** | 28.7 (26.4-29.8) | 29.9 (27.2-30.3) | 0.96× |
| 8 | **147.3** (141.9-151.4) | 113.4 (105.6-117.3) | **1.30×** | 31.0 (28.4-31.5) | 30.4 (26.4-30.9) | 1.02× |

**I due modi di pin** (2 contro 1), 3 giri: a 16 thread il core intero dà prefill 212.5 contro 178.6
e decode 27.2 contro 21.5; a 8 thread 145.9 contro 141.4 e 29.4 contro 31.1.

**Il pin al singolo processore contro nessun pin** (1 contro 0), 3 giri, la strada scartata:

| thread | prefill | senza | | decode | senza | |
|---|---|---|---|---|---|---|
| 16 | 231.9 | 191.3 | 1.21× | 25.1 | **30.5** | **0.82×** |
| 8 | 171.0 | 125.9 | 1.36× | 32.7 | 28.6 | 1.14× |
| 4 | 90.5 | 60.4 | 1.50× | 32.5 | 24.6 | 1.32× |
| 1 | 24.0 | 24.0 | 1.00× | 12.2 | 10.7 | 1.14× |

Stesso modo (1 contro 0) a prompt 2048, 16 token: 16 thread prefill 159.2 contro 147.2 (1.08×) e
decode 19.5 contro 23.4 (0.83×); 8 thread prefill 130.1 contro 86.7 (1.50×) e decode 21.4 contro
22.0. Macchina occupata (4 processi che girano a vuoto, prompt 512, 16 thread): prefill 206.5 contro
173.6 (**1.19×**), decode 20.3 contro 30.4 (0.67×).

Letture:
- **Il prefill guadagna sempre**, e tanto più quanto meno thread ci sono: a 4 thread vale 1.50×,
  perché senza affinità Windows appoggia quei 4 thread su 2 core fisici. Col default (pin al core) a
  16 thread è 1.24×. Anche con la macchina occupata il guadagno resta (1.19×): il pin non è fragile
  sotto contesa (domanda 22).
- **Il modo di pinnare decide il decode.** Legare un thread a **un processore** costa il 17-18% sul
  decode a 16 thread (0.82×, e 0.67× con la macchina occupata); legarlo al **suo core** lo riporta
  alla pari (0.96× e 1.02×, dentro lo spread) e tiene il guadagno del prefill. Quello che serve non
  è inchiodare un thread, è impedire che **due** thread finiscano sullo stesso core: il fratello
  libero è una valvola che il decode usa e il prefill no. Chiude la domanda 27.
- Non era la latenza di risveglio: allungando lo spin da 2 a 20 ms il decode a 16 thread pinnati al
  processore andava 21.0 invece di 23.6, cioè peggio.
- **Il decode preferisce 8 thread a 16** in ogni modo (31.0 contro 28.7 col default): il prefill
  vuole tutti i core, il decode no. Oggi il numero di thread è uno solo per tutte e due le fasi:
  è una leva aperta (domanda 26).
- A 1 thread il pin non cambia il prefill (non c'è niente da collocare) e dà +14% sul decode.

## Attivazioni int8 con VNNI: quanto darebbe la leva 2 (2026-09-18)

Domanda 21. Candidato scritto **nel benchmark**, non nei kernel (`tests/bench_kernels.c`): quantizza
le attivazioni a int8 a blocchi di 32 come Q8_0 e usa `_mm256_dpbusd_epi32`, con la stessa struttura
di ggml (accumulo int32 per blocco, conversione e scala in un accumulatore float, somme dei pesi del
blocco precalcolate per la correzione +128). Non è e non sarà un kernel nostro: le attivazioni a 8
bit sono un altro numero, quindi non può essere bit-identico. `make bench`, 9 run da 40 ms, mediana.

| kernel | attivazioni | n=1024 | n=2048 | n=4096 |
|---|---|---|---|---|
| `dot_row q8_0` (AVX-512) | float | 11 083 | 11 278 | 11 379 |
| `dot q8_0 x q8_0` VNNI (candidato) | int8 | 11 184 | 11 398 | 12 310 |
| `dot_row q8_0 x4` (già nostro) | float, 4 token | **28 644** | **29 969** | **30 565** |

Milioni di elementi al secondo per riga di attivazioni. Quantizzare le attivazioni costa 467 M
elem/s (scalare), che un matmul paga una volta ogni 1024 righe: trascurabile.

Letture (**corrette dalla revisione del 2026-09-18**, §Revisione avversariale e LEZIONI #65: la
tabella è giusta, la conclusione che ne era stata tirata no):
- Questa tabella confronta una riga contro **un** token, int8 contro float: lì l'int8 vale +1-8%.
  Ma il candidato ha mezza riga di lavoro scalare per blocco (half→float, catena della
  correzione), e il prefill non gira con quel kernel: gira col kernel a 4 token.
- Il nostro kernel a 4 token è 2.6-2.7× più veloce di entrambi perché riusa la riga di pesi. La
  domanda giusta era che cosa dà l'int8 **con quella stessa struttura**: misurato dopo, dà
  **1.8-2.3×** in più, mentre «blocchi più larghi» in float (8 token) non rende. Il divario con
  llama.cpp sta nelle attivazioni a 8 bit, non nei blocchi più larghi.
- A questi n il dot non è limitato dalla memoria (12 GB/s contro i ~41 misurati): è latenza e
  istruzioni per blocco. Una istruzione che fa 4 prodotti non cambia il totale solo finché le
  istruzioni per blocco restano quelle del candidato; tolte, lo cambia.

## SMT: 32 thread contro 16 (2026-09-18)

Stesso metodo, prompt 2048, 16 token, entrambi i casi **pinnati**: A = 16 thread, uno per core
fisico; B = 32 thread, uno per processore logico (i due fratelli di ogni core).

| | 16 thread (un core ciascuno) | 32 thread (fratelli SMT) |
|---|---|---|
| prefill tok/s | **163.6** | 110.2 |
| decode tok/s | **22.9** | 2.2 |

Il prefill perde un terzo, il decode crolla di **dieci volte**: due thread sullo stesso core si
tolgono la cache a vicenda e, con 464 dispatch per token, ognuno aspetta il fratello. Chiude la
domanda 3: le copie SMT non aiutano, e il default (thread = core fisici) è quello giusto.

## Speculazione dal prompt — quanto rende e quando perde (2026-09-17)

Container, volume `trochilus-models`, OLMoE-1B-7B Q8_0 intero, `run -f <prompt> -n 200`,
`--spec 0` e `--spec k` **alternati run per run** (`tools/ab_spec.sh`, LEZIONI #46), mediana di 3
giri dopo uno di riscaldamento. Il testo generato è identico in ogni giro: lo script si ferma se
non lo è. Nessun pin dei thread e misure in container: rifatte native e col pin in §Bozza adattiva,
dove il costo di una riga in più risulta 3-5 volte quello scritto qui sotto (LEZIONI #59).

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

## Bozza adattiva: quanto costa davvero una riga in più (2026-09-18)

Windows **nativo**, macchina ferma, pin al core (il default), 16 thread, `run -f <prompt> -n 200`,
`--spec 0` e `--spec 8` **alternati run per run** (`tools/ab_spec.sh`), mediana di 3 giri dopo uno
di riscaldamento, testo identico in ogni giro. `--spec-fixed` sceglie la bozza fissa.

| Prompt | Politica della bozza | `--spec 0` | `--spec 8` | | Accettate | Bozza media |
|---|---|---|---|---|---|---|
| `code-edit.txt` (il file è già nel prompt) | fissa 8 | 27.99 | **37.61** | **1.34×** | 64.4% | 8.00 |
| | adattiva, solo si accorcia | 26.96 | 32.81 | 1.22× | 70.9% | 2.86 |
| | adattiva con pausa (oggi) | 28.63 | 32.98 | **1.15×** | 69.4% | 2.34 |
| `code.txt` (codice nuovo) | fissa 8 | 29.65 | 17.67 | **0.60×** | 13.1% | 3.04 |
| | adattiva, solo si accorcia | 29.61 | 26.22 | 0.89× | 41.3% | 0.70 |
| | adattiva con pausa (oggi) | 27.49 | 27.76 | **1.01×** | 43.1% | 0.38 |

Il costo di una riga in più, ricavato dai passaggi stampati (ms per passata = 1000 · token / tok·s⁻¹
/ passaggi, meno i ~35 ms di una passata da un token, diviso la bozza media): **15 ms** con bozza
fissa 8, **20 ms** con bozza media 2.3, **27 ms** con bozza media 0.4.

Letture:
- **Su un MoE una riga in più non è quasi gratis, come sarebbe su un modello denso.** Il decode
  costa perché legge i pesi degli 8 esperti scelti in ogni strato; il token in bozza ne sceglie
  altri, e la passata legge anche quelli. Una riga in più costa così mezza passata o più, e il
  pareggio sta intorno al **60-75% di bozze accettate**, non al 15% che diceva la misura fatta in
  container (LEZIONI #59). È anche il motivo per cui la bozza lunga rende solo dove il testo è già
  nel prompt: lì gli esperti scelti sono gli stessi, perché i token sono gli stessi.
- **Accorciare la bozza non bastava**: con il 41% di accettate si perdeva ancora l'11%. Fermarla del
  tutto dopo una bozza tutta sbagliata (pausa 1, 3, 7, 15 passi, capped 16) porta il caso peggiore a
  **1.01×**, cioè «come senza», al prezzo di 1.22× → 1.15× nel caso buono. Chiude la domanda 25:
  `--spec` può diventare il default.
- La bozza fissa resta la più veloce dove il modello ricopia (1.34×): chi sa che sta riscrivendo un
  file può ancora chiedere `--spec 8 --spec-fixed`.

## Kernel: una riga di pesi contro 4 token (2026-09-17)

`make bench` (`tests/bench_kernels.c`, container, mediana di 5 × 100 ms, un thread), milioni di
elementi al secondo **per riga di input**:

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

## Leve di velocità: cosa dice la ricerca (2026-09-17)

Numeri delle fonti, su altre macchine: indicano la direzione, non promettono. Esatto = stessi
numeri; dichiarato = numeri diversi, differenza da misurare sul modello vero prima di adottarlo.

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

## Revisione avversariale (2026-09-18)

Rilettura di tutto il repository da parte di un altro modello (Fable 5.1), con l'ordine di
portare prove e non opinioni. Errori e controlli nuovi: LEZIONI #60-#68. Qui i numeri.
Container `trochilus-dev`, altri container accesi: per questo i confronti sono solo quelli con
effetti molto sopra lo spread (kernel su un core, run alternate) o **senza cronometro**
(contatori, replay). Niente di quello che segue è una misura di velocità del prodotto.

**Invarianti: reggono.** `trochilus logits` su 40-70 token, 3 tier (`TR_CPU_MAX` scalar, avx2,
avx512) × 3 numeri di thread (1, 5, 16) × 3 passate (`-b` 1, 7, 512): 27 file per modello,
identici al byte sulle righe confrontabili, per i tre modelli minuscoli (f32, f16, q8_0) e per
OLMoE vero tagliato a 2 layer (201 216 byte per riga). I test del modello passano anche sotto
`scalar` e `avx2`. Ora è un controllo: `make tier-check`. Tokenizer: 90 000 stringhe costruite
per rompere (tempeste di segni combinanti, jamo, ogni spazio Unicode, contrazioni, token aggiunti
incollati a segni e spazi) × 2 modi contro HF `tokenizers`: 0 differenze, e 12 000 decodifiche
uguali.

**Attivazioni int8, seconda misura** (domanda 21, LEZIONI #65). `make bench`, sezione «one row,
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

**Quanto legge una riga di bozza** (domanda 12, LEZIONI #67). Contatore `weight_bytes` del
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

**Le costanti della pausa, senza cronometro** (LEZIONI #60, #67). Che una bozza venga accettata
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
un controllo A/A (`tools/ab_modes.sh`, LEZIONI #66).

**Il pool con più di un pool** (LEZIONI #61-#63). Programma di prova, Linux: `H1` chiamante
prima `[0-31]`, dopo due pool creati e distrutti nell'ordine di nascita `[0,1]`; `H2` processo
ristretto a `[0,2]`, pool da 4: worker 2 e 3 su `[0]`; `H3` worker 1 di due pool vivi entrambi
su `[2,3]`; `H4` pool interno da 2, id del worker arrivato al body: 3. H1 e H2 corretti e sotto
test; H3 e H4 restano un debito dichiarato in `threads.h`.

## Tentativi

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
| 2026-09-17 | accumulatori del kernel a 4 token in un array `__m512 acc[4]` invece che in registri con nome | — | — | — | **scartato**: il compilatore li tiene sullo stack e il guadagno sparisce (LEZIONI #45) |
| 2026-09-17 | blocco di token di `tr_matmul` a 32, 64, 128 invece di 16 | prefill 180.5 tok/s (16 thread, tile 16) | 176.9 / 183.3 / 172.5 | 10-13% | scartato: tutto dentro il rumore, resta 16 |
| 2026-09-18 | una riga di pesi contro **8** token (float, identico al bit a x4), solo nel banco | x4: 25.6 / 47.6 / 114.1 ns per riga (n = 1024 / 2048 / 4096) | 1.13× / 0.79× / 0.93× | 3-8% | **scartato**: a n=2048 le attivazioni di 8 righe (64 KB) escono dalla L1 (§Revisione) |
| 2026-09-18 | dot int8 VNNI con la struttura del x4 (256 bit col trucco del segno; 512 bit a due blocchi), solo nel banco, non esatto | x4 float come sopra | 256 bit 1.64× / 1.62× / 1.96×; 512 bit **1.83× / 1.79× / 2.26×** | 3-8% | misurato, non adottato: l'int8 non è bit-identico, può essere solo un modo dichiarato. Decisione aperta (§Revisione, LEZIONI #65) |
| 2026-09-18 | tetto della pausa della bozza adattiva corretto (31 → 16) | replay esatto: 172 passate, 65 righe di bozza (`code`) | 172 passate, 66 righe | — | tenuto: è una correzione, non una leva; le costanti spostano ±2-3% (§Revisione) |
