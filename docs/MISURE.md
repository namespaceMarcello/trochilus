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
| 2 | 8 thread vincono perché stanno in un solo gruppo di core (CCD, L3 da 32 MB)? | thread fissati ai core: 8 su un CCD contro 4+4 sui due | decide come fissare i thread su tutte le CPU a due gruppi |
| 3 | Le copie SMT (32 thread logici) aiutano o peggiorano? | scenario a 32 thread contro 16 | oggi il default esclude SMT senza una misura |
| 4 | Qual è la banda reale della RAM di questa macchina? | microbenchmark di lettura sequenziale e `memcpy` da 1 GB, 1-16 thread | il tetto del decode (1.2 GB per token) si calcola da qui, non dai ~83 GB/s teorici (2×16 GB DDR5-5200); dal 2026-09-17 4 thread vanno come 16 a 41 GB/s |
| 5 | Quanti TLB miss per token, e quanto valgono le pagine da 2 MB? | Linux: pesi con `madvise(MADV_HUGEPAGE)` contro senza, stessi token/s mediani | 300 000 pagine da 4 KB toccate per token |
| 6 | Il portatile rallenta quando si scalda? | 60 s di decode continuo, token/s ogni 5 s | una misura breve può non valere per un uso reale |
| 7 | Quanto costa l'attenzione quando il contesto cresce (128, 1024, 4096 token)? | scenari con prompt lungo, zona `attention` | il coding usa contesti lunghi |
| 8 | Quanto costa il profiler acceso sul modello vero? | stessi scenari con e senza `--profile` | per fidarsi delle percentuali delle zone |
| 9 | Windows nativo e Linux sulla stessa macchina vanno alla stessa velocità? | stesso scenario nativo e in Docker/WSL | in Docker svegliare un thread costava ~20 µs |
| 10 | Più corsie (32 o 64) sbloccano la catena delle somme su un thread? | kernel sperimentale, `make bench` | AVX-512 va come AVX2 per quel limite; cambierebbe i numeri (dichiarato) |
| 11 | Quanto pesa la divisione per riga e la chiamata indiretta in `matmul_body`? | kernel che riceve tutta la matrice contro uno per riga | 1024 chiamate e 2 divisioni a 64 bit per matrice |
| 12 | Gli esperti scelti in token consecutivi si ripetono? | contare gli esperti in comune fra token vicini | se sì, i loro pesi sono ancora in cache e si può prevedere cosa caricare |

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

## Leve di velocità: cosa dice la ricerca (2026-09-17)

Numeri delle fonti, su altre macchine: indicano la direzione, non promettono. Esatto = stessi
numeri; dichiarato = numeri diversi, differenza da misurare sul modello vero prima di adottarlo.

| # | Leva | Tipo | Guadagno riportato | Per noi |
|---|---|---|---|---|
| 1 | SIMD sul prodotto riga × vettore | esatto | — | fatto: 10× sul kernel |
| 2 | Attivazioni in int8 + VNNI (`VPDPBUSD`), come llama.cpp per Q8_0 | dichiarato | llama.cpp usa questa strada per Q8_0 | alto: meno byte e meno lavoro per elemento |
| 3 | Prefill a blocchi (più token nella stessa moltiplicazione) | esatto | prefill lineare nel lotto ([discussione #18030](https://github.com/ggml-org/llama.cpp/discussions/18030)) | alto sul prefill, zero sul decode |
| 4 | Token raggruppati per esperto nel prefill | esatto | nessun numero isolato | alto sul prefill |
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

## Tentativi

| Data | Cosa | Prima | Dopo | Spread | Esito |
|---|---|---|---|---|---|
| 2026-09-17 | `dot_row q8_0` AVX-512, bit-identico | 1 957 M el/s | 19 837 M el/s | 15% / 22% | tenuto: attivo di default sulle CPU AVX-512 |
| 2026-09-17 | `dot_row q8_0` AVX2, bit-identico | 1 957 M el/s | 19 532 M el/s | 15% / 5% | tenuto: attivo sulle CPU AVX2 senza AVX-512 |
| 2026-09-17 | OLMoE-1B-7B Q8_0 decode, kernel AVX-512 | 6.87 tok/s (16 thread) | 21.08 tok/s (8 thread) | una run | tenuto; 16 thread 17.72: il default dei thread va rivisto |
| 2026-09-17 | pool di thread: attesa attiva 2 ms, uno slot per thread (Linux/Docker) | matmul esperto 16 thread 0.376 ms | 0.014 ms | 9% / 16% | tenuto; da misurare su Windows nativo e sul modello vero (mediana di 5) |
| 2026-09-17 | OLMoE-1B-7B Q8_0 decode, pool nuovo (Windows nativo, mediana di 5) | 21.08 tok/s (8 thread, una run) | 27.02 tok/s (8 thread) | 2.4% | tenuto; 16 thread = 8 thread (26.90) |
| 2026-09-17 | rope da tabella, attivazioni in parallelo, attenzione per testa + `axpy_f32` AVX-512 (logit identici al bit) | 26.34 tok/s (16 thread) | 32.78 tok/s (16 thread) | 1.4% / 2.8% | tenuto; 4 thread = 16 thread (32.93): limite della memoria |
