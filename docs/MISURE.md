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
| 13 | Il router del layer successivo, applicato allo stato del layer corrente, indovina gli esperti che verranno scelti? | traccia del routing sul modello vero (prompt di codice, ~1000 token): esperti previsti (primi 8, 12, 16) contro scelti | se prevedendone al massimo 12 se ne indovina almeno l'80%, gli esperti si precaricano dal disco prima che servano (M1) |
| 14 | Quanti byte per token restano da leggere dal disco con una cache degli esperti grande il 25, 50, 75% del modello? | simulazione LRU sulla stessa traccia | decide il budget di RAM di default della M1 |
| 15 | Streaming dal disco per layer interi (DeepSpeed) o per esperti: quanti byte per token? | stessa traccia: layer interi contro soli esperti scelti e non in cache | con un solo utente leggere tutto il modello a ogni token dà al massimo ~0.3 tok/s su 10 GB |
| 16 | Quanto legge davvero l'NVMe in blocchi grandi quanto un esperto, a posizioni casuali, senza cache? | microbenchmark `pread` senza cache del sistema (`FILE_FLAG_NO_BUFFERING`, `O_DIRECT`), al massimo 60 s | con 14 e 15 dà i token/s attesi (M1 per esperti se ≥ 5 tok/s al 50%); decide anche la KV vecchia su SSD, attesa no con attenzione piena |
| 17 | Ricaricare dal disco la KV di un file già letto costa meno del prefill? La KV in q8 cambia i token? | checkpoint: logit identici al bit dopo il ricaricamento, tempo almeno 10 volte sotto il prefill; q8: token greedy uguali ≥ 99% su 1000 token di codice | leve del coding (file già letti, contesti lunghi), M5 |
| 18 | Perché il decode perde l'11-13% da 32 a 512 token di contesto, e llama.cpp solo il 5-7%? | profilo per zone a contesto 32, 512, 2048: quota dell'attenzione e della sua parte a thread singolo | nel coding il contesto è lungo: è la differenza che cresce di più dopo il prefill |
| 19 | Decode a 4-16 thread: llama.cpp è davvero avanti dell'8-12%? | `tools/speed_compare.py` con i due motori alternati run per run nella stessa sessione (fra due sessioni la mediana di llama.cpp si è spostata del 3-7%) | dice se c'è una leva nel decode a più thread o solo rumore |
| 20 | Il prefill rende 2.0× da 4 a 16 thread e 1.16× da 8 a 16, e a 16 thread lo spread sale al 15-25%. Il profilo dice che non è il codice (moltiplicazioni 90%, attenzione 2%, seriale 8%) e la stessa matrice in cache rende 6.5× da 1 a 16 thread: è il limite di potenza del portatile? | frequenza effettiva durante una run a 16 thread contro una a 1 (contatori del sistema, o tempo di un lavoro fisso su un thread mentre gli altri 15 sono carichi); poi thread su un solo CCD (domanda 2) | dice se conviene ancora lavorare sul parallelo o solo sul lavoro per elemento |
| 21 | Quanto del divario col prefill di llama.cpp (1.57× a 1 thread) è il dot int8 VNNI contro il nostro float? | microbenchmark `dot_row q8_0 x4` (44 G elementi/s) contro un dot int8 × int8 VNNI sulla stessa riga | dice quanto resta alla strada esatta e quanto chiede la leva 2 |

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
