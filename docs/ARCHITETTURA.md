# Architettura

Trochilus fa girare modelli MoE grandi su **qualsiasi** macchina e ne usa tutto: il bersaglio è
il computer di una persona qualunque (16-32 GB di RAM, una GPU da 4-8 GB o nessuna, un SSD), non
la workstation da 128 GB. VRAM, RAM e disco lavorano insieme; gli esperti che non stanno in
memoria si leggono dal disco. Il risultato è esatto al token rispetto al riferimento. Il build
predefinito non ha dipendenze.

Macchina di riferimento per le misure: portatile Ryzen 9 7940HX, 31 GB RAM, RTX 4070 Laptop 8 GB,
NVMe 1 TB. Limite basso da tenere vivo: solo CPU, 16 GB, niente GPU.

Prende da colibri (CPU vera, disco, test esatti) e da ds4 (GPU, formato, cache KV su disco).
Da dove viene ogni file portato: `docs/ORIGINI.md`.

## Principi

0. **Si adatta da solo.** All'avvio misura la macchina (core fisici, istruzioni della CPU, RAM
   libera, VRAM, velocità del disco) e decide dove mettere cosa: pesi densi, esperti caldi, KV,
   thread. Nessuna opzione obbligatoria; le opzioni servono solo a forzare una scelta.
1. **La CPU è un motore, non un riferimento.** Ogni modello gira prima su CPU; la GPU accelera.
2. **Un binario per tutte le CPU.** Kernel scelti a runtime (cpuid / hwcap), non con `-march`:
   lo stesso eseguibile usa AVX-512 VNNI dove c'è e AVX2 o scalare dove no. Né colibri né ds4 lo fanno.
3. **Lo scalare definisce i numeri.** Ogni variante SIMD o assembly somma nello stesso ordine
   ed è **bit-identica** allo scalare, verificata da test. Una variante non identica non entra.
4. **Zero dipendenze nel nucleo**: libc, thread del sistema operativo. Niente OpenMP (su macOS e
   MinGW è una libreria in più), niente librerie di terzi. I backend GPU sono moduli caricati a
   runtime: il nucleo parte anche dove CUDA o Metal non ci sono.
5. **Il formato è GGUF v3** con i numeri di tipo standard di ggml e i nomi dei tensori di
   llama.cpp: un file scaricato da Hugging Face si apre senza conversione.
6. **Operazioni generiche, grafi per modello.** Il backend espone primitive senza nome di modello
   (matmul quantizzata, rmsnorm, rope, attenzione, top-k, swiglu). Un'operazione fusa per un
   modello è ammessa solo come accelerazione, con la composizione generica come fallback.

## C e assembly

Il motore si scrive in C, e i punti dove il tempo si consuma davvero si portano in assembly.
In un motore di inferenza quasi tutto il tempo sta in pochi kernel (prodotti scalari delle
matrici quantizzate, attenzione, dequantizzazione); caricamento, tokenizer e orchestrazione non
guadagnano niente dall'assembly e perderebbero la portabilità.

Il giro per ogni kernel:
1. versione C scalare: definisce il risultato bit per bit;
2. versione con intrinseci (AVX2, AVX-512, NEON): stessa aritmetica, test di identità;
3. **versione assembly** per la CPU di riferimento (Zen 4, AVX-512 VNNI), scritta in GAS `.S` con
   macro per le due convenzioni di chiamata (System V e Windows x64);
4. microbenchmark delle tre: la più veloce entra nella tabella di dispatch per quella CPU.

Il profiler decide l'ordine: si porta in assembly prima il kernel che pesa di più nella misura.

**Ogni ottimizzazione si prende, anche piccola**, a qualsiasi livello (algoritmo, memoria,
istruzioni, assembly), a due condizioni verificate da strumenti, non a occhio:
- **non fa danni**: i test la trovano bit-identica alla versione di prima;
- **guadagna davvero**: il benchmark (mediana di almeno 5 run, macchina a regime) la trova più
  veloce di più del rumore misurato su quel benchmark. Sotto il rumore non è un guadagno.

Ogni tentativo, riuscito o scartato, va in `docs/MISURE.md` con il numero: uno scartato oggi
non si riprova domani senza un motivo nuovo.

**Sotto l'assembly.** Il codice macchina scritto a mano non aggiunge niente all'assembly, che è già
1:1. Quello che resta a quel livello è il **JIT**: generare codice macchina al caricamento del
modello, con le dimensioni vere (lunghezza delle righe, blocchi, teste) scritte come costanti e le
istruzioni scelte per la CPU esatta. Esperimento dopo l'assembly, con pagine mai scrivibili ed
eseguibili insieme. Microcodice e firmware non si toccano; overclock e undervolt esclusi. Del
sistema si usa solo ciò che è reversibile: priorità del processo, pagine grandi, piano energetico.

## Profilazione

I dati decidono dove si lavora. Tre livelli, dal più fine al più largo:

| Livello | Strumento | Risponde a |
|---|---|---|
| kernel | `tests/bench_kernels.c` (`make bench`) | quanto è veloce un singolo kernel per tier, con il rumore |
| motore | profiler interno, `trochilus generate --profile` / `--profile-json <file>` | dove va il tempo di un token: per fase (embedding, proiezioni, norme, rope, attenzione, router, esperti, lm_head, campionamento), prefill e decode separati, attese del pool di thread, letture dal disco; byte di pesi toccati per token e banda di memoria effettiva, cioè se si è limitati dalla memoria o dal calcolo |
| scenari | `tools/profile_suite.py` (`make profile`), scenari in `bench/scenarios.json` | come va un uso reale (prompt corto → risposta lunga; file lungo → risposta corta; contesto che cresce), mediana di N run con spread, confronto automatico con la misura precedente sulla stessa macchina: migliorato, peggiorato o rumore |

Il profiler non è globale (vive nella sessione), costa un solo salto condizionato quando è spento,
e usa il contatore della CPU (RDTSC con TSC invariante, altrimenti il clock del sistema). I
risultati della suite vanno in `bench/results/<data>-<commit>-<macchina>.json`; il riassunto e
le decisioni che ne escono in `docs/MISURE.md`.

## Zona calda

Il codice che gira a ogni token sta fra `/* hot: begin */` e `/* hot: end */` (modello, kernel,
pool di thread, profiler). Il resto (caricamento, riga di comando, lettura dei file) è la zona
fredda: gira una volta, e lì valgono chiarezza e difesa dai file malformati.

Regole della zona calda, ognuna con il suo controllo:

| Regola | Controllo |
|---|---|
| nessuna allocazione né rilascio di memoria | `tools/lint.py` sul sorgente; `tests/test_hot.c` conta le chiamate all'allocatore mentre il modello genera (devono essere zero) |
| niente stringhe, stampe, file, variabili d'ambiente | `tools/lint.py` |
| niente `pow`, `sin`, `cos`, `log` per elemento: si calcolano una volta in tabella | `tools/lint.py` |
| stessi logit con ogni numero di thread e con il profiler acceso | `tests/test_hot.c` (pool da 1, 2, 3, 8 thread) |
| ogni ottimizzazione misurata prima e dopo, tenuta solo sopra il rumore | riga in `docs/MISURE.md` §Tentativi |

Un'eccezione si scrive sulla riga stessa, con il motivo: `/* hot-ok: pow -- motivo */`. Il lint
la rifiuta quando la riga non usa più quel nome. I commenti non costano niente a runtime (stesso
codice macchina con e senza, verificato): nella zona calda restano quelli che spiegano un vincolo.

## Sicurezza della macchina

- Prima di caricare si stima la memoria necessaria; se resterebbe meno di 2 GB o del 10% di RAM
  libera, il motore rifiuta e dice quanto manca. Mai swap.
- Thread al massimo quanti i core fisici; i benchmark durano al più 60 s per run.
- Nessun modello più grande della RAM finché lo streaming dal disco (M1) non è testato.

## Strati

| Cartella | Cosa fa | Da dove viene l'idea |
|---|---|---|
| `src/base/` | piattaforma (file, `pread`, O_DIRECT, tempo, memoria allineata), pool di thread, rilevamento CPU | colibri `compat.h`, `omp_tune.h` (core fisici); ds4 pool `ds4_parallel_for` |
| `src/format/` | lettore GGUF v3, tabella dei tipi, metadati | ds4 `parse_metadata` / `parse_tensors`, senza gli agganci per architettura |
| `src/kernels/` | kernel CPU per tipo quantizzato: scalare + AVX2 + AVX-512 (+VNNI) + NEON, tabella di dispatch | colibri `quant.h`, `expert_ffn.h`; ds4 riferimenti K-quant |
| `src/backend/` | interfaccia backend (tensori residenti sul dispositivo, grafo per token) e backend CPU | ds4 `ds4_gpu.h` (modello di esecuzione), ridotto alle primitive generiche |
| `src/memory/` | archivio esperti a livelli (VRAM / RAM / disco), lease, LRU O(1), pool I/O, pin appresi dall'uso | colibri `expert_store.h`, `st.h`, `route_trace.h`; ds4 streaming su VRAM |
| `src/kv/` | cache KV, riuso del prefisso, checkpoint su disco con punteggio a decadimento | colibri `kv_prefix.h`, `kv_fp8.h`; ds4 `ds4_kvstore.c` |
| `src/tokenizer/` | BPE byte-level dai metadati GGUF (famiglie di pretokenizer ammesse solo con oracolo), NFC e classi Unicode sondate da HF `tokenizers`, template di chat per architettura | idee: colibri `tok.h` (regex rigiocata in C), ds4 `vocab_load` (dal GGUF); codice nuovo |
| `src/models/` | un grafo per famiglia, costruito dalle primitive | colibri `olmoe.c`, ds4 / colibri DeepSeek V4 |
| `src/gen/` | come si sceglie il token dopo il modello: greedy, bozza dal prompt e verifica in una passata sola (poi campionamento e criteri di arresto) | idee: colibri `v4_ngram_draft`, llama.cpp `examples/lookup`; codice nuovo |
| `src/app/` | CLI, poi server (API OpenAI e Anthropic) | entrambi |
| `backends/cuda/`, `backends/metal/` | moduli GPU caricabili | ds4 `cuda/mmq` (ggml, MIT), `metal/*.metal` |
| `tools/` | convertitore HF → GGUF, generatori dei modelli minuscoli (Python, fuori dal motore) | colibri `tools/make_*_tiny.py` |
| `tests/` | test dei kernel (SIMD = scalare), oracoli minuscoli, microbenchmark | colibri |

## Esecuzione

- **Pesi**: letti con `pread` in buffer propri, non `mmap`: la memoria residente resta sotto
  controllo (colibri `st.h`, bug RSS di mmap). Denso: caricato all'avvio. Esperti: su richiesta.
- **Esperti**: `lease(layer, expert)` restituisce un puntatore valido finché non si rilascia.
  Mancante → lettura dal disco. La lettura degli esperti mancanti avviene **mentre** si calcola
  quello che è già in memoria (disciplina di ds4, `PIPE` di colibri).
- **Passate**: una chiamata di valutazione corre in passate da al più `n_batch` token (512, `-b`); il
  decode è una passata da un token, lo stesso codice. In una passata ogni numero è la stessa chiamata di
  kernel che con un token solo (un elemento di matmul = un `dot_row`; norme, RoPE, router e somma degli
  esperti per token; attenzione di un token sulle posizioni fino alla sua), quindi logit e cache sono
  identici al bit per ogni `n_batch`. Gli esperti lavorano sulle coppie (token, esperto) ordinate per
  esperto; le matrici si visitano a blocchi di token, e una riga di pesi va contro 4 token alla volta
  nei registri (`dot_row_x4`: un carico e una conversione per quattro prodotti, ognuno identico al suo
  `dot_row`). Logit dell'ultimo token, o delle ultime `n` posizioni quando servono a verificare
  una bozza (`tr_session_eval_rows`, al più `TR_LOGIT_ROWS_MAX`).
- **Speculazione dal prompt** (`src/gen/`): la bozza è la continuazione dell'ultima occorrenza
  dell'n-gramma di coda nel contesto; una passata sola verifica 1 + k posizioni e si tengono solo i
  token che il modello avrebbe scelto comunque, gli altri spariscono con `tr_session_rewind`. Poiché
  ogni riga di una passata è identica al bit alla passata da un token, i token generati sono gli
  stessi con e senza speculazione: è velocità, mai un risultato diverso.
- **Thread**: pool persistente dimensionato sui **core fisici** (colibri: +2.3x su Zen 3 contro i
  core logici), `parallel_for` su intervalli di righe. Contarli non basta: se non si fissano ai core,
  Windows ne appoggia due sullo stesso core fisico e il prefill perde il 30% (`docs/MISURE.md`
  §Dove vanno i thread).
- **GPU**: tutto il token in un solo lotto di comandi, tensori che restano sul dispositivo (ds4).
- **KV**: in memoria per sessione; riuso del prefisso per id di token; checkpoint su disco con
  punteggio `(hit decaduti + 1) × token / byte` (ds4).

## Correttezza

| Livello | Cosa confronta | Dove |
|---|---|---|
| kernel | ogni variante SIMD/asm contro lo scalare, bit per bit, su input casuali | `tests/test_kernels.c` |
| modello minuscolo | token greedy **identici** a transformers; logit entro tolleranza per posizione | `tools/make_tiny_<famiglia>.py` → `tests/oracle_<famiglia>.py` |
| modello vero | token greedy contro transformers su un prompt fisso | a mano, fuori dalla CI |
| ottimizzazione esatta | logit del modello vero prima e dopo, identici al bit, con più numeri di thread | `trochilus logits` + `cmp`, a mano |
| prefill a blocchi | logit e cache con molti token per passata identici al bit a un token per passata: `n_batch`, divisione in chiamate, thread, f32/Q8_0, rewind | `tests/test_prefill.c`; `tools/oracle.py` (`logits -b 3/64/tutto` al byte) su tiny e OLMoE a 2 layer |

## Scala dei modelli

Si parte piccoli e si sale solo quando il gradino sotto è **esatto e misurato**. A ogni gradino
si misurano token/s (prefill e decode), RAM, primo token, e lo stesso modello su **llama.cpp** e
**colibri** sulla stessa macchina: è l'unico modo di sapere quanto vale davvero Trochilus.

| Gradino | Modello | Dimensione | Cosa mette alla prova |
|---|---|---|---|
| 0 | OLMoE minuscolo, pesi casuali | 1 MB | correttezza contro transformers |
| 1 | OLMoE-1B-7B | ~7 GB Q8_0, ~4 GB Q4 | kernel CPU, thread, tokenizer: tutto in RAM |
| 2 | Qwen3-Coder-30B-A3B | ~17 GB Q4 | un MoE per il codice che sta ancora in 31 GB; prompt lunghi, file riletti |
| 3 | un MoE più grande della RAM | > 31 GB | esperti dal disco, piano VRAM + RAM + SSD |
| 4 | DeepSeek V4 Flash | centinaia di GB | il bersaglio di colibri e ds4, su una macchina normale |

## Orizzonte (dopo la M2)

Il limite di velocità in generazione è il movimento dei dati: token/s ≈ banda della memoria ÷
byte letti per token. Le direzioni che attaccano quella divisione, da provare una alla volta con
la regola «non fa danni + guadagna davvero»:

| Direzione | Cosa fa | Attacca |
|---|---|---|
| decodifica speculativa | un modello piccolo (o il testo già nel prompt) propone più token, il grande li verifica in un passaggio | più token per ogni lettura dei pesi |
| modelli a diffusione | modelli addestrati a comporre tutta la risposta insieme e raffinarla in pochi passaggi (per il codice: DiffuCoder, Dream-Coder) | un passaggio per molti token; è un'altra famiglia di modelli, va supportata a parte |
| previsione degli esperti | carica gli esperti che serviranno prima che il router li scelga | letture dal disco in attesa |
| disposizione per frequenza | esperti caldi in VRAM, tiepidi in RAM, freddi su SSD, in ordine di lettura | distanza dei dati |
| sparsità delle attivazioni | salta i neuroni che resteranno quasi a zero, previsti in anticipo | byte letti per token |
| pesi a pochi bit (2 bit, ternari) | meno byte per peso; con -1/0/+1 le moltiplicazioni diventano somme | byte letti per token |
| JIT | codice macchina generato al caricamento con le dimensioni del modello come costanti | lavoro della CPU |
| superottimizzazione | ricerca automatica della sequenza di istruzioni più veloce per i kernel minuscoli | lavoro della CPU |

## Tappe

| Tappa | Contenuto | Fatta quando |
|---|---|---|
| **M0** | base, GGUF, convertitore (F32/F16/Q8_0), backend CPU scalare + AVX2 + AVX-512 con dispatch, grafo OLMoE, greedy, CLI | oracolo minuscolo esatto su Windows e Linux; OLMoE-1B-7B vero risponde |
| M1 | esperti dal disco con budget di RAM, pool I/O, LRU O(1), pin dall'uso; **piano automatico** (misura RAM e disco, sceglie budget e thread) | budget piccolo forzato → stessi token; nessuna opzione necessaria |
| M2 | K-quant (Q4_K, Q6_K, Q2_K, IQ2_XXS) su CPU, laboratorio assembly | kernel bit-identici, microbenchmark |
| M3 | modulo CUDA (mmq di ggml via ds4), esperti caldi in VRAM, piano VRAM + RAM + disco | stessi token della CPU; un modello più grande della RAM gira sul PC di riferimento |
| M4 | DeepSeek V4 Flash | oracolo minuscolo esatto; gira sul PC di riferimento |
| M5 | checkpoint KV su disco, server, decodifica speculativa | — |
| M6 | moduli Vulkan (GPU AMD/Intel, integrate; colibri ha `backend_vulkan.c`) e Metal | stessi token della CPU |
