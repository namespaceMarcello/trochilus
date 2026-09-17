<!-- preferenze: 39f397b3 -->
# Trochilus

Motore di inferenza per modelli MoE in C, senza dipendenze: gira su CPU di ogni tipo (dispatch
dei kernel a runtime), GPU come moduli caricabili, esperti letti dal disco, risultati esatti al
token contro transformers. Nasce prendendo il meglio di colibri (Apache-2.0) e ds4 (MIT).

C11 + intrinseci (e assembly dove misurato), Makefile, gcc/clang/MinGW-w64; strumenti in Python
(`tools/.venv`: torch CPU + transformers) solo per conversione e oracoli.

---

## Come lavorare

- A ogni avvio: `/caveman ultra`, poi `docs/STATO.md`.
- Agenti: **mai Fable**. Sonnet su brief chiusi (un pezzo con la sua interfaccia e il suo test),
  Haiku per il meccanico; progettazione, interfacce e invarianti li fa l'orchestratore. Un
  agente alla volta, o in parallelo solo su file disgiunti. Pochi agenti: l'usage è il limite.
- **Fra agenti si parla inglese**, prompt e resoconti. Con Marcello in italiano.
- Brief **conciso e completo**: obiettivo, vincoli, file da toccare, forma della risposta,
  quando è finito. Il resoconto dell'agente: cosa ha fatto, file toccati, cosa resta aperto.
- Resoconto a Marcello **in due punti**: cosa è stato implementato, come si prova. Il perché
  va in `docs/STATO.md`.
- Logica e kernel: si sceglie, si costruisce, si consegna. Le domande si fanno prima, non a metà.
- **Il commit lo chiede Marcello.** Si prepara tutto (test verdi, documenti) e ci si ferma.
- Repo privato `namespaceMarcello/trochilus`: push e visibilità li decide Marcello.

### Tieni tutto sotto controllo: il ciclo di ogni passo

Obiettivo: a ogni richiesta il tasso di successo sale. Si misura in `docs/LEZIONI.md`
(colonna «Trovato da»): nessun errore deve arrivare a Marcello se un test poteva trovarlo prima.

1. **Prima**: `docs/STATO.md`, poi `grep` in `docs/LEZIONI.md` sull'area che si tocca.
2. **Errore o scoperta**, appena succede, anche piccola: una riga in `docs/LEZIONI.md`.
3. **Ogni errore diventa un controllo**: un test che lo riproduce (rosso prima del fix, verde
   dopo), o un controllo in `make check`, un pin, un hook. Una frase in un documento è
   prevenzione debole: si scrive *regola* e la lezione resta aperta.
4. **Ogni scenario nuovo entra nei test**: modello, famiglia, forma di tensore, piattaforma, uso
   (prompt lungo, contesto pieno, file malformato) diventa un caso nella suite giusta: test C
   in `tests/`, oracolo in `make oracle`, scenario di prestazioni in `bench/scenarios.json`.
5. **Prima di dire "fatto"**: `make check` verde (build 0 warning, test, oracoli, sanitizer
   dove disponibili). Un agente non chiude un passo: lo chiude la verifica dell'orchestratore.
6. **Dopo un agente**: rileggere il resoconto contro il codice, `make check`, `docker ps` e
   processi in background (LEZIONI #5).
7. **Upstream**: portando o leggendo codice di colibri e ds4, ogni bug trovato va in
   `docs/UPSTREAM.md` con la prova; PR o issue solo dopo il sì di Marcello.
8. **Documenti**, nella tabella qui sotto.

### Prima di ogni commit: documentare
| Se è cambiato… | Scrivi in |
|---|---|
| codice (ogni commit) | `docs/archivio/FATTO.md`: `### <data> — <titolo>`, cosa e come si prova |
| un errore, una scoperta, un debito | `docs/LEZIONI.md`, con la prevenzione e chi l'ha trovato |
| codice portato da colibri o ds4 | `docs/ORIGINI.md` + intestazione del file |
| decisione, debito, misura, prossimo passo | `docs/STATO.md` (sostituisci, non appendere; tetto 40 KB) |
| strati, principi, tappe | `docs/ARCHITETTURA.md` (si riscrive la riga, non si aggiunge) |
| una misura o un tentativo di ottimizzazione (anche scartato) | `docs/MISURE.md` |

---

## Leggi prima di rispondere

| Domanda su… | Apri |
|---|---|
| principi, strati, esecuzione, tappe | `docs/ARCHITETTURA.md` |
| a che punto siamo, decisioni, prossimo passo | `docs/STATO.md` |
| da dove viene un file, commit di riferimento | `docs/ORIGINI.md` |
| numeri misurati, tentativi di ottimizzazione riusciti e scartati | `docs/MISURE.md` |
| domande aperte da misurare, dove scendere nel dettaglio | `docs/MISURE.md` §Da misurare |
| regole del codice che gira a ogni token | `docs/ARCHITETTURA.md` §Zona calda |
| errori già fatti, scoperte, come si prevengono | `docs/LEZIONI.md` |
| bug trovati in colibri o ds4, segnalazioni agli autori | `docs/UPSTREAM.md` |
| profilazione: livelli, strumenti, scenari | `docs/ARCHITETTURA.md` §Profilazione |
| cosa è già stato fatto | `docs/archivio/FATTO.md` |
| come colibri o ds4 fanno una cosa | `ref/colibri`, `ref/ds4` (worktree in sola lettura, commit in ORIGINI) |

---

## Comandi

```bash
make check               # il cancello: lint, build 0 warning, test, ASan, TSan, test ripetuti, oracoli
make                     # build/trochilus (nucleo, senza dipendenze)
make test                # test C: kernel SIMD = scalare, GGUF malformati, pool, profiler
make oracle              # modelli minuscoli: genera, converte, confronta con transformers
make oracle-tokenizer    # tokenizer di OLMoE contro transformers: id, pezzi, NFC, decodifica
make oracle-real         # OLMoE vero tagliato a 2 layer contro transformers (saltato senza modello)
build/trochilus run -m <file.gguf> -f prompt.txt -n 200    # testo in entrata, generazione greedy
build/trochilus chat -m <file.gguf>                       # conversazione con il template del modello
make chat-check          # chat sul modello vero: seconda risposta = run da zero (saltato senza modello)
make spec-check          # `--spec 0/1/4/8/15` danno lo stesso testo sul modello vero a 2 layer
build/trochilus run -m <f.gguf> -f prompt.txt -n 200 --spec 8   # speculazione dal prompt (stessi token)
sh tools/ab_spec.sh <gguf> <binario> bench/prompts/code.txt 8   # quanto rende --spec, run alternate
sh tools/build_llamacpp.sh; tools/compare_llamacpp.py ...   # nel container: llama.cpp e confronto dei logit
tools/speed_compare.py ...   # nel container: velocità contro llama.cpp e colibri (modelli nel volume trochilus-models)
sh tools/ab_speed.sh <gguf> <binario A> <binario B>   # due binari alternati run per run (LEZIONI #46)
make bench               # microbenchmark dei kernel (mediana + rumore)
make profile             # scenari col profiler, mediana di N, token identici, confronto col precedente
make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json   # modello vero, 16/8/4/1 thread
tools/.venv/Scripts/python.exe tools/<script>.py    # su Linux/macOS: tools/.venv/bin/python
```

Prima di consegnare: `make check` verde. `make profile` arriva col profiler collegato al motore.
Su Windows la correttezza gira in Docker (`trochilus-dev:local`, `tools/docker/Dockerfile`):
Smart App Control blocca i binari appena compilati (LEZIONI #12).

---

## Invarianti

- **Sicurezza della macchina**: prima di caricare, il motore stima la memoria e rifiuta se
  lascerebbe meno di 2 GB o del 10% di RAM libera. Nessun modello più grande della RAM prima
  dello streaming (M1). Benchmark di al massimo 60 s per run, thread <= core fisici. Un download
  grande si annuncia (dimensione, spazio libero verificato) prima di partire.
- **Linguaggi**: C è la definizione (portabile, leggibile, riferimento bit a bit). I kernel caldi
  si riscrivono in assembly uno alla volta; la versione assembly resta solo se il test la trova
  identica al C e il benchmark la trova più veloce, e il C resta come fallback per le altre CPU.
- Il nucleo non dipende da niente oltre libc e thread del sistema operativo.
- Ogni variante di kernel (SIMD, assembly) è bit-identica allo scalare; il test lo verifica.
- Zona calda (codice che gira a ogni token): niente allocazioni, stringhe, I/O; `tools/lint.py` e
  `tests/test_hot.c` lo verificano. Ogni misura è una mediana di N run, mai una run sola.
- Un file derivato da colibri o ds4 dice nell'intestazione progetto, commit, percorso, modifica.
- Commenti e nomi nel codice in inglese; documenti in `docs/` in italiano.
- Prefisso `tr_` per i simboli pubblici; niente stato globale per modello (più modelli in un processo).
- Modelli, fixture e binari non entrano in git (`.gitignore`).

---

## Manutenzione

Quando nasce un documento in `docs/`, aggiungi la sua riga alla tabella. Quello che è successo
va in `docs/STATO.md`, non qui.
