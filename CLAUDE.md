<!-- preferenze: 39f397b3 -->
# Trochilus

Motore di inferenza per modelli MoE in C, senza dipendenze: gira su CPU di ogni tipo (dispatch
dei kernel a runtime), GPU come moduli caricabili, esperti letti dal disco, risultati esatti al
token contro transformers. Nasce prendendo il meglio di colibri (Apache-2.0) e ds4 (MIT).

C11 + intrinseci (e assembly dove misurato), Makefile, gcc/clang/MinGW-w64; strumenti in Python
(`tools/.venv`: torch CPU + transformers) solo per conversione e oracoli.

---

## Come lavorare

- A ogni avvio: `/caveman ultra`, poi `docs/STATUS.md`.
- Agenti: **mai Fable**. Sonnet su brief chiusi (un pezzo con la sua interfaccia e il suo test),
  Haiku per il meccanico; progettazione, interfacce e invarianti li fa l'orchestratore. Un
  agente alla volta, o in parallelo solo su file disgiunti. Pochi agenti: l'usage è il limite.
- **Fra agenti si parla inglese**, prompt e resoconti. Con Marcello in italiano.
- Brief **conciso e completo**: obiettivo, vincoli, file da toccare, forma della risposta,
  quando è finito. Il resoconto dell'agente: cosa ha fatto, file toccati, cosa resta aperto.
- Resoconto a Marcello **in due punti**: cosa è stato implementato, come si prova. Il perché
  va in `docs/STATUS.md`.
- Logica e kernel: si sceglie, si costruisce, si consegna. Le domande si fanno prima, non a metà.
- **Il commit lo chiede Marcello.** Si prepara tutto (test verdi, documenti) e ci si ferma.
- Repo privato `namespaceMarcello/trochilus`: push e visibilità li decide Marcello.

### Tieni tutto sotto controllo: il ciclo di ogni passo

Obiettivo: a ogni richiesta il tasso di successo sale. Si misura in `docs/LESSONS.md`
(colonna «Trovato da»): nessun errore deve arrivare a Marcello se un test poteva trovarlo prima.

1. **Prima**: `docs/STATUS.md`, poi `grep` in `docs/LESSONS.md` sull'area che si tocca.
2. **Errore o scoperta**, appena succede, anche piccola: una riga in `docs/LESSONS.md`.
3. **Ogni errore diventa un controllo**: un test che lo riproduce (rosso prima del fix, verde
   dopo), o un controllo in `make check`, un pin, un hook. Una frase in un documento è
   prevenzione debole: si scrive *regola* e la lezione resta aperta.
   **Un test si vede rosso almeno una volta, e dice quale ramo esercita**: rosso prima del fix,
   o con una mutazione (`tools/mutate_*.sh`) quando il codice nasce insieme al test; in testa al
   file sta scritto quale ramo prova, e un contatore (`TR_CHECK(n > 0)`) lo fa fallire se quel ramo
   non è stato preso. Risultati uguali non dicono quale codice ha girato (LEZIONI #43, #50, #54, #78).
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
| codice (ogni commit) | `docs/archive/DONE.md`: `### <data> — <titolo>`, cosa e come si prova |
| un errore, una scoperta, un debito | `docs/LESSONS.md`, con la prevenzione e chi l'ha trovato |
| codice portato da colibri o ds4 | `docs/ORIGINS.md` + intestazione del file |
| decisione, debito, misura, prossimo passo | `docs/STATUS.md` (sostituisci, non appendere; tetto 40 KB) |
| un passo chiuso, una decisione, una domanda nuova (con STATO) | `docs/status.json`, poi `tools/status_html.py` e ripubblica l'artifact (stesso URL) |
| strati, principi, tappe | `docs/ARCHITECTURE.md` (si riscrive la riga, non si aggiunge) |
| una misura o un tentativo di ottimizzazione (anche scartato) | `docs/MEASUREMENTS.md` |

---

## Leggi prima di rispondere

| Domanda su… | Apri |
|---|---|
| principi, strati, esecuzione, tappe | `docs/ARCHITECTURE.md` |
| a che punto siamo, decisioni, prossimo passo | `docs/STATUS.md` |
| da dove viene un file, commit di riferimento | `docs/ORIGINS.md` |
| numeri misurati, tentativi di ottimizzazione riusciti e scartati | `docs/MEASUREMENTS.md` |
| domande aperte da misurare, dove scendere nel dettaglio | `docs/MEASUREMENTS.md` §Da misurare |
| regole del codice che gira a ogni token | `docs/ARCHITECTURE.md` §Zona calda |
| errori già fatti, scoperte, come si prevengono | `docs/LESSONS.md` |
| bug trovati in colibri o ds4, segnalazioni agli autori | `docs/UPSTREAM.md` |
| profilazione: livelli, strumenti, scenari | `docs/ARCHITECTURE.md` §Profilazione |
| cosa è già stato fatto | `docs/archive/DONE.md` |
| il comando per fare una cosa (banchi, misure, mutazioni, report) | `docs/COMMANDS.md` |
| a che punto siamo, in figura (tappe, dipendenze, stato) | `docs/status.json` → `tools/status_html.py` → artifact `6mx3NS4KtQrBFPRLkAYupr`; si aggiorna insieme a `docs/STATUS.md` |
| come colibri o ds4 fanno una cosa | `ref/colibri`, `ref/ds4` (worktree in sola lettura, commit in ORIGINI) |

---

## Comandi

```bash
make check               # il cancello: lint, build 0 warning, test, ASan, TSan, oracoli
make                     # build/trochilus (nucleo, senza dipendenze)
make test                # test C: kernel SIMD = scalare, GGUF malformati, pool, profiler
make oracle              # modelli minuscoli: genera, converte, confronta con transformers
build/trochilus run -m <f.gguf> -f prompt.txt -n 200    # testo in entrata, generazione greedy
build/trochilus chat -m <file.gguf>                     # conversazione col template del modello
make profile             # scenari col profiler, mediana di N, token identici
tools/.venv/Scripts/python.exe tools/<script>.py        # su Linux/macOS: tools/.venv/bin/python
```

**Tutti gli altri comandi stanno in `docs/COMMANDS.md`**: oracoli per modello e piattaforma,
banchi (`bench-disk`, `bench-mem`, `bench-attn`, `bench-expf`), le misure native con le loro
guardie (`experts_budget.sh`, `prefill_overlap.sh`, `decode_context.sh`, `threads_phase.sh`,
`ab_modes.sh`), le mutazioni, i report Python, la mappa del progetto.

Prima di consegnare: `make check` verde. Su Windows la correttezza gira in Docker
(`trochilus-dev:local`): Smart App Control blocca i binari appena compilati (LEZIONI #12).

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
- **Misure native**: ogni script finisce ciò che ha lanciato (`tools/cleanup.lib`), niente parte con
  orfani accesi (`tools/orphans.sh`), la macchina si carica solo con `tools/busy_machine.sh`, ogni
  sessione dichiara il carico di fondo nel log e con un carico non basso le conclusioni non si tirano
  (`docs/ARCHITECTURE.md` §Profilazione, LEZIONI #84-#88).
- Un file derivato da colibri o ds4 dice nell'intestazione progetto, commit, percorso, modifica.
- Commenti e nomi nel codice in inglese; documenti in `docs/` in italiano.
- Prefisso `tr_` per i simboli pubblici; niente stato globale per modello (più modelli in un processo).
- Modelli, fixture e binari non entrano in git (`.gitignore`).

---

## Manutenzione

Quando nasce un documento in `docs/`, aggiungi la sua riga alla tabella. Quello che è successo
va in `docs/STATUS.md`, non qui.
