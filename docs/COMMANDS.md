# Comandi

Da `CLAUDE.md` (2026-09-21): l'elenco intero dei comandi del progetto. Nella mappa restano
quelli di ogni giorno; qui c'e' tutto, incluse le misure e gli strumenti che si usano di rado.

```bash
make check               # il cancello: lint, build 0 warning, test, ASan, TSan, test ripetuti, oracoli
make                     # build/trochilus (nucleo, senza dipendenze)
make test                # test C: kernel SIMD = scalare, GGUF malformati, pool, profiler
make oracle              # modelli minuscoli: genera, converte, confronta con transformers
make tier-check          # il motore sotto ogni tier (TR_CPU_MAX): test del modello e logit identici al byte
make oracle-tokenizer    # tokenizer di OLMoE contro transformers: id, pezzi, NFC, decodifica
make oracle-real         # OLMoE vero tagliato a 2 layer contro transformers (saltato senza modello)
build/trochilus run -m <f.gguf> -f prompt.txt -n 200    # testo in entrata, generazione greedy
build/trochilus chat -m <file.gguf>                       # conversazione con il template del modello
make chat-check          # chat sul modello vero: seconda risposta = run da zero (saltato senza modello)
make spec-check          # `--spec 0/1/4/8/15` danno lo stesso testo sul modello vero a 2 layer
build/trochilus run ... --spec 8   # speculazione dal prompt (stessi token)
sh tools/ab_spec.sh <gguf> <binario> bench/prompts/code.txt 8   # quanto rende --spec, run alternate
sh tools/build_llamacpp.sh; tools/compare_llamacpp.py ...   # nel container: llama.cpp, logit a confronto
tools/speed_compare.py ...   # nel container: velocità contro llama.cpp e colibri (volume trochilus-models)
sh tools/ab_speed.sh <gguf> <binario A> <binario B>   # due binari alternati run per run (LEZIONI #46)
sh tools/ab_modes.sh <giri> "a=<comando>" "b=<comando>"   # modi di un binario (env, flag), ordine a rotazione, A/A (LEZIONI #66)
build/trochilus run ... --decode-threads 8   # forza i thread del decode (default: li misura la sessione)
sh tools/threads_phase.sh sweep | widths | after <binario prima>   # thread per fase: -t 4/8/12/16, larghezze forzate
sh tools/decode_context.sh measure | widths | long | change <prima>   # decode a contesto 32/512/2048/4000: A/A, larghezze forzate, byte per zona
tools/.venv/Scripts/python.exe tools/decode_context_report.py speed|model|zones <file>   # le tabelle di MISURE dalle run; speed conta scelte e cambi
sh tools/orphans.sh      # è rimasto acceso qualcosa di nostro? make check e le misure non partono
sh tools/busy_machine.sh <n> <comando>   # il SOLO modo di caricare la macchina: i generatori muoiono con lo script
sh tools/machine_still.sh [limite] [attesa] [finestra]   # processori occupati (chi: tools/background_load.ps1): la guardia di ogni misura
sh tools/test_cleanup.sh   # uno script fermato porta via i figli; senza il trap di cleanup.lib il figlio resta
make bench               # microbenchmark dei kernel (mediana + rumore)
make bench-mem           # banda della RAM (in fila, sparsa), matmul del motore, attenzione sui due layout
make bench-disk          # il disco per chi legge esperti, senza cache del sistema (DISK_FILE=<file>)
build/trochilus run ... --route-trace <file>   # traccia del routing: esperti scelti e previsti
tools/.venv/Scripts/python.exe tools/route_trace_report.py <traccia> | --check   # previsione, cache LRU, streaming
tools/.venv/Scripts/python.exe tools/route_graph_report.py <traccia> | --compare | --mask-from   # domanda 44
sh tools/mask_quality.sh   # esperti spenti: KL e token contro il modello intero (sola misura)
sh tools/experts_budget.sh measure | misses | direct   # M1: tok/s ai 4 budget, costo di un token, cache sì/no
build/trochilus run ... --expert-budget <MiB|min>   # RAM degli esperti (default: il piano); TR_EXPERT_BUDGET_MIB nei test
sh tools/mutate_{route,tune,experts,stream}.sh | tools/mutate_reports.py   # nel container: le mutazioni, tutte rosse
make bench-attn          # l'attenzione di un prompt (512/2048/4000) su un layer, smontata per fasi, con controllo dei bit
make bench-expf          # tr_expf su tutti i 2^32 float contro il valore arrotondato e la libreria C
tools/.venv/Scripts/python.exe tools/gen_expf_table.py [--check | --scan]   # le costanti di tr_expf da mpmath (src/kernels/expf_table.h)
sh tools/expf_quality.sh | sh tools/mutate_expf.sh   # nel container: tr_expf contro il binario di prima (KL) e l'emulazione (byte)
sh tools/platform_bits.sh   # Windows e Linux danno gli stessi byte? logit e tabelle RoPE delle due piattaforme
sh tools/prefill_context.sh measure | change <binario prima>   # prefill a 512/2048/4000 in una sessione, A/A, logit al byte, zone
tools/.venv/Scripts/python.exe tools/prefill_context_report.py attn|zones <file>   # le tabelle di MISURE §Prefill su prompt lunghi
make profile             # scenari col profiler, mediana di N, token identici, confronto col precedente
make profile SCENARIOS=bench/scenarios-olmoe-1b-7b.json   # modello vero, 16/8/4/1 thread
make profile SCENARIOS=bench/scenarios-decode-context.json   # modello vero, contesto 32/512/2048/4000: ms e byte per zona
make profile SCENARIOS=bench/scenarios-prefill-context.json  # modello vero, prompt 512/2048/4000 a 16 thread e 512 a 1 thread
tools/.venv/Scripts/python.exe tools/<script>.py    # su Linux/macOS: tools/.venv/bin/python
```

Prima di consegnare: `make check` verde. `make profile` arriva col profiler collegato al motore.
Su Windows la correttezza gira in Docker (`trochilus-dev:local`, `tools/docker/Dockerfile`):
Smart App Control blocca i binari appena compilati (LEZIONI #12).

