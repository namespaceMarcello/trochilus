# Upstream

Bug e problemi trovati in colibri e ds4 mentre se ne portano i pezzi. Ogni riga dice la prova,
cosa si rompe davvero e a che punto è la segnalazione. Una PR o un'issue si apre solo dopo il sì
esplicito di Marcello, seguendo la skill `oss-contributo` nel fork di quel progetto
(`Desktop\colibri`, `Desktop\ds4`), con il suo cancello e le sue regole di contribuzione.

Decisione 2026-09-17: nessuna PR per ora, le segnalazioni si accumulano e si decide dopo.

Stati: *trovato* → *verificato* (prova riprodotta, non già segnalato) → *PR pronta* (fix + gate
verde nel fork) → *aperta #n* → *chiusa* (mergiata o rifiutata, con il motivo).

| # | Data | Progetto | Cosa | Prova | Impatto reale | Stato |
|---|---|---|---|---|---|---|
| 1 | 2026-09-17 | ds4 `8db1d1d` | `gguf_types[]` in `ds4.c` dà dimensioni di blocco sbagliate a due tipi: `iq1_s` 256/110 (giusto 256/50) e `iq4_nl` 256/50 (giusto 32/18) | il `cuda/mmq/ggml-common.h` dello stesso repo lo verifica a compilazione (`static_assert(sizeof(block_iq1_s) == sizeof(ggml_half) + QK_K/8 + QK_K/16)`); `gguf.GGML_QUANT_SIZES` concorda; trovato dal lint di Trochilus | `tensor_nbytes` calcola la dimensione di ogni tensore da questa tabella: un GGUF con tensori `iq1_s` risulta più grande del vero e può essere rifiutato come fuori dal file; con `iq4_nl` la dimensione e il riepilogo dei tipi sono sbagliati. I modelli ufficiali di ds4 non usano questi tipi: tocca chi apre GGUF di terzi | verificato (nessuna issue o PR su GitHub) |
| 2 | 2026-09-17 | colibri `a90bed9` (dev) | `make check` con gcc 13 non è a 0 warning, contro la regola di CONTRIBUTING: `g_metal_prefill` (`colibri.c:3950`), `g_direct_heat_explicit` (`colibri.c:904`), `g_k3_usage` (`kimi_k3.c:191`) inutilizzati nei build con adattatori; `-Wformat-truncation` in `st.h:452`; `-Wmaybe-uninitialized` su `stats.total_bytes` in `deepseek_v4.c:1360` | log del gate di calibrazione nel fork (`Desktop\colibri\tmp\gate\dev\`) | nessun comportamento sbagliato: il `maybe-uninitialized` è un falso positivo (la funzione riempie sempre `stats` quando riesce), il troncamento colpisce solo percorsi oltre ~1170 caratteri e l'indice è un controllo incrociato. Solo igiene del build | verificato, valore basso |
