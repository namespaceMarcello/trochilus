#!/usr/bin/env node
const { execFileSync } = require('node:child_process')
let dati = ''
process.stdin.on('data', (c) => (dati += c))
process.stdin.on('end', () => {
  let cmd = ''
  try { cmd = JSON.parse(dati || '{}').tool_input?.command || '' } catch { process.exit(0) }
  if (!/\bgit\b[\s\S]*\bcommit\b/.test(cmd)) process.exit(0)
  let staged = []
  try {
    staged = execFileSync('git', ['diff', '--cached', '--name-only'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
      .split('\n').map((r) => r.trim()).filter(Boolean)
  } catch { process.exit(0) }
  if (!staged.length) process.exit(0)
  const tocca = staged.some((f) => /^(src|tests|tools|bench|cmake)\/|^(Makefile|CMakeLists\.txt)$/.test(f))
  if (!tocca) process.exit(0)
  // Il motore cambia senza che cambi un test: si rifiuta, salvo motivo esplicito nel messaggio.
  const motore = staged.some((f) => /^src\//.test(f))
  const test = staged.some((f) => /^tests\/|^bench\/scenarios|^tools\/(make_tiny_|oracle|profile_suite)/.test(f))
  if (motore && !test && !/no-test:/.test(cmd)) {
    console.error(
      'Questo commit cambia src/ ma nessun test, oracolo o scenario.\n' +
      'Ogni comportamento nuovo o corretto entra nei test (CLAUDE.md, ciclo di controllo punto 3-4).\n' +
      'Aggiungi il test, oppure scrivi nel messaggio "no-test: <perche\' non serve>" (es. solo commenti).\n' +
      'File nel commit: ' + staged.join(', ')
    )
    process.exit(2)
  }
  if (staged.includes('docs/archive/DONE.md')) process.exit(0)
  console.error(
    'Questo commit tocca il codice ma non documenta niente.\n' +
    'Aggiungi in coda a docs/archive/DONE.md una voce di 2-5 righe:\n' +
    '  ### <data> — <titolo>\n  cosa e\' stato implementato, e come si prova.\n' +
    'Aggiorna docs/STATUS.md SOLO se e\' cambiata una decisione, un debito o un\n' +
    'prossimo passo (cancella la voce fatta, non aggiungerne una accanto).\n' +
    'Se hai portato codice da colibri o ds4, aggiorna docs/ORIGINS.md.\n' +
    'Poi metti in stage e rifai il commit.\n' +
    'File nel commit: ' + staged.join(', ')
  )
  process.exit(2)
})
