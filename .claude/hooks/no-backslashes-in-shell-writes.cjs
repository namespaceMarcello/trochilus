#!/usr/bin/env node
// LESSONS #14: testo con barre rovesciate scritto da un comando di shell (heredoc Python, echo,
// printf) arriva corrotto: le sequenze diventano tab, a capo o spariscono. Successo quattro volte.
// Rifiuta un comando Bash che scrive file da Python o shell e contiene una barra rovesciata
// seguita da una lettera di escape: quel testo va scritto con gli strumenti Write/Edit.
let dati = ''
process.stdin.on('data', (c) => (dati += c))
process.stdin.on('end', () => {
  let cmd = ''
  try { cmd = JSON.parse(dati || '{}').tool_input?.command || '' } catch { process.exit(0) }
  const scrive = /write_text\(|\.write\(|open\([^)]*['"][wa]b?['"]/.test(cmd)
  if (!scrive) process.exit(0)
  const barra = String.fromCharCode(92)
  const escape = new RegExp(barra + barra + '[ntr"\'' + barra + barra + ']')
  if (!escape.test(cmd)) process.exit(0)
  console.error(
    'Questo comando scrive un file da shell/Python e contiene barre rovesciate (es. barra+n).\n' +
    'Passando dalla shell diventano caratteri di controllo e il file si rovina (docs/LESSONS.md #14).\n' +
    'Scrivi quel testo con lo strumento Write o Edit. Per sostituzioni senza barre rovesciate va bene Python.'
  )
  process.exit(2)
})
