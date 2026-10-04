#!/usr/bin/env node
// LESSONS #267: a prediction's time written from the story ("22:00", then "23:05") where `date` said 21:44 and
// 22:52, twice in one session, after #216 had made the same rule for the date. A rule alone did not hold.
// Refuses a Bash or PowerShell command that writes to build/prep/predictions.txt and types a date with an hour
// ("2026-10-02 22:00"): the stamp comes from the clock in the same command, e.g.
//   { date '+%Y-%m-%d %H:%M, before ...'; cat <<'EOF'
//   the prediction
//   EOF
//   } >> build/prep/predictions.txt
// A date alone (no hour) and a time inside the text after the stamp ("at 21:44 the load") are not refused.
let input = ''
process.stdin.on('data', (c) => (input += c))
process.stdin.on('end', () => {
  let cmd = ''
  try { cmd = JSON.parse(input || '{}').tool_input?.command || '' } catch { process.exit(0) }
  if (!/predictions[.]txt/.test(cmd)) process.exit(0)
  // a line of the written text that starts with a typed date and hour
  if (!/(^|[\n'"])20[0-9][0-9]-[0-9][0-9]-[0-9][0-9][ T][0-9][0-9]:[0-9][0-9]/.test(cmd)) process.exit(0)
  console.error(
    'This command writes a typed date and hour into build/prep/predictions.txt (docs/LESSONS.md #267).\n' +
    'Take the stamp from the clock in the same command: { date \'+%Y-%m-%d %H:%M, before ...\'; cat <<\'EOF\'\n' +
    '...\nEOF\n} >> build/prep/predictions.txt'
  )
  process.exit(2)
})
