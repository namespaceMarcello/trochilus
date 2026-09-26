#!/usr/bin/env node
// document-before-commit.cjs — a commit that touches the code documents itself (CLAUDE.md, the table
// "Before every commit: document"): a test for every change to src/, a DONE entry for every change
// to the code, and the README for every change to src/, unless the message says why not.
const { execFileSync } = require('node:child_process')
let input = ''
process.stdin.on('data', (c) => (input += c))
process.stdin.on('end', () => {
  let cmd = ''
  try { cmd = JSON.parse(input || '{}').tool_input?.command || '' } catch { process.exit(0) }
  if (!/\bgit\b[\s\S]*\bcommit\b/.test(cmd)) process.exit(0)
  let staged = []
  try {
    staged = execFileSync('git', ['diff', '--cached', '--name-only'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
      .split('\n').map((r) => r.trim()).filter(Boolean)
  } catch { process.exit(0) }
  if (!staged.length) process.exit(0)
  const code = staged.some((f) => /^(src|tests|tools|bench|cmake)\/|^(Makefile|CMakeLists\.txt)$/.test(f))
  if (!code) process.exit(0)
  const files = 'Files in the commit: ' + staged.join(', ')
  // the engine changes and no test does: refused, unless the message says why
  const engine = staged.some((f) => /^src\//.test(f))
  const test = staged.some((f) => /^tests\/|^bench\/scenarios|^tools\/(make_tiny_|oracle|profile_suite)/.test(f))
  if (engine && !test && !/no-test:/.test(cmd)) {
    console.error(
      'This commit changes src/ but no test, oracle or scenario.\n' +
      'Every new or corrected behaviour enters the tests (CLAUDE.md, the cycle of every step, 3-4).\n' +
      'Add the test, or write "no-test: <why it is not needed>" in the message (e.g. comments only).\n' + files
    )
    process.exit(2)
  }
  if (!staged.includes('docs/archive/DONE.md')) {
    console.error(
      'This commit touches the code but documents nothing.\n' +
      'Add at the end of docs/archive/DONE.md an entry of 2-5 lines:\n' +
      '  ### <date> — <title>\n  what was implemented, and how to try it.\n' +
      'Update docs/STATUS.md ONLY if a decision, a debt or a next step changed (replace the done\n' +
      'item, do not add one beside it). Code ported from colibri or ds4: docs/ORIGINS.md too.\n' +
      'Then stage and commit again.\n' + files
    )
    process.exit(2)
  }
  // a significant implementation reaches the README in the same commit (Marcello, 2026-09-26)
  if (engine && !staged.includes('README.md') && !/no-readme:/.test(cmd)) {
    console.error(
      'This commit changes the engine (src/) but not README.md.\n' +
      'A significant implementation (a capability a user sees, a number against llama.cpp, a milestone\n' +
      'step) goes into README.md in the same commit: §What we have done, the speed table, the race with\n' +
      'llama.cpp, §What is missing. Only what is offered, never the roads left.\n' +
      'Update the README, or write "no-readme: <why it is not significant>" in the message.\n' + files
    )
    process.exit(2)
  }
  process.exit(0)
})
