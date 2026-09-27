#!/usr/bin/env node
// LESSONS #14: text with backslashes written by a shell command (a Python heredoc, echo,
// printf) arrives corrupted: the sequences become tabs, newlines or vanish. It happened four times.
// Refuses a Bash command that writes a file from Python or the shell and holds a backslash
// followed by an escape letter: that text is written with the Write/Edit tools.
let input = ''
process.stdin.on('data', (c) => (input += c))
process.stdin.on('end', () => {
  let cmd = ''
  try { cmd = JSON.parse(input || '{}').tool_input?.command || '' } catch { process.exit(0) }
  // sed -i too: a replacement holding backslash+n left the two characters in the Makefile (LESSONS #145)
  // write_bytes too: the binary-safe edit LESSONS #168 asks for still passes its text through the shell (#187)
  const writes = /write_text\(|write_bytes\(|\.write\(|open\([^)]*['"][wa]b?['"]|\bsed\s+(-[a-zA-Z]*i|--in-place)/.test(cmd)
  if (!writes) process.exit(0)
  const backslash = String.fromCharCode(92)
  const escape = new RegExp(backslash + backslash + '[ntr"\'' + backslash + backslash + ']')
  if (!escape.test(cmd)) process.exit(0)
  console.error(
    'This command writes a file from the shell/Python and holds backslashes (e.g. backslash+n).\n' +
    'Through the shell they become control characters and the file is ruined (docs/LESSONS.md #14).\n' +
    'Write that text with the Write or Edit tool. For replacements without backslashes Python is fine.'
  )
  process.exit(2)
})
