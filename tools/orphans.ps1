# orphans.ps1 <repo path> -- processes this project started and nobody ended (docs/LESSONS.md #84).
#
# One line per process: pid, start, what it is. Nothing printed means nothing is left. What counts:
#   - any `yes` (the load generator of a busy-machine test: tools/busy_machine.sh starts them and
#     ends them; four from an ad-hoc test ran for 37 hours under two days of measurements)
#   - any program running out of <repo>\build (an engine, a benchmark, a test)
#   - a measuring script of tools/ or the profile suite
# except the processes above this one (the script that is asking). Used by tools/orphans.sh.
param([string]$Repo, [string]$Callers = '')
$build = (Join-Path $Repo 'build').TrimEnd('\') + '\'
$scripts = 'tools[/\\](prefill_context|decode_context|threads_phase|remeasure|experts_budget|prefill_overlap|serve_first_prompt|ab_modes|ab_speed|ab_spec|platform_bits|busy_machine)\.sh|tools[/\\]profile_suite\.py'
$all = Get-CimInstance Win32_Process
$byId = @{}
foreach ($p in $all) { $byId[[int]$p.ProcessId] = $p }
$mine = @{}
# the callers, as MSYS sees them (tools/orphans.sh): Windows loses the chain at every MSYS exec;
# and above each of them, Windows' own parents: the outermost MSYS shell's parent is a launcher
# (Git's bin\bash.exe, whose command line holds the script it was asked to run), not an orphan
# (docs/LESSONS.md #136)
$starts = @($PID)
foreach ($c in ($Callers -split ',')) { if ($c -match '^\d+$') { $starts += [int]$c } }
foreach ($s in $starts) {
    $id = $s
    while ($id -and -not $mine.ContainsKey([int]$id)) {
        $mine[[int]$id] = $true
        if (-not $byId.ContainsKey([int]$id)) { break }
        $id = $byId[[int]$id].ParentProcessId
    }
}
foreach ($p in $all) {
    if ($mine.ContainsKey([int]$p.ProcessId)) { continue }
    $what = $null
    if ($p.Name -ieq 'yes.exe') { $what = 'a load generator' }
    elseif ($p.ExecutablePath -and $p.ExecutablePath.StartsWith($build, [StringComparison]::OrdinalIgnoreCase)) { $what = 'a program of build\' }
    elseif ($p.CommandLine -and $p.CommandLine -match $scripts -and $p.CommandLine -notmatch 'orphans\.(sh|ps1)') { $what = 'a measuring script' }
    if ($what) {
        $cmd = if ($p.CommandLine) { $p.CommandLine } else { $p.Name }
        if ($cmd.Length -gt 140) { $cmd = $cmd.Substring(0, 140) }
        '{0} pid {1}, started {2}: {3}' -f $what, $p.ProcessId, $p.CreationDate.ToString('yyyy-MM-dd HH:mm:ss'), $cmd
    }
}
