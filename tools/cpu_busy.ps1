# cpu_busy.ps1 [seconds] [-Top] -- how many logical processors are busy, machine wide.
#
# A measurement "on a still machine" has to ask the machine (docs/LESSONS.md #84: four forgotten
# `yes` processes ran at 100% for 37 hours under every native measurement of two days, and no
# script looked). Prints one number: the logical processors busy on average over the window, from
# two raw samples of the idle counter (class and property names are not localized, unlike the
# names of Get-Counter). With -Top, then one line: the three processes that used the most in the
# same window, in cores, and the kernel's System process apart (0.5-0.8 on this machine at rest),
# from the raw per-process counters, which hold the protected processes too (the antivirus, the
# search indexer, the WSL virtual machine: Get-Process cannot read them without elevation, and
# they are the likely ones, docs/LESSONS.md #241). Used by tools/machine_still.sh.
param([int]$Seconds = 2, [switch]$Top)
function Sample { Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter "Name='_Total'" }
function Procs { Get-CimInstance Win32_PerfRawData_PerfProc_Process | Where-Object { $_.Name -ne '_Total' -and $_.Name -ne 'Idle' } }
$inv = [Globalization.CultureInfo]::InvariantCulture
if ($Top) { $p0 = Procs }
$a = Sample
Start-Sleep -Seconds $Seconds
$b = Sample
if ($Top) { $p1 = Procs }
$idle = [double]($b.PercentProcessorTime - $a.PercentProcessorTime) / [double]($b.Timestamp_Sys100NS - $a.Timestamp_Sys100NS)
$busy = [double][Environment]::ProcessorCount * (1.0 - $idle)
[string]::Format($inv, "{0:F2}", [math]::Max([double]0, $busy))
if ($Top) {
    $before = @{}
    foreach ($p in $p0) { $before["$($p.IDProcess)"] = $p }
    $used = @{}
    foreach ($p in $p1) {
        $o = $before["$($p.IDProcess)"]
        if ($o -eq $null -or $p.Timestamp_Sys100NS -le $o.Timestamp_Sys100NS) { continue }
        $name = $p.Name -replace '#\d+$', ''
        # this script and the WMI host that answers its queries are the look's own work
        if ($p.IDProcess -eq $PID -or $name -eq 'WmiPrvSE') { continue }
        $cores = [double]($p.PercentProcessorTime - $o.PercentProcessorTime) / [double]($p.Timestamp_Sys100NS - $o.Timestamp_Sys100NS)
        $used[$name] = [double]$used[$name] + $cores
    }
    # (not $top: PowerShell's names ignore case, and that is the -Top switch)
    $lead = $used.GetEnumerator() | Where-Object { $_.Key -ne 'System' } | Sort-Object Value -Descending |
        Select-Object -First 3 | ForEach-Object { [string]::Format($inv, "{0} {1:F2}", $_.Key, $_.Value) }
    [string]::Format($inv, "then: {0}; the kernel's System {1:F2}", ($lead -join ', '), [double]$used['System'])
}
