# background_load.ps1 [seconds] -- what the machine is doing when nobody is measuring: one line.
#
# A native measurement runs on top of whatever else the machine does, and that is part of the
# number: every measuring script writes this line in its log before its first run and after its
# last (docs/LEZIONI.md #85: on this machine the kernel's System process alone holds 0.8 of a
# core with everything stopped). Total: logical processors busy on average over the window, from
# the raw idle counter. Then the processes that used the most, in cores, from the raw per-process
# counters: they include the protected ones (System, the antivirus, the WSL virtual machine),
# which Get-Process cannot read without elevation. Class and property names are not localized.
param([int]$Seconds = 20)
$inv = [Globalization.CultureInfo]::InvariantCulture
function Total { Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter "Name='_Total'" }
function Procs { Get-CimInstance Win32_PerfRawData_PerfProc_Process | Where-Object { $_.Name -ne '_Total' -and $_.Name -ne 'Idle' } }
$t0 = Total; $p0 = Procs
Start-Sleep -Seconds $Seconds
$t1 = Total; $p1 = Procs
$idle = [double]($t1.PercentProcessorTime - $t0.PercentProcessorTime) / [double]($t1.Timestamp_Sys100NS - $t0.Timestamp_Sys100NS)
$busy = [math]::Max([double]0, [double][Environment]::ProcessorCount * (1.0 - $idle))
$before = @{}
foreach ($p in $p0) { $before["$($p.IDProcess)"] = $p }
$used = @{}
foreach ($p in $p1) {
    $o = $before["$($p.IDProcess)"]
    if ($o -eq $null -or $p.Timestamp_Sys100NS -le $o.Timestamp_Sys100NS) { continue }
    $cores = [double]($p.PercentProcessorTime - $o.PercentProcessorTime) / [double]($p.Timestamp_Sys100NS - $o.Timestamp_Sys100NS)
    $name = $p.Name -replace '#\d+$', ''
    $used[$name] = [double]$used[$name] + $cores
}
$top = $used.GetEnumerator() | Where-Object { $_.Value -ge 0.02 -and $_.Key -ne 'System' } | Sort-Object Value -Descending |
    Select-Object -First 5 | ForEach-Object { [string]::Format($inv, "{0} {1:F2}", $_.Key, $_.Value) }
[string]::Format($inv, "background load: {0:F2} logical processors busy over {1} s, {2:F2} of them the kernel's System process (then: {3})",
    $busy, $Seconds, [double]$used['System'], ($top -join ', '))
