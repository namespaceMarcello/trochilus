# cpu_busy.ps1 [seconds] [-Top] -- how many logical processors are busy, machine wide.
#
# A measurement "on a still machine" has to ask the machine (docs/LESSONS.md #84: four forgotten
# `yes` processes ran at 100% for 37 hours under every native measurement of two days, and no
# script looked). Prints one number: the logical processors busy on average over the window, from
# two raw samples of the idle counter (class and property names are not localized, unlike the
# names of Get-Counter). With -Top, then the five processes that used the most CPU in the window.
# Used by tools/machine_still.sh.
param([int]$Seconds = 2, [switch]$Top)
function Sample { Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter "Name='_Total'" }
$inv = [Globalization.CultureInfo]::InvariantCulture
if ($Top) { $p0 = Get-Process | Select-Object Id, ProcessName, CPU }
$a = Sample
Start-Sleep -Seconds $Seconds
$b = Sample
$idle = [double]($b.PercentProcessorTime - $a.PercentProcessorTime) / [double]($b.Timestamp_Sys100NS - $a.Timestamp_Sys100NS)
$busy = [double][Environment]::ProcessorCount * (1.0 - $idle)
[string]::Format($inv, "{0:F2}", [math]::Max([double]0, $busy))
if ($Top) {
    $before = @{}
    foreach ($p in $p0) { $before[$p.Id] = $p.CPU }
    Get-Process | Where-Object { $_.CPU -ne $null -and $before.ContainsKey($_.Id) -and $before[$_.Id] -ne $null } |
        ForEach-Object { [pscustomobject]@{ Name = $_.ProcessName; Id = $_.Id; Used = $_.CPU - $before[$_.Id] } } |
        Sort-Object Used -Descending | Select-Object -First 5 |
        ForEach-Object { [string]::Format($inv, "  {0} (pid {1}): {2:F1} s of CPU", $_.Name, $_.Id, $_.Used) }
}
