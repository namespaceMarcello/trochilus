# stay_awake.ps1 <flag file> [max minutes] -- keeps Windows awake while a measurement runs.
#
# After hours without input Windows goes to modern standby and suspends every process: a
# measurement stops half way and goes on hours later on another machine (docs/LEZIONI.md #82).
# This holds a power request (ES_SYSTEM_REQUIRED) for as long as the flag file exists, and lets
# it go when the measurement removes the file, or after max minutes if the measurement died.
# It changes no setting of the machine. Started by tools/measure_guard.lib.
param([string]$Flag, [int]$MaxMinutes = 240)
Add-Type -Namespace Tr -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Tr.Power]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$deadline = (Get-Date).AddMinutes($MaxMinutes)
while ((Test-Path -LiteralPath $Flag) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Seconds 15 }
[void][Tr.Power]::SetThreadExecutionState([uint32]2147483648)   # ES_CONTINUOUS: back to normal
