<#
.SYNOPSIS
    Apex Coder - Scheduled Autonomous Flywheel Runner & Task Scheduler Helper

.DESCRIPTION
    Automates off-hours synthetic DPO preference pair generation, data pool validation,
    continual micro-SFT & DPO training, and automated quality gate adapter promotion.

.PARAMETER Count
    Number of synthetic DPO preference pairs to generate (Default: 20).

.PARAMETER DryRun
    Runs DPO generation, validation, and benchmark gates without committing adapter training.

.PARAMETER SkipDpoGen
    Skips new DPO pair generation and trains on existing pending pool.

.PARAMETER RegisterTask
    Registers this script as a daily background task in Windows Task Scheduler at 2:00 AM.

.PARAMETER UnregisterTask
    Removes the ApexCoderFlywheel task from Windows Task Scheduler.

.EXAMPLE
    .\run_flywheel.ps1 -Count 20
    .\run_flywheel.ps1 -DryRun
    .\run_flywheel.ps1 -RegisterTask
#>

param(
    [int]$Count = 20,
    [switch]$DryRun,
    [switch]$SkipDpoGen,
    [switch]$RegisterTask,
    [switch]$UnregisterTask
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "E:\apex-coder"
$PythonExe = "$ProjectRoot\venv\Scripts\python.exe"
$FlywheelScript = "$ProjectRoot\training\scheduled_flywheel.py"
$LogFile = "$ProjectRoot\training\flywheel.log"
$TaskName = "ApexCoderFlywheel"

# Task Scheduler Registration Handler
if ($RegisterTask) {
    Write-Host "[*] Registering '$TaskName' in Windows Task Scheduler (Daily at 2:00 AM)..." -ForegroundColor Cyan
    $Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectRoot\training\run_flywheel.ps1`" -Count 20"
    $Trigger = New-ScheduledTaskTrigger -Daily -At "2:00AM"
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    $Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
    Write-Host "[+] Scheduled task '$TaskName' registered successfully!" -ForegroundColor Green
    Write-Host "    Next Run Time : Daily at 2:00 AM" -ForegroundColor Gray
    Write-Host "    Action Target : $ProjectRoot\training\run_flywheel.ps1" -ForegroundColor Gray
    exit 0
}

if ($UnregisterTask) {
    Write-Host "[*] Unregistering '$TaskName' from Windows Task Scheduler..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[+] Scheduled task '$TaskName' removed successfully." -ForegroundColor Green
    exit 0
}

# Environment Configuration
$env:HF_HOME = "E:\hf_cache"
$env:UV_CACHE_DIR = "E:\uv_cache"
$env:TORCH_HOME = "E:\hf_cache\torch"
$env:PYTHONUNBUFFERED = "1"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at: $PythonExe"
    exit 1
}

# Build arguments array
$ScriptArgs = @("$FlywheelScript", "--count", "$Count")
if ($DryRun) {
    $ScriptArgs += "--dry-run"
}
if ($SkipDpoGen) {
    $ScriptArgs += "--skip-dpo-gen"
}

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "       APEX CODER - AUTONOMOUS FLYWHEEL RUNNER" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "[*] Python Executable : $PythonExe" -ForegroundColor Gray
Write-Host "[*] Project Directory : $ProjectRoot" -ForegroundColor Gray
Write-Host "[*] Log Output File   : $LogFile" -ForegroundColor Gray
Write-Host "[*] Timestamp         : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "-----------------------------------------------------------`n" -ForegroundColor Cyan

# Execute Flywheel with tee-logging
$ExitCode = 0
try {
    & $PythonExe @ScriptArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    $ExitCode = $LASTEXITCODE
}
catch {
    Write-Host "[!] Error executing flywheel: $_" -ForegroundColor Red
    $ExitCode = 1
}

if ($ExitCode -eq 0) {
    Write-Host "`n[+] Flywheel cycle completed successfully (Exit Code: 0)" -ForegroundColor Green
} else {
    Write-Host "`n[!] Flywheel cycle finished with warnings/errors (Exit Code: $ExitCode)" -ForegroundColor Yellow
}

exit $ExitCode
