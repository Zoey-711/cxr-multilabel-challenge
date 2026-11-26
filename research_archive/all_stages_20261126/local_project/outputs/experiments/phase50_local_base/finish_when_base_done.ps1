param([int]$TrainingPid)
$ErrorActionPreference = 'Stop'
$projectRoot = 'E:\计挑赛\cxr-multilabel-challenge'
$baseDir = Join-Path $projectRoot 'outputs\experiments\phase50_local_base'
$runDir = Join-Path $baseDir 'H50_BASE_F1'
try {
    Wait-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'DONE.json'))) {
        @{ state = 'BASE_INCOMPLETE'; reason = 'Training process exited without DONE.json'; pid = $TrainingPid } |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $baseDir 'finish_status.json') -Encoding UTF8
        exit 1
    }
    $pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $finalizer = Join-Path $projectRoot 'scripts\finalize_phase50_base.py'
    & $pythonExe $finalizer *>> (Join-Path $baseDir 'finalizer.log')
    if ($LASTEXITCODE -ne 0) { throw "SWA finalizer exit code $LASTEXITCODE" }
    @{ state = 'FINISHED'; pid = $TrainingPid } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $baseDir 'finish_status.json') -Encoding UTF8
} catch {
    @{ state = 'FINALIZER_FAILED'; reason = $_.Exception.Message; pid = $TrainingPid } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $baseDir 'finish_status.json') -Encoding UTF8
    exit 1
}
