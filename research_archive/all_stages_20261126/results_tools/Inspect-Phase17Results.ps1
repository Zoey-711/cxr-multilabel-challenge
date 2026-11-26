#requires -Version 5.1
<# Read-only inspection of a separately downloaded Phase17 result tree. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Root,
    [int]$ExpectedValStudies = 859,
    [int]$ExpectedTestStudies = 1414,
    [int]$ExpectedTrainStudies = 18513,
    [string]$ReportPath
)
$ErrorActionPreference = 'Stop'
$rootPath = [IO.Path]::GetFullPath($Root)
if (-not (Test-Path -LiteralPath $rootPath -PathType Container)) { throw "Result directory does not exist: $rootPath" }
$branchNames = @('D17','C17','S17','N17')
$ids = @(
    foreach ($branch in $branchNames) { foreach ($fold in 1..5) { '{0}_F{1:D2}' -f $branch,$fold } }
    'CS17_F01'; 'NM17_F01'; 'G17_FULL_01'; 'G17_FULL_02'; 'H17_DISTILL_01'; 'H17_DISTILL_02'
)
$statuses = @{}
foreach ($id in $ids) { $statuses[$id] = 'PENDING' }
$batchDir = if (Test-Path -LiteralPath (Join-Path $rootPath 'outputs/phase17_batch')) {
    Join-Path $rootPath 'outputs/phase17_batch'
} elseif (Test-Path -LiteralPath (Join-Path $rootPath 'phase17_batch')) {
    Join-Path $rootPath 'phase17_batch'
} else { $rootPath }
$hasEvidence = $false
$runningProgress = @{}
foreach ($statusFile in @('experiment_status.csv','extension_status.csv')) {
    $path = Join-Path $batchDir "reports/$statusFile"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    $hasEvidence = $true
    foreach ($row in @(Import-Csv -LiteralPath $path -Encoding UTF8)) {
        $id = [string]$row.Model_ID
        if (-not $statuses.ContainsKey($id)) { continue }
        $state = ([string]$row.State).ToLowerInvariant()
        $statuses[$id] = switch ($state) {
            'running' { 'RUNNING' }
            'complete' { 'BLOCKED' } # DONE.json is required before claiming a complete delivery.
            'failed' { 'FAILED' }
            'test_failed' { 'FAILED' }
            'blocked' { 'BLOCKED' }
            'test_pending' { 'BLOCKED' }
            default { 'PENDING' }
        }
    }
}
$snapshotCandidates = @(Get-ChildItem -LiteralPath $rootPath -Recurse -File -Filter status.json -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'phase17_batch_status' } | Sort-Object LastWriteTimeUtc -Descending)
foreach ($candidate in $snapshotCandidates) {
    try {
        $snapshotManifestPath = Join-Path $candidate.DirectoryName 'manifest.json'
        if (-not (Test-Path -LiteralPath $snapshotManifestPath -PathType Leaf)) { continue }
        $snapshotManifest = Get-Content -LiteralPath $snapshotManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $statusHashEntry = $snapshotManifest.files.PSObject.Properties['status.json']
        if (-not $statusHashEntry) { continue }
        $actualStatusHash = (Get-FileHash -LiteralPath $candidate.FullName -Algorithm SHA256).Hash
        if ($actualStatusHash -ne [string]$statusHashEntry.Value.sha256) { continue }
        $snapshot = Get-Content -LiteralPath $candidate.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($snapshot.schema -eq 'phase17_progress_v1') {
            $hasEvidence = $true
            foreach ($task in $snapshot.tasks) {
                if ($statuses.ContainsKey([string]$task.task_id)) {
                    $state = [string]$task.state
                    if ($state -in @('PENDING','RUNNING','DONE','FAILED','BLOCKED')) { $statuses[[string]$task.task_id] = $state }
                    if ($state -eq 'RUNNING') {
                        $runningProgress[[string]$task.task_id] = [pscustomobject]@{
                            Model_ID = [string]$task.task_id; GPU = [string]$task.gpu
                            Epoch = $task.current_epoch; Best_Epoch = $task.best_epoch
                            Val_AUC_Provisional = $task.val_macro_auc
                        }
                    }
                }
            }
            break
        }
    } catch { continue }
}
$experiments = Join-Path $batchDir 'experiments'
foreach ($id in $ids) {
    $run = Join-Path $experiments $id
    if (Test-Path -LiteralPath (Join-Path $run 'DONE.json') -PathType Leaf) {
        $hasEvidence = $true
        $statuses[$id] = 'DONE'
    } elseif (Test-Path -LiteralPath (Join-Path $run 'FAILED.json') -PathType Leaf) {
        $hasEvidence = $true
        $statuses[$id] = 'FAILED'
    }
}
foreach ($id in $ids) {
    if ($statuses[$id] -ne 'RUNNING' -or $runningProgress.ContainsKey($id)) { continue }
    $history = Join-Path (Join-Path $experiments $id) 'metrics.jsonl'
    if (-not (Test-Path -LiteralPath $history -PathType Leaf)) { continue }
    $last = $null
    foreach ($line in @(Get-Content -LiteralPath $history -Encoding UTF8)) {
        try { $last = $line | ConvertFrom-Json } catch { continue }
    }
    if ($last) {
        $runningProgress[$id] = [pscustomobject]@{
            Model_ID = $id; GPU = 'N/A'; Epoch = $last.epoch
            Best_Epoch = $last.best_epoch; Val_AUC_Provisional = $last.best_auc
        }
    }
}
$counts = [ordered]@{}
foreach ($state in @('DONE','RUNNING','PENDING','FAILED','BLOCKED')) {
    $counts[$state] = @($ids | Where-Object { $statuses[$_] -eq $state }).Count
}
$summaryPath = Join-Path $batchDir 'reports/experiment_summary.csv'
$metrics = @()
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
    $hasEvidence = $true
    foreach ($row in @(Import-Csv -LiteralPath $summaryPath -Encoding UTF8)) {
        $id = [string]$row.model_id
        if (-not $id) { continue }
        $metrics += [pscustomobject]@{
            Model_ID = $id
            State = if ($statuses.ContainsKey($id)) { $statuses[$id] } else { 'DERIVED' }
            Val_AUC = if ($row.val_macro_auc) { $row.val_macro_auc } elseif ($row.val_auc) { $row.val_auc } else { 'N/A' }
            OOF_AUC = if ($row.oof_macro_auc) { $row.oof_macro_auc } elseif ($row.oof_auc) { $row.oof_auc } else { 'N/A' }
            Best_Epoch = if ($row.best_epoch) { $row.best_epoch } else { 'N/A' }
        }
    }
}
$probColumns = @('Subject_id','Study_id') + @(1..10 | ForEach-Object { "Prob_$_" })
$predictionFiles = @(Get-ChildItem -LiteralPath $batchDir -Recurse -File -Filter '*.csv' -ErrorAction SilentlyContinue |
    Where-Object {
        $relativeFile = $_.FullName.Substring($batchDir.Length).TrimStart('\','/')
        $relativeFile -match '^(?i:experiments)[/\\][^/\\]+[/\\](?:ema_)?(?:val|oof|test)_predictions\.csv$' -or
        $relativeFile -match '^(?i:oof)[/\\][^/\\]+_oof_all\.csv$' -or
        $relativeFile -match '^(?i:validation|test_predictions)[/\\][^/\\]+(?:ensemble|probabilities)\.csv$'
    })
$predictions = @()
foreach ($file in $predictionFiles) {
    $relative = $file.FullName.Substring($batchDir.Length).TrimStart('\','/')
    $messages = [System.Collections.Generic.List[string]]::new()
    $status = 'PASS'
    $sha = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $provenancePath = [IO.Path]::ChangeExtension($file.FullName, 'provenance.json')
    $recorded = $null
    if (Test-Path -LiteralPath $provenancePath -PathType Leaf) {
        try {
            $provenance = Get-Content -LiteralPath $provenancePath -Raw -Encoding UTF8 | ConvertFrom-Json
            $recorded = [string]$provenance.csv_sha256
        } catch { $messages.Add('Invalid provenance JSON') }
    }
    if (-not $recorded) {
        $manifestPath = Join-Path $file.DirectoryName 'run_manifest.json'
        if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
            try {
                $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $entry = $manifest.files.PSObject.Properties[$file.Name]
                if ($entry) { $recorded = [string]$entry.Value.sha256 }
            } catch { $messages.Add('Invalid run_manifest JSON') }
        }
    }
    if ($recorded) {
        if ($sha -ne $recorded.ToLowerInvariant()) { $status = 'FAIL'; $messages.Add('SHA-256 mismatch') }
    } else {
        $messages.Add('HASH_UNVERIFIED: no source SHA-256 manifest')
        if ($status -eq 'PASS') { $status = 'HASH_UNVERIFIED' }
    }
    $studyIds = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $keyIds = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $patientIds = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $rows = @()
    try {
        $rows = @(Import-Csv -LiteralPath $file.FullName -Encoding UTF8)
        $headers = if ($rows.Count) { @($rows[0].PSObject.Properties.Name) } else {
            @((Get-Content -LiteralPath $file.FullName -TotalCount 1 -Encoding UTF8) -split ',')
        }
        if (($headers -join ',') -ne ($probColumns -join ',')) {
            $status = 'FAIL'; $messages.Add('Expected exact Subject_id,Study_id,Prob_1..Prob_10 column order')
        } else {
            foreach ($row in $rows) {
                $subject, $study = [string]$row.Subject_id, [string]$row.Study_id
                if (-not $subject -or -not $study -or -not $keyIds.Add("$subject`n$study") -or -not $studyIds.Add($study)) {
                    $status = 'FAIL'; $messages.Add('Missing or duplicate Study ID'); break
                }
                [void]$patientIds.Add($subject)
                foreach ($index in 1..10) {
                    $value = 0.0
                    if (-not [double]::TryParse([string]$row.("Prob_$index"),
                            [Globalization.NumberStyles]::Float,
                            [Globalization.CultureInfo]::InvariantCulture, [ref]$value) -or
                        [double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0 -or $value -gt 1) {
                        $status = 'FAIL'; $messages.Add('Non-finite or out-of-range probability'); break
                    }
                }
                if ($status -eq 'FAIL') { break }
            }
        }
    } catch {
        $status = 'FAIL'; $messages.Add('CSV cannot be parsed')
    }
    if ($relative -match '(?i)(^|[/\\])val(_|/)|val_predictions|val_probabilities') { $expected = $ExpectedValStudies }
    elseif ($relative -match '(?i)test') { $expected = $ExpectedTestStudies }
    elseif ($file.Name -match '_oof_all\.csv$') { $expected = $ExpectedTrainStudies }
    else { $expected = $null }
    if ($null -ne $expected -and $rows.Count -ne $expected) {
        if ($file.Name -match '_oof_all\.csv$' -and $rows.Count -lt $expected -and $status -ne 'FAIL') {
            $status = 'PARTIAL_OOF'; $messages.Add('Full OOF coverage incomplete')
        } else {
            $status = 'FAIL'; $messages.Add("Expected $expected Study rows, found $($rows.Count)")
        }
    }
    $predictions += [pscustomobject]@{
        File = $relative; Status = $status; Studies = $rows.Count; Patients = $patientIds.Count
        SHA256 = $sha; Source_SHA256 = if ($recorded) { $recorded } else { 'N/A' }
        Note = ($messages | Select-Object -Unique) -join '; '
    }
}
$oof = [ordered]@{}
foreach ($branch in $branchNames) {
    $doneFolds = @(1..5 | Where-Object { $statuses[('{0}_F{1:D2}' -f $branch,$_)] -eq 'DONE' }).Count
    $full = @($predictions | Where-Object { $_.File -match "(?i)(^|[/\\])${branch}_oof_all\.csv$" -and $_.Status -eq 'PASS' })
    $oof[$branch] = if ($doneFolds -eq 5 -and $full.Count -gt 0) { 'COMPLETE_OOF' }
                    elseif ($doneFolds -gt 0) { 'PARTIAL_OOF' } else { 'NOT_AVAILABLE' }
}
$overall = if (-not $hasEvidence -and $predictionFiles.Count -eq 0) { 'EMPTY' } else { 'AVAILABLE' }
$report = [ordered]@{
    Status = $overall; Counts = $counts; Tasks = @($ids | ForEach-Object { [pscustomobject]@{ Model_ID = $_; State = $statuses[$_] } })
    Running_Progress = @($runningProgress.Values); Metrics = $metrics; OOF = $oof; Predictions = $predictions
    Note = 'RUNNING AUC is provisional; Test labels and Test AUC are unavailable. Complete OOF requires all five folds.'
}
Write-Host "RESULTS: $overall"
Write-Host "26 TASKS: DONE=$($counts.DONE) RUNNING=$($counts.RUNNING) PENDING=$($counts.PENDING) FAILED=$($counts.FAILED) BLOCKED=$($counts.BLOCKED)"
foreach ($branch in $branchNames) { Write-Host "${branch}: $($oof[$branch])" }
if ($metrics.Count) { $metrics | Format-Table -AutoSize | Out-Host } else { Write-Host 'experiment_summary.csv: NOT_AVAILABLE' }
if ($runningProgress.Count) {
    Write-Host 'RUNNING (provisional, completed Epoch records only):'
    @($runningProgress.Values) | Format-Table -AutoSize | Out-Host
}
foreach ($item in $predictions) { Write-Host "PREDICTION $($item.Status) $($item.Studies) Studies $($item.File) $($item.Note)" }
if ($ReportPath) {
    $parent = Split-Path -Parent $ReportPath
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $temporary = "$ReportPath.$([guid]::NewGuid().ToString('N')).tmp"
    $report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $ReportPath -Force
}
if (@($predictions | Where-Object { $_.Status -eq 'FAIL' }).Count) { exit 2 }
