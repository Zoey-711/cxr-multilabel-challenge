#requires -Version 5.1
<# Download only lightweight result files into an immutable, versioned local copy. #>
[CmdletBinding()]
param(
    [string]$Root = '',
    [string]$RemoteUrl = 'git@github.com:Zoey-711/cxr-multilabel-challenge.git',
    [string]$RemoteBranch = 'results/phase17-20261024',
    [string]$ResultSubdir = 'results/phase17_batch',
    [int]$ExpectedValStudies = 859,
    [int]$ExpectedTestStudies = 1414,
    [int]$ExpectedTrainStudies = 18513
)
$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootPath = [IO.Path]::GetFullPath($(if ($Root) { $Root } else { $scriptDir }))
if ($RemoteBranch -notmatch '^[A-Za-z0-9._/-]+$' -or $ResultSubdir -notmatch '^[A-Za-z0-9._/-]+$' -or
    $RemoteBranch.Contains('..') -or $ResultSubdir.Contains('..') -or $ResultSubdir.StartsWith('/')) {
    throw 'Unsafe branch or result subdirectory'
}
$versions = Join-Path $rootPath 'versions'
$incoming = Join-Path $rootPath 'incoming'
New-Item -ItemType Directory -Path $versions,$incoming -Force | Out-Null
$originalLfs = [Environment]::GetEnvironmentVariable('GIT_LFS_SKIP_SMUDGE', 'Process')
$originalPrompt = [Environment]::GetEnvironmentVariable('GIT_TERMINAL_PROMPT', 'Process')
$env:GIT_LFS_SKIP_SMUDGE = '1'
$env:GIT_TERMINAL_PROMPT = '0'
function Invoke-GitSafe([string[]]$Arguments) {
    # PowerShell 5 turns benign native stderr (for example Git clone progress)
    # into an error record. Trust the process exit code, not stderr text.
    $prior = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = @(& git @Arguments 2>&1 | ForEach-Object { [string]$_ })
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prior }
    return [pscustomobject]@{ ExitCode = $code; Lines = $lines }
}
try {
    $lfsVersion = Invoke-GitSafe @('lfs','version')
    if ($lfsVersion.ExitCode -ne 0) { throw 'Git LFS is not installed; existing results were not changed.' }
    $remote = Invoke-GitSafe @('ls-remote',$RemoteUrl,"refs/heads/$RemoteBranch")
    if ($remote.ExitCode -ne 0) {
        throw "NOT_AVAILABLE: result repository does not exist, is not readable, or network failed. Existing downloaded versions remain intact. Git: $($remote.Lines -join ' ')"
    }
    $remoteHead = @($remote.Lines)
    if ($remoteHead.Count -eq 0 -or -not [string]$remoteHead[0]) {
        Write-Host 'NOT_AVAILABLE: result branch has no commit. Existing downloads are unchanged.'
        exit 0
    }
    $commit = (([string]$remoteHead[0]) -split '\s+')[0]
    if ($commit -notmatch '^[0-9a-fA-F]{40}$') { throw 'Remote HEAD did not return a valid commit SHA.' }
    $target = Join-Path $versions $commit
    if (Test-Path -LiteralPath $target -PathType Container) {
        Write-Host "Already downloaded commit $commit; checking the preserved local version."
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scriptDir 'Inspect-Phase17Results.ps1') `
            -Root (Join-Path $target $ResultSubdir) -ExpectedValStudies $ExpectedValStudies -ExpectedTestStudies $ExpectedTestStudies `
            -ExpectedTrainStudies $ExpectedTrainStudies
        if ($LASTEXITCODE -ne 0) { throw 'Existing version has a validation error; it was not modified.' }
        @{ commit = $commit; branch = $RemoteBranch; subdirectory = $ResultSubdir; path = "versions/$commit"; synchronized_utc = [DateTime]::UtcNow.ToString('o') } |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $rootPath 'latest.json') -Encoding UTF8
        exit 0
    }
    $staging = Join-Path $incoming ("$commit-" + [guid]::NewGuid().ToString('N'))
    Write-Host "Fetching lightweight files from $RemoteUrl at $commit ..."
    $clone = Invoke-GitSafe @('-c','filter.lfs.smudge=','-c','filter.lfs.required=false','clone',
        '--single-branch','--branch',$RemoteBranch,'--depth','1','--filter=blob:none','--no-checkout',$RemoteUrl,$staging)
    if ($clone.ExitCode -ne 0) { throw "Clone failed. Previous versions remain intact; any partial staging is at $staging. Git: $($clone.Lines -join ' ')" }
    $patterns = @('csv','json','jsonl','md','log','yaml','yml','sha256') |
        ForEach-Object { "$ResultSubdir/**/*.$_" }
    $patterns += "$ResultSubdir/.gitattributes"
    $sparse = Invoke-GitSafe (@('-C',$staging,'sparse-checkout','set','--no-cone','--') + $patterns)
    if ($sparse.ExitCode -ne 0) { throw "Sparse checkout failed; previous versions remain intact. Staging: $staging. Git: $($sparse.Lines -join ' ')" }
    $checkout = Invoke-GitSafe @('-C',$staging,'checkout','--detach',$commit)
    if ($checkout.ExitCode -ne 0) { throw "Sparse checkout materialization failed; previous versions remain intact. Staging: $staging. Git: $($checkout.Lines -join ' ')" }
    $pull = Invoke-GitSafe @('-C',$staging,'lfs','pull',
        '--include=*.csv,*.json,*.jsonl,*.md,*.log,*.yaml,*.yml,*.sha256',
        '--exclude=*.pth,*.pt,*.ckpt,*.zip,*.tar,*.gz,*.png,*.jpg,*.jpeg,*.bin')
    if ($pull.ExitCode -ne 0) { throw "Selective Git LFS pull failed; previous versions remain intact. Staging: $staging. Git: $($pull.Lines -join ' ')" }
    $resultTree = Join-Path $staging $ResultSubdir
    if (-not (Test-Path -LiteralPath $resultTree -PathType Container)) { throw "Result subdirectory missing: $ResultSubdir" }
    $lightFiles = @(Get-ChildItem -LiteralPath $resultTree -Recurse -File -ErrorAction Stop |
        Where-Object { $_.Extension -in @('.csv','.json','.jsonl','.md','.log','.yaml','.yml','.sha256') })
    foreach ($file in $lightFiles) {
        $firstLine = Get-Content -LiteralPath $file.FullName -TotalCount 1 -Encoding UTF8 -ErrorAction Stop
        if ($firstLine -eq 'version https://git-lfs.github.com/spec/v1') {
            throw "LFS pointer was not materialized: $($file.FullName). Previous versions remain intact."
        }
    }
    if ($lightFiles.Count -eq 0) {
        Write-Host "EMPTY: commit $commit contains no selected result files yet. Previous versions remain intact."
        exit 0
    }
    $report = Join-Path $staging 'sync_validation.json'
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scriptDir 'Inspect-Phase17Results.ps1') `
        -Root $resultTree -ReportPath $report -ExpectedValStudies $ExpectedValStudies `
        -ExpectedTestStudies $ExpectedTestStudies -ExpectedTrainStudies $ExpectedTrainStudies
    if ($LASTEXITCODE -ne 0) {
        throw "Downloaded prediction integrity check failed; previous versions remain intact. Staging: $staging"
    }
    # Git/LFS can keep a Windows handle on its clone directory. Publish only
    # the selected small files from a fresh non-Git directory, atomically.
    $partial = Join-Path $versions ('.' + $commit + '-' + [guid]::NewGuid().ToString('N') + '.partial')
    New-Item -ItemType Directory -Path $partial -ErrorAction Stop | Out-Null
    $copyFiles = @($lightFiles) + @(Get-Item -LiteralPath $report)
    foreach ($file in $copyFiles) {
        $relative = $file.FullName.Substring($staging.Length).TrimStart([char[]]@('\','/'))
        $destination = Join-Path $partial $relative
        $parent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $destination -ErrorAction Stop
        if ((Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) {
            throw "Copied result checksum changed: $relative. Previous versions remain intact."
        }
    }
    if (Test-Path -LiteralPath $target) { throw "Version already exists; previous versions remain intact: $target" }
    [IO.Directory]::Move($partial, $target)
    $pointer = Join-Path $rootPath ('.latest-' + [guid]::NewGuid().ToString('N') + '.tmp')
    @{ commit = $commit; branch = $RemoteBranch; subdirectory = $ResultSubdir; path = "versions/$commit"; synchronized_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $pointer -Encoding UTF8
    Move-Item -LiteralPath $pointer -Destination (Join-Path $rootPath 'latest.json') -Force
    Write-Host "Saved immutable result version: $target"
    Write-Host 'Open this folder in VS Code to inspect the uploaded CSV/JSON/Markdown files.'
    # The unique Git staging clone is disposable. A transient Windows lock may
    # leave it for later manual cleanup; never remove a published version.
    try { Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction Stop }
    catch { Write-Host "Temporary Git clone remains at $staging (safe to remove after Git exits)." }
} finally {
    [Environment]::SetEnvironmentVariable('GIT_LFS_SKIP_SMUDGE', $originalLfs, 'Process')
    [Environment]::SetEnvironmentVariable('GIT_TERMINAL_PROMPT', $originalPrompt, 'Process')
}
