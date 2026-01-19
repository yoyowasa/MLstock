param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$target = Join-Path $repoRoot "scripts\\run_weekly.py"

if (-not (Test-Path $python)) {
    Write-Error "Virtualenv python not found: $python"
    exit 1
}

Set-Location $repoRoot
& $python $target @Args
$code = $LASTEXITCODE
if ($code -ne 0) {
    exit $code
}

$bundleScript = Join-Path $repoRoot "scripts\\weekly_bundle.ps1"
if (Test-Path $bundleScript) {
    & $bundleScript
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$kpiScript = Join-Path $repoRoot "scripts\\run_deadband_kpi.py"
if (Test-Path $kpiScript) {
    & $python $kpiScript
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

exit 0
