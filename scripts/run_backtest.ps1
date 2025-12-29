param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$target = Join-Path $repoRoot "scripts\\run_backtest.py"

if (-not (Test-Path $python)) {
    Write-Error "Virtualenv python not found: $python"
    exit 1
}

Set-Location $repoRoot
& $python $target @Args
exit $LASTEXITCODE
