param(
    [string]$SelectionPath,
    [string]$OrdersPath,
    [string]$WeeklyLogPath,
    [string]$PortfolioPath = "artifacts/state/portfolio.json",
    [string]$DestRoot = "artifacts/weekly_bundle"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

function Get-LatestFile([string]$pattern) {
    $item = Get-ChildItem $pattern | Sort-Object Name | Select-Object -Last 1
    if ($null -eq $item) {
        return $null
    }
    return $item.FullName
}

function Resolve-RepoPath([string]$path) {
    if ([string]::IsNullOrWhiteSpace($path)) {
        return $null
    }
    if ([System.IO.Path]::IsPathRooted($path)) {
        return $path
    }
    return (Join-Path $repoRoot $path)
}

if (-not $SelectionPath) {
    $SelectionPath = Get-LatestFile "artifacts/orders/selection_*.json"
} else {
    $SelectionPath = Resolve-RepoPath $SelectionPath
}

if (-not $OrdersPath) {
    $OrdersPath = Get-LatestFile "artifacts/orders/orders_[0-9]*.csv"
} else {
    $OrdersPath = Resolve-RepoPath $OrdersPath
}

if (-not $WeeklyLogPath) {
    $WeeklyLogPath = Get-LatestFile "artifacts/logs/weekly_*.jsonl"
} else {
    $WeeklyLogPath = Resolve-RepoPath $WeeklyLogPath
}

$PortfolioPath = Resolve-RepoPath $PortfolioPath
$DestRoot = Resolve-RepoPath $DestRoot

if (-not (Test-Path $SelectionPath)) {
    Write-Error "Selection JSON not found: $SelectionPath"
    exit 1
}
if (-not (Test-Path $OrdersPath)) {
    Write-Error "Orders CSV not found: $OrdersPath"
    exit 1
}
if (-not (Test-Path $PortfolioPath)) {
    Write-Error "Portfolio JSON not found: $PortfolioPath"
    exit 1
}
if (-not (Test-Path $WeeklyLogPath)) {
    Write-Error "Weekly log not found: $WeeklyLogPath"
    exit 1
}

$selName = [System.IO.Path]::GetFileName($SelectionPath)
if ($selName -match 'selection_(\d{8})') {
    $stamp = $Matches[1]
} else {
    Write-Error "Cannot parse date stamp from selection file: $selName"
    exit 1
}

$dest = Join-Path $DestRoot $stamp
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item $SelectionPath, $OrdersPath, $PortfolioPath, $WeeklyLogPath -Destination $dest -Force

Write-Host "Bundled files to $dest"
Get-ChildItem $dest
