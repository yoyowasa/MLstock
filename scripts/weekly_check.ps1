param(
    [string]$SelectionPath,
    [string]$OrdersPath,
    [string]$WeeklyLogPath,
    [string]$PortfolioPath = "artifacts/state/portfolio.json",
    [int]$MaxPositions = 15,
    [int]$NgStreakThreshold = 0,
    [int]$NgStreakLookback = 0,
    [string]$BundleRoot = "artifacts/weekly_bundle"
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

function Parse-Date([string]$value) {
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $null
    }
    return [datetime]::Parse($value).Date
}

function Add-Result([string]$label, [bool]$pass, [string]$detail) {
    $script:results += [pscustomobject]@{
        Label  = $label
        Pass   = $pass
        Detail = $detail
    }
}

function Invoke-WeeklyCheck(
    [string]$SelectionPath,
    [string]$OrdersPath,
    [string]$WeeklyLogPath,
    [string]$PortfolioPath,
    [int]$MaxPositions
) {
    $script:results = @()

    $selection = Get-Content $SelectionPath -Raw | ConvertFrom-Json
    $portfolio = Get-Content $PortfolioPath -Raw | ConvertFrom-Json
    $orders = @()
    if ((Get-Item $OrdersPath).Length -gt 0) {
        $orders = Import-Csv $OrdersPath
    }

    $buy = @($selection.buy_symbols) | Where-Object { $_ }
    $sell = @($selection.sell_symbols) | Where-Object { $_ }
    $keep = @($selection.keep_symbols) | Where-Object { $_ }
    $nSelected = 0
    if ($null -ne $selection.n_selected) {
        $nSelected = [int]$selection.n_selected
    }

    # selection チェック
    $selName = [System.IO.Path]::GetFileName($SelectionPath)
    $stamp = $null
    if ($selName -match 'selection_(\d{8})') {
        $stamp = $Matches[1]
    }
    $asOfDate = Parse-Date $selection.as_of
    $weekStartDate = Parse-Date $selection.week_start
    $stampDate = $null
    if ($stamp) {
        $stampDate = [datetime]::ParseExact($stamp, "yyyyMMdd", $null).Date
    }
    $passStamp = $false
    if ($asOfDate -and $stampDate) {
        $passStamp = ($asOfDate -eq $stampDate)
    }
    Add-Result "selection: as_of とファイル日付" $passStamp "as_of=$($selection.as_of) file=$stamp"

    $passWeek = $false
    if ($asOfDate -and $weekStartDate) {
        $dow = [int]$asOfDate.DayOfWeek
        $offset = ($dow + 6) % 7
        $monday = $asOfDate.AddDays(-$offset)
        $sunday = $monday.AddDays(6)
        $passWeek = ($weekStartDate -ge $monday -and $weekStartDate -le $sunday)
    }
    Add-Result "selection: week_start が as_of の同一週" $passWeek "as_of=$($selection.as_of) week_start=$($selection.week_start)"

    $featuresDate = Parse-Date $selection.data_max_features_date
    $labelsDate = Parse-Date $selection.data_max_labels_date
    $weekMapDate = Parse-Date $selection.data_max_week_map_date
    $passDataMax = $false
    if ($weekStartDate -and $featuresDate -and $weekMapDate) {
        $labelsOk = $false
        if ($labelsDate) {
            $labelsOk = ($labelsDate -ge $weekStartDate.AddDays(-7))
        }
        $passDataMax = ($featuresDate -ge $weekStartDate -and $weekMapDate -ge $weekStartDate -and $labelsOk)
    }
    Add-Result "selection: data_max_* の更新" $passDataMax "features=$($selection.data_max_features_date) labels=$($selection.data_max_labels_date) week_map=$($selection.data_max_week_map_date)"

    $passCount = ($nSelected -eq $buy.Count)
    Add-Result "selection: n_selected と buy_symbols 数" $passCount "n_selected=$nSelected buy_symbols=$($buy.Count)"

    $overlap = @()
    $overlap += $buy | Where-Object { $sell -contains $_ -or $keep -contains $_ }
    $overlap += $sell | Where-Object { $keep -contains $_ }
    $overlap = $overlap | Sort-Object -Unique
    $passOverlap = ($overlap.Count -eq 0)
    Add-Result "selection: buy/sell/keep の重複なし" $passOverlap ("overlap=" + ($overlap -join ","))

    # orders チェック
    if ($orders.Count -eq 0) {
        $passEmpty = ($buy.Count -eq 0 -and $sell.Count -eq 0)
        Add-Result "orders: 空ファイルの整合" $passEmpty "buy=$($buy.Count) sell=$($sell.Count)"
    } else {
        $orderIssues = @()
        foreach ($row in $orders) {
            $side = ($row.side + "").ToLower()
            $symbol = ($row.symbol + "").ToUpper()
            $qtyOk = $false
            $qtyVal = 0
            if ([int]::TryParse($row.qty, [ref]$qtyVal)) {
                $qtyOk = ($qtyVal -ge 1)
            }
            if ($side -ne "buy" -and $side -ne "sell") {
                $orderIssues += "side=$($row.side)"
            }
            if (-not $qtyOk) {
                $orderIssues += "qty=$($row.qty)"
            }
            if ($side -eq "buy" -and -not ($buy -contains $symbol)) {
                $orderIssues += "buy_symbol=$symbol"
            }
            if ($side -eq "sell" -and -not ($sell -contains $symbol)) {
                $orderIssues += "sell_symbol=$symbol"
            }
        }
        $passOrders = ($orderIssues.Count -eq 0)
        Add-Result "orders: 行の整合" $passOrders ("issues=" + ($orderIssues -join "; "))
    }

    # portfolio チェック
    $passPortfolioDates = $false
    if ($portfolio.as_of -and $portfolio.week_start -and $selection.as_of -and $selection.week_start) {
        $passPortfolioDates = ($portfolio.as_of -eq $selection.as_of -and $portfolio.week_start -eq $selection.week_start)
    }
    Add-Result "portfolio: as_of/week_start が selection と一致" $passPortfolioDates "portfolio=$($portfolio.as_of)/$($portfolio.week_start)"

    $cashOk = $false
    if ($null -ne $portfolio.cash_usd) {
        $cashOk = ([double]$portfolio.cash_usd -ge 0)
    }
    Add-Result "portfolio: cash_usd >= 0" $cashOk "cash_usd=$($portfolio.cash_usd)"

    $positions = $portfolio.positions
    $posSymbols = @()
    if ($positions) {
        $posSymbols = $positions.PSObject.Properties.Name
    }
    $passMax = ($posSymbols.Count -le $MaxPositions)
    Add-Result "portfolio: 銘柄数 <= MaxPositions" $passMax "count=$($posSymbols.Count) max=$MaxPositions"

    $qtyIssues = @()
    if ($positions) {
        foreach ($p in $positions.PSObject.Properties) {
            $qOk = $false
            $qVal = 0
            if ([int]::TryParse($p.Value.ToString(), [ref]$qVal)) {
                $qOk = ($qVal -ge 1)
            }
            if (-not $qOk) {
                $qtyIssues += "$($p.Name)=$($p.Value)"
            }
        }
    }
    $passQty = ($qtyIssues.Count -eq 0)
    Add-Result "portfolio: qty が正の整数" $passQty ("issues=" + ($qtyIssues -join ","))

    # selection / portfolio 比較
    $expected = @($keep + $buy) | Sort-Object -Unique
    $actual = $posSymbols | Sort-Object -Unique
    $diff = Compare-Object $expected $actual
    $passSet = ($diff.Count -eq 0)
    Add-Result "compare: positions == keep ∪ buy" $passSet ("diff=" + ($diff | ForEach-Object { $_.InputObject }) -join ",")

    $sellInPortfolio = $sell | Where-Object { $actual -contains $_ }
    $passSell = ($sellInPortfolio.Count -eq 0)
    Add-Result "compare: sell_symbols は portfolio に無し" $passSell ("sell_in_portfolio=" + ($sellInPortfolio -join ","))

    $cashDiff = $null
    $passCashDiff = $false
    if ($null -ne $selection.cash_after_exec -and $null -ne $portfolio.cash_usd) {
        $cashDiff = [math]::Abs([double]$selection.cash_after_exec - [double]$portfolio.cash_usd)
        $passCashDiff = ($cashDiff -le 0.01)
    }
    Add-Result "compare: cash_after_exec と cash_usd" $passCashDiff "diff=$cashDiff"

    # weekly log チェック
    $logLines = Get-Content $WeeklyLogPath
    $hasError = $false
    $hasComplete = $false
    $hasValidate = $false
    foreach ($line in $logLines) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $obj = $line | ConvertFrom-Json
        } catch {
            continue
        }
        if ($obj.level -eq "ERROR") {
            $hasError = $true
        }
        if ($obj.logger -eq "weekly" -and $obj.message -eq "complete") {
            $hasComplete = $true
        }
        if ($obj.message -match "validate" -or $obj.message -match "validation_failed") {
            $hasValidate = $true
        }
    }
    $passLog = (-not $hasError -and $hasComplete -and -not $hasValidate)
    Add-Result "weekly log: ERROR無し & complete" $passLog "error=$hasError complete=$hasComplete validate=$hasValidate"

    $okCount = ($script:results | Where-Object { $_.Pass }).Count
    $ngCount = ($script:results | Where-Object { -not $_.Pass }).Count
    return [pscustomobject]@{
        Results = $script:results
        OkCount = $okCount
        NgCount = $ngCount
    }
}

function Get-BundleFiles([string]$bundleDir) {
    $selection = Get-ChildItem -Path (Join-Path $bundleDir "selection_*.json") | Sort-Object Name | Select-Object -Last 1
    if ($null -eq $selection) {
        return $null
    }
    $stamp = $null
    if ($selection.Name -match 'selection_(\d{8})') {
        $stamp = $Matches[1]
    }
    $orders = $null
    if ($stamp) {
        $orders = Get-ChildItem -Path (Join-Path $bundleDir "orders_$stamp.csv") -ErrorAction SilentlyContinue | Select-Object -First 1
    }
    if ($null -eq $orders) {
        $orders = Get-ChildItem -Path (Join-Path $bundleDir "orders_*.csv") |
            Where-Object { $_.Name -notmatch "orders_candidates" } |
            Sort-Object Name |
            Select-Object -Last 1
    }
    $weeklyLog = $null
    if ($stamp) {
        $weeklyLog = Get-ChildItem -Path (Join-Path $bundleDir "weekly_${stamp}_*.jsonl") | Sort-Object Name | Select-Object -Last 1
    }
    if ($null -eq $weeklyLog) {
        $weeklyLog = Get-ChildItem -Path (Join-Path $bundleDir "weekly_*.jsonl") | Sort-Object Name | Select-Object -Last 1
    }
    return [pscustomobject]@{
        SelectionPath = $selection.FullName
        OrdersPath = if ($orders) { $orders.FullName } else { $null }
        WeeklyLogPath = if ($weeklyLog) { $weeklyLog.FullName } else { $null }
        PortfolioPath = (Join-Path $bundleDir "portfolio.json")
    }
}

function Get-NgStreak([string]$bundleRoot, [int]$lookback, [int]$maxPositions) {
    if (-not (Test-Path $bundleRoot)) {
        return $null
    }
    $bundleDirs = Get-ChildItem -Path $bundleRoot -Directory |
        Where-Object { $_.Name -match '^\d{8}$' } |
        Sort-Object Name
    if ($lookback -gt 0) {
        $bundleDirs = $bundleDirs | Select-Object -Last $lookback
    }

    $bundleResults = @()
    foreach ($dir in $bundleDirs) {
        $files = Get-BundleFiles $dir.FullName
        if ($null -eq $files) {
            $bundleResults += [pscustomobject]@{ Bundle = $dir.Name; Ok = $false }
            continue
        }
        if (
            -not $files.SelectionPath -or -not (Test-Path $files.SelectionPath) -or
            -not $files.OrdersPath -or -not (Test-Path $files.OrdersPath) -or
            -not $files.WeeklyLogPath -or -not (Test-Path $files.WeeklyLogPath) -or
            -not $files.PortfolioPath -or -not (Test-Path $files.PortfolioPath)
        ) {
            $bundleResults += [pscustomobject]@{ Bundle = $dir.Name; Ok = $false }
            continue
        }
        $check = Invoke-WeeklyCheck -SelectionPath $files.SelectionPath -OrdersPath $files.OrdersPath -WeeklyLogPath $files.WeeklyLogPath -PortfolioPath $files.PortfolioPath -MaxPositions $maxPositions
        $bundleResults += [pscustomobject]@{ Bundle = $dir.Name; Ok = ($check.NgCount -eq 0) }
    }

    $ngStreak = 0
    foreach ($item in ($bundleResults | Sort-Object Bundle -Descending)) {
        if ($item.Ok) {
            break
        }
        $ngStreak += 1
    }
    return [pscustomobject]@{
        NgStreak = $ngStreak
        Count = $bundleResults.Count
    }
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

$check = Invoke-WeeklyCheck -SelectionPath $SelectionPath -OrdersPath $OrdersPath -WeeklyLogPath $WeeklyLogPath -PortfolioPath $PortfolioPath -MaxPositions $MaxPositions
$okCount = $check.OkCount
$ngCount = $check.NgCount

foreach ($r in $check.Results) {
    if ($r.Pass) {
        Write-Host "[OK] $($r.Label)"
    } else {
        Write-Host "[NG] $($r.Label) :: $($r.Detail)"
    }
}
Write-Host "RESULT: OK=$okCount / NG=$ngCount"

if ($NgStreakThreshold -gt 0) {
    $resolvedBundleRoot = Resolve-RepoPath $BundleRoot
    $streak = Get-NgStreak -bundleRoot $resolvedBundleRoot -lookback $NgStreakLookback -maxPositions $MaxPositions
    if ($null -eq $streak) {
        Write-Host "NG_STREAK: 0 / THRESHOLD=$NgStreakThreshold / TRIGGER=False (bundle_root_missing)"
    } else {
        $trigger = ($streak.NgStreak -ge $NgStreakThreshold)
        Write-Host "NG_STREAK: $($streak.NgStreak) / THRESHOLD=$NgStreakThreshold / TRIGGER=$trigger"
    }
}

if ($ngCount -gt 0) {
    exit 1
}
exit 0
