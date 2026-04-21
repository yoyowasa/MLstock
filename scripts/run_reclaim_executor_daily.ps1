$ErrorActionPreference = "Stop"

Set-Location "C:\BOT\MLStock"
$env:PYTHONPATH = "C:\BOT\MLStock\src"

$python = "C:\BOT\MLStock\.venv\Scripts\python.exe"
$script = "C:\BOT\MLStock\strategies\gap_d1_0935\scripts\run_reclaim_executor.py"
$tradeDate = Get-Date -Format "yyyy-MM-dd"
$commonArgs = @(
    $script,
    "--trade-date", $tradeDate,
    "--dry-run",
    "--slippage-bps-per-side", "5.0",
    "--fee-bps-round-trip", "2.0"
)

& $python @($commonArgs + @("--branch", "reclaim_first5_high"))
& $python @($commonArgs + @("--branch", "reclaim_vwap"))
& $python @($commonArgs + @("--branch", "continuation_compare"))
