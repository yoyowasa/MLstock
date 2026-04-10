$ErrorActionPreference = "Stop"

Set-Location "C:\BOT\MLStock"
$env:PYTHONPATH = "C:\BOT\MLStock\src"

& "C:\BOT\MLStock\.venv\Scripts\python.exe" "C:\BOT\MLStock\scripts\run_gap_trade.py" --skip-options --live
