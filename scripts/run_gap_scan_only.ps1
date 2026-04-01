$ErrorActionPreference = "Stop"

# ワークスペースに移動して設定解決を安定化する
Set-Location "C:\BOT\MLStock"
$env:PYTHONPATH = "C:\BOT\MLStock\src"

# タスクスケジューラは秒指定できないため、5秒待機して 09:30:05 相当にする
Start-Sleep -Seconds 5

& "C:\BOT\MLStock\.venv\Scripts\python.exe" "C:\BOT\MLStock\scripts\run_gap_trade.py" --scan-only
