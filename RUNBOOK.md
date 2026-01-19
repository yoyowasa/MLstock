# RUNBOOK.md ? 運用・稼働手順

> 本ファイルは運用/稼働の手順・確認・コマンド集。仕様は `PROJECT_SPEC.md` を参照。

---

## クイックスタート
- 参照データ未作成時の初回セットアップ（新規環境/`data/reference` が空のとき。週次運用の初回ではない）：`.\.venv\Scripts\python scripts\run_setup_reference.py`
- 週次運用（selection/注文の更新 + 自動集約 + KPI更新）：`.\scripts\run_weekly.ps1`
- 週次運用（手動集約）：`.\.venv\Scripts\python scripts\run_weekly.py` → `.\scripts\weekly_bundle.ps1`
- 週次チェック（OK/NGを自動判定）：`.\scripts\weekly_check.ps1`
- 迷ったら見る順：**selection JSON → orders CSV → portfolio JSON → weekly log**（詳細は [Runbook](#runbook実行コマンド)）

---

## 実運用コマンド一覧（最低限）
- 週次実行（自動集約 + KPI更新）：`.\scripts\run_weekly.ps1`
- 週次チェック（OK/NG判定）：`.\scripts\weekly_check.ps1`.\scripts\weekly_check.ps1 -NgStreakThreshold 2

- 連続NGカウント（補助）：`.\.venv\Scripts\python scripts\run_ng_streak_check.py`
- 強制決済（SELLのみ・注文CSV作成）：`.\.venv\Scripts\python scripts\run_force_liquidation.py`
- KPI更新（deadband監視・手動実行用）：`.\.venv\Scripts\python scripts\run_deadband_kpi.py`
- 集約だけ手動実行：`.\scripts\weekly_bundle.ps1`

---

## Runbook（実行コマンド）
> 実行は `.\.venv\Scripts\python` を推奨（mlstock未検出の再発防止）

### 手順と、その都度確認するべきファイル（クイック）
> 迷ったら「**最新の selection JSON → orders CSV → portfolio JSON → weekly log**」の順に見る。

#### 初回セットアップ（参照データ未作成時）
> 週次運用の初回ではない。`data/reference` が未作成のときだけ実行する。
- 1) reference（assets/calendar）
  - 確認ファイル：`data/reference/assets.parquet`、`data/reference/calendar.parquet`
  - 中身確認（最短）
    ```powershell
    .\.venv\Scripts\python -c "import pandas as pd; p='data/reference/assets.parquet'; df=pd.read_parquet(p); print(p, df.shape); print(df.head(3).to_string(index=False))"
    .\.venv\Scripts\python -c "import pandas as pd; p='data/reference/calendar.parquet'; df=pd.read_parquet(p); print(p, df.shape); print(df.head(3).to_string(index=False))"
    ```
- 2) seed（seed_symbols）
  - 確認ファイル：`data/reference/seed_symbols.parquet`（**SPYが含まれる**）
- 3) raw backfill（bars_daily + corp_actions）
  - 確認ファイル：`data/raw/bars_daily/{SYMBOL}.parquet`、`data/raw/corp_actions/corp_actions.parquet`
- 4) weekly snapshots（week_map/universe/features/labels）
  - 確認ファイル：`data/snapshots/weekly/week_map.parquet`、`data/snapshots/weekly/universe.parquet`、`data/snapshots/weekly/features.parquet`、`data/snapshots/weekly/labels.parquet`
- 5) backtest（監査一式）
  - 確認ファイル：`artifacts/backtest/summary.json`、`artifacts/backtest/nav.parquet`、`artifacts/backtest/trades.parquet`、`artifacts/backtest/audit_*.json`、`artifacts/backtest/top_share_by_week.csv`

#### 週次（毎週）
- 0) 設定と鍵
  - 確認ファイル：`config/config.yaml`、`config/config.local.yaml`
  - 確認：環境変数 `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`
- 1) weekly run（増分→snapshots→train→predict→orders）
  - 主要出力：`artifacts/orders/selection_YYYYMMDD.json`、`artifacts/orders/orders_YYYYMMDD.csv`、`artifacts/state/portfolio.json`
  - 注：実約定と差分が出た場合は `portfolio.json` を手動で更新する
  - 補助出力：`artifacts/orders/orders_candidates_YYYYMMDD.csv`、`artifacts/models/model_YYYYMMDD.joblib`、`artifacts/models/pred_YYYYMMDD.parquet`、`artifacts/logs/weekly_YYYYMMDD_HHMMSS.jsonl`
  - 注：ファイル名の `YYYYMMDD` は `selection_*.json` の `as_of` と一致（**週の対象は `week_start`**）
- 2) monitoring（deadband KPI）
  - 確認ファイル：`artifacts/monitoring/deadband_weekly_kpi.csv`
- 3) monitoring（運用PNLの概算）
  - 確認ファイル：`artifacts/monitoring/portfolio_nav.csv`

### ファイルの見方（最短コマンド）
```powershell
# 最新ファイル（stamp追跡）
$sel = Get-ChildItem artifacts/orders/selection_*.json | Sort-Object Name | Select-Object -Last 1
$ord = Get-ChildItem artifacts/orders/orders_[0-9]*.csv | Sort-Object Name | Select-Object -Last 1
$log = Get-ChildItem artifacts/logs/weekly_*.jsonl       | Sort-Object Name | Select-Object -Last 1
$port = "artifacts/state/portfolio.json"
$sel.FullName; $ord.FullName; $port; $log.FullName

# selection（整形表示。ここを見るのが最優先）
Get-Content $sel.FullName -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 10

# orders（空でもOK。BUY/SELLはselectionの集合と整合しているべき）
Get-Content $ord.FullName

# portfolio（週次状態。cash/positionsの真実）
Get-Content $port -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 10

# weekly log（JSONL。失敗理由/validate NGの確認に使う）
Get-Content $log.FullName -Tail 50

# parquet（pandasでhead/columnsを見る）
.\.venv\Scripts\python -c "import pandas as pd; p='data/snapshots/weekly/features.parquet'; df=pd.read_parquet(p); print(p, df.shape); print(list(df.columns)[:20]); print(df.head(3).to_string(index=False))"
```

### 出力ファイルの項目説明（実運用向け）
> null/空欄は「機能が無効」または「データ欠損」を意味する。

#### selection_YYYYMMDD.json（週次サマリ）
- `as_of`：作成日（KPI/selection生成日）
- `week_start`：対象週の開始日
- `symbols`：買い候補の互換フィールド（基本は `buy_symbols` と同じ）
- `buy_symbols`：BUY予定の銘柄一覧
- `sell_symbols`：SELL予定の銘柄一覧
- `keep_symbols`：保有継続の銘柄一覧
- `target_symbols`：目標保有の優先順位リスト（上位から買い）
- `n_selected`：BUY数
- `cash_start_usd`：週次開始時点の現金
- `cash_reserve_usd`：常に残す現金
- `cash_est_before_buys`：SELL反映後の現金見積
- `cash_est_after_buys`：BUY反映後の現金見積
- `skipped_buys_insufficient_cash`：資金不足で飛ばしたBUY数
- `buy_fill_policy`：資金不足時の埋め方（ranked_partial/strict）
- `estimate_entry_buffer_bps`：買付見積の上乗せ幅（bps）
- `missing_sell_prices`：売値欠損でSELL判定できなかった数
- `deadband_v2_enabled`：deadband適用の有無
- `deadband_abs`：deadbandの絶対値閾値
- `deadband_rel`：deadbandの相対値閾値
- `min_trade_notional`：最小取引額の閾値
- `sum_abs_dw_raw`：deadband適用前の注文量（重み）絶対値合計
- `sum_abs_dw_filtered`：deadband適用後の注文量（重み）絶対値合計
- `deadband_notional_reduction`：deadband削減率（0=削減なし）
- `filtered_trade_fraction_notional`：notionalベースの削減率
- `filtered_trade_fraction`：notionalベース削減率の互換フィールド
- `filtered_trade_fraction_count`：件数ベースの削減率
- `trade_count_raw`：deadband適用前の注文件数
- `trade_count_filtered`：deadband適用後の注文件数
- `turnover_ratio_std`：買い側turnover（互換。現行はbuyと同値）
- `turnover_ratio_buy`：買い側turnover
- `turnover_ratio_sell`：売り側turnover
- `turnover_ratio_total_abs`：buy+sell合計
- `turnover_ratio_total_half`：total_absの半分
- `cash_after_exec`：取引後の現金見積（通常は `cash_est_after_buys` と同値）
- `kept_positions`：deadband等で保持した銘柄数
- `held_positions`：取引後の想定保有銘柄数
- `data_max_features_date`：featuresの最終週
- `data_max_labels_date`：labelsの最終週
- `data_max_week_map_date`：week_mapの最終週
- `regime_gate.enabled`：gate機能ON/OFF
- `regime_gate.active`：gate発動中か（open=falseでtrue）
- `regime_gate.open`：取引許可状態
- `regime_gate.rule`：gate判定ルール名
- `regime_gate.action`：発動時の動作（例：no_trade）
- `regime_gate.source`：判定の参照データ
- `regime_gate.ma_days`：判定に使うMA期間
- `regime_gate.pred_return_floor`：予測下限（未使用ならnull）
- `vol_cap.enabled`：vol cap適用の有無
- `vol_cap.mode`：hard/soft/penalty
- `vol_cap.penalty_min`：soft系の下限
- `vol_cap.apply_stage`：適用ステージ（training/selection）
- `vol_cap.apply_to_training`：学習に適用したか
- `vol_cap.apply_to_selection`：選定に適用したか
- `vol_cap.feature_name`：ボラ判定に使う特徴量
- `vol_cap.rank_threshold`：上位何%を残すか
- `vol_cap.candidates`：候補数
- `vol_cap.excluded`：除外数（hard時）
- `vol_cap.missing`：欠損数
- `vol_cap.penalized`：罰則適用数（soft時）
- `exposure_guard.enabled`：exposure guard有効
- `exposure_guard.active`：exposure guard発動中
- `exposure_guard.trigger`：発動条件
- `exposure_guard.mode`：ガード方式
- `exposure_guard.base_source`：基準の参照元
- `exposure_guard.base_scale`：基準スケール
- `exposure_guard.cap_source`：上限計算の参照元
- `exposure_guard.cap_value`：上限値
- `exposure_guard.cap_buffer`：上限バッファ
- `exposure_guard.cap`：算出上限
- `exposure_guard.cap_enabled`：上限適用有無
- `exposure_guard.gross_exposure_raw`：ガード前の総エクスポージャ
- `exposure_guard.scale`：適用スケール
- `exposure_guard.gross_exposure_guarded`：ガード後の総エクスポージャ
- `exposure_guard.cash_est_after_buys_guarded`：ガード後の現金見積

#### orders_YYYYMMDD.csv（注文一覧）
> 空ファイルは注文ゼロ週。ヘッダー無しで空のまま。
- `side`：buy/sell
- `symbol`：銘柄
- `qty`：株数
- `type`：注文種別（market）
- `time_in_force`：有効期限（day）
- `priority`：BUYの優先順位（SELLは空）
- `est_price`：BUYの見積価格（SELLは空）
- `est_cost`：BUYの見積コスト（SELLは空）
- `required_est`：BUYの必要資金見積（SELLは空）

#### orders_candidates_YYYYMMDD.csv（候補一覧）
- `rank`：候補順位
- `symbol`：銘柄
- `pred_return`：予測リターン
- `est_price`：見積価格
- `est_cost`：見積コスト
- `required_est`：必要資金見積
- `budget_ok`：予算内かどうか
- `selected`：実際にBUYしたか

#### portfolio.json（週次ポジション）
- `as_of`：更新日
- `week_start`：対象週
- `cash_usd`：現金残高（USD）
- `positions`：銘柄→株数（qty>0のみ）

#### weekly_YYYYMMDD_HHMMSS.jsonl（週次ログ）
- `ts_utc`：時刻（UTC）
- `level`：INFO/WARN/ERROR
- `logger`：logger名（weekly）
- `message`：イベント名（start/complete/validation_failed など）
- `fields`：追加情報（buys/sells/orders など）

#### deadband_weekly_kpi.csv（KPI時系列）
- `selection_file`：集計元のselectionファイル名
- `as_of`：KPI作成日
- `week_start`：対象週の開始日
- `deadband_v2_enabled`：deadband適用の有無
- `deadband_abs`：deadbandの絶対値閾値
- `deadband_rel`：deadbandの相対値閾値
- `min_trade_notional`：最小取引額の閾値
- `sum_abs_dw_raw`：deadband適用前の注文量（重み）絶対値合計
- `sum_abs_dw_filtered`：deadband適用後の注文量（重み）絶対値合計
- `deadband_notional_reduction`：deadband削減率
- `filtered_trade_fraction_notional`：notionalベース削減率
- `filtered_trade_fraction`：notionalベース削減率の互換フィールド
- `filtered_trade_fraction_count`：件数ベース削減率
- `trade_count_raw`：deadband適用前の注文件数
- `trade_count_filtered`：deadband適用後の注文件数
- `turnover_ratio_std`：買い側turnover（互換）
- `turnover_ratio_buy`：買い側turnover
- `turnover_ratio_sell`：売り側turnover
- `turnover_ratio_total_abs`：buy+sell合計
- `turnover_ratio_total_half`：total_absの半分
- `cash_after_exec`：取引後の現金見積
- `cash_start_usd`：週次開始時点の現金
- `cash_est_before_buys`：SELL反映後の現金見積
- `cash_est_after_buys`：BUY反映後の現金見積
- `n_selected`：BUY数
- `kept_positions`：保持した銘柄数
- `held_positions`：想定保有銘柄数
- `skipped_buys_insufficient_cash`：資金不足で飛ばしたBUY数
- `data_max_features_date`：featuresの最終週
- `data_max_labels_date`：labelsの最終週
- `data_max_week_map_date`：week_mapの最終週
- `kpi_check_status`：OK/WARN/NG（NGは取引停止）
- `kpi_check_notes`：判定理由コード（`;`区切り）
- `kpi_check_detail`：判定理由の説明（`code:理由` を `;` 区切り）

### 実運用で見るポイント（最短チェックリスト）
- [ ] selection：OK＝`as_of` が `selection_YYYYMMDD` と一致、`week_start` が `as_of` の同一週（月曜開始）に入る、`data_max_features_date`/`data_max_week_map_date` が `week_start` 以上、`data_max_labels_date` は `week_start` の1週前まで許容、`n_selected == len(buy_symbols)`、`buy/sell/keep` の交差が空（NGなら取引停止）
- [ ] orders：OK＝各行の `side` は buy/sell、`qty >= 1`、`symbol` は selection の `buy_symbols`/`sell_symbols` に含まれる、空ファイルは selection の buy/sell が空のときのみ（NGなら取引停止）
- [ ] portfolio：OK＝`as_of`/`week_start` が selection と一致、`cash_usd >= 0`、`positions` の銘柄数が上限（`K_max`/現行15）以下、全qtyが正の整数（NGなら取引停止）
- [ ] weekly log：OK＝`level=ERROR` が無い、`message=complete` で終了（`WARN` は注意、ERROR/未完了は取引停止）

### selection / portfolio 見比べ（必須）
- OK：`portfolio.positions` の銘柄集合 == `selection.keep_symbols ∪ selection.buy_symbols`
- OK：`selection.sell_symbols` が `portfolio.positions` に含まれない
- OK：`abs(selection.cash_after_exec - portfolio.cash_usd) <= 0.01`
- NG：上記のどれかが外れたら取引停止

**見比べの最短コマンド**
```powershell
$selFile = Get-ChildItem artifacts/orders/selection_*.json | Sort-Object Name | Select-Object -Last 1
$portFile = "artifacts/state/portfolio.json"
$sel = Get-Content $selFile.FullName -Raw | ConvertFrom-Json
$port = Get-Content $portFile -Raw | ConvertFrom-Json
$expected = @($sel.keep_symbols + $sel.buy_symbols) | Sort-Object -Unique
$actual = $port.positions.PSObject.Properties.Name | Sort-Object
Compare-Object $expected $actual
($sel.sell_symbols | Where-Object { $actual -contains $_ })
[math]::Abs($sel.cash_after_exec - $port.cash_usd)
```

### validate NG の代表メッセージ例（見つけたら取引停止）
- `validation_failed`（例：`ingest_calendar`/`ingest_assets` のログ）
- `Calendar validation failed` / `Assets validation failed`
- `Snapshots features are empty`
- `Not enough training weeks for weekly run`
- `No eligible symbols for weekly selection`
- `No eligible symbols after vol cap filtering`
- `No eligible symbols after min_proba thresholds`

### 週次確認ファイルのまとめ（フォルダ横断）
> 週次実行後に `scripts/weekly_bundle.ps1` で自動集約する。

**自動集約（推奨）**
```powershell
.\scripts\weekly_bundle.ps1
```

**パスをまとめて表示**
```powershell
$sel = Get-ChildItem artifacts/orders/selection_*.json | Sort-Object Name | Select-Object -Last 1
$ord = Get-ChildItem artifacts/orders/orders_[0-9]*.csv | Sort-Object Name | Select-Object -Last 1
$log = Get-ChildItem artifacts/logs/weekly_*.jsonl       | Sort-Object Name | Select-Object -Last 1
$port = "artifacts/state/portfolio.json"
$sel.FullName; $ord.FullName; $port; $log.FullName
```

**1フォルダに集約（手動でやる場合）**
```powershell
$sel = Get-ChildItem artifacts/orders/selection_*.json | Sort-Object Name | Select-Object -Last 1
$ord = Get-ChildItem artifacts/orders/orders_[0-9]*.csv | Sort-Object Name | Select-Object -Last 1
$log = Get-ChildItem artifacts/logs/weekly_*.jsonl       | Sort-Object Name | Select-Object -Last 1
$port = "artifacts/state/portfolio.json"
$stamp = [IO.Path]::GetFileNameWithoutExtension($sel.Name).Split('_')[-1]
$dest = "artifacts/weekly_bundle/$stamp"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item $sel.FullName, $ord.FullName, $port, $log.FullName -Destination $dest
Get-ChildItem $dest
```

### 初回セットアップ（参照データ未作成時）
> 週次運用の初回ではない。以後は assets/calendar を更新したいときのみ実行する。
1) reference
```powershell
.\.venv\Scripts\python scripts\run_setup_reference.py
```
2) seed
```powershell
.\.venv\Scripts\python -m mlstock make-seed --n-seed 2000
```
3) raw backfill
```powershell
.\.venv\Scripts\python scripts\run_backfill_raw.py
```
4) weekly snapshots
```powershell
.\.venv\Scripts\python scripts\run_build_snapshots.py
```
5) backtest（監査一式生成）
```powershell
.\.venv\Scripts\python scripts\run_backtest.py --start 2020-07-27 --end 2025-12-15
```

### 週次運用
```powershell
# 週次実行 + 自動集約 + KPI更新
.\scripts\run_weekly.ps1

# 週次実行のみ（手動で集約する場合）
.\.venv\Scripts\python scripts\run_weekly.py
.\scripts\weekly_bundle.ps1
```

### 週次チェック（自動判定）
```powershell
.\scripts\weekly_check.ps1
```
> NGが1つでも出たら取引停止。`[NG]` の行を確認する。

連続NGを数える場合（2週連続で強制決済判断など）:
```powershell
.\scripts\weekly_check.ps1 -NgStreakThreshold 2
```

### 連続NGのカウント（補助）
```powershell
.\.venv\Scripts\python scripts\run_ng_streak_check.py
```
- 出力例：`ng_streak=2 threshold=2 trigger_liquidation=True`

### 連続NG時の強制決済（任意）
- 週次チェックが連続2週NGなら、次の寄りで **全決済のみ**（BUYなし）を実行してリセットする
- 注文CSV生成（`portfolio.json` からSELLのみ作成）
  ```powershell
  .\.venv\Scripts\python scripts\run_force_liquidation.py
  ```
  - `--update-portfolio` を付けると `portfolio.json` を現金のみへ更新（**約定後に実行**）
  - 必要なら `--cash-usd` と `--as-of` で現金と日付を指定できる
  - 出力：`artifacts/orders/orders_force_YYYYMMDD.csv`
 - 約定後は `artifacts/state/portfolio.json` を現金のみで更新する（`--update-portfolio` でも可）

### KPI更新（deadband）
- `selection_*.json` の履歴から集計し、`artifacts/monitoring/deadband_weekly_kpi.csv` を更新する
- `.\scripts\run_weekly.ps1` で **自動更新**（必要なら手動実行も可能）
- 目的は「deadbandが効き過ぎ/効かなさ過ぎ」を週次で検知すること

**KPIの見方（初心者向け・判断のしかた）**
- 1行=1週。`week_start` はその週の開始日、`as_of` はKPI作成日
- `deadband_v2_enabled`/`deadband_abs`/`deadband_rel`/`min_trade_notional`：設定値の写し。configと違うなら設定ミス
- `deadband_notional_reduction`：deadbandで削れた割合（0=削れていない）。0.10超が連発→効きすぎ疑い（追従不足）
- `trade_count_raw` と `trade_count_filtered`：適用前/後の件数。`filtered <= raw` が正常、逆転は異常
- `turnover_ratio_total_abs`：`turnover_ratio_buy + turnover_ratio_sell` と一致が必須。不一致は計算異常
- `data_max_features_date`/`data_max_week_map_date`：`week_start` より古い→データ停止（取引停止）

### 運用PNL（概算）
- `artifacts/monitoring/portfolio_nav.csv` を週次で更新（`run_weekly.py` 実行時に自動生成）
- 価格は **week_start の features.price** を使用（実約定のOPENとは差分が出る可能性あり）
- 週次の寄り約定を想定した概算。実約定価格との差分は誤差として扱う
- 列：`as_of` / `week_start` / `cash_usd` / `positions_value` / `nav` / `positions_count` / `missing_prices` / `nav_prev` / `weekly_pnl` / `pct_change`

**見方（最短）**
- `nav = cash_usd + positions_value`
- `weekly_pnl = nav - nav_prev`、`pct_change = weekly_pnl / nav_prev`
- `missing_prices > 0` の週は価格欠損があるためPNLは参考値
- 初回週は `nav_prev` が空になる（比較できない）

**各項目の説明（運用で見るための意味）**
- `as_of`：このCSVを作成した日付（作業日）
- `week_start`：対象週の開始日（この週のポジション評価に使う基準日）
- `cash_usd`：週次実行後の現金残高（USD）
- `positions_value`：`week_start` 時点の株価で評価した保有株の合計金額
- `nav`：総資産（`cash_usd + positions_value`）
- `positions_count`：保有銘柄数（qty>0 の銘柄数）
- `missing_prices`：価格が取れず評価できなかった銘柄数（>0ならPNLは参考値）
- `nav_prev`：1週前の `nav`（初回は空）
- `weekly_pnl`：週次損益（`nav - nav_prev`）
- `pct_change`：週次損益率（`weekly_pnl / nav_prev`）

**自動判定カラム（run_deadband_kpi.py で自動付与）**
- `kpi_check_status`：`OK`/`WARN`/`NG`（`NG` は取引停止）
- `kpi_check_notes`：判定理由コード（`;`区切り）
- `kpi_check_detail`：判定理由の説明（`code:理由` を `;` 区切り）
  - `features_date_old`：`data_max_features_date < week_start`
  - `week_map_date_old`：`data_max_week_map_date < week_start`
  - `labels_date_old`：`data_max_labels_date` が1週以上古い
  - `data_max_missing`：`data_max_*` が欠落
  - `trade_count_filtered_gt_raw`：件数が逆転
  - `turnover_mismatch`：`total_abs != buy + sell`
  - `turnover_missing`：turnover値が欠落
  - `deadband_reduction_high`：`deadband_notional_reduction >= 0.10`
  - `deadband_reduction_missing`：値が欠落
  - `week_start_missing`：`week_start` 欠落

### Execution Deadband v2 週次チェックリスト（運用・1ページ版）
> 毎週 `run_weekly.py` 実行後に、**データ鮮度／設定／注文整合／turnover分解／deadband効き具合／gate状態** を最短で確認し、異常時は kill switch で即回避する。

**参照する出力**
- `artifacts/orders/selection_YYYYMMDD.json`（週次サマリ・設定・指標・symbol集合）
- `artifacts/orders/orders_YYYYMMDD.csv`（生成注文）
- `artifacts/monitoring/deadband_weekly_kpi.csv`（週次KPI時系列。`run_deadband_kpi.py` で更新）

**定型コマンド**
```powershell
# 1) 週次実行（データ増分 → snapshots → selection/orders）
.\.venv\Scripts\python scripts\run_weekly.py

# 2) 週次KPI更新（selection履歴 → CSV再集計）
.\.venv\Scripts\python scripts\run_deadband_kpi.py

# 3) 最新の週次サマリ/注文（stamp確認用）
Get-ChildItem artifacts/orders/selection_*.json | Sort-Object Name | Select-Object -Last 1
Get-ChildItem artifacts/orders/orders_[0-9]*.csv | Sort-Object Name | Select-Object -Last 1
```

**チェックリスト（OK/NGを即判定）**
- [ ] 1) **データ鮮度（最優先）**：`week_start` が直近、`data_max_features_date`/`data_max_labels_date`/`data_max_week_map_date` が更新され `week_start` と整合（止まっていたら売買判断は保留推奨）
- [ ] 2) **deadband v2 設定**：`deadband_v2_enabled==true`、`deadband_abs==0.0025`、`deadband_rel==0.0`、`min_trade_notional==0.0`（異常時は `execution.deadband_v2.enabled=false` で即OFF）
- [ ] 3) **注文と集合の整合**：`orders_*.csv` の buy/sell が `buy_symbols`/`sell_symbols` に含まれる、`keep_symbols` と `sell_symbols` が不自然に重ならない（注文ゼロ週は `orders.csv` 空でOK）
- [ ] 4) **turnover分解（監視の本命）**：`turnover_ratio_total_abs == turnover_ratio_buy + turnover_ratio_sell`、`turnover_ratio_total_half == 0.5*turnover_ratio_total_abs`
- [ ] 5) **売りだけ週（必須の正常系）**：`turnover_ratio_buy==0` でも `turnover_ratio_sell>0` かつ `turnover_ratio_total_abs>0`（`turnover_ratio_std==0` は仕様上OK）
- [ ] 6) **deadband効き具合**：`deadband_notional_reduction ≈ 1-(filtered/raw)`（raw>0）、`trade_count_filtered<=trade_count_raw`、`filtered_trade_fraction_count` は 0〜1（ゼロ割しない）
- [ ] 7) **gate状態**：`regime_gate.enabled` と `regime_gate.active` を混同しない（`enabled=false` のとき `active=false`、`active=true` のときのみ `action` が運用に影響）

**警戒ライン（目安）**
- `deadband_notional_reduction` が **10%超** が連発 → 効きすぎ（追従不足の疑い）
- `filtered_trade_fraction_count` が **70%超** が連発 → 取引を止めすぎの疑い
- `cash_after_exec` が急増し、注文ゼロ週が続く → 候補不足／データ不足／deadband効きすぎの疑い

**スモーク（推奨：変更時・違和感時）**
```powershell
# kill switch（OFF同値）
.\.venv\Scripts\python -m pytest tests\test_deadband_kill_switch.py -k off_smoke

# 売りだけ週（再集計で売り>0/総量>0 を確認）
.\.venv\Scripts\python -m pytest tests\test_deadband_sell_only_week_smoke.py
```

**即時回避（kill switch）**
- `config.local.yaml` で `execution.deadband_v2.enabled: false` にして素通し（監視/ログは継続、execution変換のみ即停止）

### Deadband v2 運用前チェック（OFF同値/監視KPI/カナリア）
1) OFF同値テスト（最低限）
```powershell
.\.venv\Scripts\python -m pytest tests\test_deadband_kill_switch.py -k off_smoke
```
2) （任意）rollingのゴールデン確認
```powershell
.\.venv\Scripts\python -m pytest tests\test_deadband_golden_metrics.py
```
3) 監視KPI定型出力（週次の履歴をCSV化）
```powershell
.\.venv\Scripts\python scripts\run_deadband_kpi.py
```
4) カナリア → 段階拡大（例）
   - `config.local.yaml` で `selection.max_positions` を 5→10→15 の順に2週ずつ
   - 週次で `artifacts/monitoring/deadband_weekly_kpi.csv` を確認
   - 異常時は `execution.deadband_v2.enabled=false` で即OFF

### compare（例）
- regime
```powershell
.\.venv\Scripts\python scripts\run_backtest.py --start 2022-01-01 --end 2023-12-31 --compare-regime
```
- volcap（gate ON基準）
```powershell
.\.venv\Scripts\python scripts\run_backtest.py --start 2020-07-27 --end 2025-12-15 --compare-volcap --thr 0.70
```

---
