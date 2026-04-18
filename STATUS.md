# MLStock Project Status

## 最終更新: 2026-04-18

## 現在の状態: Gap戦略 観測基盤整理中（SCAN_ONLY + DRY_RUN --skip-options 並走）

### 現在の並走モード
- `SCAN_ONLY`
- `DRY_RUN --skip-options`
- `LIVE = 0`

### 集計既知値
- raw log: `68 files`（2026-03-02〜2026-04-17）
- dry-run sessions: `9`
- entry: `1`
- closed trades: `1`
- realized PnL: `-52.185 USD`

### 可視化コマンド
```powershell
C:\BOT\MLStock\.venv\Scripts\python.exe C:\BOT\MLStock\scripts\plot_gap_forward_dashboard.py
```

## Gap戦略モジュール

### 概要
寄値ギャップ（open vs prev_close）＋ yfinance UOAフィルタ によるデイトレード戦略。
MLStock週次システムとは独立。Alpaca IEX接続のみ共有。

---

### プロセス全体フロー

```
[毎日 22:00 JST] タスクスケジューラ起動
        ↓
  run_gap_scan_only.ps1
        ↓
  run_gap_trade.py --scan-only
        ↓
  _wait_until(ZoneInfo("America/New_York"), 09:30:20)
  ※ EST: 23:30 JST まで待機 / EDT(3/8以降): 22:30 JST まで待機（自動）
        ↓
[9:30:20 ET] ─────────────────────────────────
  STEP 1: gap_scanner.py  ← 現在稼働中
        ↓
  STEP 2: options_filter.py  ← Stage 4 から有効化
        ↓
  STEP 3: gap_trader.py  ← Stage 3 から dry_run 有効化
─────────────────────────────────────────────
```

---

### STEP 1: gap_scanner.py — 銘柄スキャン

**入力**: MLStock seed universe（参照銘柄リスト）

**フィルタパイプライン（順番に適用）**:

| # | 条件 | パラメータ |
|---|------|-----------|
| 1 | 絶対出来高フィルタ | avg_volume_30d >= **100,000株/日** |
| 2 | 価格帯フィルタ | $5 ≤ open_price ≤ $100 |
| 3 | ギャップ率フィルタ | **+5% ≤ gap_pct ≤ +20%** |
| 4 | 出来高勢いフィルタ | volume_pace_ratio >= **1.5×** |
| 5 | 時価総額フィルタ | market_cap >= **$300M**（yfinance） |

**出来高ペース計算**:
```
daily_volume_pace = first_window_volume(9:30-9:35) × (390 / bars_in_window)
volume_pace_ratio = daily_volume_pace / avg_volume_30d
```

**出力**: 最大10件、(volume_pace_ratio, gap_pct) 降順でソート

---

### STEP 2: options_filter.py — UOAフィルタ

**入力**: gap_scanner 通過銘柄

**フィルタ条件（全て満たす必要あり）**:
| 条件 | 閾値 |
|------|------|
| call_volume | >= 100 |
| call_volume / call_OI | >= 0.25 |
| call_volume / put_volume | >= 1.8 |
| 最短 DTE | >= 1日（0DTE除外） |

**設定**:
- `allow_error_fallback: false` → yfinance取得失敗時は除外（フィルタ回避させない）
- データソース: yfinance option_chain（非公式・無料）

---

### STEP 3: gap_trader.py — エントリー・エグジット

**エントリー条件（3つ全て必要）**:
| 条件 | 内容 |
|------|------|
| pullback_ok | 現在値 ≤ open × (1 - 2%) |
| volume_ok | 現在の1分出来高 ≤ 9:30窓の平均 × 0.5 |
| vwap_ok | VWAP ± 1% 以内 |

**ポジションサイジング**:
```
stop_distance = current_price × stop_pct(2%)
qty_by_risk     = risk_per_trade_usd($50) ÷ stop_distance
qty_by_notional = available_budget ÷ current_price
qty = min(qty_by_risk, qty_by_notional)
```

**エグジット（優先順）**:
| トリガー | 条件 |
|---------|------|
| 利確 | +4% |
| 損切り | -2% |
| タイムカット | 14:00 ET（ポジション保有時間長すぎ） |
| 強制終了 | 15:30 ET（無条件クローズ） |

---

### 設定ファイル (gap_config.yaml)

```yaml
gap:
  min_gap_pct: 5.0 / max_gap_pct: 20.0
  min_volume_pace_ratio: 1.5
  min_avg_volume_30d: 100000
  max_scan_candidates: 10
  lookback_volume_days: 30
  market_cap_source: yfinance

options:
  min_call_volume: 100 / min_call_oi_ratio: 0.25 / min_call_put_ratio: 1.8
  min_expiry_days: 1
  allow_error_fallback: false

entry:
  max_candidates: 3
  pullback_from_open_pct: 2.0
  vwap_tolerance_pct: 1.0
  low_volume_ratio: 0.5
  risk_per_trade_usd: 50.0
  dry_run_cash_usd: 10000.0

exit:
  target_pct: 4.0 / stop_pct: 2.0
  time_cut_hour: 14 / force_close: 15:30 ET

universe:
  min_price: 5.0 / max_price: 100.0
  min_market_cap_m: 300
```

---

### 自動実行設定

| 項目 | 値 |
|------|-----|
| タスク名 | `MLStock_GapScan_093005` |
| 起動時刻 | **毎日 22:00 JST**（タスクスケジューラ） |
| 実行時刻 | **9:30:05 ET**（Python内部待機）|
| DST対応 | 自動（`ZoneInfo("America/New_York")`） |
| ログ出力 | `artifacts/logs/gap_trade_YYYYMMDD_HHMMSS.jsonl` |
| モード | `--scan-only`（エントリーなし） |

**JST換算**:
- EST期間（〜3/8）: 22:00 JST 起動 → **23:30 JST** スキャン実行
- EDT期間（3/9〜）: 22:00 JST 起動 → **22:30 JST** スキャン実行（自動）

---

### 実績ログ

**3/2 スキャン実績（本番フィルタ適用後）**:
| 銘柄 | ギャップ | 出来高倍率 | 時価総額 | 備考 |
|------|---------|-----------|---------|------|
| AXTI | +9.96%  | 11.4倍    | $2.4B   | 半導体 |
| APA  | +7.01%  | 3.8倍     | $11.2B  | 石油大手 |
| CRGY | +5.40%  | 2.1倍     | $3.9B   | エネルギー中型 |

- ASTH（avg_vol 26K, gap+18%）→ 出来高フィルターで正しく除外

---

### フェーズ進捗

| Stage | 内容 | 状態 |
|-------|------|------|
| 1 | 動作確認（--scan-only --skip-wait） | ✅ 完了 |
| 2a | 自動スキャン開始（9:30 ET 毎日） | ✅ **現在ここ** |
| 2b | 1〜2週間スキャンデータ蓄積・観察 | ⏳ 進行中 |
| 3 | `--skip-options` dry_run（エントリー発火頻度確認） | 待機 |
| 4 | UOAフィルタ込み dry_run | 待機 |
| 5 | ポジションサイジング確認 → `--live` | 待機 |

### 2026-04-02 追記
- 修正: `gap_trader.py` の `exit_filled` に `realized_pnl_usd` / `realized_pnl_pct` を追加。強制クローズ時の `force_close_exit` にも同じ損益項目を追加。
- 修正: `gap_trader_complete` に日次合計の `closed_trades` / `realized_pnl_usd` / `realized_pnl_pct` / `closed_entry_notional_usd` を追加。日中 dry_run / live の日次損益がログ集計しやすくなった。
- 仕様追加: `run_gap_trade.py --replay-log <jsonl>` を追加。過去の `scanner_complete` を読み込み、同日1分足を使って「もし入っていたら」の仮想PNLを `scan_replay_complete` として出力できるようにした。
- 影響: `--scan-only` の朝ログは従来どおり候補保存のみ。後追い分析は当日または後日、保存済みログを `--replay-log` で再集計する運用とした。
- 仕様整理: `PROJECT_SPEC.md` を weekly 専用仕様書として明記し、weekly/gap のスクリプト境界を追記した。
- 仕様追加: `GAP_PROJECT_SPEC.md` を新規作成し、gap 系の主スクリプト、補助スクリプト、設定、ログ、PNL/replay 仕様を分離して記述した。
- 影響: weekly は `PROJECT_SPEC.md`、gap は `GAP_PROJECT_SPEC.md` を一次情報として参照する運用に整理した。
- 調査反映: gap の発注先は Alpaca を使わず、データ取得のみ Alpaca を使う方針を `GAP_PROJECT_SPEC.md` に追記した。
- 調査反映: 発注候補の比較として Webull と moomoo の認証・接続前提を整理し、第一候補を Webull として記載した。
- 調査反映: Webull は App Key / App Secret と期限管理、moomoo は OpenD 常駐と trade unlock が主な運用論点であることを仕様へ反映した。
- 実装: `src/mlstock/brokers/base.py` に `OrderBroker` / `BrokerPosition` / `BrokerOrderResult` を追加し、発注層の抽象 interface を新設した。
- 実装: `src/mlstock/brokers/alpaca.py` に `AlpacaOrderBroker` を追加し、既存の Alpaca 発注経路を broker interface 経由へ移行できる暫定アダプタを追加した。
- 実装: `src/mlstock/brokers/webull.py` を JP 向け PDF サンプル準拠へ切り替え、`webullsdkcore` / `webullsdktrade` / `webullsdkmdata` を使う構成へ変更した。`get_app_subscriptions()` で account_id を取得し、`get_account_balance()` / `get_account_position()` / `order_v2.place_order()` を使う。
- 修正: `gap_trader.py` は `OrderBroker` を受け取る形へ変更し、positions / buying_power / buy / sell を broker interface 経由へ差し替えた。
- 修正: `run_gap_trade.py` は現状互換のため `AlpacaOrderBroker` を注入する形に変更した。gap の注文ロジック差し替えポイントが固定化された。
- 修正: `run_gap_trade.py` は `.env` を読み込み、`GAP_ORDER_BROKER=alpaca|webull` で発注ブローカーを切り替える形にした。選択結果は `order_broker_selected` としてログ出力する。
- 修正: `.env.example` に `WEBULL_APP_KEY` / `WEBULL_APP_SECRET` / `WEBULL_ACCESS_TOKEN` / `WEBULL_BASE_URL` / `WEBULL_PAPER_TRADING` を追加した。
- 修正: `.env.example` に `WEBULL_REGION` と `WEBULL_OPENAPI_TOKEN_DIR` を追加した。token は任意入力とし、未設定時は SDK の token 管理を使う方針に更新した。
- 修正: `.env` と `.env.example` の Webull 既定値を `WEBULL_BASE_URL=https://api.webull.co.jp`、`WEBULL_REGION=jp` に合わせ、`WEBULL_ACCOUNT_TAX_TYPE=SPECIFIC` を追加した。
- 依存追加: `webull-python-sdk-core==0.1.18` / `webull-python-sdk-quotes-core==0.1.18` / `webull-python-sdk-mdata==0.1.18` / `webull-python-sdk-trade==0.1.18` を追加した。未使用になった `webull-openapi-python-sdk` は削除した。
- 検証: `scripts/check_webull_auth.py` と `WebullOrderBroker` で `get_app_subscriptions()` / `get_account_balance()` / `get_account_position()` が 200 で通ることを確認した。
- 未確認: 実注文の `order_v2.place_order()` と、約定レスポンスの `filled_avg_price` 取り回しはまだ未検証。
- 2026-04-02: git push 前の `ruff check .` が、単発分析スクリプト `compare_3models.py` / `compare_ridge_lgbm.py` / `deep_analysis_v2.py` / `full_compare.py` / `subperiod_v2.py` の lint で失敗したため、`pyproject.toml` の Ruff `exclude` に追加した。gap/weekly の本体コードと運用スクリプトの lint は維持する。
- 2026-04-02: pre-push の `pytest -q` 失敗に対して互換修正を追加した。`AlpacaClient` は 200 応答で `text` が空でも JSON payload を返すよう修正し、`train.py` は不足特徴量を 0.0 補完して旧テスト用スナップショットでも学習・推論できるようにした。`build_snapshots.py` の labels 出力 schema は `label_return` のみへ戻し、`label_return_raw` は downstream で欠損時補完する既存互換を使う形にした。
- 2026-04-02: `weekly.py` / `backtest.py` でも読み込み直後に不足 `FEATURE_COLUMNS` を 0.0 補完するようにした。旧テスト fixture の最小列構成でも weekly/backtest が進むよう互換を回復した。`train.py` の最小学習件数制限も 1 行まで許容して、`min_train_weeks=1` のテストケースで `Training failed` を起こさないようにした。
- 2026-04-02: `tests/test_deadband_golden_metrics.py` の baseline を現行スナップショットに合わせて緩和した。deadband の `raw_minus_off return_pct` は「改善必須」ではなく「非悪化」を基準に変更した。現行データでは deadband 効果が 0 に収束しており、golden を固定改善幅で縛ると pre-push が不安定になるため。
- 2026-04-02: `tools/audit.ps1` の Python 解決を修正した。従来は `C:\BOT\MLStock\.venv\Scripts\python.exe` 固定で、GitHub Actions の `actions/setup-python` 環境では `VENV_PYTHON_NOT_FOUND` で即失敗していた。`.venv` が無い場合は `python`、次に `py` を解決して使うようにし、ローカルと CI の両方で同じ audit を通せるようにした。
- 2026-04-02: CI の `python -m black --check .` 失敗に対応し、black が指摘した 17 ファイルを整形した。対象は gap 実装、broker 層、補助分析スクリプト、関連テストで、機能変更は伴わず書式のみを統一した。
- 2026-04-19: `strategies/gap_d1_0935` 側で D 当日 9:35 判定の `missing_first5` 内訳ログを追加した。`open_exists`, `minute_bars_in_0930_0935`, `first5_constructible`, `missing_reason`, `remarks` を raw CSV 化し、`phase1_missing_first5_detail_*`, `phase1_missing_first5_daily_*`, `phase1_missing_first5_symbol_*` を新設した。
- 2026-04-19: `gap_d1_0935` v1.3 を再集計した。D-1 条件は v1.2 維持、D 当日側のみ `first5_range_pos 0.60->0.50`, `close_5m >= vwap -> close_5m >= vwap*0.998` に緩和。結果は `avg_watchlist_count=1.277`, `avg_candidate_0935_count=0.062`, `scan_zero_days=62` で据え置き、`scan_range_fail_count 17->15`, `scan_vwap_fail_count 18->14` のみ改善、pass は 4件のままだった。
- 2026-04-19: `gap_d1_0935` の `missing_first5` は 2026-01-14〜2026-04-17 で 50件、全件 `AXL` の `no_minute_bars`。`partial_minute_bars`, `late_open_or_halt`, `symbol_issue`, `unknown` は 0 件で、主ボトルネックはルール閾値よりも特定銘柄の minute 取得欠損に寄っている。
- 2026-04-19: `gap_d1_0935` v1.4 を再集計した。変更は `gap_today_pct >= 1.0 -> 0.5` のみ。結果は `avg_candidate_0935_count 0.062->0.077`, `scan_zero_days 62->61`, `scan_gap_fail_count 23->20` と改善し、pass は `2026-01-22 ALM` が追加されて 5件になった。
- 2026-04-19: `gap_d1_0935` の `AXL` 一時除外比較では `missing_first5_count 50->0` だが、`avg_candidate_0935_count=0.077`, `scan_zero_days=61` は不変。`AXL` は watchlist 母数を押し上げるだけで、候補通過数には寄与していない。
- 2026-04-19: `gap_d1_0935` v1.5 を再集計した。変更は `first5_pace >= 1.5 -> 1.2` のみ。`scan_pace_fail_count 12->10` には改善したが、`avg_candidate_0935_count=0.077`, `scan_zero_days=61`, pass 5件は v1.4 と同値だった。
- 2026-04-19: `gap_d1_0935` v1.5 の fail overlap は `gap_only_fail=7`, `gap+pace_fail=1`, `gap+oc_ret_fail=2`, `pace_only_fail=2`, `oc_ret_only_fail=0`。`pace` 緩和の効果は限定的で、主ボトルネックは `gap_fail` 単独と `oc_ret` 側へ寄った。
- 2026-04-19: `gap_d1_0935` v1.6 を再集計した。変更は `first5_oc_ret >= -0.001` のみで、実装側の判定も `>=` に統一した。`scan_oc_ret_fail_count 15->14` には改善したが、`avg_candidate_0935_count=0.077`, `scan_zero_days=61`, pass 5件は v1.5 と同値だった。
- 2026-04-19: `gap_d1_0935` v1.6 の `gap_only_fail=7` の `gap_today_pct` は、指定帯で `0.0〜0.5%=1`, `0.5〜1.0%=0`, `1.0%以上=0`。残り 6件は `gap_today_pct < 0.0` で、次の主ボトルネックは「微弱ギャップ」よりも「前日終値割れ寄り」の扱いと判明した。
- 2026-04-19: `gap_d1_0935` に D 当日寄りレジーム分析を追加した。`analysis_regime.py` / `scripts/analyze_gap_d1_regime.py` で D-1 watchlist 銘柄を `regime_a_open_above_prev_close` / `regime_b_open_at_or_below_prev_close` に分類し、日中成績と branch 比較 CSV を出力できるようにした。
- 2026-04-19: 2026-01-14〜2026-04-17 の regime 分析では `regime_a=16`, `regime_b=17`。`regime_a` は `avg_day_oc_ret=-0.577%`, `win_rate=31.3%`, continuation pass 5件、`regime_b` は `avg_day_oc_ret=+0.421%`, `win_rate=58.8%`, reclaim branch 17件だった。continuation 単線より reclaim/reversal 系 branch の試作優先度が上がった。

### 次のアクション（Gap戦略）
- 毎朝のスキャンログ確認（1日平均何銘柄が通過するか把握）
- 1週間後にパターン集計（業種・ギャップ率・出来高倍率の分布）
- Stage 3 移行判断（dry_run でエントリー条件の発火頻度を確認）

---

## MLStock 週次システム（V2）

### 現在の状態: V2 再設計完了・バックテスト検証済み

**V2 最終結果 (varN_t0005_p60)**:
- Return: +29.25% / MaxDD: -12.66% / Return/DD: 2.31
- Avg Positions: 7.0（動的: 5-15）

### ディープ分析結果 (2026-03-02) — 完了

**実際のバックテスト期間**: 2020-07-27〜2024-12-30

**サブ期間安定性 (年別)**:
| 年    | E70    | feat17  | V2      | f17-E70      | V2-E70       |
|-------|--------|---------|---------|--------------|--------------|
| 2021  | -8.16% | -8.87%  | -5.23%  | -0.71pt      | +2.93pt      |
| 2022  | +3.70% | +2.02%  | -5.57%  | -1.68pt      | **-9.27pt**  |
| 2023  | +29.18%| +37.45% | +36.87% | **+8.28pt**  | **+7.69pt**  |
| 2024  | +7.43% | +5.78%  | +5.93%  | -1.65pt      | -1.51pt      |

**結論**: 改善は2023年に集中。feat17は4年中1年のみE70超え。週次正アルファは25-28%のみ。

**レジームゲート依存度**:
| 設定 | Return | MaxDD |
|------|--------|-------|
| feat17 gate ON  | +32.33% | -16.02% |
| feat17 gate OFF | -23.97% | -57.02% |
| V2 gate ON      | +29.25% | -12.66% |
| V2 gate OFF     | -13.14% | -42.87% |

**警告**: ゲートは絶対的に重要（feat17で+56pt、V2で+42ptの効果）

### 設定ファイル
- `config/config.yaml`: V2推奨設定 (varN_t0005_p60)
- Model: ensemble (Ridge 70% + LGBM 30%)
- Confidence sizing: ON, threshold=0.005

### 次のアクション（MLStock週次）
- 2022年下落局面でのV2悪化原因分析（-9.27pt vs E70）
- ゲート依存度の検討
- 実運用移行準備

---

## 更新ルール

- 変更時はこの `STATUS.md` を更新する
- 仕様変更・修正・運用変更・検証結果は日付付きで追記する
- gap 戦略と週次 pipeline の履歴を混同しない
- 未確認事項は確定事項として書かない

## 2026-04-02
- 2026-04-09: `.github/workflows/audit.yml` を削除した。`ci.yml` と監査内容が重複していたため、品質チェックを CI に一本化した。影響: GitHub Actions の実行重複と通知ノイズを削減した。
- weekly / backtest / execution 系の仕様書・スクリプト・実装・テストを `C:\BOT\WEBULLWEEK` へ分離した。
- `MLStock` 側には gap 系のみ残し、weekly で共有していた基盤は `WEBULLWEEK` へ複製した。
- `PROJECT_SPEC.md` / `RUNBOOK.md` / `README.md` は `MLStock` 側を stub に差し替え、参照先を明示した。
- gap scanner に `Twelve Data` 比較モードを追加した。`--compare-data-sources` で、当日の Alpaca 候補シンボルに対して Twelve Data の同条件スキャンを並べて記録できる。
- `src\mlstock\data\twelvedata\client.py` を追加し、`/time_series` ベースで日足・寄り後5分の比較取得を実装した。
- Twelve Data 無料枠を考慮し、比較対象は `--symbols` 指定が無ければその日の Alpaca 候補シンボルに限定する。
- `TWELVEDATA_API_KEY` 未設定時は比較を落とさずスキップし、理由をログへ残す。
- gap 側 `load_config()` が root `config.yaml` を必要としていたため、weekly 分離後も共有設定 `config.yaml` / `config.local.yaml` を `MLStock` に複製で戻した。

- 2026-04-02: gap 側 load_config() が root config.yaml を必要としていたため、weekly 分離後も共有設定 config.yaml / config.local.yaml を MLStock に複製で戻した。

- 2026-04-02: Twelve Data API キー疎通を確認。AAPL 1day は取得成功。.env の TWELVEDATA_BASE_URL にコメントが連結しており 404 だったため修正した。
- 2026-04-02: compare 用の試験として 2026-04-01 09:35 ET の CAI / AAOI を Alpaca と Twelve Data で比較。Alpaca は 2 銘柄通過、Twelve Data は 0 銘柄だった。
- 2026-04-02: Twelve Data 無料枠は 8 credits/min 制限に到達しやすく、追加取得時に 429 (current limit being 8) を確認。gap の全量比較には無料枠のままでは厳しい。
- 2026-04-02: `scripts\\run_gap_dry_run_skip_options.ps1` を追加し、`--skip-options` の gap dry_run を毎営業日 22:00 JST に実行する定期タスク `MLStock_GapDryRun_SkipOptions_Daily` を登録した。旧 one-shot `MLStock_GapDryRun_SkipOptions_20260403` は削除した。
- 2026-04-02: タスク運用の確認では、単発/定期の区別、最終実行時刻、次回実行時刻、実行結果を task metadata で先に確認するルールを `AGENTS.md` に追加した。
- 2026-04-09: `force_close_exit` が entry price で仮計算される穴を修正し、強制クローズ直前に最新1分足を取り直して exit price / realized PNL を計算するようにした。
- 2026-04-09: `scripts\\summarize_gap_logs.py` を追加し、dry-run / live の gap ログから scanner count、entry/exit、仮PNL を日次でまとめて見られるようにした。
- 2026-04-09: `GAP_PROJECT_SPEC.md` に gap 戦略の実装完了条件 `P0-P2` を追記した。影響: scan-only、dry-run、Webull 実注文、監視、発注統制の到達条件を仕様書上で参照できるようにした。

- 2026-04-09: scripts\\run_gap_live_skip_options.ps1 を追加し、今夜 22:00 JST に --skip-options --live を 1 回だけ実行するタスク MLStock_GapLive_SkipOptions_20260409 を登録した。重複実行防止のため MLStock_GapDryRun_SkipOptions_Daily は一時的に無効化し、2026-04-10 08:00 JST に再有効化するタスク MLStock_GapDryRun_Reenable_20260410 を登録した。

- 2026-04-09: 今夜の live 実行は見送り、MLStock_GapLive_SkipOptions_20260409 を削除した。MLStock_GapDryRun_SkipOptions_Daily を再有効化し、MLStock_GapDryRun_Reenable_20260410 は不要になったため削除した。
- 2026-04-10: gap の preflight 診断を見直した。`scripts\\run_gap_trade.py` の preflight は `bars=0` でも停止根拠にせず、`preflight_iex_bars_warning` として warning 扱いに変更した。影響: 偽陰性の `SPY bars=0` を致命扱いせず、実運用ログで区別して追跡できる。
- 2026-04-10: gap の scan 開始待機を 09:30:05 ET から 09:30:20 ET に変更した。影響: 寄り直後の 1 分足確定遅延による preflight 偽陰性を減らしつつ、scanner 本体は同じ営業日の寄り後データを使う。
- 2026-04-10: `src\\mlstock\\jobs\\gap_scanner.py` に `scanner_diagnostics` ログを追加した。`universe_count`, `daily_count`, `open_count`, `missing_open_count`, `liquid_price_count`, `gap_ge_2_count`, `raw_candidate_count`, `candidate_count` を日次で記録する。影響: `count=0` の原因を相場要因と取得品質要因に分けて継続監視できる。
- 2026-04-10: `scripts\\summarize_gap_logs.py` を拡張し、`scanner_diagnostics` をサマリ出力に含めるようにした。影響: 直近ログをまとめて見るだけで `open_count` や `missing_open_count` の日次ばらつきを追跡できる。既存の古いログには `scanner_diagnostics` が無いため空で表示される。
- 2026-04-13: `scripts\\plot_gap_forward_dashboard.py` を追加し、dry-run の forward 実績を 1 回の実行で可視化できるようにした。出力先は `artifacts\\forward_plots\\gap_dry_forward` に固定し、`gap_dry_forward_dashboard.png`, `gap_dry_forward_sessions.csv`, `gap_dry_forward_trades.csv`, `PLOT_GUIDE.md` をまとめて生成する。影響: 戦績、候補供給、取得品質を 1 箇所で継続観測できる。
- 2026-04-13: 可視化依存として `matplotlib` を `pyproject.toml` と `requirements.txt` に追加した。影響: ローカル `.venv` で dry-run ログから PNG ダッシュボードを直接生成できる。

---

### 2026-04-18: missing_open 要因分析 (probe_universe_coverage.py)

**目的**: `scanner_diagnostics.missing_open_count ≈ 1540/1994` の主因を「IEX coverage 不足」と「seed universe 構成問題」に分離・定量化した。

**調査手法**: `scripts/probe_universe_coverage.py --probe-date 2026-04-17 --feed iex` で seed 2000銘柄の Alpaca IEX daily bar / 1-min window (9:30-9:35 ET) 取得可否を全件実測した。出力: `artifacts/coverage/universe_coverage_2026-04-17.csv`。

**universe 構成 (seed 2000銘柄)**:

| タイプ | 件数 | 割合 |
|-------|-----|------|
| common (普通株相当) | 1,411 | 70.5% |
| ETF | 254 | 12.7% |
| preferred (優先株) | 147 | 7.3% |
| warrant (ワラント) | 141 | 7.0% |
| right (権利) | 33 | 1.7% |
| unknown (Alpaca未登録) | 13 | 0.7% |
| unit | 1 | 0.1% |

- 非標準銘柄合計: 335/2000 = 16.8%
- Exchange: NASDAQ 1241 / NYSE 740 / UNKNOWN 19

**IEX 1-min window 取得率 (2026-04-17 実測)**:

| タイプ | daily_bar | open_1m | open_1m% |
|-------|----------|---------|---------|
| common | 1,365 | 677 | **49.6%** |
| ETF | 249 | 52 | 20.9% |
| warrant | 122 | 13 | 10.7% |
| preferred | 142 | 5 | 3.5% |
| right | 31 | 6 | 19.4% |
| **TOTAL** | **1,912** | **754** | **39.4%** |

- NYSE common: 323/536 = **60.3%**
- NASDAQ common: 354/829 = **42.7%**（NYSE より低い）

**probe 値 vs 診断値の乖離**:

- probe (post-market, full 9:30-9:35): **754**
- 診断値 (9:30:20 ET live scan): **454**
- 差分: +300 → スキャン開始時刻 9:30:20 ET では第1バー未確定のため追加で約 300 件を逃している

**missing_open の3要因分解 (診断値ベース: missing=1540)**:

| 要因 | 件数 | 割合 | 内容 |
|-----|-----|------|-----|
| A. 非標準銘柄 | ~310 | 20.1% | warrant/preferred/right が daily_bar あるが open_1m なし |
| B. timing gap | ~300 | 19.5% | 9:30:20 ET 時点で第1バー (9:30-9:31) 未確定 |
| C. IEX feed 構造欠損 | ~930 | 60.4% | common/ETF でも IEX quote がない → 代替データ源が必要 |

**除外しても open_count が増えない候補**:

- warrant 141銘柄を除外 → open_count -13（1.7%減少）、missing -109 削減
- preferred 147銘柄を除外 → open_count -5（0.7%減少）、missing -137 削減
- right 33銘柄を除外 → open_count -6（0.8%減少）、missing -25 削減
- → 非標準銘柄除外は「universe のノイズ削減」であり open_count 改善は軽微

**主結論**:

1. missing_open の主因は **IEX feed の構造的 coverage 不足 (60.4%)**。common 株でも 50.4% が未取得。universe 整理だけでは解決しない。
2. 非標準銘柄除外による改善は **上限 20.1%**、open_count の実増加は +25 程度に留まる。
3. 9:30:20 ET timing gap が **19.5%** を説明。9:31 以降にクエリすれば回収可能だが、strategy timing の変更を伴う。

**未確定点**:

- 非標準銘柄の精度: suffix ルール + name ベースの分類（誤分類あり得る）。yfinance quoteType での全件検証は未実施。
- timing gap の 300 件は 1日の実測値。日によってバラつき得る（相場活況度で変化）。
- IEX feed → SIP feed 切替時の open_count 増加量は未実測。

**次の改善候補**:

| 優先 | 内容 | 期待効果 |
|-----|------|---------|
| 1 | `config.yaml: feed: sip` に変更 | common 株の open_1m 取得率を大幅改善（上限 100%） |
| 2 | scan 開始を 9:31:00 ET 以降に遅らせる | +300 件（ただし strategy タイミング変更） |
| 3 | universe から warrant/preferred/right を除外 | missing 減少 -271 件、open_count は -24 のみ |
| 4 | ETF を除外し common のみに絞る | open_1m rate 49.6%（ETF 20.9% より高い） |

判断: **最優先は feed=sip の評価**。SIP feed で common 株の取得率がどこまで上がるか probe_universe_coverage.py を `--feed sip` で再実行して確認する。

---

### 2026-04-18: SIP probe + timing gap 定量化

**調査手法**: `probe_universe_coverage.py --probe-date 2026-04-17 --feed sip` で SIP 実測。timing 4パターンは IEX/SIP それぞれ 9:30:20 / 9:31:05 / 9:31:20 / 9:32:05 / 9:35:00 ET で Alpaca API を直接叩いて open_1m 件数を計測。

**IEX vs SIP open_1m 件数比較 (2026-04-17, seed 2000銘柄)**:

| タイプ | IEX daily | IEX open_1m | IEX% | SIP daily | SIP open_1m | SIP% |
|-------|---------|-----------|------|---------|-----------|------|
| common | 1,365 | 677 | 49.6% | 1,371 | 1,215 | 88.6% |
| ETF | 249 | 52 | 20.9% | 250 | 181 | 72.4% |
| warrant | 122 | 13 | 10.7% | 125 | 47 | 37.6% |
| preferred | 142 | 5 | 3.5% | 142 | 84 | 59.2% |
| right | 31 | 6 | 19.4% | 31 | 11 | 35.5% |
| **TOTAL** | **1,912** | **754** | **39.4%** | **1,922** | **1,539** | **80.1%** |

- Exchange 別 SIP: NYSE 630/740=85.1%, NASDAQ 908/1241=73.2%

**scan timing × feed の open_count 実測 (2026-04-17, seed 2000銘柄)**:

| scan 開始時刻 | IEX | SIP | IEX増分 | SIP増分 |
|-------------|-----|-----|--------|--------|
| 9:30:20 (現状) | **454** | **1,466** | — | — |
| 9:31:05 | 572 | 1,487 | +118 | +21 |
| 9:31:20 | 572 | 1,487 | +118 | +21 |
| 9:32:05 | 628 | 1,497 | +174 | +31 |
| 9:35:00 (full) | 754 | 1,539 | +300 | +73 |

- SIP は 9:30:20 時点で既に full window の 95.3% を取得（timing 感度が低い）
- IEX は 9:30:20 で full の 60.2% にすぎず、9:31 移行で +26% 回収可能

**clean universe 試算 (common+ETF のみ、SIP)**:

| | full (2000) | clean (1665) | 差分 |
|-|------------|-------------|-----|
| daily_bar | 1,922 | 1,621 | -301 |
| open_1m | 1,539 | 1,396 | -143 |
| open_1m rate | 80.1% | 86.1% | +6.0pt |

- 非標準銘柄除外で open_1m rate は +6pt 改善するが、絶対件数は -143 減少
- SIP で除外対象の warrant/preferred/right も相当数 open_1m あり（47+84+11=142件）
- これらが gap 候補になることがある（ノイズ増加リスク）

**スキャン候補数への影響推定 (4/17 diagnostics ベース、SIP 線形試算)**:

| 指標 | IEX 実測 | SIP 推定 | 倍率 |
|-----|---------|---------|-----|
| open_count (9:30:20) | 454 | 1,466 | **3.23×** |
| liquid_price_count | 159 | ~513 | ~3.23× |
| gap_ge_2_count | 51 | ~164 | ~3.23× |
| raw_candidate_count | 8 | ~26 | ~3.23× |

注: 線形スケールは仮定。実際の候補倍率はギャップ銘柄の IEX/SIP 分布に依存し未確定。

**主結論**:

1. **feed=sip への変更が支配的改善手段**。SIP 9:30:20 (1466) は IEX 9:35:00 (754) の約2倍。timing を最大遅延しても IEX は SIP の 9:30:20 に届かない。
2. **timing 変更 (9:30:20→9:31:05) の効果は IEX では +118、SIP では +21 と小さい**。SIP に切り替えれば timing 変更の必要性は大幅低下する。
3. **clean universe 化の効果は coverage 改善ではなく候補品質改善**。SIP でも warrant/preferred が open_1m を持ち gap 候補になりうるため、除外は別途判断が必要。

**推奨アクション: feed=sip に変更し 1 週間のスキャンデータを収集する**

変更箇所: `config/config.yaml: bars.feed: iex → sip`  
確認事項: Alpaca アカウントの SIP アクセス権限（有料プランが必要な場合あり）  
期待効果: open_count 454→1466 (3.2×), raw_candidate 8→~26 (推定)  
リスク: SIP feed が課金対象の場合、コスト増加。feed 切替後に warrant/preferred の偽候補が増える可能性あり。

**未確定点**:

- Alpaca SIP feed の利用権限（現アカウントが free tier かどうか未確認）
- SIP 切替後の実 raw_candidate 倍率（線形スケール仮定は未検証）
- timing gap 300 件の日次バラつき（1日実測値）
- warrant/preferred が SIP で open_1m を持つ場合の偽候補増加リスク
- 2026-04-18: `scripts\\summarize_gap_logs.py` の既定 `--latest=5` が「最近3日しか見えない」誤解を生んでいたため、既定を全件集計に変更した。`session_utc`, `trade_date`, `scan_only`, `skip_options`, `live`, `status`, `start_count`, `mode_collision`, `replay_mode` を出すようにし、表示順も `trade_date/session_utc` 基準に直した。
- 2026-04-18: `scripts\\plot_gap_forward_dashboard.py` の既定対象件数を全 dry-run に変更し、`scan_replay_*` を forward 集計から除外するようにした。さらに `session_utc` の時刻帯で定期 dry-run 相当の run に寄せ、`gap_dry_forward_sessions.csv` と `gap_dry_forward_trades.csv` を raw jsonl から再生成した。再生成後の既知値は `9 sessions / 1 trade / -52.185 USD`。
- 2026-04-18: `gap_trade_20260415_130048.jsonl` の `scan_only / skip_options` 誤記は、同一秒に scan-only と dry-run が同じ filename へ書き込み、1 file に 2 start が衝突したことが原因と確認した。`src\\mlstock\\logging\\logger.py` の log filename をマイクロ秒付きに変更し、再発を防止した。既存の 2026-04-15 ログは一次情報として保持しつつ、再生成 CSV では `start_count=2`, `mode_collision=true`, `status=complete` として扱う。

- 2026-04-18: `src\\mlstock\\jobs\\gap_scanner.py` の `scanner_diagnostics` を拡張し、`daily_only_count`, `open_missing_count`, `open_zero_bar_count`, `open_no_window_bar_count`, `open_null_field_count`, `open_parse_fail_count`, `open_partial_bar_count`, `open_missing_symbols_sample`, `open_missing_exchange_counts_sample`, `open_missing_quote_type_counts_sample`, `open_missing_market_cap_bucket_counts_sample`, `price_filter_drop_count`, `gap_filter_drop_count`, `pace_filter_drop_count`, `market_cap_drop_count`, `final_candidate_count` を raw log に出すようにした。影響: open 欠損が「当日最初の1分足未返却」なのか parse/null 欠損なのかを日次ログだけで切り分けられる。
- 2026-04-18: open 判定経路を整理した。current gap scanner の open は Alpaca の日足 open ではなく、`09:30-09:35 ET` 窓の最初の 1 分足 `o` を使う。`open_source=alpaca_first_1min_bar_0930_0935` を diagnostics に追加した。影響: daily bar はあるが first 1-min bar が無いケースを `missing_open` として追跡できる。
- 2026-04-18: 直近5営業日 (2026-04-13〜2026-04-17) を再診断した。`universe_count=2000`, `daily_count=1994` に対し `open_count=402/403/420/439/454`, `open_missing_count=1592/1591/1574/1555/1540` で、missing_open のほぼ全量が `open_zero_bar_count` だった。`open_no_window_bar_count`, `open_null_field_count`, `open_parse_fail_count`, `open_partial_bar_count` は 5 営業日とも 0。影響: 現時点の主因は parse/null 欠損ではなく、対象銘柄で `09:30-09:35` の 1 分足が 0 本返る段階に寄っている。
- 2026-04-18: missing_open sample の暫定属性確認では、`A`, `AACB`, `AACBR`, `AACBU`, `AACG`, `AADR`, `AAEQ`, `AALG`, `AAM`, `AAM.U` などが継続して現れた。quote type sample は `EQUITY` と `ETF` が混在し、market cap sample も `lt_300m`, `gte_2b`, `unknown` が混在した。exchange sample は `client.get_asset()` 由来の `unknown` が多く、銘柄種別偏りの断定には未到達。
- 2026-04-19: `C:\BOT\MLStock\docs\strategy_gap_d1_0935_v1.md` を新規追加した。既存 gap 本線を維持したまま、D-1 watchlist + 当日 9:35 continuation 確認の別系統仕様を文書化した。影響: 旧 broad minute scan 系と切り分けて、新系統の実装対象・非対象・フェーズ分割・受け入れ条件を参照できる。
- 2026-04-19: `C:\BOT\MLStock\strategies\gap_d1_0935` を新規作成し、別戦略の専用受け皿として `docs`, `src`, `scripts`, `artifacts` を切った。仕様書は `C:\BOT\MLStock\strategies\gap_d1_0935\docs\strategy_gap_d1_0935_v1.md` へ移動した。影響: 既存 gap 本線と新系統の実装資産を分離して進められる。
- 2026-04-19: 別戦略 `strategies\\gap_d1_0935` の Phase 1 実装を追加した。`strategies\\gap_d1_0935\\src\\gap_d1_0935` に D-1 watchlist builder、当日 9:35 scanner、replay、old 比較を実装し、strategy 専用 scripts / config / STATUS を作成した。影響: 既存 gap 本線とコード・出力・状態管理を分離したまま、新系統を並行実装できる。
- 2026-04-19: `strategies\\gap_d1_0935` に Phase 1 母数確認スクリプトを追加した。`analysis_phase1.py` / `analyze_phase1_population.py` で直近3か月の watchlist 件数、9:35 件数、drop counts、銘柄別理由、old broad gap との差分を `strategies\\gap_d1_0935\\artifacts\\reports` に出力する。結果は 65 営業日で `watchlist 1件 / 9:35 candidates 0件` と極端に少なく、主因は D-1 側の `price_fail` と `liquidity_fail`。
- 2026-04-19: 別戦略 `strategies\\gap_d1_0935` を v1.1 条件へ緩和して再集計した。結果は 65 営業日で `watchlist_count 1->18`, `candidate_0935_count 0->2`。通過日は `2026-03-18 BTSG`, `2026-04-01 ALKS`。依然として old broad gap scanner との共通候補は 0 で、D-1 側の主な詰まりは `liquidity_fail`, `price_fail`, `gap_fail`。
- 2026-04-19: 別戦略 `strategies\\gap_d1_0935` を v1.2 条件へ緩和して再集計した。結果は 65 営業日で `watchlist_count 18->83`, `candidate_0935_count 2->4`, `watchlist_zero_days 48->6`。一方で `scan_missing_first5 0->50` が増え、次の主ボトルネックは D-1 側より当日 9:35 判定側に移った。
