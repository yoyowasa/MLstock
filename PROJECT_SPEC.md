# PROJECT_SPEC.md — US Weekly ML Stock MVP (1-share, $1,000, Long-only)

> 本ファイルは、このチャットで確定した **設計・仕様・固定関係・運用ルール・検証/監査・比較実験** を、リポジトリに置ける **単一の仕様書**として整理したもの。  
> **コードは含めない**。実装は本仕様に従う。

---

## Table of Contents
1. [目的](#目的)  
2. [スコープ](#スコープ)  
3. [固定要件（不変）](#固定要件不変)  
4. [追加の固定仕様（このチャットで確定）](#追加の固定仕様このチャットで確定)  
5. [リポジトリと実行環境](#リポジトリと実行環境)  
6. [設定（config）規約](#設定config規約)  
7. [データソースとデータ契約](#データソースとデータ契約)  
8. [パイプライン（初回バックフィル→週次運用）](#パイプライン初回バックフィル週次運用)  
9. [バックテスト仕様（現金・リザーブ・約定）](#バックテスト仕様現金リザーブ約定)  
10. [検証・監査（監査JSON/比較実験）](#検証監査監査json比較実験)  
11. [採用済みリスク制御と結論](#採用済みリスク制御と結論)  
12. [未確定事項とバックログ](#未確定事項とバックログ)  
13. [Runbook（実行コマンド）](#runbook実行コマンド)  
14. [用語・定義](#用語定義)  

---

## 目的
米国株を対象に、**週次で銘柄選定 → 注文CSV生成 → 1週間保有**を回すMLベースのMVPを構築する。  
目的は「予測精度」単体ではなく、**資金制約・1株制約・コスト・分割・休場・欠損**込みで **E2E稼働**すること。

**Done（合格条件）**
- 初回バックフィルから週次運用までE2E完走
- バックテスト完走 + 以下が成立
  - `cash_usd >= 0`
  - `cash_usd - reserve_usd >= 0`
  - NAV系列 `NaN == 0`
  - 監査一式（max/min/DD/top-share）が **バックテスト1回**で生成され算術整合が取れる
- 週次実行で `orders_YYYYMMDD.csv` を生成し、BUYが予算内（見積/実行ルール準拠）

---

## スコープ
### スコープ内（MVPで実装・運用する）
- データ収集（初期バックフィル＋増分更新）
- データ保存（reference / raw / weekly snapshots）
- データ検証（欠損・重複・異常値・分割整合の最低限）
- 週次スナップショット生成（week_map / universe / features / labels）
- 学習（週次・ローリング）→予測→銘柄選定→注文CSV生成
- バックテスト（現金・リザーブ・コスト込み）
- 監査出力（max/min/DD/top-share）と比較実験（regime/volcap）

### スコープ外（後付け）
- ブローカーへの自動発注（MVPは「注文案CSV」まで）
- 分足・板などの高頻度領域
- CRSP等の有料データベンダ統合（差し替え抽象化は可能だがMVPでは不要）

---

## 固定要件（不変）
### 市場・時間軸・保有
- 市場：**米国株**
- 時間軸：**週次**
- 保有期間：**1週間**

### 売買タイミング（不変）
- 観測（特徴量確定）：**金曜引け（Fri Close）**
  - 金曜休場なら、その週の **最終取引日引け**
- エントリー（建て）：**月曜寄り（Mon Open）**
  - 月曜休場なら **次の取引日寄り**
- エグジット（手仕舞い）：**翌週の月曜寄り（次のentry）**
- つまり評価区間は **Open(entry_date) → Open(exit_date)**

### 取引制約（不変）
- **現物のみ（ロングオンリー）**
- **各銘柄 1株**
- 資金：**$1,000**

---

## 追加の固定仕様（このチャットで確定）
### 資金・リザーブ（固定）
- `reserve_usd = 100`
- 購入可能額：`cash_usd - reserve_usd`
- BUYは必要額（価格＋コスト）を満たす場合のみ実行
- 資金不足時：BUYは **p順（priority順）で買える分だけ**（ranked_partial）

### ポートフォリオ制約（固定/推奨）
- 最大保有数：`K_max = 15`
- 価格上限：`P_max = 60`（$1,000・1株での偏り抑制）

### Execution改善（deadband v2 採用）
- 仕様: target_set固定 / dwフィルタ / バックフィル無し / renorm無し（余りはcash）
- 採用値: `deadband_abs=0.0025`、`deadband_rel=0.0`、`min_trade_notional=0.0`
  - min_tradeは「rebalance-only」設計だが、現行backtestは週次フル入替（w_cur=0）なので発火しないため不採用
- 運用監視KPI（週次ログ/レポート固定）
  - `deadband_notional_reduction`（`1 - sum_abs_dw_filtered/sum_abs_dw_raw`）
  - `filtered_trade_fraction`（互換維持、実体はnotional削減率）
  - `filtered_trade_fraction_notional`（`deadband_notional_reduction`の別名）
  - `filtered_trade_fraction_count`（件数ベース）
  - `trade_count_raw` / `trade_count_filtered`
  - `cash_after_exec`（または`avg_cash_ratio`）
  - `turnover_ratio_std`（買い側のみ・互換維持）
  - `turnover_ratio_buy` / `turnover_ratio_sell`
  - `turnover_ratio_total_abs` / `turnover_ratio_total_half`
  - `data_max_features_date` / `data_max_labels_date` / `data_max_week_map_date`
- KPI定型出力: `scripts/run_deadband_kpi.py` で週次KPIをCSV化
  - 出力: `artifacts/monitoring/deadband_weekly_kpi.csv`
  - 列: `as_of`, `week_start`, `deadband_v2_enabled`, `deadband_abs`, `deadband_rel`, `min_trade_notional`,
    `sum_abs_dw_raw`, `sum_abs_dw_filtered`, `deadband_notional_reduction`, `filtered_trade_fraction`,
    `filtered_trade_fraction_notional`, `filtered_trade_fraction_count`, `trade_count_raw`, `trade_count_filtered`,
    `turnover_ratio_std`, `turnover_ratio_buy`, `turnover_ratio_sell`, `turnover_ratio_total_abs`,
    `turnover_ratio_total_half`, `cash_after_exec`, `cash_start_usd`, `cash_est_before_buys`,
    `cash_est_after_buys`, `n_selected`, `kept_positions`, `held_positions`, `skipped_buys_insufficient_cash`,
    `data_max_features_date`, `data_max_labels_date`, `data_max_week_map_date`
- 緊急OFF手順: `execution.deadband_v2.enabled=false` で deadband/min_trade を完全無効化（素通し）
- rolling-valid（四半期更新／校正104週）最終検証（bps=0/1/5/10/20）  
  - ret_diff（raw-off）: 0bp +0.255pp / 5bp +0.246pp / 10bp +0.236pp / 20bp +0.218pp  
  - maxDD_diff（raw-off）: 0bp +0.248pp / 5bp +0.239pp / 10bp +0.231pp / 20bp +0.214pp  
  - turnover_ratio_std: Δ(raw-off) -0.139（減少）  
  - turnover_ratio: Δ(raw-off) -0.275（減少）
- 局所探索（0.15/0.20/0.25/0.30/0.35%）で ret_diff_5bp 最大は 0.25%（0.30%以降はretが低下）

参照:
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0025_final_summary.csv
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0025_final_cost.csv
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0025_final_updates.csv
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0025_final_quarterly.csv
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0015_summary.csv
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0020_summary.csv
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0025_summary.csv
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0030_summary.csv
- artifacts/backtest/execution_rolling_deadband_v2_rel0_0_0035_summary.csv

### セーフティ（固定）
- **検証NGの週は取引停止（orders空）**が可能であること
- 取得は **冪等**（同期間再取得しても重複なし）
- 遅延/修正吸収：`直近N日を再取得→上書き`（bars/corp actions）

---

## リポジトリと実行環境
### 置き場所（固定）
- `E:\MLStock` に作成して運用する

### Python/venv（固定）
- 実行は **必ず venv を使用**（`python`直叩きで `mlstock` 未検出が発生したため）  
- 推奨：`.\.venv\Scripts\python` を利用（PS1ラッパーを用意してもよい）

---

## 設定（config）規約
### 基本
- 設定は **YAML**（`config.yaml`）に寄せる
- ローカル上書き：`config.local.yaml`（gitignore推奨）
- 秘密情報（Alpaca APIキー）は **環境変数**から読む（configに入れない）

### 設定例（最小サンプル）
> 実際のキー名は実装に合わせる。重要なのは **概念と固定関係**。

```yaml
project:
  timezone: America/New_York
  start_date: "2016-01-01"        # 取得開始（実データ上はlabels最古=2020-07-27）

portfolio:
  cash_start_usd: 1000
  reserve_usd: 100
  max_positions: 15
  price_cap: 60

cost_model:
  enabled: true
  bps_per_side: 5
  min_cost_usd_per_side: 0.01

seed_selection:
  n_seed: 2000
  include_spy: true              # SPYを必ず入れる（ベンチ/ゲート用）

weekly:
  labels:
    benchmark_symbol: SPY
    delta_bps: 30                # y=1の超過リターン閾値

risk:
  regime_gate:
    enabled: true
    spy_symbol: SPY
    rule: "spy_close_above_ma60"
    action: "no_trade"

  vol_cap:
    enabled: false               # 比較実験でON/OFF
    rank_threshold: 0.70         # 高ボラ上位30%除外（候補）
    apply_stage: selection       # 初期はselectionのみ

paths:
  data_dir: data
  artifacts_dir: artifacts
  reference_assets: data/reference/assets.parquet
  reference_calendar: data/reference/calendar.parquet
  reference_seed: data/reference/seed_symbols.parquet
```

### Alpacaキー（環境変数）
- `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` を読む（推奨）

---

## データソースとデータ契約
### データ種別
- `assets`：取引可能銘柄一覧
- `calendar`：取引日・open/close（週境界決定に使用）
- `bars_daily`：日次OHLCV（raw保存）
- `corp_actions`：主に株式分割（split）
- `weekly snapshots`：week_map / universe / features / labels
- `backtest`：NAV/trades + 監査一式
- `weekly run`：orders + portfolio_state

### 保存パス契約（固定）
**reference**
- `data/reference/assets.parquet`
- `data/reference/calendar.parquet`
- `data/reference/seed_symbols.parquet`（SPY含むこと）

**raw**
- `data/raw/bars_daily/...`（銘柄別parquet）
- `data/raw/corp_actions/...`

**weekly**
- `data/processed/weekly/...`（week_map/universe/features/labels）

**artifacts/backtest**
- `summary.json`
- `nav.parquet`
- `trades.parquet`
- 監査一式：
  - `audit_max_jump.json`
  - `audit_min_jump.json`
  - `audit_max_drawdown.json`
  - `audit_top_share_profile.json`
  - `top_share_by_week.csv`

**artifacts/orders & state**
- `artifacts/orders/orders_YYYYMMDD.csv`
- `artifacts/orders/selection_YYYYMMDD.json`
- `portfolio.json`（`cash_usd` と `positions` を保持）

### SPYの扱い（固定）
- SPYは **seedに必ず含める**（ベンチ/ゲート用に必要）
- SPYは **売買ユニバースから除外**（トレード対象に混ぜない）

---

## パイプライン（初回バックフィル→週次運用）
### 初回バックフィル（E2E）
1. reference取得：assets/calendar
2. seed生成：`seed_symbols.parquet`（SPY強制追加）
3. rawバックフィル：bars_daily + corp_actions
4. weekly snapshots 生成：week_map / universe / features / labels
5. backtest実行：NAV/trades/監査一式を生成

### 週次運用（毎週）
- 直近N日を再取得して上書き（bars/corp_actions）
- validate NGなら停止（orders空）
- 週次スナップショット更新（必要分）
- 学習（過去のみ）→予測 → 選定（制約）
- `orders_YYYYMMDD.csv` と `portfolio.json` を更新

---

## バックテスト仕様（現金・リザーブ・約定）
### 状態
- `cash_usd`
- `positions`（symbol→shares=1）
- `reserve_usd = 100`

### 約定
- 週ごとに `entry_date open` で SELL→BUY を処理
- BUYは `cash_usd - reserve_usd >= required` を満たすもののみ実行
- 資金不足時：priority順に買える分だけ（ranked_partial）
- コスト：片道bps＋最低額を現金に反映

### NAV更新
- `nav = cash_usd + positions_value`
- positions_value は `exit_date open` で評価（open-to-openの週次）

### 健全性要件（固定）
- `cash_usd` が負にならない
- `cash_usd - reserve_usd` が負にならない（`reserve_violation_count == 0`）

---

## 検証・監査（監査JSON/比較実験）
### prev基準の監査フィールド（固定）
監査JSONは **prev基準で統一**し、説明基準を固定する。

- `nav_prev`
- `cash_prev`
- `positions_value_prev`
- `weekly_pnl = nav - nav_prev`
- `pct_change = weekly_pnl / nav_prev`

### 監査一式（バックテスト1回で生成）
- `audit_max_jump.json`：`abs(pct_change)` 最大週
- `audit_min_jump.json`：`pct_change` 最小週
- `audit_max_drawdown.json`：ピーク→谷（drawdown最小週）
- `audit_top_share_profile.json`：全週top-shareプロファイル
- `top_share_by_week.csv`：週次明細（昇順・一意・shareは0〜1）

### 寄与分解の残差（固定の健全性指標）
- `residual_pnl = weekly_pnl - sum(symbol_pnl)`
- 浮動小数誤差レベルであること（実績：~3.8e-13）

### 比較実験（片側だけ変える）
**原則：同一期間、同一データ、片側だけ変更して差分を出す。**

- `--compare-regime`：`regime_gate OFF` vs `ON`（SPY必須）
- `--compare-volcap`：`vol_cap OFF` vs `ON(thr=...)`（gateはON固定で比較）

**評価期間の注意**
- labelsが存在する期間のみ比較対象  
  （実データ上：labels最古は 2020-07-27）

---

## 採用済みリスク制御と結論
### レジームゲート（採用確定）
- SPYが無いと差分が出ない問題があり、**SPYをseedに追加＋バックフィル**で解消
- 有効ラベル期間の分割比較でも **全期間でDD改善＆return改善**を確認 → 採用確定

### 高ボラ除外（vol cap）
- gate ON固定で `thr=0.80` / `thr=0.70` を比較
- 全期間では thr=0.70 が強い改善（return/DD）を示す一方、3分割ではトレードオフが発生
- さらに `avg_cash_ratio` が系統的に変化（投下比率が上がる）ため、差分は「銘柄質」＋「投下量変化」が混在

### vol cap（rolling-valid採用見送り）
- 採用ゲート: rolling-valid（四半期更新／校正104週）で valid 0bp/5bp ともに ret_diff>=0（DD条件は未設定）
- 試したもの: hard / hysteresis / EMA / soft penalty（p_min=0.3/0.5/0.7） / regime gate（HV premium L=13 theta=0 q80-q20）
- 結果: すべてゲート未達 → 不採用
- 運用方針: 実験コードは feature-flag 下のみ実行（`vol_cap.enabled=true` 時のみ有効、デフォルトOFF）

参照:
- artifacts/backtest/exposure_rolling_regime_hv13_summary.csv
- artifacts/backtest/exposure_rolling_regime_hv13_cost.csv
- artifacts/backtest/exposure_rolling_regime_hv13_updates.csv
- artifacts/backtest/exposure_rolling_regime_hv13.json
- artifacts/backtest/exposure_rolling_soft_pmin_0_30_summary.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_30_cost.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_30_updates.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_30.json
- artifacts/backtest/exposure_rolling_soft_pmin_0_50_summary.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_50_cost.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_50_updates.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_50.json
- artifacts/backtest/exposure_rolling_soft_pmin_0_70_summary.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_70_cost.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_70_updates.csv
- artifacts/backtest/exposure_rolling_soft_pmin_0_70.json
- artifacts/backtest/exposure_rolling_summary.csv
- artifacts/backtest/exposure_rolling_cost.csv
- artifacts/backtest/exposure_rolling_updates.csv
- artifacts/backtest/exposure_rolling.json

### Volatility Cap 導入時の Exposure Guard（Gross Exposure Cap）
#### 背景と問題設定
`vol_cap`（rank_threshold=0.70）を有効化すると、銘柄選択（selection）の質が改善する一方で、ポートフォリオの投下資本比率（gross exposure）が恒常的に増加し、結果として損益・DDが増幅して成績を毀損するケースが確認された。

特に 2022–2023 では、vol_cap ON により以下が発生（trades / avg_n_positions は不変）:
- avg_gross_exposure: 0.3777 → 0.5320（+0.1543）
- avg_position_weight: 0.0689 → 0.0971（+0.0282）
- gross_exposure_p95: 0.4419 → 0.6197（+0.1778）
- gross_exposure_max: 0.4756 → 0.6730（+0.1974）

また、全期間（2020-07-27→2025-12-15）の分解により、
- 生ON（ON−OFF）は return -5.07pp / maxDD -3.83pp（悪化）
- 投下ニュートラル（neutral_avg−OFF）は return +3.50pp / maxDD +4.73pp（改善）

となり、「vol_cap自体の選択効果」はプラス寄りだが、「gross exposure 増加」がそれ以上にマイナスであることが示唆される。

よって、vol_cap の効果を活かすには、vol_cap 有効時に同時発生する gross exposure の上振れを抑えるガードが必要である。

#### 目的
vol_cap による「選択効果（銘柄/タイミングの質改善）」を維持したまま、vol_cap が誘発する gross exposure の増加（投下増）による損益・DDの増幅を抑制する。

具体的には、gross_exposure_p95 を vol_cap OFF 水準に寄せる cap を導入し、投下分布を安定化させる。

#### 提案仕様（Exposure Guard）
##### 概要
vol_cap ON のときのみ、最終的なポジションサイズ決定後に gross exposure の上限（cap）を適用し、上限を超える場合は **全ポジションを比例スケールダウン**して cash に戻す。

##### 定義
- gross_exposure_t = Σ|w_i,t|（long-only の場合は Σw_i,t と同値）
- cash_ratio_t = 1 − gross_exposure_t
- gross_cap = 上限値（推奨：OFFの gross_exposure_p95）

##### アルゴリズム
1) selection / sizing により日次ターゲットウェイト `w_raw` を得る  
2) `gross_raw = Σ|w_raw|` を計算  
3) `gross_raw > gross_cap` の場合  
   - `scale = gross_cap / gross_raw`
   - `w_final = w_raw * scale`
   - `cash = 1 − Σ|w_final|`
4) `gross_raw <= gross_cap` の場合はそのまま  
5) `scale` をログ出力（分析・デバッグ用）

比例スケールにすることで「銘柄間の相対順位（相対配分）」は維持され、vol_cap の選択効果を極力壊さずに投下量だけを制御できる。

#### cap の決め方（推奨順）
**推奨：OFF p95 追従（cap_source=off_p95）**  
`gross_cap = gross_exposure_p95_off + cap_buffer`  
cap_buffer は 0.00〜0.02 程度を想定（バックテストで調整）  
「gross_exposure_p95 を OFF 水準に寄せる」という要求を最も直接的に満たす。

**代替：OFF avg 追従（cap_source=off_avg）**  
`gross_cap = avg_gross_exposure_off + cap_buffer`  
（より強い制御。過度にキャッシュ化しやすい）

**代替：固定値（cap_source=fixed）**  
`gross_cap = cap_value`（運用で安定した上限を置きたい場合）

#### 運用モード
- mode: daily（推奨）  
  リバランス日ごとに cap を適用し、上振れを即時抑制する
- mode: weekly_aligned（分析互換モード）  
  週単位で scale を固定し、週次整合のニュートラル比較に近い挙動を再現する（検証用）

#### 設定（config.yaml 追記案）
```yaml
vol_cap:
  enabled: true
  rank_threshold: 0.70
  apply_stage: selection
  # training+selection は、2022–2023および全期間で return/DD が悪化したため本採用しない

exposure_guard:
  enabled: true
  trigger: vol_cap_enabled     # vol_cap.enabled=true のときのみ適用
  mode: daily                  # daily | weekly_aligned
  base_source: off_avg_on_avg  # off_avg_on_avg | fixed | none
  base_scale: null             # base_source=fixed のとき使用
  cap_source: off_p95          # off_p95 | off_avg | fixed
  cap_value: null              # cap_source=fixed のとき使用
  cap_buffer: 0.00             # (例) 0.00〜0.02
  log_scale: true
```

#### 出力・ログ（compare/診断の最低要件）
以下を compare JSON（または run summary）に保存する：
- avg_cash_ratio / avg_gross_exposure / avg_position_weight
- gross_exposure_p95 / gross_exposure_max
- exposure_guard_scale（平均・p95・min・max）
- exposure_neutral（neutral_avg / neutral_weekly）は維持

#### 受け入れ基準（Acceptance Criteria）
Exposure Guard 実装後、全期間＋3分割で次を満たすこと：

**投下分布の制御**
- gross_exposure_p95_on_with_guard <= gross_exposure_p95_off + cap_buffer
- avg_gross_exposure_on_with_guard が OFF 水準に十分近い  
  （例：差分±0.02以内、運用要件で調整）

**性能の整合（原因切り分けの確認）**
- ON_with_guard が ON_neutral に近づく  
  （= ON−neutral の悪化分が縮小）

**最低限の成績要件（プロジェクト側のKPIに合わせて設定）**
- 例：全期間で return が OFF を下回らない、または maxDD が OFF より悪化しない 等  
  ※最終KPI（return/DD優先度）は運用方針に依存するため、数値閾値は別途決定

#### 非目標（Non-goals）
- training データ分布を vol_cap で削る（apply_stage=training+selection）は本仕様の対象外  
  （既に検証で悪化が確認されたため）
- 2022–2023 の「選択効果そのもの（neutralでも負ける部分）」の解消は本仕様の主目的ではない  
  → 必要なら別途、thr再探索・soft penalty化・レジーム条件化で対処する

**Decision（変更理由）**  
vol_cap ON の主な劣化要因は “選択の質” ではなく “gross exposure の上振れ（投下増）” による増幅であるため、vol_cap を活かすには exposure_guard をセットで導入する。

### Exposure Guard 拡張仕様（Neutral 追従のための OffAvg Cap / Base Scaling + Tail Cap）
#### 追加背景（p95 cap の限界が確認された）
cap_source=off_p95（gross_exposure_p95 を OFF 水準に合わせる cap）を全期間＋3分割で適用した結果、p95 の一致自体は達成した（cap=OFF p95 で一致）一方で、投下ニュートラル（neutral_avg）に対して成績が一貫して劣後することが確認された。

ON_with_guard − neutral_avg（return / maxDD）は以下の通り、全期間・全分割でマイナス：
- 全期間：return -0.0527 / maxDD -0.0532
- 2020–2021：return -0.0059 / maxDD -0.0069
- 2022–2023：return -0.0098 / maxDD -0.0159
- 2024–2025：return -0.0107 / maxDD -0.0184

このことは、p95 cap が「テール（上位数％の過剰投下）」には効くが、平常時を含む投下水準（分布の中心）を neutral_avg 相当に落とし切れていないことを意味する。  
よって、Exposure Guard は「p95 テール制御」だけでなく、平均（または中心）投下水準を OFF に寄せる制御を追加する。

#### 目的（拡張）
vol_cap による選択効果（相対配分・ランキング）を保ったまま、neutral_avg に近い投下（gross exposure）水準を実装で再現し、ON_with_guard − neutral_avg の残差（投下起因の劣後）を縮小する。

#### 方式A：OffAvg Cap（中心投下の抑制を優先）
##### 概要
cap_source=off_avg を追加し、gross_cap を OFF の avg_gross_exposure に設定する。  
p95 cap より強く、平常時レンジも含めて投下上限を抑制する。

##### 定義
- gross_exposure_t = Σ|w_i,t|
- gross_cap = avg_gross_exposure_off + cap_buffer

##### 日次適用ルール（既存と同様）
gross_raw > gross_cap の場合、全ポジションを比例縮小：  
w_final = w_raw * (gross_cap / gross_raw)

##### 想定される効果
- p95 cap で残った ON_with_guard − neutral_avg のマイナス（残存投下効果）を縮小しやすい
- 一方で、抑制が強くなるため 過度なキャッシュ化のリスクがある（cap_buffer で調整）

#### 方式B：Base Scaling + Tail Cap（neutral_avg 追従を優先：推奨）
##### 概要（推奨）
neutral_avg が実質的に行っている「常時薄め（一定倍率のスケール）」を、明示的な base_scale として実装する。  
その上で、異常日に備えて tail cap（例：off_p95）を安全弁として残す。

##### アルゴリズム
selection / sizing 後のターゲット w_raw を得る（vol_cap 適用済み）

**Base Scaling（常時適用）**
- w_base = w_raw * base_scale
- base_scale は「投下水準を OFF に寄せる」目的で決める（後述）

**Tail Cap（安全弁）**
- gross_base = Σ|w_base|
- gross_base > gross_cap の場合：  
  w_final = w_base * (gross_cap / gross_base)
- そうでなければ：w_final = w_base

cash = 1 − Σ|w_final|

ログに base_scale と cap 由来の scale_cap、および scale_total を保存する。

比例縮小のみを使うため、銘柄間の相対順位・相対配分は維持され、vol_cap の選択効果を壊しにくい。

##### base_scale の決め方（ルックアヘッド禁止）
本番・OOS を想定し、base_scale は 校正（calibration）期間で推定して固定するか、過去のみを使うローリング推定を用いる。

**固定（推奨：実装が簡単で検証が安定）**
- 校正期間で avg_gross_exposure_on_raw と avg_gross_exposure_off を計測し  
  base_scale = avg_gross_exposure_off / avg_gross_exposure_on_raw
- 以後、OOS ではこの base_scale を固定して適用する

**ローリング（環境変化に追従したい場合）**
- 過去 N 日（例：252営業日）の実績から同様の比率を更新
- ただし過剰適応を避けるため base_scale は clip_min/clip_max でクリップ推奨

##### gross_cap（tail cap）の推奨
gross_cap = gross_exposure_p95_off + cap_buffer（安全弁としてのテール制御）

方式A（off_avg cap）を採らない場合でも、tail cap を残すことで 極端な投下上振れを抑えられる。

#### 設定（config.yaml 追記案）
```yaml
exposure_guard:
  enabled: true
  trigger: vol_cap_enabled
  mode: daily                     # daily | weekly_aligned（検証互換）
  scheme: base_scale_plus_cap      # cap_only | base_scale_plus_cap

  # Base Scaling（neutral_avg 追従の中核）
  base_scale:
    enabled: true
    source: calibrated_fixed       # calibrated_fixed | rolling
    value: null                    # source=calibrated_fixed の場合、校正で算出して埋める
    rolling_window_days: 252       # source=rolling の場合
    clip_min: 0.70
    clip_max: 1.00

  # Tail Cap（安全弁）
  cap_source: off_p95              # off_p95 | off_avg | fixed
  cap_value: null
  cap_buffer: 0.00
  log_scale: true
```

方式A（OffAvg Cap）を使う場合は scheme: cap_only かつ cap_source: off_avg に設定する。

#### 出力・ログ（追加要件）
以下を compare JSON（または run summary）に保存する：
- avg_gross_exposure / gross_exposure_p95 / gross_exposure_max
- guard_applied_count（cap適用回数）
- base_scale_value（固定なら単一値、rollingなら avg/p95/min/max）
- scale_cap と scale_total（avg/p95/min/max）
- ON_with_guard と exposure_neutral (neutral_avg / neutral_weekly) の比較値  
  （投下要因が潰せたかを直接判定するため）

#### 受け入れ基準（Acceptance Criteria：拡張分）
方式Aまたは方式Bの適用後、全期間＋3分割で以下を満たすこと：

**投下水準の整合**
- 方式A（off_avg cap）：avg_gross_exposure_on_with_guard ≈ avg_gross_exposure_off
- 方式B（base scaling）：avg_gross_exposure_on_with_guard が neutral_avg 相当に近づく

**neutral との残差縮小（最重要）**
- ON_with_guard − neutral_avg の差分が、各期間で 0 に近づく（残存投下効果の縮小）  
  ※目標許容幅はプロジェクトKPI（return/DD優先度）に合わせて別途設定する

**安全弁の維持**
- gross_exposure_p95_on_with_guard <= gross_exposure_p95_off + cap_buffer（tail cap を採用する場合）

#### 非目標（Non-goals）
neutral_avg − OFF がマイナスとなる期間（例：2022–2023 等）の selection 起因の劣後を、この Exposure Guard 拡張だけで解消することは目的としない  
（必要なら thr 再探索、soft penalty、レジーム条件化など別施策で対応する）

#### Exposure Guard 検証結果（cap_source=off_avg）
全期間＋3分割にて exposure_guard.base_source=none / cap_source=off_avg を適用したところ、ON_with_guard は投下ニュートラル基準（neutral_avg）をほぼ完全に再現した。

ON_with_guard − neutral_avg（return / maxDD）は全期間・全分割で ≈ 0 に収束：
- 全期間：return +0.003690 / maxDD +0.003736
- 2020–2021：return +0.000250 / maxDD +0.000292
- 2022–2023：return +0.000282 / maxDD +0.000443
- 2024–2025：return +0.000291 / maxDD +0.000507

この結果より、off_avg cap は「投下要因（gross exposure 増加）を消す」目的に対して最短で有効であり、以後の評価では **ON_with_guard（off_avg） vs OFF の差分を selection 起因として解釈できる。**

#### 運用上の注意（ルックアヘッド回避）
cap_source=off_avg における off_avg は、同一評価期間の OFF 結果から算出するとルックアヘッドとなり得る。  
従って、運用・OOS 評価では以下のいずれかを採用する：

- calibrated_fixed：校正期間（training/validation 等）の OFF 平均投下（avg_gross_exposure_off）から gross_cap を算出し、OOS では固定適用
- rolling：過去 N 日（例：252営業日）の実績から gross_cap を推定し、将来情報を使わずに更新（必要に応じてクリップ）

分析用途（原因分解）としてのみ、同期間の OFF から算出した off_avg を使用してよい。

#### 推奨設定（分析用途・原因分解）
```yaml
exposure_guard:
  enabled: true
  base_source: none
  cap_source: off_avg
  cap_buffer: 0.00
```

#### 推奨設定（OOS/運用用途）
```yaml
exposure_guard:
  enabled: true
  base_source: none
  cap_source: fixed              # または calibrated_fixed / rolling 相当の実装
  cap_value: <calibrated_off_avg>
  cap_buffer: 0.00
```

### 正式採用判断用サマリ（ON_with_guard[off_avg] − OFF）
本比較は exposure_guard.base_source=none / cap_source=off_avg により投下要因を実質的に排除したため、**差分はほぼ selection 効果**として解釈できる。

※差分は pp（×100）。maxDD は「プラス＝DD改善（浅くなる）」。

| 期間 | return差 | maxDD差 | 解釈 |
|---|---:|---:|---|
| 全期間 | +2.58pp | +3.05pp | 全体では selection 効果はプラス |
| 2020–2021 | +5.06pp | +5.36pp | 強いプラス |
| 2022–2023 | -4.95pp | +0.35pp | return で明確に負け＝selection 課題 |
| 2024–2025 | -2.09pp | -1.65pp | return/DD ともに悪化 |

結論：  
- 投下要因は **off_avg guard により概ね排除済み**。  
- 以後は **ON_with_guard vs OFF の差分を selection 問題として扱う**。  
- 2022–2023（および 2024–2025）で負けているため、**次の改善は selection 側**に集中する。  
  - 例：thr 再探索 / soft penalty（高ボラ・回転抑制）/ レジーム条件化。

運用上の注意：  
off_avg は分析用途として「同期間の OFF から算出」できるが、OOS では **校正固定（calibrated_fixed）か rolling 推定**でルックアヘッドを避ける。

### 推奨（次の一手）
#### 推奨採用：thr = 0.95
理由は3つ。

1) 全ゲート通過（全期間DD悪化なし、2024–2025非悪化、2022–2023 return非マイナス）
2) これまで課題だった 2022–2023 と 2024–2025で、guard-only（thr=1.00）に対しても上乗せがプラス
3) excluded_rate 0.053 で運用負荷（置換・歪み）が小さい

#### 0.90 は「攻め案」
全期間の改善は最大だが、直近2期間で上乗せがマイナス。  
“歴史全体の最適化”に寄りやすく、直近レジームに対しては頑健性が落ちやすい。

#### 1.00 は「guardだけ採用」の比較ベース
volcapを入れない/入れられない場合でも、off_avg guard自体は有効（少なくともこのバックテストでは）。  
常に戻り先として優秀。

#### 次にやるべきこと（実装・仕様としての“確定作業”）
1) OOSでの cap の作り方を確定  
   - 分析では cap_source=off_avg が最短で効くのは確定  
   - 運用では「同期間OFF平均」参照は避け、校正期間で算出して固定 or 過去のみローリングで cap を作る
2) v2（base_scale + off_p95）で thr=0.95 を再現確認  
   - off_avg guard固定のときに最適なthrが、v2でも同じ最適点になるとは限らない  
   - thr=0.95 を1点だけ v2 で通してゲート確認（全期間＋3分割）  
   - ここでズレるなら、base_scale と cap_buffer の校正を優先

#### 結論
本採用候補は thr=0.95 が最も筋が良い。  
次は thr=0.95 を固定して、運用想定（v2）で再現・ゲート確認し、  
問題なければ **「推奨thr=0.95（excluded_rate≈5%）」として確定**で進める。

### OOS校正方針（運用想定・ルックアヘッド禁止／Walk-forward校正）
#### 推奨設定（結論）
- thr: 0.95（固定）
- Exposure Guard: v2（base_scale + off_p95）
- 校正: 過去データのみで算出 → 次のOOS区間では固定（更新は月次/四半期/年次のいずれか）

これが一番「OOSで再現できて、過学習もしづらい」方針。

#### 校正手順（1回のOOS区間に対して）
OOS開始日を T0、校正窓長を L 日（例：252〜504営業日）とする。

1) 校正期間を決める  
   - 校正期間：[T0-L, T0)（OOS開始日前までのみ）  
   - OOS期間：[T0, T1)（次の更新日まで）

2) 校正期間で3本のランを取る（同一期間）  
   同じ [T0-L, T0) で以下を回して必要統計を取る。  
   - OFF（volcap無効）：OFFの投下分布（ターゲット）  
     - avg_gross_off  
     - gross_p95_off  
   - ON raw（volcap有効・guard無効）：volcapが誘発する投下増の測定  
     - avg_gross_on_raw  
   - （任意）ON v2（volcap有効・guard有効）：校正区間での sanity check

3) base_scale を算出（ルックアヘッド無し）  
   - 推奨式：base_scale = avg_gross_off / avg_gross_on_raw  
   - クリップ例：base_scale = clip(base_scale, 0.80, 1.00)  
   - ロバスト化（任意）：平均が外れ値に弱い場合は trimmed_mean / median へ置換

4) cap（off_p95）を設定  
   - 推奨式：cap = gross_p95_off + cap_buffer  
   - cap_buffer はまず 0.00（必要なら +0.01〜0.02）

5) OOS区間は「固定パラメータ」で走らせる  
   - OOSの [T0, T1) は base_scale と cap を固定して運用  
   - 更新は T1（次の校正タイミング）でのみ行う

#### 更新頻度（おすすめ）
迷うならまずこれで十分。  
- 推奨：四半期更新  
- 校正窓：過去504営業日（約2年）  
- base_scale クリップ：[0.80, 1.00]  
- cap_buffer：0.00  

理由：年次更新はレジーム変化に鈍く、月次更新はノイズに振られやすい。四半期更新が折衷案として妥当。

#### OOSでの合格基準（運用監視）
性能そのもの以上に、設計意図どおり exposure が制御されているかを確認する。

最低限の監視KPI：
- avg_gross_on_with_guard が avg_gross_off に近い（±数pp程度）
- gross_p95_on_with_guard <= cap（capの定義どおり）
- cap適用率（days_applied / total_days）が高すぎない  
  - 目安：5〜20%程度  
  - 30%超が続くなら base_scale が高い / cap が低い可能性

#### 意思決定（短い決定文）
- 推奨設定：thr=0.95、apply_stage=selection、exposure_guard=v2(base_scale + off_p95)
- OOS校正：base_scale = avg_gross_off / avg_gross_on_raw（校正期間はOOS開始日前まで）、cap = gross_p95_off + buffer
- 更新：四半期ごと、校正窓は過去2年、base_scaleはクリップ

#### 次の実装（再現性の担保）
校正を回すスクリプト/モード（例：run_exposure_calibration.py）を用意し、
1回の更新で **base_scale / cap と校正窓（start/end）**をJSONに保存 → OOSランがそれを読む、の形にする。

### OOS結果の要約（thr=0.95 / v2 / 固定校正）
#### 1) OOS結果の要約（OFF基準の改善量）
OOS-1（TEST=2022-2023 / CAL=2020-2021固定）

OFF → ON_guard_fixed
- return: -0.02399 → -0.00352（+0.02046 / +2.046pp）
- maxDD: -0.12401 → -0.10902（+0.01499 / +1.499pp）

exposure整合
- avg_gross: 0.13803 → 0.13872（差 +0.00070、ほぼ一致）
- p95: 0.44190 → 0.43257（-0.00933：capがOFF p95より少し保守的）
- max: 0.47560 → 0.43257（capで上振れを確実に抑制）

運用的観点
- guard_applied: 38週（rate 0.3654）
  - capが実際に働いている点は良いシグナル（ただし過剰拘束かは要監視）

OOS-2（TEST=2024-2025 / CAL=2022-2023固定）

OFF → ON_guard_fixed
- return: -0.03735 → +0.01199（+0.04934 / +4.934pp）
- maxDD: -0.09098 → -0.06767（+0.02331 / +2.331pp）

exposure整合
- avg_gross: 0.11926 → 0.11418（差 -0.00509：やや薄め）
- p95: 0.37923 → 0.34170（-0.03753：OFFよりかなり保守的）
- max: 0.51337 → 0.42330（上振れ抑制が効いている）

ON_rawも強い
- ON_rawですでに return/DD がOFFより良い
- guardは returnをほぼ維持しつつDDをさらに改善（設計意図どおり）

#### 2) ここまでで「確定したこと」
- 校正値（base_scale / cap）を次期間に固定しても破綻しない（ルックアヘッド無しでOOSに耐える）
- v2の役割分担がOOSでも成立
  - base_scale: 平均投下（avg_gross）を合わせに行く
  - off_p95 cap: 投下の尻尾（p95/max）を切る
  - OOS-1はcap寄り、OOS-2はbase_scale寄りで機能
- guard_applied率の解釈に注意
  - base_scale適用とcap拘束が混ざっている可能性が高い
  - 次段では base_scale適用とcap拘束（bind）を別カウントに分離する

#### 3) 次にやること（最短の“本番形”確認）
Next-1: ローリング更新で 2020-2025 を一本で回す
- 更新頻度: 四半期更新（推奨）
- 校正窓: 過去2年（約504営業日）
- 各更新点で base_scale / cap を算出 → 次の四半期は固定
- 1本の成績（return/DD/turnover/applied率）で採用判断を完成させる

Next-2: ログ項目を監視用に整理
- base_scale_value
- cap_value
- cap_bind_days（gross_base > cap の日数）
- cap_bind_rate
- scale_total_avg/p95/min/max
- avg_gross_guard / p95_guard / max_guard

Next-3: 保守性が強すぎないかの確認
- avg_gross_guard がOFF比で恒常的に低すぎないか（例: -5pp以上が継続）
- 低すぎる場合の調整案
  - base_scale のクリップ上限を緩める（例: max=1.02）
  - cap_buffer を +0.01〜0.02

#### 4) 今日の到達点としての判断
thr=0.95 + v2 + 校正→次期間固定は、OOS-1/OOS-2の両方で return改善 & DD改善が出ている。  
次は「rolling更新を含む全期間1本の検証」と「ログ分離」で、運用設計として完成させる段階。

---

## 未確定事項とバックログ

---

## Runbook（実行コマンド）
> 実行は `.\.venv\Scripts\python` を推奨（mlstock未検出の再発防止）

### 初回
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
.\.venv\Scripts\python scripts\run_weekly.py
```

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
Get-ChildItem artifacts/orders/orders_*.csv    | Sort-Object Name | Select-Object -Last 1
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

## 用語・定義
- **week_end_date**：その週の最終取引日（calendar基準）
- **entry_date**：week_end の次の取引日（通常月曜）
- **exit_date**：次週のentry（= 1週間後の寄り相当）
- **weekly_pnl**：`nav - nav_prev`
- **pct_change**：`weekly_pnl / nav_prev`
- **cash_ratio**：`cash_usd / nav`
- **top-share**：週次寄与（銘柄別PnL）に対する上位寄与比率（gross基準推奨）

---

> 本仕様は「このチャットで決めた固定関係」を一次情報として記述している。  
> 追加の設計変更（例えば volcap を training にも適用する等）は、`未確定事項とバックログ` を実施した上で仕様更新する。
