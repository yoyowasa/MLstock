# GAP_PROJECT_SPEC.md — Gap + Options Intraday Strategy

> 本ファイルは `C:\BOT\MLStock` 内の **gap 戦略専用仕様書** とする。  
> weekly パイプラインの仕様は `PROJECT_SPEC.md` を参照する。  
> 実装は本仕様に従い、運用ログと変更履歴は `STATUS.md` に追記する。

---

## Table of Contents
1. [目的](#目的)
2. [仕様対象とスクリプト境界](#仕様対象とスクリプト境界)
3. [スコープ](#スコープ)
4. [設定ファイル](#設定ファイル)
5. [ブローカー方針](#ブローカー方針)
6. [接続認証比較](#接続認証比較)
7. [実行モード](#実行モード)
8. [スキャン仕様](#スキャン仕様)
9. [オプションフィルタ仕様](#オプションフィルタ仕様)
10. [トレーダー仕様](#トレーダー仕様)
11. [PNL とリプレイ仕様](#pnl-とリプレイ仕様)
12. [自動実行仕様](#自動実行仕様)
13. [ログ契約](#ログ契約)
14. [weekly 系との境界](#weekly-系との境界)
15. [未解決事項](#未解決事項)

---

## 目的
米国株の寄付き gap を対象に、寄り直後の候補抽出、必要ならオプション需給確認、日中の押し目エントリー、利確・損切・強制クローズまでを一連で扱う。

主目的は次の3点。
- 毎朝の `scan-only` で、どの銘柄がギャップ条件に引っかかるかを継続収集する
- dry-run で intraday entry/exit 条件の発火頻度と実現損益を観測する
- live 前に、候補頻度・約定頻度・PNL を同一ログ系列で検証可能にする

---

## 仕様対象とスクリプト境界

### gap 系の主スクリプト
- `scripts/run_gap_trade.py`

### gap 系の補助スクリプト
- `scripts/run_gap_scan_only.ps1`
- `scripts/check_webull_auth.py`

### gap 系の主モジュール
- `src/mlstock/jobs/gap_scanner.py`
- `src/mlstock/jobs/options_filter.py`
- `src/mlstock/jobs/gap_trader.py`

### gap 系の設定
- `config/gap_config.yaml`

### gap 系の主出力
- `artifacts/logs/gap_trade_YYYYMMDD_HHMMSS.jsonl`

### 対象外
- 週次選定、週次バックテスト、注文CSV、portfolio.json は本仕様の対象外
- それらは `PROJECT_SPEC.md` の対象とする

---

## スコープ

### スコープ内
- 当日寄付き gap 候補の抽出
- 30日平均出来高と出来高ペースによる流動性確認
- yfinance を用いた時価総額確認
- yfinance option chain を用いた UOA フィルタ
- intraday 1分足での entry/exit 判定
- dry-run / live の共通ログ出力
- `--replay-log` による後追い仮想PNL集計

### スコープ外
- 数日保有、スイング保有
- weekly portfolio との資金統合
- 高頻度執行最適化
- 板情報やティックベース執行

---

## 設定ファイル
設定は `config/gap_config.yaml` を正とする。主要項目は次のとおり。

### gap
- `min_gap_pct`
- `max_gap_pct`
- `min_volume_pace_ratio`
- `min_avg_volume_30d`
- `max_scan_candidates`
- `lookback_volume_days`
- `market_cap_source`

### options
- `min_call_volume`
- `min_call_oi_ratio`
- `min_call_put_ratio`
- `min_expiry_days`
- `yfinance_delay_sec`
- `allow_error_fallback`

### entry
- `max_candidates`
- `pullback_from_open_pct`
- `vwap_tolerance_pct`
- `low_volume_ratio`
- `risk_per_trade_usd`
- `dry_run_cash_usd`
- `max_notional_per_trade_usd`
- `min_order_qty`

### exit
- `target_pct`
- `stop_pct`
- `time_cut_hour`
- `force_close_hour`
- `force_close_minute`

### universe
- `use_mlstock_seed`
- `min_price`
- `max_price`
- `min_market_cap_m`

---

## ブローカー方針

### データ取得
- データ取得は Alpaca を使う
- 対象は日足、当日1分足、clock とする

### 発注・決済
- gap 戦略では Alpaca に発注しない
- 発注・決済は別ブローカーへ分離する

### 候補順位
1. Webull
2. moomoo

### 採用方針
- Webull を第一候補とする
- Webull の認証運用が実運用に耐えない場合のみ moomoo を再評価する

### broker interface の責務
- `get_buying_power`
- `list_positions`
- `submit_market_buy`
- `submit_market_sell`
- `close_position`

### 実装モジュール
- `src/mlstock/brokers/base.py`
- `src/mlstock/brokers/alpaca.py`
- `src/mlstock/brokers/webull.py`

### 認証診断スクリプト
- `scripts/check_webull_auth.py`
- 目的:
  - account list の 401/200 切り分け
  - token create/check の単体確認
  - `.env` の Webull 設定確認

### 設計ルール
- scanner と minute bar 取得は Alpaca に残す
- trader の注文処理だけを broker interface 経由に置換する
- dry-run / replay は broker 実装に依存しない

### 現在の実装段階
- `OrderBroker` 抽象 interface は実装済み
- `AlpacaOrderBroker` は既存挙動維持のための暫定アダプタとして実装済み
- `WebullOrderBroker` は SDK 接続と基本注文系まで実装済み
- 実口座の疎通確認は未実施

### WebullOrderBroker の設計仕様

#### 目的
- gap trader の売買処理を Alpaca から切り離し、Webull 発注へ差し替え可能にする

#### クラス責務
- Webull API 認証情報の保持
- 残高照会の正規化
- 保有ポジション照会の正規化
- 成行買い注文の送信
- 成行売り注文の送信
- ポジションクローズの送信

#### 入力
- `base_url`
- `app_key`
- `app_secret`
- `access_token` または同等の再利用可能 token
- `paper_trading`

#### 環境変数
- `GAP_ORDER_BROKER`
- `WEBULL_REGION`
- `WEBULL_APP_KEY`
- `WEBULL_APP_SECRET`
- `WEBULL_ACCESS_TOKEN`
- `WEBULL_PAPER_TRADING`
- `WEBULL_BASE_URL`
- `WEBULL_OPENAPI_TOKEN_DIR`
- `WEBULL_ACCOUNT_TAX_TYPE`

#### broker 選択ルール
- `GAP_ORDER_BROKER=alpaca`
  - 既存互換の暫定経路
- `GAP_ORDER_BROKER=webull`
  - WebullOrderBroker を選択
  - Webull JP SDK 経由で残高取得、保有一覧、成行買い、成行売り、クローズを実行する

#### 返却契約
- 残高は `float`
- ポジション一覧は `BrokerPosition[]`
- 注文結果は `BrokerOrderResult`

#### エラー方針
- 認証情報不足は即時例外
- API 接続失敗、認証失効、注文拒否は例外で上位へ返す
- `gap_trader.py` 側で `entry_failed` / `exit_failed` としてログ化する

#### 認証ライフサイクル
- App Key / App Secret は管理画面発行の値を使う
- 現在の JP 実装では access token は必須にしていない
- `WEBULL_OPENAPI_TOKEN_DIR` は将来 token 保存が必要になった場合の予約項目として残す
- App Secret 失効時は broker 実装内で自動再生成せず、運用手順で更新する

#### 非目標
- Webull の key reset 自動化
- 2FA フローそのものの自動突破
- 板情報や複雑注文

---

## 接続認証比較

### Webull

#### 接続前提
- Webull 口座が必要
- API access の申請と審査が必要
- 承認後に App Key / App Secret を発行する
- 現在の実装は `JP OpenAPI` フローを前提とする

#### 認証方式
- OpenAPI は App Key / App Secret による署名方式
- 現在確認できた JP フローでは、`get_app_subscriptions()` / `get_account_balance()` / `get_account_position()` に access token は不要

#### 運用上の重要点
- App Secret の既定有効期限は 1 日
- 次回 reset に対して最大 7 日まで有効期間を設定できる
- Secret 期限切れ後は reset が必要

#### 評価
- 毎回フルログインする構造ではない
- 一方で、App Secret の期限管理は必須
- 長期自動運用では key 更新手順を運用仕様に入れる必要がある

### moomoo

#### 接続前提
- OpenD の常駐が必要
- API クライアントは OpenD に接続して発注する

#### 認証方式
- live 取引では `unlock_trade` が必要
- `unlock_trade` には取引パスワードを使う
- paper では unlock 不要

#### 運用上の重要点
- `moomoo token` を有効にしていると OpenAPI の unlock が失敗する
- API 認証よりも OpenD 常駐の運用負荷が大きい

#### 評価
- 毎回フルログインを要求する構造ではない
- ただし OpenD の生存監視と trade unlock の管理が必要
- Windows 常駐前提のため、運用は Webull より重い

### 比較結論
- 実装しやすさ: Webull が優位
- 運用の単純さ: Webull が優位
- 認証の手離れ: Webull は Secret 更新が必要、moomoo は OpenD 常駐が必要
- 現時点の第一候補は Webull とする

---

## 実行モード

### 1. scan-only
用途:
- 毎朝の候補収集
- Stage 2 のログ蓄積

起動例:
- `scripts/run_gap_trade.py --scan-only`

動作:
- market open 確認
- scanner 実行
- `scanner_complete` を出して終了

### 2. dry-run intraday
用途:
- Stage 3 以降の entry/exit 条件確認
- 実注文なしで日中挙動を確認

起動例:
- `scripts/run_gap_trade.py --skip-options`
- `scripts/run_gap_trade.py`

動作:
- scanner
- 必要なら options filter
- trader 実行
- `entry_filled` / `exit_filled` / `gap_trader_complete`

### 3. live
用途:
- paper または live 環境で実注文を送る

起動例:
- `scripts/run_gap_trade.py --live`

注意:
- デフォルトは dry-run
- `--live` 指定時のみ注文送信

### 4. replay-log
用途:
- 過去の scan-only ログを読み込み、その日の1分足で仮想PNLを再計算する

起動例:
- `scripts/run_gap_trade.py --replay-log <jsonl>`

出力:
- `scan_replay_complete`

---

## スキャン仕様

### 入力
- 前日終値
- 30日平均出来高
- 当日寄り後の最初の5分窓
- price / market cap 情報

### フィルタ順
1. `avg_volume_30d >= min_avg_volume_30d`
2. `min_price <= open_price <= max_price`
3. `min_gap_pct <= gap_pct <= max_gap_pct`
4. `volume_pace_ratio >= min_volume_pace_ratio`
5. `market_cap_m >= min_market_cap_m`

### 指標
- `gap_pct = (open_price - prev_close) / prev_close * 100`
- `daily_volume_pace = first_window_volume * (390 / bars_in_window)`
- `volume_pace_ratio = daily_volume_pace / avg_volume_30d`

### 時価総額
- yfinance 系の取得を使う
- 取得不能銘柄は安全側で除外
- 失敗状況は `market_cap_fetch_summary` に記録
- 個別除外は `market_cap_filter_drop` に記録

### 出力
- `scanner_complete`
- 最大件数は `gap.max_scan_candidates`

---

## オプションフィルタ仕様

### 実行タイミング
- `--scan-only` では未実行
- `--skip-options` 指定時は未実行
- それ以外では scanner 後に実行

### 判定対象
- scanner 通過銘柄のうち上位 `entry.max_candidates` 件まで

### 条件
- `calls.volume.sum() >= min_call_volume`
- `calls.volume.sum() / calls.openInterest.sum() >= min_call_oi_ratio`
- `calls.volume.sum() / puts.volume.sum() >= min_call_put_ratio`
- 最短限月は `min_expiry_days` 以上

### 失敗時
- `allow_error_fallback=false` を既定とする
- その場合、yfinance 失敗銘柄は通さない

---

## トレーダー仕様

### 監視粒度
- 1分ごと

### エントリー条件
以下を全て満たす場合に entry 候補とする。
- `current_price <= open_price * (1 - pullback_from_open_pct / 100)`
- `last_1m_volume <= avg_prev_five_volume * low_volume_ratio`
- `current_price >= VWAP * (1 - vwap_tolerance_pct / 100)`

### サイジング
数量は次の小さい方を採用する。
- リスク上限ベース
- 予算上限ベース

概念式:
- `qty_by_risk = risk_per_trade_usd / (current_price * stop_pct)`
- `qty_by_notional = per_trade_budget / current_price`
- `qty = min(qty_by_risk, qty_by_notional)`

### 新規エントリー制約
- `entry.max_candidates` 件まで
- `exit.time_cut_hour` 以降は新規エントリーしない

### エグジット
- 利確: `current_price >= entry_price * (1 + target_pct / 100)`
- 損切: `current_price <= entry_price * (1 - stop_pct / 100)`
- 強制クローズ: `force_close_hour:force_close_minute`

---

## PNL とリプレイ仕様

### 実運用・dry-run の損益ログ
- `exit_filled`
  - `realized_pnl_usd`
  - `realized_pnl_pct`
- `force_close_exit`
  - `realized_pnl_usd`
  - `realized_pnl_pct`
- `gap_trader_complete`
  - `closed_trades`
  - `realized_pnl_usd`
  - `realized_pnl_pct`
  - `closed_entry_notional_usd`

### scan-only 後追い分析
- `--replay-log` で `scanner_complete` を読み込む
- 同日の1分足を再取得する
- 現行 entry/exit ロジックで仮想売買する
- `scan_replay_complete` を出力する

### 注意
- replay は「その日このロジックで入っていたら」の再現であり、実約定結果ではない
- replay 結果は当日の候補品質と entry 条件の厳しさを観測する目的で使う

---

## 自動実行仕様

### タスク
- タスク名: `MLStock_GapScan_093005`
- 起動時刻: 毎日 `22:00 JST`

### 実行順
1. タスクスケジューラが `run_gap_scan_only.ps1` を起動
2. PowerShell から `run_gap_trade.py --scan-only` を起動
3. Python 側が `America/New_York 09:30:05` まで待機
4. scanner 実行
5. `scanner_complete` を保存して終了

### DST
- `ZoneInfo("America/New_York")` による自動追従
- タスク時刻の季節変更は不要

---

## ログ契約

### 主なイベント
- `start`
- `wait_for_scan`
- `clock_status`
- `preflight_iex_bars`
- `market_cap_fetch_summary`
- `market_cap_filter_drop`
- `scanner_complete`
- `options_filter_complete`
- `entry_check`
- `entry_filled`
- `exit_filled`
- `force_close_exit`
- `gap_trader_complete`
- `scan_replay_complete`

### ログの用途
- scanner 通過頻度の観測
- options filter の脱落率確認
- entry 条件の発火頻度確認
- 実現損益と仮想損益の比較

---

## weekly 系との境界

### shared
- Alpaca 接続設定
- seed symbols の参照
- artifacts / logging の共通基盤

### 分離
- weekly:
  - `PROJECT_SPEC.md`
  - `scripts/run_weekly.ps1`
  - `scripts/run_weekly.py`
- gap:
  - `GAP_PROJECT_SPEC.md`
  - `scripts/run_gap_trade.py`
  - `scripts/run_gap_scan_only.ps1`

### 禁止
- gap 用仕様を `PROJECT_SPEC.md` に混在させない
- weekly 用注文CSV仕様を本書に混在させない

---

## 未解決事項
- options filter を live 前にどこまで厳格化するか
- market cap 取得の代替ソースを追加するか
- force close 時の約定価格扱いをどこまで厳密化するか
- replay 集計の対象日を複数日バッチ処理する補助スクリプトを作るか

---

## 実装完了条件（P0-P2）

### P0
- `scan-only` が毎営業日安定稼働し、`scanner_complete` の日次確認ができる
- `dry-run` が安定稼働し、entry/exit と日次PNLを同一ログ系列で追える
- `--replay-log` で後追い再計算ができる
- Webull broker で残高取得、保有取得、成行 buy/sell、close が通る

### P1
- Webull 実注文の最小検証が完了する
- `entry_failed` / `exit_failed` の扱いと再実行ルールが確定する
- 実約定価格とログ価格の差分を確認できる
- 1日損失上限、同時建玉上限、連続発注防止を実装する

### P2
- UOA フィルタ込みの本番可否を判断できる
- 候補頻度、発火頻度、実約定頻度、実PNL の監視指標を固定する
- 戦略横断の発注統制レイヤーへ接続する
