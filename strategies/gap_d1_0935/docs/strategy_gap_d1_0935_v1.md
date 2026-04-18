# 前日ギャップ候補 × 当日9:35確認 デイトレ戦略 仕様書 v1.0

## 1. 目的

本戦略は、前営業日に需給・材料・出来高が集中した中小型株を翌営業日に監視し、
当日寄り後5分の値動きで continuation を確認してからエントリーし、当日中に全決済することを目的とする。

本仕様の前提:
- 無料データ前提
- 持ち越しなし
- premarket movers 自動取得なし
- broad 2000銘柄の寄り直後リアルタイム scan なし
- 既存 gap strategy 本線は維持し、新規系統として実装する

---

## 2. 戦略思想

探索と執行確認を分ける。

- D-1（前営業日）: 日足で候補を作る
- D（当日）: 9:35 ET に最初の5分足で continuation を確認する
- Entry: 9:36〜10:00
- Exit: 当日中のみ

従来の「当日朝に broad universe を寄り直後 scan」する方式は、無料データでは coverage と安定性に問題がある。
そのため、本戦略では日足ベースの watchlist を前日確定し、当日は watchlist のみを見る。

---

## 3. 非対象

以下は本仕様では扱わない。

- 持ち越し
- premarket movers 自動取得
- broad 2000銘柄 minute scan
- 超初動 9:30 直後スキャル
- 板 / tick ベース執行
- short
- fade / reversion
- moomoo 単独での本線運用

---

## 4. データソース方針

### 4.1 本線
- Alpaca Basic(IEX)
- 用途:
  - D-1 日足
  - D 当日 minute / 5m
  - replay / backtest

### 4.2 補助
- yfinance
- 用途:
  - market cap
  - sector
  - security type
  - 静的補完
  - 欠損時の補助確認

### 4.3 補助2
- moomoo
- 本線には使わない
- 用途:
  - large/mid の snapshot 照合
  - snapshot.open_price の確認
  - Unknown stock を使った補助除外

---

## 5. Universe

### 基本対象
米国普通株のみ

### 除外
- ETF
- ADR
- preferred
- warrant
- right
- unit
- OTC
- 極端な低流動株
- 極端な低位株

### 初期フィルタ
- 3 <= close <= 30
- avg_volume_20 >= 300000
- avg_dollar_volume_20 >= 2000000
- 50M <= market_cap <= 10B

---

## 6. D-1 watchlist 仕様

### 6.1 目的
前営業日 D-1 に、翌営業日 D に監視する候補銘柄を抽出する。

### 6.2 必要データ
- open_D-1
- high_D-1
- low_D-1
- close_D-1
- close_D-2
- volume_D-1
- avg_volume_20
- avg_dollar_volume_20
- market_cap
- sector
- security_type

### 6.3 特徴量
- prev_gap_pct = (open_D-1 - close_D-2) / close_D-2 * 100
- rel_vol_prev = volume_D-1 / avg_volume_20
- close_in_range_prev = (close_D-1 - low_D-1) / (high_D-1 - low_D-1)
- oc_ret_prev = (close_D-1 - open_D-1) / open_D-1 * 100

### 6.4 候補条件
- prev_gap_pct >= +5%
- rel_vol_prev >= 2.0
- close_in_range_prev >= 0.70
- oc_ret_prev > 0

### 6.5 地合い列
- index_ret_D-1
- sector_ret_D-1

### 6.6 出力ファイル
`artifacts/watchlist/watchlist_gap_d1_YYYYMMDD.csv`

### 6.7 出力列
- symbol
- trade_date
- open_D-1
- high_D-1
- low_D-1
- close_D-1
- close_D-2
- prev_gap_pct
- rel_vol_prev
- close_in_range_prev
- oc_ret_prev
- market_cap
- avg_volume_20
- avg_dollar_volume_20
- sector
- security_type
- index_ret_D-1
- sector_ret_D-1
- selected_reason

### 6.8 ログ
- universe_count
- excluded_non_common_count
- excluded_price_count
- excluded_liquidity_count
- excluded_market_cap_count
- excluded_gap_count
- excluded_rel_vol_count
- excluded_close_strength_count
- selected_count

---

## 7. 当日9:35判定仕様

### 7.1 目的
D-1 watchlist を当日寄り後5分で再確認し、continuation 候補だけを残す。

### 7.2 判定時刻
- 09:35:05 ET 以降
- first 5m bar 確定後

### 7.3 使用データ
- 当日始値 open_D
- 前日終値 close_D-1
- 09:30-09:35 の first 5m bar
- intraday VWAP または近似値
- D-1 watchlist 情報

### 7.4 特徴量
- gap_today_pct = (open_D - close_D-1) / close_D-1 * 100
- first5_open
- first5_high
- first5_low
- first5_close
- first5_volume
- first5_range_pos = (close_5m - low_5m) / (high_5m - low_5m)
- first5_oc_ret = (close_5m - open_5m) / open_5m * 100
- first5_pace = (volume_5m * 78) / avg_volume_20
- vwap

### 7.5 continuation 条件
- open_D > close_D-1
- gap_today_pct >= +1.0%
- first5_range_pos >= 0.60
- first5_oc_ret > 0
- first5_pace >= 1.5
- close_5m >= vwap

### 7.6 fail reason
- missing_open
- missing_first5
- gap_fail
- range_fail
- oc_ret_fail
- pace_fail
- vwap_fail

### 7.7 出力ファイル
`artifacts/scans/gap_0935_candidates_YYYYMMDD.csv`

### 7.8 出力列
- symbol
- trade_date
- open_D
- close_D-1
- gap_today_pct
- first5_open
- first5_high
- first5_low
- first5_close
- first5_volume
- first5_pace
- first5_range_pos
- first5_oc_ret
- vwap
- pass
- fail_reason

### 7.9 ログ
- watchlist_count
- open_ok_count
- first5_ok_count
- gap_pass_count
- range_pass_count
- pace_pass_count
- vwap_pass_count
- final_candidate_count

---

## 8. エントリー仕様

### 対象
- long continuation のみ
- breakout のみ

### 時間帯
- 09:36:00 ET 〜 10:00:00 ET

### パターン
- first5_high breakout
- 9:35判定通過後、現在値が first5_high を上抜いたら entry

---

## 9. ポジションサイズ

- stop_price = first5_low
- risk_per_share = entry_price - stop_price
- qty = floor(risk_per_trade_usd / risk_per_share)

制約:
- 1銘柄最大 notional 上限
- 同時建玉上限
- 極端な価格飛び銘柄は除外

---

## 10. エグジット仕様

### 損切り
- first5_low 割れ

### 利確
- 2R

### 時間切れ
- 15:55 ET 強制決済
- no overnight

### exit reason
- stop
- target
- time_exit

---

## 11. バックテスト仕様

### シグナル生成
- D-1 日足で watchlist 作成
- D の first 5m で continuation 判定

### 執行仮定
- breakout は first5_high 突破時に entry
- 次バー始値またはスリッページ加味の近似

### コスト
- 手数料
- スリッページ
- broker 想定コスト

### 検証単位
- 日次
- 月次
- 年次
- bull / bear / high vol / low vol
- sector 別
- gap帯別

---

## 12. 評価指標

- watchlist_count/day
- final_candidate_count/day
- entry_count/day
- win_rate
- avg_pnl_per_trade
- expectancy
- max_drawdown
- holding_time
- gap帯別成績
- sector別成績
- regime別成績

---

## 13. 追加特徴量候補

### 推奨
- index_ret_D-1
- sector_ret_D-1
- theme_strength_label
- ATR比
- float / float_turnover
- news category

### 後回し
- option flow
- borrow fee
- SNS trend
- 板 / tick マイクロ構造

---

## 14. 実装ファイル

### D-1 watchlist
- src/mlstock/jobs/build_gap_d1_watchlist.py
- scripts/run_build_gap_d1_watchlist.py

### 当日9:35判定
- src/mlstock/jobs/gap_0935_watchlist_scanner.py
- scripts/run_gap_scan_0935_watchlist.py

### replay / 比較
- scripts/replay_gap_0935_watchlist.py
- scripts/compare_gap_old_vs_0935.py

### 将来フェーズ
- src/mlstock/jobs/gap_0935_executor.py
- scripts/backtest_gap_0935.py

---

## 15. 出力物

- artifacts/watchlist/watchlist_gap_d1_YYYYMMDD.csv
- artifacts/scans/gap_0935_candidates_YYYYMMDD.csv
- artifacts/reports/gap_0935_compare_YYYYMMDD.csv
- artifacts/reports/gap_0935_compare_summary.csv
- artifacts/logs/gap_0935_scan_YYYYMMDD_HHMMSS.jsonl

---

## 16. フェーズ分割

### Phase 1
- D-1 watchlist builder
- D 当日 9:35 scanner
- replay / old 比較

### Phase 2
- breakout entry backtest
- stop / target / time_exit

### Phase 3
- 押し目 entry
- score 化
- sector / theme 加点

### Phase 4
- fade / short を別戦略として追加

---

## 17. 受け入れ条件

### Phase 1 完了条件
- D-1 watchlist CSV が出る
- 9:35 candidates CSV が出る
- fail_reason が銘柄単位で出る
- replay が 1日以上通る
- old vs 0935 の比較 CSV が出る

### Phase 2 完了条件
- breakout entry の backtest が通る
- stop / target / time_exit がログ化される
- 日次 PnL が集計できる

---

## 18. 現時点の運用判断

### 本線
- Alpaca Basic(IEX) を本線に使う
- yfinance は補助のみ
- moomoo は補助のみ

### 捨てる前提
- premarket movers 自動取得
- free での broad 2000 銘柄 minute scan
- 9:30 bar 完全依存

### 残す思想
- 中小型 gap を狙う
- 朝の強い継続だけを取る
- no overnight

---

## 19. 一文まとめ

前日に強く gap した中小型株を候補化し、翌日 9:35 の first 5m で continuation を確認し、9:36〜10:00 に breakout でロングし、当日中に閉じる。
