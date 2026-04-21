# STATUS.md

## 目的
- `gap_d1_0935` 戦略専用の実装・修正・分析履歴を本線と分離して管理する。
- 本線 gap strategy とは別系統として、D-1 watchlist + 当日 9:35 continuation 確認の進捗を記録する。

## 現在状態
- 運用観測フェーズ
- main `reclaim_first5_high`
- compare `reclaim_vwap`
- reference `continuation_compare`
- strategy 専用 folder 作成済み
- 仕様書配置済み
- 本線 `src/mlstock/jobs/gap_*` は未変更で維持

## 実装履歴
- 2026-04-19: strategy 専用 folder を作成した。

## 修正履歴
- なし

## 仕様変更履歴
- 2026-04-19: `docs/strategy_gap_d1_0935_v1.md` を本戦略の一次仕様として固定した。

## 未解決事項
- market cap / sector / security_type の yfinance 依存をどこまで許容するかは運用で再確認が必要。
- index_ret_D-1 / sector_ret_D-1 の精度は簡易版で、将来の厳密化余地あり。
- compare の old 系 count は root `artifacts/logs` の `scanner_complete` に依存する。

## 更新ルール
- 本戦略に関する変更・修正・分析はここへ追記する。
- 本線 gap strategy の履歴はここへ書かない。
- 未確認事項は確定事項として書かない。
- 追記は日付付き・簡潔に行う。
- 2026-04-19: `src/gap_d1_0935/build_gap_d1_watchlist.py`, `gap_0935_watchlist_scanner.py`, `replay_gap_0935_watchlist.py`, `compare_gap_old_vs_0935.py` と strategy 専用 scripts を追加した。出力先は `strategies/gap_d1_0935/artifacts` に固定し、本線 artifacts と混在しないようにした。
- 2026-04-19: 最小検証として `AAL, AAOI, AXTI, CRGY` の 4銘柄で `trade_date=2026-04-17` を replay し、watchlist CSV / 9:35 scan CSV が生成されることを確認した。結果は `watchlist 0 / scan 0` で、経路確認まで完了。
- 2026-04-19: `src/gap_d1_0935/analysis_phase1.py` と `scripts/analyze_phase1_population.py` を追加し、直近3か月の D-1 watchlist / 当日9:35 candidates / drop counts / symbol理由 / old broad gap との差分を CSV 出力できるようにした。出力先は `strategies/gap_d1_0935/artifacts/reports` に固定した。
- 2026-04-19: 2026-01-14〜2026-04-17 の 65 営業日で母数確認を実施した。結果は `avg_watchlist_count=0.015`, `avg_candidate_0935_count=0.0`, `watchlist_zero_days=64`, `scan_zero_days=65`。watchlist 成立は 2026-01-30 の `AMPX` 1件のみで、当日 9:35 では `gap_fail|pace_fail` で不通過だった。
- 2026-04-19: D-1 側の主な drop は `price_fail` と `liquidity_fail` で、3か月累計は `price_fail=64000`, `liquidity_fail=61673`, `gap_fail=1849`, `market_cap_fail=1147`, `non_common_fail=208`。old broad gap scanner との overlap 26 営業日では、新戦略候補は 0、old 側平均 1.92 件/日で、差分主因は `price_fail`, `liquidity_fail`, `gap_fail`, `market_cap_fail` だった。
- 2026-04-19: config を v1.1 に緩和した。変更は `max_close 30->60`, `min_avg_volume_20 300000->100000`, `min_avg_dollar_volume_20 2000000->1000000`, `max_market_cap 10B->20B`, `min_prev_gap_pct 5%->3%`。`rel_vol_prev`, `close_in_range_prev`, `oc_ret_prev` は維持した。
- 2026-04-19: v1.1 で 2026-01-14〜2026-04-17 を再集計した。`avg_watchlist_count=0.277`, `watchlist_zero_days=48`, `candidate_0935_count_total=2` まで改善した。通過日は `2026-03-18 BTSG`, `2026-04-01 ALKS`。一方で old broad gap scanner との共通候補は依然 0。
- 2026-04-19: v1.0 比で `watchlist_count 1->18`, `candidate_0935_count 0->2` に改善した。まだ D-1 側では `liquidity_fail`, `price_fail`, `gap_fail` が大きく、scan 側では `gap_fail`, `vwap_fail`, `range_fail` が主な詰まり。
- 2026-04-19: config を v1.2 に緩和した。変更は `max_close 60->80`, `min_avg_dollar_volume_20 1000000->500000`, `max_market_cap 20B->30B`, `min_prev_gap_pct 3%->2%`。`rel_vol_prev`, `close_in_range_prev`, `oc_ret_prev` は維持した。
- 2026-04-19: v1.2 で 2026-01-14〜2026-04-17 を再集計した。`avg_watchlist_count=1.277`, `watchlist_zero_days=6`, `candidate_0935_count_total=4`。通過日は `2026-01-26 ALM`, `2026-03-18 BTSG`, `2026-04-01 ALKS`, `2026-04-01 AMPX`。old broad gap scanner との共通候補は依然 0。
- 2026-04-19: v1.1 比で `watchlist_count 18->83`, `candidate_0935_count 2->4` に改善した。一方で D 当日側 `missing_first5`, `gap_fail`, `vwap_fail`, `range_fail` が増え、次の主ボトルネックは 9:35 continuation 判定側へ寄った。
- 2026-04-19: `gap_0935_watchlist_scanner.py` に `missing_first5` の内訳項目 `open_exists`, `minute_bars_in_0930_0935`, `first5_constructible`, `missing_reason`, `remarks` を追加した。`analysis_phase1.py` も同項目を集計し、`phase1_missing_first5_detail_*`, `phase1_missing_first5_daily_*`, `phase1_missing_first5_symbol_*` を出力するよう更新した。
- 2026-04-19: config を v1.3 に調整した。D-1 条件は v1.2 を維持し、D 当日 9:35 側のみ `min_first5_range_pos 0.60->0.50`, `min_close_vs_vwap_ratio 1.0->0.998` を適用した。
- 2026-04-19: v1.3 で 2026-01-14〜2026-04-17 を再集計した。`avg_watchlist_count=1.277`, `avg_candidate_0935_count=0.062`, `scan_zero_days=62` で v1.2 と同値。`scan_range_fail_count 17->15`, `scan_vwap_fail_count 18->14` には改善したが、通過銘柄は `ALM`, `BTSG`, `ALKS`, `AMPX` の 4件で増加なし。
- 2026-04-19: `missing_first5` 50件は全件 `AXL` の `no_minute_bars` だった。`partial_minute_bars`, `late_open_or_halt`, `symbol_issue`, `unknown` は 0 件で、現時点の主因は「特定銘柄の minute 欠損」であり、条件緩和では解消しないことを確認した。
- 2026-04-19: config を v1.4 に調整した。変更は D 当日 9:35 側の `min_gap_today_pct 1.0->0.5` のみで、他条件は v1.3 を維持した。
- 2026-04-19: v1.4 で 2026-01-14〜2026-04-17 を再集計した。`avg_candidate_0935_count 0.062->0.077`, `scan_zero_days 62->61`, `scan_gap_fail_count 23->20` と改善し、pass は `2026-01-22 ALM` が 1件追加されて計 5件になった。
- 2026-04-19: `AXL` を一時除外した比較では `missing_first5_count 50->0` だが、`avg_candidate_0935_count=0.077`, `scan_zero_days=61` は不変だった。`AXL` は watchlist 母数を押し上げるだけで、候補通過数には寄与していない。
- 2026-04-19: v1.4 の fail overlap は `gap_only_fail=6`, `gap+pace_fail=2`, `gap+oc_ret_fail=2`, `gap+range_fail=0`, `gap+vwap_fail=0`。`gap_fail` は単独または `oc_ret/pace` との重なりが中心で、`range/vwap` との単純重なりは主因でない。
- 2026-04-19: config を v1.5 に調整した。変更は D 当日 9:35 側の `min_first5_pace 1.5->1.2` のみで、他条件は v1.4 を維持した。
- 2026-04-19: v1.5 で 2026-01-14〜2026-04-17 を再集計した。`scan_pace_fail_count 12->10` には改善したが、`avg_candidate_0935_count=0.077`, `scan_zero_days=61`, pass 5件は v1.4 と同値だった。
- 2026-04-19: `AXL` を一時除外した v1.5 比較でも `missing_first5_count 50->0` 以外は不変だった。`AXL` は引き続き candidate 通過数の主因ではない。
- 2026-04-19: v1.5 の fail overlap は `gap_only_fail=7`, `gap+pace_fail=1`, `gap+oc_ret_fail=2`, `pace_only_fail=2`, `oc_ret_only_fail=0`。`pace` を緩めても pass は増えず、次の主ボトルネックは `gap_fail` 単独と `oc_ret` 側に寄った。
- 2026-04-19: config を v1.6 に調整した。変更は D 当日 9:35 側の `first5_oc_ret > 0` 相当を `first5_oc_ret >= -0.001` に微緩和したもの。実装側も `>` 判定を `>=` に合わせ、`analysis_phase1.py` の `symbol_detail` に `gap_today_pct`, `first5_oc_ret`, `first5_pace` を追加した。
- 2026-04-19: v1.6 で 2026-01-14〜2026-04-17 を再集計した。`scan_oc_ret_fail_count 15->14` には改善したが、`avg_candidate_0935_count=0.077`, `scan_zero_days=61`, pass 5件は v1.5 と同値だった。
- 2026-04-19: v1.6 の `gap_only_fail=7` の `gap_today_pct` 分布は、指定帯では `0.0〜0.5%=1`, `0.5〜1.0%=0`, `1.0%以上=0`。残り 6件は `gap_today_pct < 0.0` で、主因は「微弱ギャップ不足」ではなく「前日終値割れ寄り」だった。
- 2026-04-19: `analysis_regime.py` と `scripts/analyze_gap_d1_regime.py` を追加した。D-1 watchlist 銘柄の D 当日寄りを `regime_a_open_above_prev_close` / `regime_b_open_at_or_below_prev_close` に分類し、`regime_detail_*`, `regime_summary_*`, `regime_branch_compare_*` を出力する。
- 2026-04-19: 2026-01-14〜2026-04-17 の regime 分析では、watchlist 33件のうち `regime_a=16`, `regime_b=17`。`regime_a` は `avg_day_oc_ret=-0.577%`, `win_rate=31.3%`, continuation pass 5件、`regime_b` は `avg_day_oc_ret=+0.421%`, `win_rate=58.8%`, reclaim branch 17件だった。
- 2026-04-19: current continuation pass 5件はすべて `regime_a` に属したが、平均 `day_oc_ret=-2.083%` で日中成績は弱かった。一方、簡易 reclaim branch は `count=17`, `avg_day_oc_ret=+0.421%`, `win_rate=58.8%` で、reclaim/reversal 系の別 branch 試作価値が高いと判断した。
- 2026-04-19: `analysis_regime.py` を拡張し、reclaim trigger 別の entry proxy と `reclaim v0.1` 比較を追加した。trigger は `prev_close reclaim`, `vwap reclaim`, `first5_high reclaim`。出力は `regime_trigger_summary_*` と、`continuation_compare` を含む `regime_branch_compare_*`。
- 2026-04-19: `reclaim v0.1` の gate は `regime B`, `close_5m >= vwap`, `first5_range_pos >= 0.50`, `first5_pace >= 1.0`。entry は `prev_close reclaim` と `first5_high reclaim` を別々に評価し、stop は `first5_low`, target は `2R`, exit は日中最終 bar close 近似で比較した。
- 2026-04-19: 2026-01-14〜2026-04-17 の trigger 集計では `prev_close reclaim count=6, avg_trade_ret=+0.405%, win_rate=50.0%`, `vwap reclaim count=7, avg_trade_ret=+1.053%, win_rate=57.1%`, `first5_high reclaim count=7, avg_trade_ret=+0.907%, win_rate=57.1%`。比較用 `continuation_compare` は `count=5, avg_trade_ret=-3.626%, win_rate=0%` で明確に劣後した。
- 2026-04-21: `src/gap_d1_0935/reclaim_branch_backtest.py` と `scripts/run_reclaim_branch_backtest.py` を追加した。reclaim を本線候補、continuation を compare branch として、`2026-01-14..2026-04-17` を replay/backtest できるようにした。出力は `reclaim_branch_trades_*`, `reclaim_branch_summary_*`, `continuation_vs_reclaim_compare_*`。
- 2026-04-21: reclaim v0.1 backtest は `regime B`, `close_5m >= vwap`, `first5_range_pos >= 0.50`, `first5_pace >= 1.0` を gate に使い、entry は `reclaim_first5_high` と `reclaim_vwap` を別比較、stop は `first5_low`, target は `2R`, force exit は `15:55 ET`, no overnight とした。
- 2026-04-21: backtest 結果は `reclaim_first5_high trade_count=7, win_rate=57.1%, avg_trade_ret=+1.054%, expectancy=+1.054%`, `reclaim_vwap trade_count=7, win_rate=57.1%, avg_trade_ret=+1.053%, expectancy=+1.053%`, `continuation_compare trade_count=3, win_rate=33.3%, avg_trade_ret=-0.668%, expectancy=-0.668%`。本線候補は reclaim に昇格、continuation は compare branch に格下げした。
- 2026-04-21: `analysis_regime.py` の helper 重複を整理し、`run_reclaim_branch_backtest.py` の既定期間を `12か月` に延長した。backtest には `slippage_bps_per_side=5`, `fee_bps_round_trip=2` を追加し、net return ベースで比較できるようにした。
- 2026-04-21: `2025-04-10..2026-04-17` の 256 営業日でコスト込み backtest を再集計した。`reclaim_first5_high trade_count=24, win_rate=50.0%, avg_trade_ret=+0.582%, expectancy=+0.582%`, `reclaim_vwap trade_count=26, win_rate=53.8%, avg_trade_ret=+0.839%, expectancy=+0.839%`, `continuation_compare trade_count=19, win_rate=36.8%, avg_trade_ret=-2.672%, expectancy=-2.672%`。
- 2026-04-21: コスト込みでも reclaim 2系統は continuation compare を上回った。暫定本線 `reclaim_first5_high` は採用維持、比較候補 `reclaim_vwap` は成績上はやや優位だが、trigger 安定性と運用解釈のしやすさを見て compare 扱いのまま残す。
- 2026-04-21: `src/gap_d1_0935/reclaim_executor.py` と `scripts/run_reclaim_executor.py` を追加し、`dry_run` 専用 executor を実装した。branch は `reclaim_first5_high`, `reclaim_vwap`, `continuation_compare` を切り替え可能で、`branch_executor_start`, `branch_candidate_loaded`, `branch_entry_check`, `branch_entry_filled`, `branch_exit_filled`, `branch_force_exit`, `branch_executor_complete` を jsonl に出す。
- 2026-04-21: executor は `09:36-10:00 ET` の entry window、`stop=first5_low`, `target=2R`, `force exit=15:55 ET`, `no overnight` を実装した。`risk_per_trade_usd`, `max_notional_per_trade_usd`, `min_order_qty` を config から読み、strategy 専用 artifacts に `reclaim_executor_daily_YYYYMMDD.csv`, `reclaim_branch_compare_YYYYMMDD.csv` を出力する。
- 2026-04-21: `2026-04-01` の最小 dry-run 検証では 3 branch とも正常完走した。`reclaim_first5_high` / `reclaim_vwap` は全候補 `not_regime_b`、`continuation_compare` は `no_trigger/gate_fail` で entry 0。branch 横並び CSV と jsonl ログが出力され、executor 経路の疎通を確認した。
- 2026-04-21: `reclaim_executor.py` に cost/fill モデルを追加した。`entry/exit slippage`, `fee_bps_round_trip`, `time_exit` 名称を backtest に寄せ、`run_reclaim_executor.py` も `--slippage-bps-per-side`, `--fee-bps-round-trip` を受けるよう更新した。既定値は backtest script と同じ `5.0 / 2.0`。
- 2026-04-21: `2025-04-10` で executor と backtest を再突合した。`reclaim_first5_high` は `entry_time`, `entry_price`, `exit_time`, `exit_price`, `exit_reason`, `pnl_pct` が一致。`reclaim_vwap` も `entry/exit/pnl` は一致した。残差分は `target_price` 列のみで、executor は `effective_entry/effective_stop` ベース、backtest CSV は `effective_entry/raw_stop` ベースで列化しているため `~0.0029` の差が残る。
- 2026-04-21: `reclaim_executor.py` に期間 replay helper を追加し、`run_reclaim_executor.py` から `--months`, `--end-date`, `--branch all` で 12か月 replay を回せるようにした。`daily/profile` は期間内で1回だけ読み、branch 別の日次集計と summary を `reclaim_executor_replay_daily_*`, `reclaim_executor_replay_summary_*`, `reclaim_executor_replay_compare_*` に出力する。
- 2026-04-21: `2025-04-10..2026-04-17` の 256営業日 replay を実行した。候補が出たのは 107営業日、candidate_count は全 branch 共通で 145。結果は `reclaim_first5_high entry_count=21, realized_pnl_usd=+502.44, avg_pnl_pct=+0.875`, `reclaim_vwap entry_count=26, realized_pnl_usd=+696.80, avg_pnl_pct=+0.910`, `continuation_compare entry_count=16, realized_pnl_usd=-127.79, avg_pnl_pct=-2.580`。
- 2026-04-21: `run_reclaim_executor.py` の `--from-scanner-csv` は single-day/period とも未対応のまま残した。period replay の本線は完了したが、scanner CSV 再構成 replay は次段で追加実装が必要。
- 2026-04-21: `run_reclaim_executor.py` の `--from-scanner-csv` を single-day/period の両方で受けるように更新した。scan CSV から `trade_date`, `symbol`, `open_D`, `close_D-1`, `first5_*`, `vwap` を読み、当日 minute bars を再取得して trigger 時刻と `reclaim_vwap_trigger_price` を再構成する。
- 2026-04-21: scan CSV が空でも落ちずに空 report を出すようにした。現存の `artifacts/scans/gap_0935_candidates_20260417.csv` は header only で、`--from-scanner-csv` period replay も empty report を正常生成した。
- 2026-04-21: 単日 executor の候補 0 日を正常系に変更した。`skip_reason=no_candidates` の empty row と `candidate_count=0` の日次サマリを出すため、連続 dry-run 監視に使える。
- 2026-04-21: `reclaim_branch_backtest.py` の `target_price` 列を executor と揃え、`_simulate_trade` の返却値から直接出すように修正した。`target_price` の定義は `raw_entry + 2 * (effective_entry - effective_stop)` で統一した。
- 2026-04-21: 直近5営業日 `2026-04-13..2026-04-17` の dry-run を `reclaim_first5_high`, `reclaim_vwap`, `continuation_compare` で実行した。結果は全 branch とも `candidate_count=2, entry_count=0, closed_trades=0, realized_pnl_usd=0`。候補があったのは `2026-04-13`, `2026-04-16` のみで、entry 発火は無かった。
- 2026-04-21: 戦略選定フェーズを終了し、本線 `reclaim_first5_high`、compare `reclaim_vwap`、reference `continuation_compare` で固定した。以後は新機能追加を止め、運用確認フェーズへ移行する。
- 2026-04-21: `reclaim_executor.py` の日次 summary に `gate_pass_count`, `trigger_touch_count`, `no_trigger_count`, `invalid_risk_count`, `entry_window_closed_count` を追加した。`compare` 行にも `gate_pass`, `trigger_touched` を持たせ、entry 0 理由を日次・branch 単位で追えるようにした。
- 2026-04-21: 直近20営業日 `2026-03-20..2026-04-17` の dry-run を 3 branch で再集計した。`reclaim_first5_high trade_days=20, candidate_count=7, gate_pass_count=1, trigger_touch_count=1, entry_count=1, realized_pnl_usd=+89.86`, `reclaim_vwap trade_days=20, candidate_count=7, gate_pass_count=1, trigger_touch_count=1, entry_count=1, realized_pnl_usd=+101.91`, `continuation_compare trade_days=20, candidate_count=7, gate_pass_count=2, trigger_touch_count=0, entry_count=0, no_trigger_count=2`。
- 2026-04-21: 直近20営業日の `gate_fail` を分解し、`reclaim_executor_gate_fail_daily_20260320_20260417.csv` と `reclaim_executor_gate_fail_symbol_20260320_20260417.csv` を追加出力した。branch 別合計は `reclaim_first5_high: regime_fail=4, pace_fail=3, range_fail=3, vwap_fail=2, compound_fail=4`, `reclaim_vwap: 同値`, `continuation_compare: pace_fail=4, range_fail=3, regime_fail=3, vwap_fail=2, compound_fail=3`。
- 2026-04-21: `gate_fail_symbol` では、`BBWI` が `regime|vwap|range|pace` の複合 fail、`ASST` は reclaim 側で `vwap|range` fail、`ANNX` は reclaim 側で `range|pace` fail、`CMPX` は reclaim 側で `regime|pace` fail と確認した。現時点では reclaim 2 branch の trigger 差は無く、entry 差ではなく exit 側差が主因。
- 2026-04-21: 観測継続フェーズへ固定した。以後は `reclaim_first5_high / reclaim_vwap / continuation_compare` の dry-run 継続監視を優先し、条件探索・branch 追加・本線変更は行わない。
- 2026-04-21: `reclaim_executor.py` の `pd.concat` 由来 `FutureWarning` を局所 suppress し、CSV 出力互換を維持したまま 62営業日 replay を warning 無しで完走させた。
- 2026-04-21: `2026-01-20..2026-04-17` の 62営業日 replay を再更新した。`reclaim_first5_high trade_count=7, realized_pnl_usd=+148.50, realized_pnl_pct=+0.927`, `reclaim_vwap trade_count=7, realized_pnl_usd=+250.97, realized_pnl_pct=+1.117`, `continuation_compare trade_count=1, realized_pnl_usd=-42.98`。
- 2026-04-21: 直近5営業日 `2026-04-13..2026-04-17` は全 branch `entry_count=0`。主な gate_fail 常連は `ASST`, `ANNX`, `BBWI`, `CMPX` で変わらず、reclaim 2 branch の差は引き続き exit 側に寄った。
- 2026-04-21: 本線再判定条件の進捗は `30営業日条件=到達済み`, `reclaim_first5_high trade_count=7/10`。再判定用の比較軸は `trade_count`, `win_rate`, `realized_pnl_usd`, `realized_pnl_pct`, `avg_hold_min`, `exit_reason breakdown` で固定した。
- 2026-04-21: 旧 gap 本線の定期タスク `MLStock_GapScan_093005` と `MLStock_GapDryRun_SkipOptions_Daily` を `Disabled` にした。`gap_d1_0935` への切り替え準備として、本線の scan-only / dry-run 自動実行を停止した。
- 2026-04-21: [C:\BOT\MLStock\scripts\run_reclaim_executor_daily.ps1](C:\BOT\MLStock\scripts\run_reclaim_executor_daily.ps1) を追加した。`reclaim_first5_high`, `reclaim_vwap`, `continuation_compare` を同一 `trade_date` で順番に `--dry-run` 実行する daily 起動スクリプトで、slippage/fee は既定 `5.0 / 2.0` を明示した。
- 2026-04-21: `MLStock_GapD1_0935_DryRun_Daily` の新規登録を試行したが、`Register-ScheduledTask` は `アクセスが拒否されました` で失敗した。OS 側の管理者権限不足が原因で、このセッションでは task 作成まで完了できていない。
- 2026-04-21: [C:\BOT\MLStock\scripts\run_reclaim_executor_daily.ps1](C:\BOT\MLStock\scripts\run_reclaim_executor_daily.ps1) を手動実行し、`reclaim_executor_daily_20260421.csv` と `reclaim_branch_compare_20260421.csv` の生成を確認した。初回結果は全 branch `candidate_count=0`, `skip_reason=no_candidates`。
