# STATUS.md

## 目的
- `gap_d1_0935` 戦略専用の実装・修正・分析履歴を本線と分離して管理する。
- 本線 gap strategy とは別系統として、D-1 watchlist + 当日 9:35 continuation 確認の進捗を記録する。

## 現在状態
- Phase 1 実装着手
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
