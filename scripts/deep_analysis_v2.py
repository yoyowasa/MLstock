"""V2改善の深堀り分析スクリプト。
サブ期間安定性, 特徴量重要度, ルックアヘッドバイアス, 市場レジーム分解を行う。
"""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path("C:/BOT/MLStock")
BACKTEST_DIR = BASE_DIR / "artifacts" / "backtest"

# ──────────────────────────────────────────────────────────────────────────────
# ① ルックアヘッドバイアス 厳密検証
# ──────────────────────────────────────────────────────────────────────────────
def verify_timing():
    print("=" * 70)
    print("① ルックアヘッドバイアス検証")
    print("=" * 70)

    features = pd.read_parquet(BASE_DIR / "data/snapshots/weekly/features.parquet")
    labels   = pd.read_parquet(BASE_DIR / "data/snapshots/weekly/labels.parquet")

    features["week_start"] = pd.to_datetime(features["week_start"]).dt.date
    labels["week_start"]   = pd.to_datetime(labels["week_start"]).dt.date

    merged = features.merge(labels, on=["week_start","symbol"], how="inner")

    # 特徴量は week_start の月曜 open 時点で入手可能
    # ラベル = 翌週 open / 今週 open - 1 → 翌週月曜まで不明
    spy_cols = ["spy_ret_1w", "spy_ret_4w", "spy_vol_4w", "market_breadth"]

    # (A) SPY特徴量 vs ラベル の単純相関
    print("\n[A] 各特徴量とラベル(label_return)の単純相関:")
    print(f"{'Feature':<20} {'corr(label)':>12} {'corr(label_raw)':>16}")
    print("-" * 52)
    for col in spy_cols + ["ret_1w", "ret_4w", "vol_4w"]:
        if col in merged.columns:
            c_ex  = merged[col].corr(merged["label_return"])
            c_raw = merged[col].corr(merged["label_return_raw"])
            print(f"{col:<20} {c_ex:>12.4f} {c_raw:>16.4f}")

    # (B) SPY特徴量の遅延検証: spy_ret_1w の構造確認
    # spy_ret_1w[t] = SPY price[t] / SPY price[t-1] - 1  (全銘柄同じ)
    # label_return_raw[t] = next_open[t] / price[t] - 1  (翌週の話)
    # 同一週の値がどの程度相関するか vs 1週ずらした場合
    spy_by_week = merged.groupby("week_start")["spy_ret_1w"].first().rename("spy_ret_1w")
    label_by_week = merged.groupby("week_start")["label_return_raw"].mean().rename("label_raw_mean")
    df_w = pd.DataFrame({"spy_ret_1w": spy_by_week, "label_raw_mean": label_by_week}).dropna()
    c_same  = df_w["spy_ret_1w"].corr(df_w["label_raw_mean"])
    c_lag1  = df_w["spy_ret_1w"].corr(df_w["label_raw_mean"].shift(-1))  # 1週先ラベル
    c_lead1 = df_w["spy_ret_1w"].corr(df_w["label_raw_mean"].shift(1))   # 1週前ラベル(逆チェック)
    print(f"\n[B] spy_ret_1w の週次相関構造:")
    print(f"  同一週ラベル(正常):   {c_same:.4f}  ← これが大きいとOK(モメンタム)")
    print(f"  1wk-ahead label:      {c_lag1:.4f}  <- if ~= same-week corr -> LEAK suspect")
    print(f"  1wk-before label:     {c_lead1:.4f}  <- if large -> LEAK confirmed")

    if abs(c_lag1) > abs(c_same) * 0.9:
        print("  ⚠️  WARNING: spy_ret_1w が翌週ラベルと強相関 → 確認要")
    else:
        print("  ✓  spy_ret_1w は翌週ラベルとの相関が弱い → リークなし")

    # (C) market_breadth も同様にチェック
    breadth_by_week = merged.groupby("week_start")["market_breadth"].first()
    df_b = pd.DataFrame({"breadth": breadth_by_week, "label_raw": label_by_week}).dropna()
    c_b_same = df_b["breadth"].corr(df_b["label_raw"])
    c_b_lag1 = df_b["breadth"].corr(df_b["label_raw"].shift(-1))
    print(f"\n  market_breadth — 同一週相関: {c_b_same:.4f}, 翌週相関: {c_b_lag1:.4f}")
    if abs(c_b_lag1) > abs(c_b_same) * 0.9:
        print("  ⚠️  WARNING: market_breadth と翌週ラベルの相関が高い")
    else:
        print("  ✓  market_breadth リーク問題なし")


# ──────────────────────────────────────────────────────────────────────────────
# ② サブ期間安定性チェック
# ──────────────────────────────────────────────────────────────────────────────
def subperiod_stability():
    print("\n" + "=" * 70)
    print("② サブ期間安定性チェック (年別)")
    print("=" * 70)

    import shutil, yaml
    from mlstock.config.loader import load_config
    from mlstock.jobs import backtest
    from mlstock.config.schema import AppConfig

    config_path = BASE_DIR / "config" / "config.yaml"
    with open(config_path) as f:
        base_yaml = yaml.safe_load(f)

    sub_periods = [
        ("2018", "2018-01-01", "2018-12-31"),
        ("2019", "2019-01-01", "2019-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2021", "2021-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
    ]

    def run_for_config(yaml_cfg, start, end, label_str):
        import io, contextlib, tempfile
        with open(config_path, "w") as f:
            yaml.dump(yaml_cfg, f, default_flow_style=False)
        cfg = load_config()
        from datetime import date
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = backtest.run(cfg, start=s, end=e)
        nav = pd.read_parquet(BACKTEST_DIR / "nav.parquet")
        nav_s = pd.to_numeric(nav["nav"], errors="coerce")
        dd = float((nav_s / nav_s.cummax() - 1).min()) if len(nav_s) > 1 else 0.0
        return summary.get("return_pct", 0.0), dd, nav

    def restore_config():
        with open(config_path, "w") as f:
            yaml.dump(base_yaml, f, default_flow_style=False)

    # Configs to compare
    # A: E_70 old (baseline w/o new features)
    cfg_e70 = copy.deepcopy(base_yaml)
    cfg_e70["selection"]["confidence_sizing"] = False
    cfg_e70["selection"]["price_cap"] = 60
    cfg_e70["snapshots"]["min_price"] = 1.0

    # B: baseline_feat17 (new features, no variable N)
    cfg_feat17 = copy.deepcopy(base_yaml)
    cfg_feat17["selection"]["confidence_sizing"] = False
    cfg_feat17["selection"]["price_cap"] = 60
    cfg_feat17["snapshots"]["min_price"] = 5.0

    # C: V2 (new features + variable N)
    cfg_v2 = copy.deepcopy(base_yaml)  # current recommended

    results = {}
    configs = [("E_70_old", cfg_e70), ("feat17_fixedN", cfg_feat17), ("V2_varN", cfg_v2)]

    print(f"\n{'Year':<8}", end="")
    for cname, _ in configs:
        print(f"  {cname:>14}", end="")
    print(f"  {'feat17-E70':>12}  {'V2-E70':>10}")
    print("-" * 75)

    all_results = {cname: {} for cname, _ in configs}
    for year_label, start, end in sub_periods:
        row = f"{year_label:<8}"
        for cname, cfg in configs:
            ret, dd, _ = run_for_config(cfg, start, end, cname)
            all_results[cname][year_label] = {"ret": ret, "dd": dd}
            row += f"  {ret*100:>12.2f}%"
        delta_feat = (all_results["feat17_fixedN"][year_label]["ret"] -
                      all_results["E_70_old"][year_label]["ret"]) * 100
        delta_v2   = (all_results["V2_varN"][year_label]["ret"] -
                      all_results["E_70_old"][year_label]["ret"]) * 100
        row += f"  {delta_feat:>+10.2f}pt  {delta_v2:>+8.2f}pt"
        print(row)

    restore_config()

    # Summary stats
    print("\n[サブ期間安定性サマリー]")
    for cname in [c for c, _ in configs]:
        rets = [v["ret"] for v in all_results[cname].values()]
        wins_vs_e70 = sum(
            1 for yr in all_results[cname]
            if all_results[cname][yr]["ret"] > all_results["E_70_old"][yr]["ret"]
        ) if cname != "E_70_old" else None
        print(f"  {cname:<20}: mean={np.mean(rets)*100:+.2f}%  std={np.std(rets)*100:.2f}%"
              + (f"  wins vs E70: {wins_vs_e70}/7" if wins_vs_e70 is not None else ""))

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# ③ 特徴量重要度チェック (Ridge係数 + 直近モデル)
# ──────────────────────────────────────────────────────────────────────────────
def feature_importance_check():
    print("\n" + "=" * 70)
    print("③ 特徴量重要度チェック (Ridge係数ベース)")
    print("=" * 70)

    from mlstock.config.loader import load_config
    from mlstock.model.train import train_ridge_model
    from mlstock.model.features import FEATURE_COLUMNS

    cfg = load_config()
    features = pd.read_parquet(BASE_DIR / "data/snapshots/weekly/features.parquet")
    labels   = pd.read_parquet(BASE_DIR / "data/snapshots/weekly/labels.parquet")
    features["week_start"] = pd.to_datetime(features["week_start"]).dt.date
    labels["week_start"]   = pd.to_datetime(labels["week_start"]).dt.date

    merged = features.merge(labels, on=["week_start","symbol"], how="inner")
    merged = merged.dropna(subset=list(FEATURE_COLUMNS) + ["label_return"])

    # 最新4年分のデータでモデルを1回訓練
    all_weeks = sorted(merged["week_start"].unique())
    recent_weeks = all_weeks[-208:]  # ~4年分
    train_df = merged[merged["week_start"].isin(recent_weeks)]

    model = train_ridge_model(train_df, FEATURE_COLUMNS, "label_return", alpha=1.0)
    if model is None:
        print("  Ridge model training failed")
        return

    coef = model["coef"]
    feat_importance = sorted(
        zip(FEATURE_COLUMNS, coef),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print(f"\n{'Feature':<25} {'Coefficient':>14} {'AbsRank':>8}")
    print("-" * 50)
    spy_cols = {"spy_ret_1w", "spy_ret_4w", "spy_vol_4w", "market_breadth"}
    for rank, (feat, coef_val) in enumerate(feat_importance, 1):
        marker = " ← NEW" if feat in spy_cols else ""
        print(f"{feat:<25} {coef_val:>14.6f} {rank:>8d}{marker}")

    # Fraction of variance explained by new vs old features
    X = train_df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = train_df["label_return"].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    # Standardize
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    c_arr = np.array(coef)
    preds = X_std.dot(c_arr)
    # Contribution per feature group
    spy_idx = [i for i, f in enumerate(FEATURE_COLUMNS) if f in spy_cols]
    old_idx = [i for i, f in enumerate(FEATURE_COLUMNS) if f not in spy_cols]

    pred_spy = X_std[:, spy_idx].dot(c_arr[spy_idx])
    pred_old = X_std[:, old_idx].dot(c_arr[old_idx])
    var_total = np.var(preds)
    var_spy   = np.var(pred_spy)
    var_old   = np.var(pred_old)
    print(f"\n[予測分散への寄与 (標準化係数ベース)]")
    print(f"  旧13特徴量:         {var_old/var_total*100:6.1f}%")
    print(f"  新4 SPY特徴量:      {var_spy/var_total*100:6.1f}%")
    print(f"  (注: 共線性あるため合計≠100%)")


# ──────────────────────────────────────────────────────────────────────────────
# ④ 市場レジーム分解: SPY上昇期 vs 下落期
# ──────────────────────────────────────────────────────────────────────────────
def regime_decomposition():
    print("\n" + "=" * 70)
    print("④ 市場レジーム分解 (SPY上昇期 vs 下落期)")
    print("=" * 70)

    # E_70 old vs feat17 vs V2 の NAV から週次リターンを計算
    configs_navs = [
        ("E_70_old",    BACKTEST_DIR / "nav_ensemble_70_old.parquet"),
        ("feat17_fixedN", BACKTEST_DIR / "sweep_v2" / "nav_baseline_feat17.parquet"),
        ("V2_varN",     BACKTEST_DIR / "sweep_v2" / "nav_varN_t0005_p60.parquet"),
    ]

    # SPYデータ取得
    spy_bars = pd.read_parquet(BASE_DIR / "data/raw/bars/SPY/bars.parquet")
    spy_bars["date"] = pd.to_datetime(spy_bars["date"]).dt.date
    spy_bars = spy_bars.sort_values("date")
    # 週次に変換 (月曜 open ベース)
    features = pd.read_parquet(BASE_DIR / "data/snapshots/weekly/features.parquet")
    spy_week = features[features["symbol"] == "SPY"]["week_start"] if False else None

    # SPY week map from features (spy_ret_1w is already computed)
    feat_df = pd.read_parquet(BASE_DIR / "data/snapshots/weekly/features.parquet")
    feat_df["week_start"] = pd.to_datetime(feat_df["week_start"]).dt.date
    spy_context = feat_df.groupby("week_start")[["spy_ret_1w", "spy_vol_4w", "market_breadth"]].first().reset_index()

    results_by_regime = {}
    for name, nav_path in configs_navs:
        if not nav_path.exists():
            print(f"  {name}: ファイルなし ({nav_path.name})")
            continue
        nav_df = pd.read_parquet(nav_path)
        nav_df["week_start"] = pd.to_datetime(nav_df["week_start"]).dt.date
        nav_s = pd.to_numeric(nav_df["nav"], errors="coerce")
        nav_df["weekly_ret"] = (nav_s / nav_s.shift(1) - 1).values

        merged = nav_df.merge(spy_context, on="week_start", how="left")
        # 上昇週 = spy_ret_1w > 0, 下落週 = spy_ret_1w <= 0
        up_weeks = merged[merged["spy_ret_1w"] > 0]
        dn_weeks = merged[merged["spy_ret_1w"] <= 0]
        # High vol = spy_vol_4w > median
        vol_median = merged["spy_vol_4w"].median()
        hi_vol = merged[merged["spy_vol_4w"] > vol_median]
        lo_vol = merged[merged["spy_vol_4w"] <= vol_median]

        results_by_regime[name] = {
            "all_mean":     merged["weekly_ret"].mean(),
            "up_mean":      up_weeks["weekly_ret"].mean(),
            "dn_mean":      dn_weeks["weekly_ret"].mean(),
            "hi_vol_mean":  hi_vol["weekly_ret"].mean(),
            "lo_vol_mean":  lo_vol["weekly_ret"].mean(),
            "n_up":         len(up_weeks),
            "n_dn":         len(dn_weeks),
        }

    print(f"\n{'Model':<22} {'All':>8} {'SPY-Up':>8} {'SPY-Dn':>8} {'Hi-Vol':>8} {'Lo-Vol':>8}  {'n_up/dn'}")
    print("-" * 75)
    for name, r in results_by_regime.items():
        print(f"{name:<22} "
              f"{r['all_mean']*100:>7.3f}% "
              f"{r['up_mean']*100:>7.3f}% "
              f"{r['dn_mean']*100:>7.3f}% "
              f"{r['hi_vol_mean']*100:>7.3f}% "
              f"{r['lo_vol_mean']*100:>7.3f}%  "
              f"{r['n_up']}/{r['n_dn']}")

    print("\n[解釈]")
    if "feat17_fixedN" in results_by_regime and "E_70_old" in results_by_regime:
        delta_up = (results_by_regime["feat17_fixedN"]["up_mean"] -
                    results_by_regime["E_70_old"]["up_mean"]) * 100
        delta_dn = (results_by_regime["feat17_fixedN"]["dn_mean"] -
                    results_by_regime["E_70_old"]["dn_mean"]) * 100
        if abs(delta_up) > abs(delta_dn) * 1.5:
            print("  → 改善は主に上昇週に集中 (体制依存の可能性あり)")
        elif abs(delta_dn) > abs(delta_up) * 1.5:
            print("  → 改善は主に下落週に集中 (下落防御が効いている)")
        else:
            print("  → 改善は上昇・下落週で均等 (体制非依存でロバスト)")
        print(f"  上昇週デルタ: {delta_up:+.3f}pt, 下落週デルタ: {delta_dn:+.3f}pt")


# ──────────────────────────────────────────────────────────────────────────────
# ⑤ regime_gate との相互作用
# ──────────────────────────────────────────────────────────────────────────────
def regime_gate_interaction():
    print("\n" + "=" * 70)
    print("⑤ Regime Gate の重複性チェック")
    print("=" * 70)
    print("  (spy系特徴量はgate OFFでも機能するか？)")

    import io, contextlib, yaml
    from mlstock.config.loader import load_config
    from mlstock.jobs import backtest
    from datetime import date

    config_path = BASE_DIR / "config" / "config.yaml"
    with open(config_path) as f:
        base_yaml = yaml.safe_load(f)

    def run_cfg(yaml_cfg):
        with open(config_path, "w") as f:
            yaml.dump(yaml_cfg, f, default_flow_style=False)
        cfg = load_config()
        s = date.fromisoformat("2018-01-01")
        e = date.fromisoformat("2024-12-31")
        with contextlib.redirect_stdout(io.StringIO()):
            summary = backtest.run(cfg, start=s, end=e)
        nav_df = pd.read_parquet(BACKTEST_DIR / "nav.parquet")
        nav_s = pd.to_numeric(nav_df["nav"], errors="coerce")
        dd = float((nav_s / nav_s.cummax() - 1).min())
        return summary.get("return_pct", 0.0), dd

    configs = {
        "feat17 + gate=ON":  (False, True),
        "feat17 + gate=OFF": (False, False),
        "V2    + gate=ON":   (True,  True),
        "V2    + gate=OFF":  (True,  False),
    }

    print(f"\n{'Config':<25} {'Return%':>9} {'MaxDD%':>9} {'Return/DD':>10}")
    print("-" * 55)
    for label, (sizing, gate_on) in configs.items():
        cfg = copy.deepcopy(base_yaml)
        cfg["selection"]["confidence_sizing"] = sizing
        cfg["selection"]["price_cap"] = 60
        cfg["snapshots"]["min_price"] = 5.0
        cfg["risk"]["regime_gate"]["enabled"] = gate_on
        ret, dd = run_cfg(cfg)
        ratio = ret*100 / abs(dd*100) if dd != 0 else 0
        print(f"{label:<25} {ret*100:>9.2f} {dd*100:>9.2f} {ratio:>10.2f}")

    # Restore
    with open(config_path, "w") as f:
        yaml.dump(base_yaml, f, default_flow_style=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MLStock V2 深堀り分析レポート")
    print("="*70)

    verify_timing()
    feature_importance_check()
    regime_decomposition()
    regime_gate_interaction()
    subperiod_stability()   # 最後 (一番時間かかる)

    print("\n\n=== 分析完了 ===")
