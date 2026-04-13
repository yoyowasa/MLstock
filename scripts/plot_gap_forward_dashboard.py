from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dry-run forward dashboard for gap strategy logs")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("artifacts/logs"),
        help="Directory that contains gap_trade_*.jsonl logs",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=30,
        help="Number of most recent dry-run logs to include",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/forward_plots/gap_dry_forward"),
        help="Directory to save dashboard, markdown guide, and CSV exports",
    )
    return parser.parse_args()


def _iter_gap_logs(log_dir: Path) -> List[Path]:
    return sorted(log_dir.glob("gap_trade_*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_session(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    session: Dict[str, Any] = {
        "log": str(path),
        "session_utc": None,
        "trade_date": None,
        "scan_only": None,
        "skip_options": None,
        "live": None,
        "preflight_warning": 0,
        "preflight_bars": None,
        "scanner_count": 0,
        "options_selected_count": 0,
        "entries": 0,
        "exits": 0,
        "closed_trades": 0,
        "realized_pnl_usd": 0.0,
        "realized_pnl_pct": 0.0,
        "open_positions_count": 0,
        "stop_reason": "",
        "universe_count": 0,
        "daily_count": 0,
        "open_count": 0,
        "missing_open_count": 0,
        "liquid_price_count": 0,
        "gap_ge_2_count": 0,
        "raw_candidate_count": 0,
        "candidate_count": 0,
    }
    trades: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            message = payload.get("message")
            if session["session_utc"] is None and isinstance(payload.get("ts_utc"), str):
                session["session_utc"] = payload["ts_utc"]
            if message == "start":
                session["scan_only"] = bool(payload.get("scan_only"))
                session["skip_options"] = bool(payload.get("skip_options"))
                session["live"] = bool(payload.get("live"))
            elif message == "preflight_iex_bars":
                session["preflight_bars"] = _safe_int(payload.get("bars"))
            elif message == "preflight_iex_bars_warning":
                session["preflight_warning"] = 1
                session["preflight_bars"] = _safe_int(payload.get("bars"))
            elif message == "scanner_diagnostics":
                session["trade_date"] = payload.get("trade_date")
                session["universe_count"] = _safe_int(payload.get("universe_count"))
                session["daily_count"] = _safe_int(payload.get("daily_count"))
                session["open_count"] = _safe_int(payload.get("open_count"))
                session["missing_open_count"] = _safe_int(payload.get("missing_open_count"))
                session["liquid_price_count"] = _safe_int(payload.get("liquid_price_count"))
                session["gap_ge_2_count"] = _safe_int(payload.get("gap_ge_2_count"))
                session["raw_candidate_count"] = _safe_int(payload.get("raw_candidate_count"))
                session["candidate_count"] = _safe_int(payload.get("candidate_count"))
            elif message == "scanner_complete":
                session["scanner_count"] = _safe_int(payload.get("count"))
            elif message in {"options_skipped", "options_filter_complete"}:
                selected = payload.get("selected", [])
                if isinstance(selected, list):
                    session["options_selected_count"] = len(selected)
            elif message == "entry_filled":
                session["entries"] += 1
            elif message in {"exit_filled", "force_close_exit"}:
                session["exits"] += 1
                trades.append(
                    {
                        "log": str(path),
                        "session_utc": session["session_utc"],
                        "trade_date": session["trade_date"],
                        "symbol": str(payload.get("symbol", "")).upper(),
                        "reason": str(payload.get("reason") or message).lower(),
                        "qty": _safe_int(payload.get("qty")),
                        "entry_price": _safe_float(payload.get("entry_price")),
                        "exit_price": _safe_float(payload.get("exit_price")),
                        "realized_pnl_usd": _safe_float(payload.get("realized_pnl_usd")),
                        "realized_pnl_pct": _safe_float(payload.get("realized_pnl_pct")),
                    }
                )
            elif message == "gap_trader_complete":
                session["entries"] = _safe_int(payload.get("entries"), session["entries"])
                session["exits"] = _safe_int(payload.get("exits"), session["exits"])
                session["closed_trades"] = _safe_int(payload.get("closed_trades"))
                session["realized_pnl_usd"] = _safe_float(payload.get("realized_pnl_usd"))
                session["realized_pnl_pct"] = _safe_float(payload.get("realized_pnl_pct"))
                open_positions = payload.get("open_positions", [])
                session["open_positions_count"] = len(open_positions) if isinstance(open_positions, list) else 0
            elif isinstance(message, str) and message.startswith("stop_"):
                session["stop_reason"] = message

    return session, trades


def _load_dry_run_data(log_dir: Path, latest: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sessions: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    for path in _iter_gap_logs(log_dir):
        session, trade_rows = _load_session(path)
        if session["scan_only"] is not False:
            continue
        if session["live"] is not False:
            continue
        sessions.append(session)
        trades.extend(trade_rows)
        if len(sessions) >= max(latest, 1):
            break

    sessions_df = pd.DataFrame(sessions)
    trades_df = pd.DataFrame(trades)
    if not sessions_df.empty:
        sessions_df["session_utc"] = pd.to_datetime(sessions_df["session_utc"], utc=True, errors="coerce")
        sessions_df["trade_date"] = pd.to_datetime(sessions_df["trade_date"], errors="coerce")
        sessions_df["plot_date"] = sessions_df["trade_date"].fillna(sessions_df["session_utc"].dt.tz_convert(None))
        sessions_df = sessions_df.sort_values("session_utc").reset_index(drop=True)
        sessions_df["cum_realized_pnl_usd"] = sessions_df["realized_pnl_usd"].cumsum()
        equity = sessions_df["cum_realized_pnl_usd"]
        sessions_df["drawdown_usd"] = equity - equity.cummax()
        sessions_df["open_coverage_pct"] = (
            sessions_df["open_count"] / sessions_df["daily_count"].replace({0: pd.NA}) * 100.0
        ).fillna(0.0)
    if not trades_df.empty:
        trades_df["session_utc"] = pd.to_datetime(trades_df["session_utc"], utc=True, errors="coerce")
        trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"], errors="coerce")
        trades_df = trades_df.sort_values(["session_utc", "symbol"]).reset_index(drop=True)
    return sessions_df, trades_df


def _plot_dashboard(sessions: pd.DataFrame, trades: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle("Gap Dry-Run Forward Dashboard", fontsize=18, fontweight="bold")

    if sessions.empty:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, "No dry-run sessions found.", ha="center", va="center", fontsize=16)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    x = sessions["plot_date"].dt.strftime("%Y-%m-%d").fillna("unknown")

    ax = axes[0, 0]
    ax.plot(x, sessions["cum_realized_pnl_usd"], marker="o", linewidth=2, color="#1f77b4", label="Cumulative PnL (USD)")
    ax.set_title("Equity Curve")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=45)
    ax2 = ax.twinx()
    ax2.fill_between(x, sessions["drawdown_usd"], 0, color="#d62728", alpha=0.2, label="Drawdown")
    ax2.set_ylabel("Drawdown USD")

    ax = axes[0, 1]
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in sessions["realized_pnl_usd"]]
    ax.bar(x, sessions["realized_pnl_usd"], color=colors, alpha=0.85, label="Daily realized PnL")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Daily PnL")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 0]
    ax.plot(x, sessions["open_count"], marker="o", label="open_count")
    ax.plot(x, sessions["liquid_price_count"], marker="o", label="liquid_price_count")
    ax.plot(x, sessions["gap_ge_2_count"], marker="o", label="gap_ge_2_count")
    ax.plot(x, sessions["candidate_count"], marker="o", label="candidate_count")
    ax.set_title("Scanner Funnel Trend")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1, 1]
    ax.bar(x, sessions["open_coverage_pct"], color="#4c78a8", alpha=0.8, label="open coverage %")
    warning_idx = sessions["preflight_warning"] > 0
    if warning_idx.any():
        ax.scatter(
            sessions.loc[warning_idx, "plot_date"].dt.strftime("%Y-%m-%d").fillna("unknown"),
            sessions.loc[warning_idx, "open_coverage_pct"],
            color="#ff7f0e",
            s=80,
            marker="x",
            label="preflight warning",
        )
    ax.set_title("Data Quality")
    ax.set_ylabel("Open Coverage %")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2, 0]
    if trades.empty:
        ax.text(0.5, 0.5, "No exits yet", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        trade_pcts = trades["realized_pnl_pct"].astype(float)
        bins = min(max(len(trade_pcts), 5), 15)
        ax.hist(trade_pcts, bins=bins, color="#9467bd", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Trade Return Distribution")
        ax.set_xlabel("Realized PnL %")
        ax.set_ylabel("Trades")

    ax = axes[2, 1]
    if trades.empty:
        ax.text(0.5, 0.5, "No exit reasons yet", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        reasons = Counter(str(item).lower() for item in trades["reason"].tolist())
        labels = list(reasons.keys())
        values = list(reasons.values())
        ax.bar(labels, values, color="#8c564b", alpha=0.85)
        ax.set_title("Exit Reason Mix")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_guide_md(sessions: pd.DataFrame, trades: pd.DataFrame, output_dir: Path) -> str:
    session_count = len(sessions)
    closed_trades = int(trades.shape[0]) if not trades.empty else 0
    total_pnl = float(sessions["realized_pnl_usd"].sum()) if not sessions.empty else 0.0
    win_days = int((sessions["realized_pnl_usd"] > 0).sum()) if not sessions.empty else 0
    warning_days = int((sessions["preflight_warning"] > 0).sum()) if not sessions.empty else 0
    avg_open_coverage = float(sessions["open_coverage_pct"].mean()) if not sessions.empty else 0.0

    return f"""# Gap Dry-Run Forward Plot Guide

出力先: `{output_dir}`

## このダッシュボードの目的

この図は、dry-run のフォワード挙動を 1 枚で確認するためのものです。  
戦績だけでなく、候補供給量とデータ取得品質を同時に見る前提で作っています。

## 今回の集計レンジ

- 対象セッション数: {session_count}
- クローズ済みトレード数: {closed_trades}
- 累計 realized PnL (USD): {total_pnl:.2f}
- プラス日数: {win_days}
- preflight warning 日数: {warning_days}
- 平均 open coverage: {avg_open_coverage:.2f}%

## PLOT 1: Equity Curve

累積 realized PnL を見ます。  
右軸の drawdown とセットで、勝っているかだけでなく、どの程度沈むかを見ます。

## PLOT 2: Daily PnL

日次 realized PnL の棒グラフです。  
単発の大勝ち・大負け依存か、平準的に積めているかを見ます。

## PLOT 3: Scanner Funnel Trend

以下を同じ時間軸で並べます。

- `open_count`
- `liquid_price_count`
- `gap_ge_2_count`
- `candidate_count`

これで、候補が出ない理由が

- 市場全体でギャップが少ない
- 流動性条件で落ちる
- scanner 後段で落ちる

のどこかを見分けやすくなります。

## PLOT 4: Data Quality

`open_count / daily_count` の coverage を見ます。  
`preflight warning` が付いた日と coverage 低下が連動しているかを確認します。

## PLOT 5: Trade Return Distribution

各トレードの realized PnL % の分布です。  
勝率だけではなく、利小損大か、損小利大かを見ます。

## PLOT 6: Exit Reason Mix

`stop`, `target`, `force_close` の比率を見ます。  
現状ロジックがどこで終わりやすいかを把握します。

## 併せて出しているファイル

- `gap_dry_forward_dashboard.png`: ダッシュボード本体
- `gap_dry_forward_sessions.csv`: セッション単位の集計
- `gap_dry_forward_trades.csv`: トレード単位の集計
- `PLOT_GUIDE.md`: この説明
"""


def main() -> None:
    args = _parse_args()
    sessions, trades = _load_dry_run_data(args.log_dir, args.latest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard_path = output_dir / "gap_dry_forward_dashboard.png"
    guide_path = output_dir / "PLOT_GUIDE.md"
    sessions_csv = output_dir / "gap_dry_forward_sessions.csv"
    trades_csv = output_dir / "gap_dry_forward_trades.csv"

    _plot_dashboard(sessions, trades, dashboard_path)
    sessions.to_csv(sessions_csv, index=False)
    trades.to_csv(trades_csv, index=False)
    guide_path.write_text(_build_guide_md(sessions, trades, output_dir), encoding="utf-8")

    print(f"dashboard: {dashboard_path}")
    print(f"guide: {guide_path}")
    print(f"sessions_csv: {sessions_csv}")
    print(f"trades_csv: {trades_csv}")


if __name__ == "__main__":
    main()
