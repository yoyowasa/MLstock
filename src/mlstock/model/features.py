from __future__ import annotations

FEATURE_COLUMNS = [
    # モメンタム系（既存）
    "ret_1w",
    "ret_4w",
    # モメンタム系（追加）
    "ret_8w",  # 8週リターン（中期トレンド）
    "ret_13w",  # 13週リターン（四半期トレンド）
    "ret_26w",  # 26週リターン（半年トレンド）
    # ボラティリティ系（既存）
    "vol_4w",
    # ボラティリティ系（追加）
    "vol_8w",  # 8週ボラティリティ
    "vol_13w",  # 13週ボラティリティ
    # 価格位置系
    "ma_ratio_10w",  # 10週MA乖離率 = price / MA(10w) - 1
    "ma_ratio_20w",  # 20週MA乖離率
    "high_low_range_4w",  # 4週間の高値安値レンジ / price
    # 出来高系
    "volume_ratio_4w",  # 直近1週出来高 / 4週平均出来高
    # リバーサル系
    "ret_1w_rank",  # ret_1w の週次クロスセクショナルランク（percentile）
    # マーケットコンテキスト系（全銘柄同一値 — 市場レジーム認識用）
    "spy_ret_1w",  # SPY 1週リターン
    "spy_ret_4w",  # SPY 4週リターン
    "spy_vol_4w",  # SPY 4週ボラティリティ
    "market_breadth",  # 同一週の ret_1w > 0 の銘柄割合
]
