"""指数移动均线回踩反弹 — EMA21/34/55/89 多头排列 + 回踩 EMA21/34/55 支撑 + 缩量确认"""
import polars as pl

LOOKBACK_DAYS = 150
ENTRY_SIGNALS = ["signal_ma20_breakout", "signal_ma_golden_5_20"]
EXIT_SIGNALS = ["signal_ma20_breakdown", "signal_ma_dead_5_20"]
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 15
ALERTS = []

META = {
    "id": "ema_pullback_rebound",
    "name": "指数移动均线回踩反弹",
    "description": "EMA21/34/55/89 多头排列 + 价格回踩均线支撑 + 缩量确认反弹",
    "tags": ["均线", "EMA", "回踩", "反弹", "多头", "趋势"],
    "basic_filter": {
        "price_min": 3,
        "price_max": 300,
        "market_cap_min": 10e8,
        "amount_min": 0.5e8,
        "exclude_st": True,
        "exclude_new_days": 60,
    },
    "params": [
        {"id": "pullback_ema_near_pct", "label": "回踩EMA接近度(%)", "type": "float",
         "default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5},
        {"id": "volume_shrink_ratio", "label": "缩量比例上限", "type": "float",
         "default": 0.7, "min": 0.3, "max": 1.0, "step": 0.05},
        {"id": "trend_days_min", "label": "多头排列最少天数", "type": "int",
         "default": 5, "min": 1, "max": 30, "step": 1},
        {"id": "require_positive_close", "label": "要求收阳线", "type": "bool",
         "default": True},
    ],
    "params_defaults": {
        "pullback_ema_near_pct": 3.0,
        "volume_shrink_ratio": 0.7,
        "trend_days_min": 5,
        "require_positive_close": True,
    },
    "scoring": {
        "momentum_5d": 0.3,
        "vol_ratio_5d": 0.3,
        "turnover_rate": 0.2,
        "momentum_20d": 0.2,
    },
    "entry_signals": ["signal_ma20_breakout", "signal_ma_golden_5_20"],
    "exit_signals": ["signal_ma20_breakdown", "signal_ma_dead_5_20"],
    "stop_loss": -0.05,
    "max_hold_days": 15,
    "order_by": "score",
    "descending": True,
    "limit": 100,
}


def _ema_alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def _calc_ema(values: list[float], span: int) -> list[float]:
    alpha = _ema_alpha(span)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """对每只股票检测 EMA 多头排列 + 回踩支撑 + 缩量确认。"""
    if df.is_empty():
        return pl.DataFrame()

    df = df.sort(["symbol", "date"])

    pullback_near_pct = params.get("pullback_ema_near_pct", 3.0) / 100.0
    volume_shrink_ratio = params.get("volume_shrink_ratio", 0.7)
    trend_days_min = params.get("trend_days_min", 5)
    require_positive = params.get("require_positive_close", True)

    symbols = df["symbol"].unique().to_list()
    matched: list[dict] = []

    for sym in symbols:
        sub = df.filter(pl.col("symbol") == sym).sort("date")
        n = sub.height
        if n < 90:
            continue

        closes = sub["close"].to_list()
        volumes = sub["volume"].to_list()
        dates = sub["date"].to_list()
        opens = sub["open"].to_list()

        ema21 = _calc_ema(closes, 21)
        ema34 = _calc_ema(closes, 34)
        ema55 = _calc_ema(closes, 55)
        ema89 = _calc_ema(closes, 89)

        c = closes[-1]
        v = volumes[-1]
        e21 = ema21[-1]
        e34 = ema34[-1]
        e55 = ema55[-1]
        e89 = ema89[-1]

        # 条件1: EMA21 > EMA34 > EMA55 > EMA89 多头排列
        if not (e21 > e34 > e55 > e89):
            continue

        # 条件2: 收盘价在 EMA89 上方（长期上升趋势）
        if c <= e89:
            continue

        # 条件3: 多头排列至少持续 trend_days_min 天
        aligned_days = 0
        for i in range(n - 1, max(0, n - trend_days_min - 1), -1):
            if ema21[i] > ema34[i] > ema55[i] > ema89[i]:
                aligned_days += 1
            else:
                break
        if aligned_days < trend_days_min:
            continue

        # 条件4: 收盘价回踩到 EMA21 / EMA34 / EMA55 附近
        dist_to_21 = abs(c - e21) / e21
        dist_to_34 = abs(c - e34) / e34
        dist_to_55 = abs(c - e55) / e55
        near_ema21 = dist_to_21 <= pullback_near_pct
        near_ema34 = dist_to_34 <= pullback_near_pct
        near_ema55 = dist_to_55 <= pullback_near_pct
        if not (near_ema21 or near_ema34 or near_ema55):
            continue

        # 选择最近的支撑均线
        dists = []
        if near_ema21:
            dists.append(("ema21", dist_to_21))
        if near_ema34:
            dists.append(("ema34", dist_to_34))
        if near_ema55:
            dists.append(("ema55", dist_to_55))
        support_ema, support_dist = min(dists, key=lambda x: x[1])

        # 条件5: 缩量（当日成交量 < 5日均量 * 缩量比例上限）
        vol_ma5 = sum(volumes[-6:-1]) / 5 if n >= 6 else v
        vol_shrink = v / vol_ma5 if vol_ma5 > 0 else 1.0
        if vol_shrink > volume_shrink_ratio:
            continue

        # 条件6: 收阳线（可选）
        if require_positive and c <= opens[-1]:
            continue

        name = sub["name"][0] if "name" in sub.columns else ""
        latest = sub.tail(1).to_dicts()[0]

        row = {
            "symbol": sym,
            "name": name or latest.get("name", ""),
            "date": latest.get("date"),
            "close": c,
            "support_ema": support_ema,
            "support_dist_pct": round(support_dist * 100, 2),
            "vol_shrink_ratio": round(vol_shrink, 2),
            "aligned_days": aligned_days,
            # 合并标准 enriched 字段
            "change_pct": latest.get("change_pct"),
            "amount": latest.get("amount"),
            "vol_ratio_5d": latest.get("vol_ratio_5d"),
            "turnover_rate": latest.get("turnover_rate"),
            "change_amount": latest.get("change_amount"),
            "prev_close": latest.get("prev_close"),
            # 评分: 回踩越精准分越高 + 缩量越明显分越高 + 排列越久分越高
            "score": round(
                max(0, (pullback_near_pct - support_dist) / pullback_near_pct) * 3
                + max(0, (volume_shrink_ratio - vol_shrink) / max(0.01, volume_shrink_ratio)) * 2
                + min(1.0, aligned_days / max(1, trend_days_min * 2)) * 3,
                2,
            ),
        }
        matched.append(row)

    if not matched:
        return pl.DataFrame()

    return pl.DataFrame(matched)
