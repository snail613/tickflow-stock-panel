"""龙回头 — 识别龙回头形态：前顶→回落→放量突破前顶→主升浪→回踩前顶"""
import polars as pl

# 策略引擎需要读取的历史窗口天数（应 ≥ analyze_lookback_days）
LOOKBACK_DAYS = 250

META = {
    "id": "dragon_rebound",
    "name": "龙回头",
    "description": "识别龙回头形态：前顶→回落→放量突破前顶→主升浪→回踩前顶得到支撑",
    "tags": ["形态", "龙回头", "突破", "支撑", "回踩"],
    "basic_filter": {
        "price_min": 3,
        "price_max": 300,
        "market_cap_min": 10e8,
        "amount_min": 0.5e8,
        "exclude_st": True,
        "exclude_new_days": 60,
    },
    "params": [
        {"id": "analyze_lookback_days", "label": "分析数据范围(天)", "type": "int",
         "default": 250, "min": 100, "max": 500, "step": 10},
        {"id": "rally_gain_min", "label": "主升浪最低涨幅(%)", "type": "float",
         "default": 50.0, "min": 20.0, "max": 200.0, "step": 5.0},
        {"id": "pullback_range_min", "label": "距前顶下限(%)", "type": "float",
         "default": -3.0, "min": -3.0, "max": 0.0, "step": 1.0},
        {"id": "pullback_range_max", "label": "距前顶上限(%)", "type": "float",
         "default": 5.0, "min": 0.0, "max": 20.0, "step": 1.0},
        {"id": "use_lookback", "label": "启用涨停回溯", "type": "bool",
         "default": False},
        {"id": "lookback_days", "label": "涨停回溯天数", "type": "int",
         "default": 50, "min": 20, "max": 100, "step": 10,
         "depends_on": "use_lookback"},
    ],
    "params_defaults": {
        "analyze_lookback_days": 250,
        "rally_gain_min": 50.0,
        "pullback_range_min": -3.0,
        "pullback_range_max": 5.0,
        "use_lookback": False,
        "lookback_days": 50,
    },
    "scoring": {
        "momentum_5d": 0.4,
        "vol_ratio_5d": 0.3,
        "turnover_rate": 0.3,
    },
    "entry_signals": ["signal_limit_up"],
    "exit_signals": ["signal_ma20_breakdown"],
    "stop_loss": -0.06,
    "max_hold_days": 20,
    "order_by": "score",
    "descending": True,
    "limit": 100,
}


# ───────────────────────── 形态识别核心 ─────────────────────────

def _detect_peaks(prices: list[float], window: int = 20) -> list[int]:
    """滑动窗口局部最高点（左右各 window 根，共 2*window+1 根）。"""
    n = len(prices)
    peaks: list[int] = []
    for i in range(window, n - window):
        peak = True
        for j in range(i - window, i + window + 1):
            if j == i:
                continue
            if prices[j] > prices[i]:
                peak = False
                break
        if peak:
            peaks.append(i)
    return peaks


def _score_stock(sub: pl.DataFrame, peak_idx: int, params: dict) -> dict | None:
    """对一只股票的候选峰值执行龙回头 A→J 条件链。"""
    n = sub.height
    if n < 100:
        return None

    highs = sub["high"].to_list()
    lows = sub["low"].to_list()
    closes = sub["close"].to_list()
    change_pcts = sub["change_pct"].to_list()
    dates = sub["date"].to_list()
    current_close = closes[-1]
    current_date = dates[-1]

    peak_price = highs[peak_idx]

    # A: 峰值为该日期前的绝对历史最高价
    if max(highs[: peak_idx + 1]) > peak_price:
        return None

    # B: 峰值后 10 天内无更高点
    end_b = min(n, peak_idx + 11)
    if max(highs[peak_idx + 1 : end_b]) > peak_price:
        return None

    # C/D: 涨停回溯（可选）
    use_lookback = params.get("use_lookback", False)
    lookback_days = params.get("lookback_days", 50)
    if use_lookback:
        if peak_idx < lookback_days:
            return None
        start_c = max(0, peak_idx - lookback_days)
        has_limit = any(
            pct is not None and pct >= 0.095 for pct in change_pcts[start_c:peak_idx]
        )
        if not has_limit:
            return None

    # E: 峰值不位于数据末尾
    if peak_idx >= n - 1:
        return None

    # F: 从峰值到后续最高点的涨幅 ≥ rally_gain_min
    after_highs = highs[peak_idx:]
    rally_offset = after_highs.index(max(after_highs))
    rally_idx = peak_idx + rally_offset
    rally_high = highs[rally_idx]
    rally_gain = (rally_high - peak_price) / peak_price
    rally_gain_min = params.get("rally_gain_min", 50.0) / 100.0
    if rally_gain < rally_gain_min:
        return None

    # G: 存在某天最高价突破前顶价格
    breakout_idx: int | None = None
    for j in range(peak_idx + 1, rally_idx + 1):
        if highs[j] > peak_price:
            breakout_idx = j
            break
    if breakout_idx is None:
        return None

    # H: 突破日次日到主升浪高点期间，收盘价不得跌破前顶
    for j in range(breakout_idx + 1, rally_idx + 1):
        if closes[j] < peak_price:
            return None

    # I: 主升浪高点之后，所有日期最低价不得跌破前顶
    for j in range(rally_idx + 1, n):
        if lows[j] < peak_price:
            return None

    # J: 当前收盘价距前顶的比例在 [pullback_range_min, pullback_range_max]
    price_distance_pct = (current_close - peak_price) / peak_price
    pr_min = params.get("pullback_range_min", -3.0) / 100.0
    pr_max = params.get("pullback_range_max", 5.0) / 100.0
    if not (pr_min <= price_distance_pct <= pr_max):
        return None

    # 回踩最低点：主升浪高点之后到当前的最低价及其日期
    pullback_low = min(lows[rally_idx + 1 :])
    pullback_offset = lows[rally_idx + 1 :].index(pullback_low)
    pullback_date = dates[rally_idx + 1 + pullback_offset]

    return {
        "peak_date": dates[peak_idx],
        "peak_price": peak_price,
        "rally_high_date": dates[rally_idx],
        "main_rally_high": rally_high,
        "rally_gain_pct": round(rally_gain * 100, 2),
        "pullback_date": pullback_date,
        "pullback_low": pullback_low,
        "price_distance_pct": round(price_distance_pct * 100, 2),
        "current_price": current_close,
        "current_date": current_date,
    }


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """历史选股入口：对 df 中每只股票执行龙回头形态识别。"""
    if df.is_empty():
        return pl.DataFrame()

    analyze_lookback_days = params.get("analyze_lookback_days", 250)
    df = df.sort(["symbol", "date"])

    # 确保 change_pct 列存在（enriched parquet 存储列不含此字段，需运行时计算）
    if "change_pct" not in df.columns:
        df = df.with_columns(
            pl.col("close").over("symbol").pct_change().alias("change_pct")
        )

    matched_latest_rows: list[dict] = []

    for symbol in df["symbol"].unique():
        sub = df.filter(pl.col("symbol") == symbol).sort("date")
        # 数据量不足时直接跳过（至少要有足够识别前顶+主升浪+回踩的窗口）
        if sub.height < 100:
            continue

        # 取最近 analyze_lookback_days 条进行分析；若实际数据不足，则使用全部可用数据
        sub = sub.tail(min(analyze_lookback_days, sub.height))
        n = sub.height

        highs = sub["high"].to_list()
        lows = sub["low"].to_list()
        change_pcts = sub["change_pct"].to_list()

        # 前置条件 P2: 至少一个涨停
        has_limit_up = any(pct is not None and pct >= 0.095 for pct in change_pcts)
        if not has_limit_up:
            continue

        # 前置条件 P3: 从历史最低到最高涨幅 ≥ rally_gain_min
        max_h = max(highs)
        min_l = min(lows)
        overall_gain = (max_h - min_l) / min_l if min_l > 0 else 0
        if overall_gain < params.get("rally_gain_min", 50.0) / 100.0:
            continue

        # 识别局部峰值（需 ≥ 2 个峰值）
        peaks = _detect_peaks(highs, window=20)
        if len(peaks) < 2:
            continue

        # 从后续峰值中找形态匹配（优先靠后的峰值，即更近期的前顶）
        best: dict | None = None
        for peak_idx in reversed(peaks):
            res = _score_stock(sub, peak_idx, params)
            if res:
                best = res
                break

        if best:
            name = sub["name"][0] if "name" in sub.columns else ""
            latest = sub.tail(1).to_dicts()[0]
            # 合并标准 enriched 字段，确保通用列（现价/涨跌幅/成交额/量比等）能正常展示
            std_fields = {k: latest.get(k) for k in (
                "close", "change_pct", "amount", "vol_ratio_5d", "turnover_rate",
                "change_amount", "prev_close",
            )}
            row = {
                "symbol": symbol,
                "name": name or latest.get("name", ""),
                "date": latest.get("date"),
                "score": round(
                    best["rally_gain_pct"] * 0.5
                    + max(0, -best["price_distance_pct"]) * 2
                    + (latest.get("vol_ratio_5d", 1.0) or 1.0) * 5,
                    2,
                ),
                **std_fields,
                **best,
            }
            matched_latest_rows.append(row)

    if not matched_latest_rows:
        return pl.DataFrame()

    return pl.DataFrame(matched_latest_rows)


# 本策略通过 filter_history 使用历史窗口识别形态，不定义 filter 函数。
# 引擎在 filter_history 后会自动按目标日期过滤，无需再次执行单日选股。
