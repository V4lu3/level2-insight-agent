"""特征工程特征层：L1 日级 + L3 时段 + L4 时序。

模块结构：
  1. 共享归一化
  2. L1 特征
  3. L3 特征
  4. L4 特征

注意：quote 价格单位为 元×10000；trade/order 价格同。
      _normalize_trade / _normalize_order 会 ÷10000 转为元；
      _normalize_quote 保留原始单位，仅在 _l1_od 中 ÷10000 对齐。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import core as C
from .core import (
    time_to_seconds, safe_div, safe_mean, safe_std, safe_skew, safe_kurt,
    autocorr, assign_time_bucket, assign_l3_window, classify_order,
    compute_hurst_aggregate, compute_fft_spectral_entropy, compute_runs_stat,
)


# ============================================================
#  1. 共享归一化
# ============================================================
def _normalize_trade(trade: pd.DataFrame) -> pd.DataFrame:
    """逐笔成交归一化：价格 ÷10000→元；BS→+1/-1/0；时间→秒。
    过滤 price<=0 的无效记录。
    """
    if trade is None or len(trade) == 0:
        return pd.DataFrame(columns=["time_s", "price", "volume", "bs", "amount"])
    df = trade.copy()
    df["time_s"] = time_to_seconds(df[C.T_COL_TIME])
    df["price"] = df[C.T_COL_PRICE].astype("float64") / 10000.0
    df["volume"] = df[C.T_COL_VOL].astype("float64")
    df["bs"] = df[C.T_COL_BS].astype("str").str.strip().map(
        {"B": 1, "b": 1, "S": -1, "s": -1}).fillna(0).astype("int8")
    df["amount"] = df["price"] * df["volume"]
    df = df[df["price"] > 0].reset_index(drop=True)
    return df


def _normalize_order(order: pd.DataFrame) -> pd.DataFrame:
    """逐笔委托归一化：价格 ÷10000→元；分类器统一 5 类。"""
    if order is None or len(order) == 0:
        return pd.DataFrame(columns=["time_s", "price", "volume", "amount", "type_code"])
    df = order.copy()
    df["time_s"] = time_to_seconds(df[C.O_COL_TIME])
    df["price"] = df[C.O_COL_PRICE].astype("float64") / 10000.0
    df["volume"] = df[C.O_COL_VOL].astype("float64")
    df["amount"] = df["price"] * df["volume"]
    df["type_code"] = classify_order(df[C.O_COL_TYPE])
    return df


def _normalize_quote(quote: pd.DataFrame) -> pd.DataFrame:
    """行情快照归一化：时间→秒 + 价差/不平衡/OBPI/OFI/深度变化/K线形态。"""
    if quote is None or len(quote) == 0:
        return pd.DataFrame(columns=[
            "time_s", "last", "volume", "amount", "spread",
            "imbalance", "depth_decay", "obpi",
            "bid_shape", "ask_shape", "ofi", "ofi_cont",
            "depth_change_bid", "depth_change_ask",
            "prev_close", "is_upper", "amplitude", "gap_ratio",
            "bid_skew", "ask_skew", "mid_price",
        ])
    df = quote.copy().sort_values(C.Q_COL_TIME).reset_index(drop=True)
    df["time_s"] = time_to_seconds(df[C.Q_COL_TIME])

    df["last"] = df[C.Q_COL_LAST].astype("float64")
    df["amount"] = df[C.Q_COL_AMOUNT].astype("float64")
    df["volume"] = df[C.Q_COL_VOLUME].astype("float64")
    df["prev_close"] = df[C.Q_COL_PREV_CLOSE].astype("float64")

    df["change_pct"] = safe_div(df["last"] - df["prev_close"], df["prev_close"]) * 100.0
    df["is_upper"] = (df["change_pct"] >= 9.95).astype("int8")
    df["amplitude"] = safe_div(
        df[C.Q_COL_HIGH].astype("float64") - df[C.Q_COL_LOW].astype("float64"),
        df[C.Q_COL_OPEN].astype("float64"))
    df["gap_ratio"] = safe_div(
        df[C.Q_COL_OPEN].astype("float64") - df["prev_close"], df["prev_close"])

    # 十档盘口
    bid_vols = [df[f"申买量{i}"].astype("float64") for i in range(1, 11)]
    ask_vols = [df[f"申卖量{i}"].astype("float64") for i in range(1, 11)]
    bid_prices = [df[f"申买价{i}"].astype("float64") for i in range(1, 11)]
    ask_prices = [df[f"申卖价{i}"].astype("float64") for i in range(1, 11)]

    bid1 = bid_prices[0]
    ask1 = ask_prices[0]
    bid1_v = bid_vols[0]
    ask1_v = ask_vols[0]

    df["spread"] = (ask1 - bid1).clip(lower=0)
    tot_bid = sum(bid_vols)
    tot_ask = sum(ask_vols)
    df["imbalance"] = safe_div(tot_bid - tot_ask, tot_bid + tot_ask)
    df["depth_decay"] = safe_div(sum(bid_vols[:5]), tot_bid)

    # OBPI
    weights = np.array([1.0 / i for i in range(1, 11)])
    obp_buy = sum(weights[i] * bid_vols[i] for i in range(10))
    obp_sell = sum(weights[i] * ask_vols[i] for i in range(10))
    df["obpi"] = safe_div(obp_buy - obp_sell, obp_buy + obp_sell)

    df["bid_shape"] = safe_div(bid1_v, tot_bid)
    df["ask_shape"] = safe_div(ask1_v, tot_ask)

    # 买卖盘偏度：各档深度的分布偏斜
    # 越均匀（量化）→ skew≈0；越集中在 best 档（游资）→ skew>0
    bid_vol_arr = np.column_stack([bv.values for bv in bid_vols])
    ask_vol_arr = np.column_stack([av.values for av in ask_vols])
    bid_skew = np.array([safe_skew(bid_vol_arr[i]) for i in range(len(df))])
    ask_skew = np.array([safe_skew(ask_vol_arr[i]) for i in range(len(df))])
    df["bid_skew"] = bid_skew
    df["ask_skew"] = ask_skew

    df["mid_price"] = (bid1 + ask1) / 2.0

    # OFI
    df["ofi"] = (df[C.Q_COL_TOTAL_BID].astype("float64")
                 - df[C.Q_COL_TOTAL_ASK].astype("float64"))

    # OFI Cont(2014)
    mid = (bid1 + ask1) / 2.0
    sign = np.sign(mid.diff()).fillna(0)
    df["ofi_cont"] = np.where(sign > 0, bid1_v.diff().fillna(0),
                               np.where(sign < 0, -ask1_v.diff().fillna(0), 0))

    df["depth_change_bid"] = tot_bid.diff()
    df["depth_change_ask"] = tot_ask.diff()

    return df


# ============================================================
#  2. L1 特征
# ============================================================
def _l1_basic(trade: pd.DataFrame) -> dict:
    """基础量价。"""
    n = len(trade)
    if n == 0:
        return {k: 0.0 for k in [
            "l1_volume_total_log", "l1_amount_total_log", "l1_txn_count_log",
            "l1_avg_trade_size_log", "l1_vwap", "l1_amihud_illiquidity"]}
    vol = trade["volume"].sum()
    amt = trade["amount"].sum()
    vwap = safe_div(amt, vol)
    avg_size = safe_div(vol, n)
    prices = trade["price"].values
    rets = np.abs(np.diff(np.log(np.clip(prices, 1e-6, None))))
    amihud = float(np.nanmean(rets)) / (float(amt) if amt > 0 else 1.0) * 1e9 if len(rets) > 0 else 0.0
    return {
        "l1_volume_total_log": float(np.log1p(vol)),
        "l1_amount_total_log": float(np.log1p(amt)),
        "l1_txn_count_log": float(np.log1p(n)),
        "l1_avg_trade_size_log": float(np.log1p(avg_size)),
        "l1_vwap": float(vwap),
        "l1_amihud_illiquidity": float(amihud),
    }


def _l1_rs_split(trade: pd.DataFrame) -> dict:
    """RS 拆单与爆发特征：量化拆单均匀性 + 游资突发性。

    第一性原理：量化算法拆单使成交量均匀分布（低CV、高自相关、偏度≈0）；
    游资集中扫货使成交量分布尖峰厚尾（高CV、高偏度、高峰度）；
    散户随机成交介于两者之间。
    """
    if len(trade) < 3:
        return {k: 0.0 for k in [
            "l1_rs_split_similarity", "l1_rs_burst_ratio",
            "l1_rs_volume_skew", "l1_rs_volume_kurt",
            "l1_rs_top5_volume_pct", "l1_rs_top10_volume_pct"]}
    vol = trade["volume"].values
    cv = safe_std(vol) / (np.mean(vol) + 1e-6)

    # 拆单相似度：CV 越小（量化越均匀）→ 越接近 1
    similarity = 1.0 / (1.0 + cv)

    # 爆发比：最大成交量 / 均值
    burst = float(vol.max() / (vol.mean() + 1e-6))

    # 成交量分布形状
    skew = safe_skew(vol)
    kurt = safe_kurt(vol)

    # 前 N 笔占比
    n5 = min(5, len(vol))
    n10 = min(10, len(vol))
    top5 = float(np.partition(vol, -n5)[-n5:].sum() / (vol.sum() + 1e-6))
    top10 = float(np.partition(vol, -n10)[-n10:].sum() / (vol.sum() + 1e-6))

    return {
        "l1_rs_split_similarity": similarity,
        "l1_rs_burst_ratio": burst,
        "l1_rs_volume_skew": skew,
        "l1_rs_volume_kurt": kurt,
        "l1_rs_top5_volume_pct": top5,
        "l1_rs_top10_volume_pct": top10,
    }


def _l1_oss(trade: pd.DataFrame) -> dict:
    """单笔金额分级（大单/小单/超大单）。"""
    if len(trade) == 0:
        return {k: 0.0 for k in [
            "l1_oss_big_amount_pct", "l1_oss_small_count_pct",
            "l1_oss_mega_amount_pct", "l1_quant_small_pct"]}
    amt = trade["amount"].values
    total = amt.sum() + 1e-12
    return {
        "l1_oss_mega_amount_pct": float((amt >= C.MEGA_AMOUNT_THRESHOLD).sum() / total),
        "l1_oss_big_amount_pct": float((amt >= C.BIG_AMOUNT_THRESHOLD).sum() / total),
        "l1_oss_small_count_pct": float((amt < C.SMALL_AMOUNT_THRESHOLD).mean()),
        "l1_quant_small_pct": float(amt[amt < C.SMALL_AMOUNT_THRESHOLD].sum() / total),
    }


def _l1_trd_interval(trade: pd.DataFrame) -> dict:
    """成交节奏：间隔变异系数 CV。"""
    if len(trade) < 3:
        return {"l1_trd_interval_cv": 0.0}
    dt = np.diff(trade["time_s"].values)
    dt = dt[dt > 0]
    if len(dt) < 2:
        return {"l1_trd_interval_cv": 0.0}
    return {"l1_trd_interval_cv": float(np.std(dt, ddof=1) / (np.mean(dt) + 1e-6))}


def _l1_ap(trade: pd.DataFrame) -> dict:
    """主动成交：方向/连续性/扫单/对倒/大单净流入/方向主导度/净方向强度。

    第一性原理：游资方向极端（主动买 >> 主动卖或反之），量化方向中性接近 0.5。
    """
    if len(trade) == 0:
        return {k: v for k, v in {
            "l1_ap_active_buy_pct": 0.5, "l1_ap_max_consecutive": 0.0,
            "l1_ap_sweep_ratio": 0.0, "l1_ap_big_net_inflow": 0.0,
            "l1_ap_wash_trade_ratio": 0.0,
            "l1_ap_max_consecutive_buy": 0.0, "l1_ap_max_consecutive_sell": 0.0,
            "l1_ap_dominant_direction": 0.0, "l1_ap_active_net_direction": 0.0,
        }.items()}
    bs = trade["bs"].values
    amt = trade["amount"].values
    big_mask = amt >= C.BIG_AMOUNT_THRESHOLD
    total = amt.sum() + 1e-12

    buy_amt = amt[bs > 0].sum()
    sell_amt = amt[bs < 0].sum()
    active_buy_pct = buy_amt / total

    # 方向主导度：max(buy_pct, 1-buy_pct) → 值越大单边性越强
    dominant = max(active_buy_pct, 1.0 - active_buy_pct)
    # 净方向强度：abs(buy - sell) / (buy + sell)
    net_direction = abs(buy_amt - sell_amt) / (buy_amt + sell_amt + 1e-12)

    # 连续同向 run-length
    sign = np.sign(bs)
    sign = sign[sign != 0]
    max_cons = max_cons_buy = max_cons_sell = 0
    if len(sign) > 0:
        diffs = np.diff(sign) != 0
        runs = [r for r in np.split(sign, np.where(diffs)[0] + 1) if len(r) > 0]
        max_cons = max(len(r) for r in runs)
        max_cons_buy = max((len(r) for r in runs if r[0] > 0), default=0)
        max_cons_sell = max((len(r) for r in runs if r[0] < 0), default=0)

    thresh = np.percentile(amt, 95) if len(amt) >= 20 else np.max(amt)
    sweep_ratio = float((amt >= thresh).mean())
    wash = float(min(buy_amt, sell_amt) / (max(buy_amt, sell_amt) + 1e-6))
    big_net = (amt[big_mask & (bs > 0)].sum() - amt[big_mask & (bs < 0)].sum()) / total

    return {
        "l1_ap_active_buy_pct": float(active_buy_pct),
        "l1_ap_max_consecutive": float(max_cons),
        "l1_ap_sweep_ratio": sweep_ratio,
        "l1_ap_big_net_inflow": float(big_net),
        "l1_ap_wash_trade_ratio": wash,
        "l1_ap_max_consecutive_buy": float(max_cons_buy),
        "l1_ap_max_consecutive_sell": float(max_cons_sell),
        "l1_ap_dominant_direction": float(dominant),
        "l1_ap_active_net_direction": float(net_direction),
    }


def _l1_cb(order: pd.DataFrame, trade: pd.DataFrame) -> dict:
    """撤单链路：撤单量比/快速撤单比/大单撤单占比/部分撤单比/闪电撤单/买卖撤单分化/撤单委托比。

    第一性原理：量化高频撤单多（闪电撤单<2s），游资撤单分化（挂单试探后撤买单或卖单），散户撤单少。
    """
    if order is None or len(order) == 0:
        return {k: 0.0 for k in [
            "l1_cb_cancel_volume_ratio", "l1_cb_fast_cancel_ratio",
            "l1_cb_big_cancel_amount_pct", "l1_cb_partial_cancel_ratio",
            "l1_cb_cancel_imbalance",
            "l1_cb_flash_cancel_ratio", "l1_cb_cancel_order_ratio",
            "l1_cb_cancel_buy_ratio", "l1_cb_cancel_sell_ratio"]}
    types = order["type_code"].values
    amts = order["amount"].values
    times = order["time_s"].values
    total_vol = float(trade["volume"].sum() + 1e-6)
    total_order_amt = amts.sum() + 1e-6

    cancel_mask = types == 1
    partial_mask = types == 3
    cancel_vol = amts[cancel_mask].sum()

    # 快速撤单
    fast_cancel = 0
    if cancel_mask.any():
        cancel_times = times[cancel_mask]
        if len(cancel_times) > 1:
            fast_cancel = int((np.diff(cancel_times) < C.FAST_CANCEL_INTERVAL_SEC).sum())

    # 闪电撤单（< 2秒，高频量化特征）
    flash_cancel = 0
    if cancel_mask.any():
        cancel_times = times[cancel_mask]
        if len(cancel_times) > 1:
            flash_cancel = int((np.diff(cancel_times) < C.FLASH_CANCEL_INTERVAL_SEC).sum())

    big_cancel = amts[cancel_mask & (amts >= C.BIG_AMOUNT_THRESHOLD)].sum()
    partial_amt = amts[partial_mask].sum()
    cancel_total = amts[cancel_mask].sum() + amts[partial_mask].sum()

    # 有条件地判断撤单买卖方向：委托类型编码中不含买卖方向信息
    # 对于撤单，我们无法区分买卖，这里用委托金额的 balance 作为近似
    # 更精确的买卖方向需结合 order 的买卖方向字段（如果有）
    # 这里用 type_code 0=主动 vs 1=撤单，撤单无买卖方向，设为0
    cancel_buy_ratio = 0.0
    cancel_sell_ratio = 0.0

    # 撤单/委托比：撤单次数 / 总委托次数
    cancel_order_ratio = float(cancel_mask.sum() / max(len(order), 1))

    return {
        "l1_cb_cancel_volume_ratio": float(cancel_vol / total_vol),
        "l1_cb_fast_cancel_ratio": float(fast_cancel / max(int(cancel_mask.sum()), 1)),
        "l1_cb_big_cancel_amount_pct": float(big_cancel / (cancel_vol + 1e-6)),
        "l1_cb_partial_cancel_ratio": float(partial_amt / (cancel_vol + partial_amt + 1e-6)),
        "l1_cb_cancel_imbalance": float(np.abs(amts[cancel_mask].sum() - amts[partial_mask].sum())
                                         / (cancel_total + 1e-6)),
        "l1_cb_flash_cancel_ratio": float(flash_cancel / max(int(cancel_mask.sum()), 1)),
        "l1_cb_cancel_order_ratio": cancel_order_ratio,
        "l1_cb_cancel_buy_ratio": cancel_buy_ratio,
        "l1_cb_cancel_sell_ratio": cancel_sell_ratio,
    }


def _l1_od(order: pd.DataFrame, quote: pd.DataFrame) -> dict:
    """主动委托：大单占比 + 委托价相对VWAP偏离。"""
    _zero = {"l1_od_active_big_ratio": 0.0, "l1_od_price_deviation_mean": 0.0}
    if order is None or len(order) == 0 or quote is None or len(quote) == 0:
        return _zero
    types = order["type_code"].values
    active_mask = types == 0
    if active_mask.sum() == 0:
        return _zero
    active_big = float((order["amount"].values[active_mask] >= C.BIG_AMOUNT_THRESHOLD).mean())
    q_last = quote["last"].astype("float64").values / 10000.0
    q_vol = quote["volume"].astype("float64").values
    vwap = float(np.sum(q_last * q_vol) / (np.sum(q_vol) + 1e-6))
    dev = float(np.mean(np.abs(order["price"].values[active_mask] - vwap) / (vwap + 1e-6)))
    return {"l1_od_active_big_ratio": active_big, "l1_od_price_deviation_mean": dev}


def _l1_obp(quote: pd.DataFrame) -> dict:
    """盘口特征：价差/不平衡/深度衰减/OBPI/形状/偏度/中间价。

    第一性原理：量化做市使盘口均匀（偏度≈0），游资/散户集中在 best 档（偏度>0）。
    """
    if len(quote) == 0:
        return {k: 0.0 for k in [
            "l1_obp_spread_mean", "l1_obp_imbalance_mean", "l1_obp_depth_decay_mean",
            "l1_obpi_mean", "l1_bid_shape_mean", "l1_ask_shape_mean",
            "l1_obp_bid_skew_mean", "l1_obp_ask_skew_mean", "l1_obp_mid_price_mean"]}
    return {
        "l1_obp_spread_mean": float(safe_mean(quote["spread"].values)),
        "l1_obp_imbalance_mean": float(safe_mean(quote["imbalance"].values)),
        "l1_obp_depth_decay_mean": float(safe_mean(quote["depth_decay"].values)),
        "l1_obpi_mean": float(safe_mean(quote["obpi"].values)),
        "l1_bid_shape_mean": float(safe_mean(quote["bid_shape"].values)),
        "l1_ask_shape_mean": float(safe_mean(quote["ask_shape"].values)),
        "l1_obp_bid_skew_mean": float(safe_mean(quote["bid_skew"].values)),
        "l1_obp_ask_skew_mean": float(safe_mean(quote["ask_skew"].values)),
        "l1_obp_mid_price_mean": float(safe_mean(quote["mid_price"].values)),
    }


def _l1_ofi(quote: pd.DataFrame) -> dict:
    """OFI：简化版均值/正占比 + Cont(2014) 连续版累积/标准差。"""
    if len(quote) == 0:
        return {"l1_ofi_mean": 0.0, "l1_ofi_positive_ratio": 0.5,
                "l1_ofi_cont_cumsum_last": 0.0, "l1_ofi_cont_std": 0.0}
    ofi = quote["ofi"].values
    ofi_c = quote["ofi_cont"].values
    return {
        "l1_ofi_mean": float(np.nanmean(ofi)),
        "l1_ofi_positive_ratio": float((ofi > 0).mean()),
        "l1_ofi_cont_cumsum_last": float(np.nansum(ofi_c)),
        "l1_ofi_cont_std": float(np.nanstd(ofi_c)),
    }


def _l1_dc(quote: pd.DataFrame) -> dict:
    """深度变化：突变占比 + 卖/买盘最大变化 + 买盘平均变化。"""
    _zero = {"l1_dc_sudden_change_ratio": 0.0, "l1_dc_ask_change_max": 0.0,
             "l1_dc_bid_change_max": 0.0, "l1_dc_bid_change_mean": 0.0}
    if len(quote) < 5:
        return _zero
    ask_change = quote["depth_change_ask"].values
    ask_change = ask_change[~np.isnan(ask_change)]
    bid_change = quote["depth_change_bid"].values
    bid_change = bid_change[~np.isnan(bid_change)]
    if len(ask_change) < 2:
        return _zero
    sd = np.std(ask_change) + 1e-6
    return {
        "l1_dc_sudden_change_ratio": float((np.abs(ask_change) > C.DEPTH_CHANGE_SUDDEN_SIGMA * sd).mean()),
        "l1_dc_ask_change_max": float(np.max(np.abs(ask_change))),
        "l1_dc_bid_change_max": float(np.max(np.abs(bid_change))) if len(bid_change) > 0 else 0.0,
        "l1_dc_bid_change_mean": float(safe_mean(np.abs(bid_change))),
    }


def _l1_pd(trade: pd.DataFrame) -> dict:
    """价格行为：动量/波动/跳跃/收益分布形状/最大回撤/上涨比。

    第一性原理：游资制造极端收益分布（高偏度、高峰度、大回撤），
    量化收益分布更接近正态，散户介于之间。
    """
    if len(trade) < 3:
        return {"l1_pd_price_momentum": 0.0, "l1_pd_return_std": 0.0,
                "l1_pd_price_jump_count": 0.0,
                "l1_pd_return_skew": 0.0, "l1_pd_return_kurt": 0.0,
                "l1_pd_max_drawdown": 0.0, "l1_pd_up_ratio": 0.5,
                "l1_pd_price_jump_max": 0.0}
    prices = trade["price"].values
    rets = np.diff(np.log(np.clip(prices, 1e-6, None)))
    momentum = (prices[-1] - prices[0]) / (prices[0] + 1e-6)
    thresh = (np.percentile(np.abs(rets), C.JUMP_PERCENTILE * 100) if len(rets) >= 20
              else np.mean(np.abs(rets)) + 2 * np.std(rets))

    # 收益分布形状
    ret_skew = safe_skew(rets)
    ret_kurt = safe_kurt(rets)

    # 最大回撤
    cumlog = np.cumsum(np.log(np.clip(prices[1:] / prices[:-1], 1e-6, None)))
    running_max = np.maximum.accumulate(cumlog)
    drawdown = cumlog - running_max
    max_dd = float(np.abs(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    # 上涨比
    up_ratio = float((rets > 0).mean())

    return {
        "l1_pd_price_momentum": float(momentum),
        "l1_pd_return_std": float(np.std(rets)),
        "l1_pd_price_jump_count": float((np.abs(rets) > thresh).sum()),
        "l1_pd_return_skew": ret_skew,
        "l1_pd_return_kurt": ret_kurt,
        "l1_pd_max_drawdown": max_dd,
        "l1_pd_up_ratio": up_ratio,
        "l1_pd_price_jump_max": float(np.max(np.abs(rets))) if len(rets) > 0 else 0.0,
    }


def _l1_pi(trade: pd.DataFrame) -> dict:
    """时段分布：开盘30min/收盘10min/尾盘冲刺/峰值集中度/集合竞价占比/VWAP偏离。

    第一性原理：游资尾盘拉升（close_rush 高），量化均匀分布（peak_ratio 低），
    散户早盘活跃（auction_volume 高）。
    """
    if len(trade) == 0:
        return {k: 0.0 for k in [
            "l1_pi_open_30min_amount_pct", "l1_pi_close_10min_amount_pct",
            "l1_pi_close_rush_amount_pct", "l1_pi_peak_amount_ratio",
            "l1_pi_auction_volume_pct", "l1_pi_vwap_deviation",
            "l1_micro_hhi_5min"]}
    sec = trade["time_s"].values
    amt = trade["amount"].values
    vol = trade["volume"].values
    prices = trade["price"].values
    total = amt.sum() + 1e-6
    bucket = assign_time_bucket(sec)

    open_pct = amt[bucket == 1].sum() / total
    close_pct = amt[bucket == 3].sum() / total

    # 尾盘冲刺：收盘前最后 10 分钟占全天比
    rush_mask = (sec >= C.TRADING_END_SEC - 600) & (sec < C.TRADING_END_SEC)
    close_rush_pct = amt[rush_mask].sum() / total

    # 峰值集中度：最大 30min 金额占比
    in_trade = (sec >= C.TRADING_START_SEC) & (sec < C.TRADING_END_SEC)
    amt_t = amt[in_trade]
    sec_t = sec[in_trade]
    if len(sec_t) == 0:
        peak_ratio = 0.0
    else:
        bucket_30m = ((sec_t - C.TRADING_START_SEC) // 1800).astype(int)
        sums = np.bincount(bucket_30m, weights=amt_t,
                           minlength=bucket_30m.max() + 1 if len(bucket_30m) > 0 else 1)
        peak_ratio = float(sums.max() / (sums.sum() + 1e-6))

    # 集合竞价占比
    auction_mask = (sec >= C.AUCTION_START_SEC) & (sec < C.AUCTION_END_SEC)
    auction_vol_pct = float(amt[auction_mask].sum() / total)

    # VWAP 偏离：成交价 vs VWAP 的加权平均偏离
    prices_t = prices[in_trade]
    amt_t2 = amt[in_trade]
    vwap = float(np.sum(prices_t * amt_t2) / (np.sum(amt_t2) + 1e-6))
    vwap_dev = float(np.mean(np.abs(prices_t - vwap) / (vwap + 1e-6)))

    # 5 分钟 HHI
    if len(sec_t) == 0:
        hhi = 0.0
    else:
        bucket_5m = ((sec_t - C.TRADING_START_SEC) // 300).astype(int)
        sums = np.bincount(bucket_5m, weights=amt_t,
                           minlength=bucket_5m.max() + 1 if len(bucket_5m) > 0 else 1)
        p = sums / (sums.sum() + 1e-6)
        hhi = float((p ** 2).sum())

    return {
        "l1_pi_open_30min_amount_pct": float(open_pct),
        "l1_pi_close_10min_amount_pct": float(close_pct),
        "l1_pi_close_rush_amount_pct": float(close_rush_pct),
        "l1_pi_peak_amount_ratio": peak_ratio,
        "l1_pi_auction_volume_pct": float(auction_vol_pct),
        "l1_pi_vwap_deviation": vwap_dev,
        "l1_micro_hhi_5min": hhi,
    }


def _l1_micro(trade: pd.DataFrame) -> dict:
    """微观结构补充：30min HHI、集中度、价格冲击弹性。

    第一性原理：量化时间分散（30min HHI 低），游资集中攻击（30min HHI 高）。
    """
    if len(trade) == 0:
        return {k: 0.0 for k in [
            "l1_micro_hhi_30min", "l1_micro_concentration_top10",
            "l1_micro_price_impact_elasticity"]}
    sec = trade["time_s"].values
    amt = trade["amount"].values
    vol = trade["volume"].values
    prices = trade["price"].values
    total_amt = amt.sum() + 1e-6

    # 30min HHI
    in_trade = (sec >= C.TRADING_START_SEC) & (sec < C.TRADING_END_SEC)
    sec_t = sec[in_trade]
    amt_t = amt[in_trade]
    if len(sec_t) == 0:
        hhi_30 = 0.0
    else:
        bucket_30m = ((sec_t - C.TRADING_START_SEC) // 1800).astype(int)
        sums = np.bincount(bucket_30m, weights=amt_t,
                           minlength=bucket_30m.max() + 1 if len(bucket_30m) > 0 else 1)
        p = sums / (sums.sum() + 1e-6)
        hhi_30 = float((p ** 2).sum())

    # 集中度：交易额最大的前10笔占比
    n10 = min(10, len(amt))
    conc_top10 = float(np.partition(amt, -n10)[-n10:].sum() / total_amt)

    # 价格冲击弹性：Amihud 的变体，衡量成交额对价格的影响
    # 越低→流动性越好（量化做市），越高→冲击越大（游资扫货）
    if len(prices) >= 5 and total_amt > 0:
        log_rets = np.abs(np.diff(np.log(np.clip(prices, 1e-6, None))))
        impact = float(np.mean(log_rets) / (np.log1p(total_amt) + 1e-6)) * 1e6
    else:
        impact = 0.0

    return {
        "l1_micro_hhi_30min": hhi_30,
        "l1_micro_concentration_top10": conc_top10,
        "l1_micro_price_impact_elasticity": impact,
    }


def _l1_limit(quote: pd.DataFrame) -> dict:
    """涨跌停：是否触及涨停/封板强度/触及跌停/近涨停比。

    第一性原理：游资打板推至涨停并封板（seal_upper_strength 高），
    量化几乎不参与涨停板交易，散户追涨（near_upper_ratio 高）。
    """
    if len(quote) == 0:
        return {k: 0.0 for k in [
            "l1_limit_touch_upper", "l1_limit_seal_upper_strength",
            "l1_limit_touch_lower", "l1_limit_near_upper_ratio"]}
    last = quote["last"].values
    prev_close = quote["prev_close"].values
    change_pct = safe_div(last - prev_close, prev_close) * 100.0

    touch_upper = float((change_pct >= 9.95).max())
    touch_lower = float((change_pct <= -9.95).max())

    # 封板强度：涨停后成交价在涨停价附近的快照占比
    if touch_upper > 0:
        at_upper = change_pct >= 9.95
        seal_strength = float(at_upper.mean())
    else:
        seal_strength = 0.0

    # 近涨停比：涨幅 > 8% 的快照占比（游资试探行为）
    near_upper = float((change_pct >= 8.0).mean())

    return {
        "l1_limit_touch_upper": touch_upper,
        "l1_limit_seal_upper_strength": seal_strength,
        "l1_limit_touch_lower": touch_lower,
        "l1_limit_near_upper_ratio": near_upper,
    }


def _l1_kl(quote: pd.DataFrame) -> dict:
    """K线形态：振幅/跳空/实体比/影线比/十字星/收盘回报。

    第一性原理：游资制造长上影线（拉高出货）或长下影线（洗盘），
    量化交易 K 线形态更接近随机。
    """
    if len(quote) == 0:
        return {k: 0.0 for k in [
            "l1_kl_amplitude", "l1_kl_gap_ratio",
            "l1_kl_body_ratio", "l1_kl_upper_shadow_ratio", "l1_kl_lower_shadow_ratio",
            "l1_kl_doji_score", "l1_kl_body_direction", "l1_kl_close_return"]}
    open_p = quote[C.Q_COL_OPEN].astype("float64").values
    high_p = quote[C.Q_COL_HIGH].astype("float64").values
    low_p = quote[C.Q_COL_LOW].astype("float64").values
    close_p = quote["last"].values if "last" in quote.columns else open_p

    # 使用当天最后一笔快照
    o, h, l, c = open_p[-1], high_p[-1], low_p[-1], close_p[-1]

    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    total_range = h - l + 1e-6

    body_ratio = body / total_range
    upper_shadow = upper / total_range
    lower_shadow = lower / total_range
    doji = 1.0 if body < (total_range * 0.1) else 0.0
    body_dir = 1.0 if c > o else -1.0 if c < o else 0.0
    close_return = float((c - o) / (o + 1e-6))

    # 振幅和跳空（已有）
    prev_close = quote["prev_close"].values[-1] if len(quote) > 0 else open_p[-1]
    amplitude = float(safe_div(high_p[-1] - low_p[-1], open_p[-1]))
    gap = float(safe_div(open_p[-1] - prev_close, prev_close))

    return {
        "l1_kl_amplitude": amplitude,
        "l1_kl_gap_ratio": gap,
        "l1_kl_body_ratio": float(body_ratio),
        "l1_kl_upper_shadow_ratio": float(upper_shadow),
        "l1_kl_lower_shadow_ratio": float(lower_shadow),
        "l1_kl_doji_score": float(doji),
        "l1_kl_body_direction": float(body_dir),
        "l1_kl_close_return": close_return,
    }


def _l1_wavg_spread(quote: pd.DataFrame) -> dict:
    """加权叫买卖价差：均值/标准差/与中间价比。"""
    _zero = {"l1_wavg_spread_mean": 0.0, "l1_wavg_spread_std": 0.0,
             "l1_wavg_spread_mid_ratio": 0.0}
    if len(quote) == 0:
        return _zero
    sp = (quote[C.Q_COL_WEIGHTED_ASK].astype("float64").values
          - quote[C.Q_COL_WEIGHTED_BID].astype("float64").values)
    sp_pos = sp[sp > 0]
    if len(sp_pos) == 0:
        return _zero
    spread_mean = float(safe_mean(sp_pos))
    spread_std = float(safe_std(sp_pos, ddof=1))

    # 价差/中间价比：价差占中间价的比例
    mid = (quote[C.Q_COL_WEIGHTED_BID].astype("float64").values
           + quote[C.Q_COL_WEIGHTED_ASK].astype("float64").values) / 2.0
    mid_pos = mid[sp > 0]
    sp_ratio = float(np.mean(sp_pos / (mid_pos + 1e-6))) if len(mid_pos) > 0 else 0.0

    return {"l1_wavg_spread_mean": spread_mean, "l1_wavg_spread_std": spread_std,
            "l1_wavg_spread_mid_ratio": sp_ratio}


def _l1_sweep(trade: pd.DataFrame) -> dict:
    """扫单比 = 成交额 > P95 占比。"""
    if len(trade) < 20:
        return {"l1_swp_quote_sweep_ratio": 0.0}
    amt = trade["amount"].values
    return {"l1_swp_quote_sweep_ratio": float((amt > np.percentile(amt, 95)).mean())}


def _l1_quant_derived(d: dict) -> dict:
    """量化/游资派生标量（从已算好的字段组合）。"""
    return {
        "l1_quant_uniform_score": float(1.0 / (1.0 + d.get("l1_trd_interval_cv", 0.0))),
        "l1_yz_burst_score": float(d.get("l1_ap_max_consecutive", 0) / 100.0
                                   * d.get("l1_pi_open_30min_amount_pct", 0)),
    }


def _l1_scoring_derived(d: dict) -> dict:
    """量化/游资评分补充：中性度、方向强度、游资大单占比。

    第一性原理：量化交易方向中性（net_direction≈0）、无方向偏好（dominant≈0.5）；
    游资方向极端（net_direction>0.6、dominant>0.7）。
    """
    r = d.get("l1_rs_split_similarity", 0.5)
    t = d.get("l1_trd_interval_cv", 1.0)
    nd = d.get("l1_ap_active_net_direction", 0.0)
    dom = d.get("l1_ap_dominant_direction", 0.5)

    # 中性度：均匀(r高) + 方向中性(nd低) + 节奏规律(t低)
    neutrality = float(r / (1.0 + nd + t))

    # 方向强度 = 极端方向性
    dir_strength = float(dom * (1.0 + nd))

    # 游资大单占比 = 大单占比 × 方向性
    big_pct = d.get("l1_oss_big_amount_pct", 0.0)
    yz_big = float(big_pct * dom)

    return {
        "l1_quant_neutrality": neutrality,
        "l1_yz_direction_strength": dir_strength,
        "l1_yz_big_pct": yz_big,
    }


def extract_l1_features(quote_raw, trade_raw, order_raw) -> dict:
    """单只股票单日 → L1 特征 dict。"""
    trade = _normalize_trade(trade_raw)
    order = _normalize_order(order_raw)
    quote = _normalize_quote(quote_raw)

    d = {}
    d.update(_l1_basic(trade))
    d.update(_l1_rs_split(trade))
    d.update(_l1_oss(trade))
    d.update(_l1_trd_interval(trade))
    d.update(_l1_ap(trade))
    d.update(_l1_cb(order, trade))
    d.update(_l1_od(order, quote))
    d.update(_l1_obp(quote))
    d.update(_l1_ofi(quote))
    d.update(_l1_dc(quote))
    d.update(_l1_pd(trade))
    d.update(_l1_pi(trade))
    d.update(_l1_micro(trade))
    d.update(_l1_limit(quote))
    d.update(_l1_kl(quote))
    d.update(_l1_wavg_spread(quote))
    d.update(_l1_sweep(trade))
    d.update(_l1_quant_derived(d))
    d.update(_l1_scoring_derived(d))

    d["dq_missing_trade_ratio"] = 1.0 - min(len(trade) / 5000.0, 1.0)

    for k in C.L1_P0:
        d.setdefault(k, 0.0)
    return d


# ============================================================
#  3. L3 特征（10 维）
# ============================================================
def _l3_window_core(trade: pd.DataFrame, order: pd.DataFrame) -> np.ndarray:
    """构建 16×13 矩阵：每 15min 窗口汇总 13 个统计量。"""
    mat = np.zeros((16, 13), dtype=np.float64)
    if len(trade) == 0:
        return mat

    # 过滤 price<=0 的无效记录
    trade = trade[trade[C.T_COL_PRICE].astype("float64") > 0]
    if len(trade) == 0:
        return mat

    sec = time_to_seconds(trade[C.T_COL_TIME]).values
    prices = trade[C.T_COL_PRICE].astype("float64").values / 10000.0
    vol = trade[C.T_COL_VOL].astype("float64").values
    amt = prices * vol
    bs = trade[C.T_COL_BS].astype("str").str.strip().map(
        {"B": 1, "S": -1}).fillna(0).values
    win_idx = assign_l3_window(sec)
    big_mask = amt >= C.BIG_AMOUNT_THRESHOLD

    flash_cnt = np.zeros(16, dtype=np.float64)
    cancel_mask_o = np.zeros(0, dtype=bool)
    o_amt = np.zeros(0, dtype=np.float64)
    o_win = np.full(0, -1, dtype=np.int32)
    if order is not None and len(order) > 0:
        o_sec = time_to_seconds(order[C.O_COL_TIME]).values
        o_amt = (order[C.O_COL_PRICE].astype("float64").values / 10000.0
                 * order[C.O_COL_VOL].astype("float64").values)
        o_types = classify_order(order[C.O_COL_TYPE]).values
        o_win = assign_l3_window(o_sec)
        cancel_mask_o = o_types == 1
        for w in range(16):
            idx = np.where((o_win == w) & cancel_mask_o)[0]
            if len(idx) >= 2:
                flash_cnt[w] = int((np.diff(o_sec[idx]) < C.FLASH_CANCEL_INTERVAL_SEC).sum())

    max_jump = np.zeros(16, dtype=np.float64)
    if len(trade) > 2:
        rets = np.abs(np.diff(np.log(np.clip(prices, 1e-6, None))))
        for w in range(16):
            w_rets = rets[(win_idx == w)[1:]]
            if len(w_rets) > 0:
                max_jump[w] = w_rets[np.argmax(w_rets)]

    for w in range(16):
        m = win_idx == w
        if m.sum() == 0:
            continue
        w_amt = amt[m].sum()
        w_vol = vol[m].sum()
        w_buy = amt[m & (bs > 0)].sum()
        w_vol_arr = vol[m]
        o_m = o_win == w
        w_cancel_amt = o_amt[o_m & cancel_mask_o].sum() if len(o_amt) > 0 else 0.0
        w_cancel_cnt = int((o_m & cancel_mask_o).sum()) if len(o_amt) > 0 else 0
        mat[w] = [
            w_amt, w_vol, int(m.sum()), safe_div(w_amt, w_vol),
            safe_div(w_buy, w_amt), 0.0,
            amt[m & big_mask].sum() / (w_amt + 1e-6),
            w_cancel_cnt, w_cancel_amt, safe_div(w_cancel_amt, w_amt),
            flash_cnt[w], max_jump[w],
            safe_div(w_vol_arr.max(), w_vol_arr.mean() + 1e-6),
        ]
    return mat


def _l2_from_l3(mat: np.ndarray) -> dict:
    """从 16 窗矩阵 → 5 维 L2 摘要。"""
    n_seg = 8
    seg = 16 // n_seg
    amt_8 = np.array([mat[i*seg:(i+1)*seg, 0].sum() for i in range(n_seg)])
    buy_8 = np.array([mat[i*seg:(i+1)*seg, 4].mean() for i in range(n_seg)])
    flash_8 = np.array([mat[i*seg:(i+1)*seg, 10].sum() for i in range(n_seg)])

    x = np.arange(n_seg, dtype=np.float64)
    if amt_8.std() < 1e-12:
        amt_trend = 0.0
    else:
        xm, am = x.mean(), amt_8.mean()
        amt_trend = float(((x - xm) * (amt_8 - am)).sum() / ((x - xm) ** 2).sum() / (am + 1e-6))

    total = amt_8.sum() + 1e-6
    p = amt_8 / total
    p = p[p > 0]

    return {
        "l2_amt_trend": amt_trend,
        "l2_buy_range": float(buy_8.max() - buy_8.min()),
        "l2_time_hhi": float(((amt_8 / total) ** 2).sum()),
        "l2_flash_cancel_total": float(flash_8.sum()),
        "l2_mat_entropy": float(-(p * np.log(p + 1e-12)).sum()),
    }


def _l3_scalars(mat: np.ndarray) -> dict:
    """从 16 窗矩阵 → 5 维 L3 标量。"""
    burst = mat[:, 12]
    jump = mat[:, 11]
    cancel = mat[:, 9]
    buy = mat[:, 4]
    vol = mat[:, 1]

    sign = np.sign(buy - 0.5)
    diffs = np.diff(sign) != 0
    runs = [r for r in np.split(sign, np.where(diffs)[0] + 1) if len(r) > 0]
    max_run = max((len(r) for r in runs), default=0)
    seq_z = float((max_run - 8) / (4 + 1e-6))

    if len(vol) > 2 and vol.std() > 1e-6:
        v0 = vol[:-1] - vol.mean()
        vol_acf = float((v0 * (vol[1:] - vol.mean())).sum() / ((v0 ** 2).sum() + 1e-6))
    else:
        vol_acf = 0.0

    return {
        "l3_burst_max": float(burst.max()),
        "l3_jump_max": float(jump.max()),
        "l3_cancel_mean": float(cancel.mean()),
        "l3_mat_seq_pattern_score": seq_z,
        "l3_mat_vol_clustering_acf": vol_acf,
    }


def extract_l3_features(trade_raw, order_raw) -> dict:
    """单只股票单日 → 10 维 L3 特征 dict。"""
    mat = _l3_window_core(trade_raw, order_raw)
    d = {}
    d.update(_l2_from_l3(mat))
    d.update(_l3_scalars(mat))
    for k in C.L3_P0:
        d.setdefault(k, 0.0)
    return d


# ============================================================
#  4. L4 特征（12 维：时序摘要 + 高级时序分析）
# ============================================================
def extract_l4_features(quote_raw, trade_raw) -> dict:
    """单只股票单日 → 12 维 L4 时序摘要 dict。

    新增 Hurst 指数、FFT 频谱熵、游程检验、方差比、序列相关、趋势拟合
    从时序结构层面区分三类交易者：
    - 量化做市 → H≈0.5(随机游走)、频谱熵高(复杂)、无趋势
    - 游资拉升 → H>0.5(趋势持续)、频谱熵低(规律)、显著趋势
    - 散户追涨 → H波动大、游程检验显著
    """
    if quote_raw is None or len(quote_raw) < 5 or trade_raw is None or len(trade_raw) < 5:
        return {k: 0.0 for k in C.L4_P0}

    q = quote_raw.sort_values(C.Q_COL_TIME).reset_index(drop=True)
    d = {}

    # --- 原有 5 维 ---
    ofi = (q[C.Q_COL_TOTAL_BID].astype("float64").values
           - q[C.Q_COL_TOTAL_ASK].astype("float64").values)
    d["l4_ofi_autocorr_lag1"] = float(autocorr(ofi, lag=1))

    tot = (q[C.Q_COL_TOTAL_BID].astype("float64").values
           + q[C.Q_COL_TOTAL_ASK].astype("float64").values)
    diff = np.abs(np.diff(tot))
    d["l4_depth_change_cv"] = float(diff.std() / diff.mean()) if len(diff) > 2 and diff.mean() > 1e-6 else 0.0

    vol = trade_raw[C.T_COL_VOL].astype("float64").values
    d["l4_acf_volume_lag1"] = float(autocorr(vol.astype(np.float64), lag=1))

    prices = trade_raw[C.T_COL_PRICE].astype("float64").values / 10000.0
    high_ratio = 0.0
    if len(prices) > 5:
        rets = np.abs(np.diff(np.log(np.clip(prices, 1e-6, None))))
        if len(rets) > 60:
            rstd = pd.Series(rets).rolling(60, min_periods=10).std().values
            valid = rstd[~np.isnan(rstd)]
            if len(valid) > 5:
                high_ratio = float((valid > np.percentile(valid, 90)).mean())
    d["l4_vol_high_ratio"] = high_ratio

    entropy = 0.0
    if len(prices) > 10:
        rets_p = np.diff(prices)
        if rets_p.std() > 1e-6:
            bins = np.linspace(-1.5, 1.5, 6) * rets_p.std()
            _, counts = np.unique(np.digitize(rets_p, bins), return_counts=True)
            p = counts / counts.sum()
            entropy = float(-(p * np.log(p + 1e-12)).sum())
    d["l4_rand_symbol_entropy"] = entropy

    # --- 新增 7 维：高级时序分析 ---
    # 使用成交价序列
    ts = prices
    rets_l4 = np.diff(np.log(np.clip(ts, 1e-6, None)))

    # 1) Hurst 指数（聚合方差法）
    d["l4_rand_hurst_gain"] = compute_hurst_aggregate(ts) - 0.5

    # 2) FFT 频谱熵
    d["l4_fft_spectral_entropy"] = compute_fft_spectral_entropy(ts)

    # 3) 游程检验
    d["l4_rand_runs_stat"] = compute_runs_stat(ts)

    # 4) 方差比：VR(4) = Var(r_t(4)) / (4 * Var(r_t))
    # VR>1 正序列相关（趋势），VR<1 负序列相关（均值回归）
    if len(ts) >= 20:
        q_vr = 4
        n_vr = len(ts)
        var_1 = np.var(np.diff(ts))
        # 重叠聚合
        agg = np.array([ts[i + q_vr] - ts[i] for i in range(n_vr - q_vr)])
        var_q = np.var(agg)
        vr = var_q / (q_vr * var_1 + 1e-12) if var_1 > 1e-12 else 1.0
        d["l4_rand_variance_ratio"] = float(vr)
    else:
        d["l4_rand_variance_ratio"] = 1.0

    # 5) 序列相关（lag-1 自相关）
    if len(rets_l4) > 1:
        d["l4_rand_serial_corr"] = float(autocorr(rets_l4, lag=1))
    else:
        d["l4_rand_serial_corr"] = 0.0

    # 6) 趋势线性拟合
    if len(ts) >= 3:
        x = np.arange(len(ts))
        try:
            coeff = np.polyfit(x, ts, 1)
            d["l4_trend_linear"] = float(coeff[0])
            pred = np.polyval(coeff, x)
            ss_res = ((ts - pred) ** 2).sum()
            ss_tot = ((ts - ts.mean()) ** 2).sum()
            d["l4_trend_r_squared"] = float(1.0 - ss_res / (ss_tot + 1e-12))
        except Exception:
            d["l4_trend_linear"] = 0.0
            d["l4_trend_r_squared"] = 0.0
    else:
        d["l4_trend_linear"] = 0.0
        d["l4_trend_r_squared"] = 0.0

    for k in C.L4_P0:
        d.setdefault(k, 0.0)
    return d