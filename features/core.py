"""特征工程核心层：字段映射 + 常量 + 工具函数 + 委托分类器。

模块结构：
  1. 原始 parquet 字段名
  2. 时间窗口常量
  3. 特征列清单（P0/P1/L3/L4/DQ/ENH / NEUTRALIZE_COLS / OUTPUT_COLUMNS）
  4. 业务阈值常量
  5. 工具函数
  6. 委托分类器
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
#  1. 原始 parquet 字段名（与 原始数据/{date}/*.parquet 严格一致）
# ============================================================
Q_COL_CODE = "万得代码"
Q_COL_TIME = "时间"
Q_COL_PREV_CLOSE = "前收盘"
Q_COL_OPEN = "开盘价"
Q_COL_HIGH = "最高价"
Q_COL_LOW = "最低价"
Q_COL_LAST = "成交价"
Q_COL_VOLUME = "成交量"
Q_COL_AMOUNT = "成交额"
Q_COL_WEIGHTED_BID = "加权平均叫买价"
Q_COL_WEIGHTED_ASK = "加权平均叫卖价"
Q_COL_TOTAL_BID = "叫买总量"
Q_COL_TOTAL_ASK = "叫卖总量"

T_COL_CODE = "万得代码"
T_COL_TIME = "时间"
T_COL_PRICE = "成交价格"
T_COL_VOL = "成交数量"
T_COL_BS = "BS标志"

O_COL_CODE = "万得代码"
O_COL_TIME = "时间"
O_COL_PRICE = "委托价格"
O_COL_VOL = "委托数量"
O_COL_TYPE = "委托类型"

RAW_QUOTE_COLS = [
    Q_COL_CODE, Q_COL_TIME, Q_COL_PREV_CLOSE, Q_COL_OPEN, Q_COL_HIGH, Q_COL_LOW,
    Q_COL_LAST, Q_COL_VOLUME, Q_COL_AMOUNT,
    Q_COL_WEIGHTED_BID, Q_COL_WEIGHTED_ASK, Q_COL_TOTAL_BID, Q_COL_TOTAL_ASK,
] + [f"申买价{i}" for i in range(1, 11)] \
  + [f"申卖价{i}" for i in range(1, 11)] \
  + [f"申买量{i}" for i in range(1, 11)] \
  + [f"申卖量{i}" for i in range(1, 11)]

RAW_TRADE_COLS = [T_COL_CODE, T_COL_TIME, T_COL_PRICE, T_COL_VOL, T_COL_BS]
RAW_ORDER_COLS = [O_COL_CODE, O_COL_TIME, O_COL_PRICE, O_COL_VOL, O_COL_TYPE]


# ============================================================
#  2. 时间窗口常量
# ============================================================
TRADING_START_SEC = 9 * 3600 + 30 * 60   # 34200
TRADING_END_SEC   = 15 * 3600             # 54000
AUCTION_START_SEC = 9 * 3600 + 15 * 60    # 33300  集合竞价起
AUCTION_END_SEC   = 9 * 3600 + 25 * 60    # 33900  集合竞价止

L3_WINDOW_LEN_SEC = 15 * 60
L3_WINDOWS = [
    (TRADING_START_SEC + i * L3_WINDOW_LEN_SEC,
     TRADING_START_SEC + (i + 1) * L3_WINDOW_LEN_SEC)
    for i in range(16)
]


# ============================================================
#  3. 特征列清单
# ============================================================
L1_P0 = [
    # ----- 基础量价 -----
    "l1_volume_total_log", "l1_amount_total_log", "l1_txn_count_log",
    "l1_avg_trade_size_log", "l1_vwap", "l1_amihud_illiquidity",
    # ----- RS 拆单与爆发（新增：量化拆单均匀性 + 游资突发性）-----
    "l1_rs_split_similarity", "l1_rs_burst_ratio",
    "l1_rs_volume_skew", "l1_rs_volume_kurt",
    "l1_rs_top5_volume_pct", "l1_rs_top10_volume_pct",
    # ----- OSS 大单分级 -----
    "l1_oss_big_amount_pct", "l1_oss_small_count_pct", "l1_trd_interval_cv",
    # ----- AP 主动成交（新增：方向主导度 + 净方向强度）-----
    "l1_ap_active_buy_pct", "l1_ap_max_consecutive", "l1_ap_sweep_ratio",
    "l1_ap_big_net_inflow", "l1_ap_wash_trade_ratio",
    "l1_ap_dominant_direction", "l1_ap_active_net_direction",
    # ----- CB 撤单（新增：闪电撤单、买卖撤单分化、撤单/委托比）-----
    "l1_cb_cancel_volume_ratio", "l1_cb_fast_cancel_ratio",
    "l1_cb_big_cancel_amount_pct", "l1_cb_partial_cancel_ratio",
    "l1_cb_flash_cancel_ratio", "l1_cb_cancel_order_ratio",
    "l1_cb_cancel_buy_ratio", "l1_cb_cancel_sell_ratio",
    # ----- OD 委托深度 -----
    "l1_od_active_big_ratio", "l1_od_price_deviation_mean",
    # ----- OBP 盘口（新增：买卖盘偏度、中间价）-----
    "l1_obp_spread_mean", "l1_obp_imbalance_mean", "l1_obp_depth_decay_mean",
    "l1_obpi_mean", "l1_bid_shape_mean", "l1_ask_shape_mean",
    "l1_obp_bid_skew_mean", "l1_obp_ask_skew_mean", "l1_obp_mid_price_mean",
    # ----- OFI -----
    "l1_ofi_mean", "l1_ofi_positive_ratio", "l1_ofi_cont_cumsum_last", "l1_ofi_cont_std",
    # ----- DC 深度变化（新增：买盘变化）-----
    "l1_dc_sudden_change_ratio", "l1_dc_ask_change_max",
    "l1_dc_bid_change_max", "l1_dc_bid_change_mean",
    # ----- PD 价格行为（新增：收益分布形状、最大回撤、上涨比、跳跃最大值）-----
    "l1_pd_price_momentum", "l1_pd_return_std", "l1_pd_price_jump_count",
    "l1_pd_return_skew", "l1_pd_return_kurt",
    "l1_pd_max_drawdown", "l1_pd_up_ratio", "l1_pd_price_jump_max",
    # ----- PI 时段分布（新增：尾盘冲刺、峰值集中度、集合竞价占比、VWAP偏离）-----
    "l1_pi_open_30min_amount_pct", "l1_pi_close_10min_amount_pct",
    "l1_pi_close_rush_amount_pct", "l1_pi_peak_amount_ratio",
    "l1_pi_auction_volume_pct", "l1_pi_vwap_deviation",
    "l1_micro_hhi_5min",
    # ----- 微观结构（新增：30min HHI、集中度、价格冲击弹性）-----
    "l1_micro_hhi_30min", "l1_micro_concentration_top10", "l1_micro_price_impact_elasticity",
    # ----- 涨跌停（新增：封板强度、触及跌停、近涨停比）-----
    "l1_limit_touch_upper", "l1_limit_seal_upper_strength",
    "l1_limit_touch_lower", "l1_limit_near_upper_ratio",
    # ----- K线形态（新增：实体比、影线比、十字星、收盘回报）-----
    "l1_kl_amplitude", "l1_kl_gap_ratio",
    "l1_kl_body_ratio", "l1_kl_upper_shadow_ratio", "l1_kl_lower_shadow_ratio",
    "l1_kl_doji_score", "l1_kl_body_direction", "l1_kl_close_return",
    # ----- 加权价差（新增：价差标准差、价差/中间价比）-----
    "l1_wavg_spread_mean", "l1_wavg_spread_std", "l1_wavg_spread_mid_ratio",
    # ----- 扫单 -----
    "l1_swp_quote_sweep_ratio",
    # ----- 量化/游资评分（新增：中性度、方向强度、大单占比）-----
    "l1_yz_burst_score", "l1_quant_uniform_score",
    "l1_quant_neutrality", "l1_yz_direction_strength", "l1_yz_big_pct",
]

L1_P1 = [
    "l1_cb_cancel_imbalance",
    "l1_quant_small_pct",
    "l1_oss_mega_amount_pct",
    "l1_ap_max_consecutive_buy",
    "l1_ap_max_consecutive_sell",
]

L3_P0 = [
    "l2_amt_trend", "l2_buy_range", "l2_time_hhi", "l2_flash_cancel_total",
    "l2_mat_entropy", "l3_burst_max", "l3_jump_max", "l3_cancel_mean",
    "l3_mat_seq_pattern_score", "l3_mat_vol_clustering_acf",
]

L4_P0 = [
    # 原有 5 维
    "l4_ofi_autocorr_lag1", "l4_depth_change_cv",
    "l4_acf_volume_lag1", "l4_vol_high_ratio", "l4_rand_symbol_entropy",
    # 新增 7 维：Hurst、FFT频谱熵、游程检验、方差比、序列相关、趋势
    "l4_rand_hurst_gain", "l4_fft_spectral_entropy",
    "l4_rand_runs_stat", "l4_rand_variance_ratio",
    "l4_rand_serial_corr", "l4_trend_linear", "l4_trend_r_squared",
]

DQ_P0 = ["dq_missing_trade_ratio"]

ENH_FEATS = [
    # 原有 4 维
    "enh_quant_score", "enh_yz_score", "enh_bigflow_x_dir", "enh_burst_x_direction",
    # 新增 4 维：OFI×买卖、OFI×深度变化、撤单×尾盘、波动率×HHI
    "enh_ofi_x_buy", "enh_ofi_x_depth_cv",
    "enh_cancel_x_close", "enh_volatility_x_hhi",
]

P0_FEATS = L1_P0 + L3_P0 + L4_P0 + DQ_P0 + ENH_FEATS
P1_FEATS = L1_P1
FINAL_FEAT_LIST = P0_FEATS + P1_FEATS

# 需做市值中性化的连续型特征（已冻结，与 CSV 输出严格一致）
# 选取规则：P0 中连续值且非已归一化的 _pct/_ratio/_score 等
NEUTRALIZE_COLS = [
    "l1_volume_total_log", "l1_amount_total_log", "l1_txn_count_log",
    "l1_avg_trade_size_log", "l1_vwap", "l1_amihud_illiquidity",
    "l1_rs_burst_ratio", "l1_rs_volume_skew", "l1_rs_volume_kurt",
    "l1_ap_max_consecutive", "l1_ap_big_net_inflow",
    "l1_od_price_deviation_mean",
    "l1_obp_spread_mean", "l1_obp_imbalance_mean", "l1_obp_depth_decay_mean",
    "l1_obpi_mean", "l1_bid_shape_mean", "l1_ask_shape_mean",
    "l1_obp_bid_skew_mean", "l1_obp_ask_skew_mean", "l1_obp_mid_price_mean",
    "l1_ofi_mean", "l1_ofi_cont_cumsum_last", "l1_ofi_cont_std",
    "l1_dc_ask_change_max", "l1_dc_bid_change_max", "l1_dc_bid_change_mean",
    "l1_pd_price_momentum", "l1_pd_return_std", "l1_pd_price_jump_count",
    "l1_pd_return_skew", "l1_pd_return_kurt", "l1_pd_max_drawdown",
    "l1_pd_up_ratio", "l1_pd_price_jump_max",
    "l1_pi_vwap_deviation",
    "l1_micro_hhi_5min", "l1_micro_hhi_30min",
    "l1_micro_concentration_top10", "l1_micro_price_impact_elasticity",
    "l1_kl_amplitude", "l1_kl_gap_ratio",
    "l1_kl_body_ratio", "l1_kl_upper_shadow_ratio", "l1_kl_lower_shadow_ratio",
    "l1_kl_doji_score", "l1_kl_close_return",
    "l1_wavg_spread_mean", "l1_wavg_spread_std",
    "l1_swp_quote_sweep_ratio",
]

NEU_FEAT_LIST = [f"{c}_neu" for c in NEUTRALIZE_COLS]
OUTPUT_COLUMNS = ["stock_code", "date"] + FINAL_FEAT_LIST + NEU_FEAT_LIST


# ============================================================
#  4. 业务阈值常量
# ============================================================
BIG_AMOUNT_THRESHOLD = 100_000
MEGA_AMOUNT_THRESHOLD = 1_000_000
SMALL_AMOUNT_THRESHOLD = 10_000
FAST_CANCEL_INTERVAL_SEC = 3.0
FLASH_CANCEL_INTERVAL_SEC = 2.0
DEPTH_CHANGE_SUDDEN_SIGMA = 3.0
JUMP_PERCENTILE = 0.95
WINSOR_LOW = 0.05
WINSOR_HIGH = 0.95
# 盘口偏度：计算买卖盘各档深度的分布偏斜
N_OBP_LEVELS = 10


# ============================================================
#  5. 工具函数
# ============================================================
def time_to_seconds(t: pd.Series) -> pd.Series:
    """HHMMSSmmm（毫秒） → 距 00:00:00 的秒数。"""
    t = t.astype("int64")
    h = t // 10_000_000
    m = (t // 100_000) % 100
    s = (t // 1000) % 100
    ms = t % 1000
    return (h * 3600 + m * 60 + s + ms / 1000.0).astype("float64")


def safe_div(a, b, fill=0.0):
    """向量安全除法：0 分母 → fill。"""
    mask = np.abs(b) > 1e-12
    safe_b = np.where(mask, b, 1.0)
    return np.where(mask, a / safe_b, fill)


def safe_mean(x):
    """安全均值：空数组或全 NaN → 0.0。"""
    if len(x) == 0:
        return 0.0
    v = np.nanmean(x)
    return 0.0 if np.isnan(v) else float(v)


def safe_std(x, ddof=0):
    """安全标准差：空数组或全 NaN → 0.0。"""
    if len(x) < 2:
        return 0.0
    v = np.nanstd(x, ddof=ddof)
    return 0.0 if np.isnan(v) else float(v)


def safe_skew(x):
    """安全偏度：样本数不足或全 NaN → 0.0。"""
    if len(x) < 3:
        return 0.0
    x = x[~np.isnan(x) & ~np.isinf(x)]
    if len(x) < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s < 1e-12:
        return 0.0
    return float(np.mean((x - m) ** 3) / (s ** 3))


def safe_kurt(x):
    """安全峰度（超额峰度）：样本数不足或全 NaN → 0.0。"""
    if len(x) < 4:
        return 0.0
    x = x[~np.isnan(x) & ~np.isinf(x)]
    if len(x) < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s < 1e-12:
        return 0.0
    return float(np.mean((x - m) ** 4) / (s ** 4) - 3.0)


def winsorize(s: pd.Series, low=0.05, high=0.95) -> pd.Series:
    """截面 Winsorize：[low, high] 分位截尾。"""
    if len(s) == 0:
        return s
    return s.clip(s.quantile(low), s.quantile(high))


def zscore(s: pd.Series) -> pd.Series:
    """截面 Z-Score。"""
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def assign_l3_window(sec: np.ndarray) -> np.ndarray:
    """秒数 → 16 个 15min 窗口编号（-1 = 不在交易时段）。"""
    win = np.full(len(sec), -1, dtype=np.int32)
    for i, (lo, hi) in enumerate(L3_WINDOWS):
        win[(sec >= lo) & (sec < hi)] = i
    return win


def assign_time_bucket(sec: np.ndarray) -> np.ndarray:
    """时段标签：0=集合竞价, 1=开盘30min, 2=盘中, 3=收盘30min, -1=其它。"""
    out = np.full(len(sec), -1, dtype=np.int32)
    out[(sec >= AUCTION_START_SEC) & (sec < AUCTION_END_SEC)] = 0
    out[(sec >= TRADING_START_SEC) & (sec < TRADING_START_SEC + 1800)] = 1
    out[(sec >= TRADING_END_SEC - 1800) & (sec < TRADING_END_SEC)] = 3
    out[(sec >= TRADING_START_SEC + 1800) & (sec < TRADING_END_SEC - 1800)] = 2
    return out


def autocorr(x: np.ndarray, lag: int = 1) -> float:
    """自相关系数（lag 阶）。"""
    if len(x) <= lag:
        return 0.0
    x0 = x[:-lag] - np.nanmean(x)
    x1 = x[lag:] - np.nanmean(x)
    s0 = np.nanstd(x0) * np.sqrt(len(x0))
    s1 = np.nanstd(x1) * np.sqrt(len(x1))
    if s0 < 1e-12 or s1 < 1e-12:
        return 0.0
    return float(np.nansum(x0 * x1) / (s0 * s1))


def theil_sen_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Theil-Sen 鲁棒回归残差（用于市值中性化）。"""
    n = len(y)
    if n < 3 or np.nanstd(x) < 1e-12:
        return y - np.nanmean(y)
    n_pairs = n * (n - 1) // 2
    if n_pairs > 5000:
        rng = np.random.default_rng(42)
        idx_i = rng.integers(0, n, 5000)
        idx_j = rng.integers(0, n, 5000)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)
    slopes = (y[idx_i] - y[idx_j]) / (x[idx_i] - x[idx_j] + 1e-12)
    slope = np.nanmedian(slopes)
    intercept = np.nanmedian(y - slope * x)
    return y - (slope * x + intercept)


def compute_hurst_aggregate(ts: np.ndarray, max_k: int = 50) -> float:
    """聚合方差法 Hurst 指数（快速近似，O(n) 复杂度）。"""
    ts = np.asarray(ts, dtype=float)
    ts = ts[~np.isnan(ts) & ~np.isinf(ts)]
    n = len(ts)
    if n < 20:
        return 0.5
    max_k = min(max_k, n // 4)
    if max_k < 2:
        return 0.5
    variances, ks = [], []
    for k in range(2, max_k + 1):
        n_blocks = n // k
        if n_blocks < 1:
            continue
        truncated = ts[:n_blocks * k].reshape(n_blocks, k)
        block_means = truncated.mean(axis=1)
        if len(block_means) >= 2 and np.std(block_means) > 0:
            variances.append(np.var(block_means, ddof=1))
            ks.append(k)
    if len(ks) < 3:
        return 0.5
    try:
        log_ks = np.log(ks)
        log_vars = np.log(np.abs(variances) + 1e-8)
        slope = np.polyfit(log_ks, log_vars, 1)[0]
        h = np.clip(1.0 - slope / 2.0, 0.01, 0.99)
        return float(h)
    except Exception:
        return 0.5


def compute_fft_spectral_entropy(ts: np.ndarray) -> float:
    """FFT 频谱熵：衡量序列的随机性/可预测性。"""
    ts = np.asarray(ts)
    ts = ts[~np.isnan(ts) & ~np.isinf(ts)]
    if len(ts) < 50:
        return 0.0
    power = np.abs(np.fft.fft(ts))[:len(ts) // 2] ** 2
    if power.sum() == 0:
        return 0.0
    p = power / power.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + 1e-8)))


def compute_runs_stat(ts: np.ndarray) -> float:
    """游程检验统计量：衡量序列的随机性。"""
    ts = np.asarray(ts)
    ts = ts[~np.isnan(ts) & ~np.isinf(ts)]
    if len(ts) < 10:
        return 0.0
    median = np.median(ts)
    signs = (ts > median).astype(int)
    n1, n2 = signs.sum(), len(signs) - signs.sum()
    if n1 == 0 or n2 == 0 or (n1 + n2) < 2:
        return 0.0
    runs = 1 + (np.diff(signs) != 0).sum()
    expected = 2 * n1 * n2 / (n1 + n2) + 1
    var_runs = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    if var_runs <= 0:
        return 0.0
    return float((runs - expected) / np.sqrt(var_runs))


# ============================================================
#  6. 委托分类器
# ============================================================
_ORDER_MAP = {
    "A": 0, "0": 0, "1": 0, "2": 0,
    "D": 1, "3": 1, "d": 1,
    "S": 2, "4": 2, "s": 2,
    "U": 3, "u": 3, "5": 3,
}
_ORDER_OTHER = {"I", "O", "J", "C", "6", "7", "8", "9"}


def classify_order(type_series: pd.Series) -> pd.Series:
    """委托类型 Series（str）→ 0/1/2/3/4 编码。"""
    s = type_series.astype("str").str.strip()
    out = s.map(_ORDER_MAP)
    out = out.where(~s.isin(_ORDER_OTHER), 4)
    return out.fillna(4).astype(np.int8)