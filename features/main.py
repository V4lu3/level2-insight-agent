"""特征工程主流程：IO + 特征提取 + 后处理 + CLI。

流程：
  1. 多日并行读取 parquet → 逐股提取 L1+L3+L4 特征
  2. 合并市值 → 后处理（Z-Score + Winsorize + enh 交互 + 22 维 _neu）
  3. 输出 CSV + 可选 meta 冻结

入口：
  - run()     编程接口
  - main()    CLI 入口
  - postprocess()  独立后处理
"""
from __future__ import annotations
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import core as C
from .features import extract_l1_features, extract_l3_features, extract_l4_features
from .core import winsorize, zscore, theil_sen_residual

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("特征工程")

__version__ = "v2.0"

__all__ = [
    "run", "main", "postprocess", "save_meta", "load_meta",
    "P0_FEATS", "P1_FEATS", "NEUTRALIZE_COLS", "FINAL_FEAT_LIST",
    "OUTPUT_COLUMNS", "ENH_FEATS", "L3_WINDOWS",
    "RAW_QUOTE_COLS", "RAW_TRADE_COLS", "RAW_ORDER_COLS", "NEU_FEAT_LIST",
]

# 包级 API 别名（从 _core 直接引用）
P0_FEATS = C.P0_FEATS
P1_FEATS = C.P1_FEATS
NEUTRALIZE_COLS = C.NEUTRALIZE_COLS
FINAL_FEAT_LIST = C.FINAL_FEAT_LIST
OUTPUT_COLUMNS = C.OUTPUT_COLUMNS
ENH_FEATS = C.ENH_FEATS
L3_WINDOWS = C.L3_WINDOWS
RAW_QUOTE_COLS = C.RAW_QUOTE_COLS
RAW_TRADE_COLS = C.RAW_TRADE_COLS
RAW_ORDER_COLS = C.RAW_ORDER_COLS
NEU_FEAT_LIST = C.NEU_FEAT_LIST


# ============================================================
#  后处理
# ============================================================
def _zscore_per_date(df: pd.DataFrame, cols: list, meta_in: dict | None = None) -> pd.DataFrame:
    """单日截面 Z-Score（train 现场算 / infer 读 meta）。"""
    if meta_in is None:
        for c in cols:
            if c in df.columns:
                df[c] = df.groupby("date")[c].transform(zscore)
    else:
        for c in cols:
            if c not in df.columns or c not in meta_in.get("zscore", {}):
                continue
            stats = meta_in["zscore"][c]
            mu = df["date"].map(stats.get("mean", {})).fillna(0.0).values
            sd = df["date"].map(stats.get("std", {})).fillna(1.0).values
            sd = np.where(sd < 1e-8, 1.0, sd)
            df[c] = (df[c].values - mu) / sd
    return df


def _winsorize_per_date(df: pd.DataFrame, cols: list, low: float, high: float,
                        meta_in: dict | None = None) -> pd.DataFrame:
    """单日截面 Winsorize（train 现场算 / infer 读 meta）。"""
    if meta_in is None:
        for c in cols:
            if c in df.columns:
                df[c] = df.groupby("date")[c].transform(lambda s: winsorize(s, low, high))
    else:
        for c in cols:
            if c not in df.columns or c not in meta_in.get("winsor", {}):
                continue
            bounds = meta_in["winsor"][c]
            df[c] = df[c].clip(lower=bounds["low"], upper=bounds["high"])
    return df


def _neutralize_batch(df: pd.DataFrame, cols: list,
                      log_mcap_col: str = "log_market_cap") -> pd.DataFrame:
    """对每个日期截面，用 log_market_cap 做 Theil-Sen 残差 → {c}_neu。"""
    if log_mcap_col not in df.columns:
        for c in cols:
            df[f"{c}_neu"] = np.nan
        return df
    for c in cols:
        if c not in df.columns:
            df[f"{c}_neu"] = np.nan
            continue
        out = np.full(len(df), np.nan, dtype=np.float64)
        for date, idx in df.groupby("date").groups.items():
            sub = df.loc[idx, [c, log_mcap_col]].dropna()
            if len(sub) < 3:
                continue
            res = theil_sen_residual(sub[c].values, sub[log_mcap_col].values)
            out[df.index.get_indexer(sub.index)] = res
        df[f"{c}_neu"] = out
    return df


def _add_enh_features(df: pd.DataFrame) -> pd.DataFrame:
    """8 维 enh_ 交互特征（4 原有 + 4 新增）。"""
    # ----- 原有 4 维 -----
    if "l1_trd_interval_cv" in df.columns:
        df["enh_quant_score"] = (-df["l1_trd_interval_cv"]
                                 * (1.0 - df.get("l1_ap_active_buy_pct", pd.Series(0.5))
                                     .sub(0.5).abs().mul(2)))
    if all(c in df.columns for c in ["l1_pi_open_30min_amount_pct",
                                     "l1_ap_max_consecutive", "l1_oss_big_amount_pct"]):
        df["enh_yz_score"] = (df["l1_oss_big_amount_pct"]
                              * df["l1_ap_max_consecutive"] / 100.0
                              * df["l1_pi_open_30min_amount_pct"])
    if all(c in df.columns for c in ["l1_ap_big_net_inflow", "l1_ap_active_buy_pct"]):
        df["enh_bigflow_x_dir"] = (df["l1_ap_big_net_inflow"]
                                   * (df["l1_ap_active_buy_pct"] - 0.5).abs())
    if all(c in df.columns for c in ["l1_pi_open_30min_amount_pct", "l1_pd_price_momentum"]):
        df["enh_burst_x_direction"] = (df["l1_pi_open_30min_amount_pct"]
                                       * df["l1_pd_price_momentum"].abs())

    # ----- 新增 4 维：OFI×买卖、OFI×深度变化、撤单×尾盘、波动率×HHI -----
    if all(c in df.columns for c in ["l1_ofi_mean", "l1_ap_active_buy_pct"]):
        df["enh_ofi_x_buy"] = (df["l1_ofi_mean"]
                               * (df["l1_ap_active_buy_pct"] - 0.5).abs())
    if all(c in df.columns for c in ["l1_ofi_cont_std", "l4_depth_change_cv"]):
        df["enh_ofi_x_depth_cv"] = (df["l1_ofi_cont_std"]
                                    * df["l4_depth_change_cv"].clip(upper=10))
    if all(c in df.columns for c in ["l1_cb_flash_cancel_ratio", "l1_pi_close_rush_amount_pct"]):
        df["enh_cancel_x_close"] = (df["l1_cb_flash_cancel_ratio"]
                                    * df["l1_pi_close_rush_amount_pct"])
    if all(c in df.columns for c in ["l1_pd_return_std", "l1_micro_hhi_5min"]):
        df["enh_volatility_x_hhi"] = (df["l1_pd_return_std"]
                                      * df["l1_micro_hhi_5min"])
    return df


def postprocess(df: pd.DataFrame, mode: str = "train",
                do_neutralize: bool = True,
                winsor_low: float = C.WINSOR_LOW,
                winsor_high: float = C.WINSOR_HIGH,
                meta_in: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """后处理主入口：Z-Score → Winsorize → enh 交互 → _neu 中性化 → 强制列序。

    mode: "train" 现场算统计量 + 写 meta_out；"infer" 读 meta_in。
    返回 (df_out, meta_out)。
    """
    df = df.copy()
    base_cols = [c for c in C.FINAL_FEAT_LIST if c in df.columns]

    # 1) Z-Score 截面
    df = _zscore_per_date(df, base_cols, meta_in)
    # 2) Winsorize 截面
    df = _winsorize_per_date(df, base_cols, winsor_low, winsor_high, meta_in)
    # 3) enh_ 交互
    df = _add_enh_features(df)
    # 4) _neu 中性化
    meta_out = {}
    if do_neutralize:
        df = _neutralize_batch(df, C.NEUTRALIZE_COLS)

    # 5) 冻结统计量（仅 train）
    if mode == "train":
        meta_out["zscore"] = {
            c: df.groupby("date")[c].agg(["mean", "std"]).to_dict()
            for c in base_cols if c in df.columns
        }
        meta_out["winsor"] = {
            c: df.groupby("date")[c].agg(
                low=lambda x: x.quantile(winsor_low),
                high=lambda x: x.quantile(winsor_high)
            ).to_dict()
            for c in base_cols if c in df.columns
        }

    # 6) 强制列序
    out_cols = ["stock_code", "date"] + [c for c in C.FINAL_FEAT_LIST if c in df.columns]
    if do_neutralize:
        out_cols += [f"{c}_neu" for c in C.NEUTRALIZE_COLS if f"{c}_neu" in df.columns]
    for c in C.ENH_FEATS:
        if c in df.columns and c not in out_cols:
            out_cols.append(c)
    df = df[out_cols]
    return df, meta_out


def save_meta(meta: dict, path: str) -> None:
    """保存 meta 到 JSON（numpy 类型 → python 原生）。"""
    def _to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(meta), f, ensure_ascii=False, indent=2)


def load_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
#  IO：批量读取（每张 parquet 只 open 1 次）
# ============================================================
def _build_rg_map(pf: "pq.ParquetFile", code_col: str) -> dict:
    """row_group → stock_code 索引（只扫 1 列）。"""
    return {
        pf.read_row_group(i, columns=[code_col]).column(0)[0].as_py(): i
        for i in range(pf.num_row_groups)
    }


def _list_stocks(parquet_dir: Path, date: str) -> list[str]:
    """从行情快照的 row_group 数 → 股票列表。"""
    qpath = parquet_dir / date / "行情快照.parquet"
    if not qpath.exists():
        return []
    return list(_build_rg_map(pq.ParquetFile(str(qpath)), C.Q_COL_CODE).keys())


# ============================================================
#  单日处理
# ============================================================
def _extract_one(code: str, date: str, quote, trade, order) -> dict | None:
    """单只股票特征提取（纯计算，无 IO）。"""
    try:
        if quote is None or len(quote) == 0:
            return None
        d = {"stock_code": code, "date": date}
        d.update(extract_l1_features(quote, trade, order))
        d.update(extract_l3_features(trade, order))
        d.update(extract_l4_features(quote, trade))
        return d
    except Exception as e:
        log.warning(f"{date}/{code} 特征提取失败: {e}")
        return None


def _process_one_date(args) -> list[dict]:
    """单日处理：每张 parquet open 1 次，按 50 只/批流式读取 + 提特征。"""
    parquet_dir, date, stock_list, max_stocks = args
    t0 = time.time()
    codes = stock_list if stock_list else _list_stocks(Path(parquet_dir), date)
    if max_stocks and max_stocks > 0:
        codes = codes[:max_stocks]
    if not codes:
        return []

    BATCH_SIZE = 50
    log.info(f"[{date}] 共 {len(codes)} 只，分批处理（{BATCH_SIZE}/批）")

    base = Path(parquet_dir) / date
    qpath = base / "行情快照.parquet"
    tpath = base / "逐笔成交.parquet"
    opath = base / "逐笔委托.parquet"
    if not qpath.exists():
        log.warning(f"[{date}] 行情快照不存在，跳过")
        return []

    # 每张表 open 1 次 + 建 1 次索引
    pf_q = pq.ParquetFile(str(qpath))
    rg_q = _build_rg_map(pf_q, C.Q_COL_CODE)
    pf_t = pq.ParquetFile(str(tpath)) if tpath.exists() else None
    rg_t = _build_rg_map(pf_t, C.T_COL_CODE) if pf_t else {}
    pf_o = pq.ParquetFile(str(opath)) if opath.exists() else None
    rg_o = _build_rg_map(pf_o, C.O_COL_CODE) if pf_o else {}

    out = []
    n_batches = (len(codes) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        chunk = codes[bi * BATCH_SIZE: (bi + 1) * BATCH_SIZE]
        # 批量读取
        batch_data = {}
        for c in chunk:
            rg = rg_q.get(c)
            if rg is None:
                continue
            q = pf_q.read_row_group(rg, columns=C.RAW_QUOTE_COLS).to_pandas()
            if len(q) == 0:
                continue
            tr = (pf_t.read_row_group(rg_t[c], columns=C.RAW_TRADE_COLS).to_pandas()
                  if pf_t and c in rg_t else None)
            od = (pf_o.read_row_group(rg_o[c], columns=C.RAW_ORDER_COLS).to_pandas()
                  if pf_o and c in rg_o else None)
            batch_data[c] = (q, tr, od)
        # 特征提取
        for c in chunk:
            if c in batch_data:
                r = _extract_one(c, date, *batch_data[c])
                if r is not None:
                    out.append(r)
        del batch_data
        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            log.info(f"[{date}] 批次 {bi+1}/{n_batches}，累计 {len(out)} 只，{time.time()-t0:.1f}s")

    log.info(f"[{date}] 完成 {len(out)}/{len(codes)} 只，{time.time()-t0:.1f}s")
    return out


# ============================================================
#  市值加载
# ============================================================
def _load_market_cap(market_cap_csv, parquet_dir, out_dir) -> pd.DataFrame | None:
    """加载市值 CSV，自动扫描三种来源（合并后按 stock_code,date 去重）：
      1. 显式传入的 market_cap_csv（单文件或目录）
      2. out_dir/market_cap.csv
      3. parquet_dir 同级 '特征工程' 目录下所有 *market_cap.csv
    """
    frames = []
    seen = set()

    def _try_load(p):
        p = Path(p)
        key = str(p.resolve())
        if key in seen or not p.exists():
            return
        if p.is_file() and p.suffix.lower() == ".csv":
            seen.add(key)
            try:
                tmp = pd.read_csv(p)
            except Exception as e:
                log.warning(f"读取市值文件失败 {p}: {e}")
                return
            if {"stock_code", "date", "market_cap"}.issubset(tmp.columns):
                frames.append(tmp)
                log.info(f"  市值文件命中: {p.name} ({len(tmp)} 行)")
        elif p.is_dir():
            for f in sorted(p.glob("*market_cap.csv")):
                _try_load(f)

    if market_cap_csv:
        _try_load(market_cap_csv)
    else:
        _try_load(out_dir / "market_cap.csv")
    # 特征工程目录（训练期合并文件 + 推理期分日期文件）
    fe_dir = Path(parquet_dir).parent / "特征工程"
    if fe_dir.is_dir():
        for f in sorted(fe_dir.glob("*market_cap.csv")):
            _try_load(f)

    if not frames:
        return None
    mcap = pd.concat(frames, ignore_index=True)
    mcap["date"] = mcap["date"].astype(str)
    mcap = mcap.drop_duplicates(subset=["stock_code", "date"], keep="first")
    log.info(f"市值合并完成: {len(mcap)} 行，覆盖 {mcap['date'].min()}~{mcap['date'].max()}")
    return mcap


# ============================================================
#  主入口
# ============================================================
def run(parquet_daily_dir: str,
        output_dir: str,
        dates: list[str] | None = None,
        train_end_date: str | None = None,
        mode: str = "train",
        stock_list: list[str] | None = None,
        n_jobs: int = 6,
        do_neutralize: bool = True,
        feature_meta_in: str | None = None,
        feature_meta_out: str | None = None,
        max_stocks: int = 0,
        market_cap_csv: str | None = None) -> pd.DataFrame:
    """特征工程主入口。"""
    parquet_dir = Path(parquet_daily_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 解析 dates
    if dates is None:
        all_dates = sorted(p.name for p in parquet_dir.iterdir()
                           if p.is_dir() and p.name.isdigit() and len(p.name) == 8)
        if mode == "train" and train_end_date:
            dates = [d for d in all_dates if d <= train_end_date]
        elif mode == "infer" and train_end_date:
            dates = [d for d in all_dates if d > train_end_date]
        else:
            dates = all_dates
    if not dates:
        log.warning("没有可用日期")
        return pd.DataFrame()
    log.info(f"将处理 {len(dates)} 个交易日：{dates[0]} ~ {dates[-1]}")

    # 2) 多日并行
    t0 = time.time()
    args_iter = [(str(parquet_dir), d, stock_list, max_stocks) for d in dates]
    if n_jobs <= 1:
        all_records = [r for a in args_iter for r in _process_one_date(a)]
    else:
        all_records = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_process_one_date, a) for a in args_iter]
            for fut in as_completed(futs):
                all_records += fut.result()

    log.info(f"特征计算完成，{len(all_records)} 行，{time.time()-t0:.1f}s")
    if not all_records:
        return pd.DataFrame()
    df = pd.DataFrame(all_records)
    df["date"] = df["date"].astype(str)

    # 3) 加载市值
    if do_neutralize:
        mcap = _load_market_cap(market_cap_csv, parquet_dir, out_dir)
        if mcap is not None and len(mcap) > 0:
            mcap["date"] = mcap["date"].astype(str)
            df = df.merge(mcap, on=["stock_code", "date"], how="left")
            df["log_market_cap"] = np.log(df["market_cap"].astype(float) + 1.0)
            log.info(f"已合并市值: {int(df['market_cap'].notna().sum())}/{len(df)} 行")
        else:
            log.warning("未找到市值文件，_neu 列将全部 NaN")

    # 4) 加载 meta（推理时）
    meta_in = (load_meta(feature_meta_in)
               if feature_meta_in and Path(feature_meta_in).exists() else None)

    # 5) 后处理
    df_out, meta_out = postprocess(df, mode=mode, do_neutralize=do_neutralize,
                                   meta_in=meta_in)

    # 6) 写文件
    tag = f"{dates[0]}_{dates[-1]}" if len(dates) > 1 else dates[0]
    out_csv = out_dir / f"features_{tag}_{mode}.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info(f"已写出: {out_csv}, shape={df_out.shape}")

    # 7) 写 meta
    if mode == "train" and feature_meta_out and meta_out:
        save_meta(meta_out, feature_meta_out)
        log.info(f"已冻结 meta: {feature_meta_out}")

    return df_out


# ============================================================
#  CLI
# ============================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="特征工程 v2：70 维特征，单日/多日兼容，字段 100% 对齐。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--parquet_daily_dir", required=True, help="原始数据根目录")
    p.add_argument("--output_dir", required=True, help="输出根目录")
    p.add_argument("--dates", nargs="+", default=None,
                   help="显式指定日期列表（YYYYMMDD）")
    p.add_argument("--mode", choices=["train", "infer", "full"], default="train")
    p.add_argument("--train_end_date", default=None, help="训练/推理切分日")
    p.add_argument("--stock_list", default=None, help="股票池文件路径")
    p.add_argument("--n_jobs", type=int, default=6, help="跨日并行 worker 数")
    p.add_argument("--do_neutralize", action="store_true", default=True)
    p.add_argument("--no_neutralize", dest="do_neutralize", action="store_false")
    p.add_argument("--feature_meta_in", default=None, help="推理期 meta JSON")
    p.add_argument("--feature_meta_out", default=None, help="训练期写出 meta JSON")
    p.add_argument("--max_stocks", type=int, default=0, help="调试：每日期最多 N 只")
    p.add_argument("--market_cap_csv", default=None, help="市值 CSV 路径")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    stock_list = None
    if args.stock_list:
        sl_path = Path(args.stock_list)
        if sl_path.suffix in (".xlsx", ".xls"):
            stocks = pd.read_excel(sl_path)
        else:
            stocks = pd.read_csv(sl_path, header=None)
        stock_list = stocks[stocks.columns[0]].astype(str).str.zfill(6).tolist()
        log.info(f"从 {args.stock_list} 读取 {len(stock_list)} 只股票")
    run(parquet_daily_dir=args.parquet_daily_dir,
        output_dir=args.output_dir,
        dates=args.dates,
        train_end_date=args.train_end_date,
        mode=args.mode,
        stock_list=stock_list,
        n_jobs=args.n_jobs,
        do_neutralize=args.do_neutralize,
        feature_meta_in=args.feature_meta_in,
        feature_meta_out=args.feature_meta_out,
        max_stocks=args.max_stocks,
        market_cap_csv=args.market_cap_csv)


if __name__ == "__main__":
    main()
