# -*- coding: utf-8 -*-
"""
数据加载与稳健预处理引擎 (Data Engine)
======================================
负责跨日期/训练集/推理集特征的加载、格式规范化、0.5%-99.5% 分位数 Winsorizing 截断与 NaN/Inf 稳健清洗。
"""
import os
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import config

logger = logging.getLogger(__name__)

DROP_COLS = {"stock_code", "date", "transaction_date", "capital_type", "capital_intention", "pattern_type"}


def format_stock_code(code) -> str:
    """标准化股票代码格式 (如 600000.SH, 000001.SZ)"""
    s = re.sub(r"\.(SH|SZ|BJ)$", "", str(code)).strip()
    if not s.isdigit():
        return "000001.SZ"
    s = s.zfill(6)
    if s.startswith("688") or s.startswith("689") or s.startswith("6"):
        return s + ".SH"
    return s + ".SZ"


def clean_and_winsorize(df: pd.DataFrame, low_q=0.005, high_q=0.995) -> pd.DataFrame:
    """
    极度稳健的数值清洗：
    1. 将 ±inf 转换为 NaN
    2. 数值列按 [0.5%, 99.5%] 分位数做 Winsorization 截断，防止离群点污染模型
    3. 用 0.0 填充最终 NaN
    """
    df = df.copy()
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    
    # 转换为 float64 并替换 inf
    for c in feat_cols:
        arr = df[c].values.astype(np.float64)
        arr[~np.isfinite(arr)] = np.nan
        
        # 分位数 Winsorize
        if np.count_nonzero(~np.isnan(arr)) > 10:
            q_low = np.nanpercentile(arr, low_q * 100)
            q_high = np.nanpercentile(arr, high_q * 100)
            arr = np.clip(arr, q_low, q_high)
            
        # 兜底填充 0
        arr = np.nan_to_num(arr, nan=0.0)
        df[c] = arr
        
    return df


def load_features(date: str) -> pd.DataFrame:
    """加载指定交易日的推理特征 CSV"""
    # 格式化日期，如 7.22 -> 7.22
    d_str = str(date).replace("20260", "").replace("2026", "")
    if len(d_str) == 4 and d_str.isdigit():
        d_str = f"{int(d_str[:2])}.{d_str[2:]}"
        
    path_cand = [
        config.INFER_FEAT_DIR / d_str / "Task1_features_infer.csv",
        config.INFER_FEAT_DIR / str(date) / "Task1_features_infer.csv",
        config.INFER_FEAT_DIR / d_str / "Task2_features_infer.csv",
    ]
    
    target_path = None
    for p in path_cand:
        if p.exists():
            target_path = p
            break
            
    if target_path is None:
        raise FileNotFoundError(f"未找到日期 {date} 的特征 CSV，尝试路径: {path_cand}")
        
    logger.info(f"加载特征文件: {target_path}")
    df = pd.read_csv(target_path)
    df["stock_code"] = df["stock_code"].apply(format_stock_code)
    df["date"] = str(date)
    return clean_and_winsorize(df)


def load_train_features() -> pd.DataFrame:
    """加载 6 个月完整训练集特征文件"""
    train_files = list(config.TRAIN_FEAT_DIR.glob("*.csv"))
    if not train_files:
        raise FileNotFoundError(f"未在 {config.TRAIN_FEAT_DIR} 找到训练集 CSV")
        
    logger.info(f"加载训练集文件: {train_files[0]}")
    df = pd.read_csv(train_files[0])
    df["stock_code"] = df["stock_code"].apply(format_stock_code)
    df["date"] = df["date"].astype(str)
    return clean_and_winsorize(df)
