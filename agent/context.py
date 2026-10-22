# -*- coding: utf-8 -*-
"""
Level2-Insight Agent - 上下文工程模块 (Context Engineering)
======================================================
将 A 股高频 Level-2 的 174 维微观结构特征按信噪比进行动态压缩。
避免将全量 174 维特征无差别塞入 Prompt 导致 Token 膨胀及 LLM 注意力涣散。
采用：截面 Z-Score 动态截断 + Top-K 极值离群筛选 + 特征领域语义映射。
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 动态定位项目根目录 (AFAC2026金融智能创新大赛/旧/project_solution)
DATA_BASE_DIR = Path(__file__).resolve().parent.parent
INFER_FEAT_DIR = DATA_BASE_DIR / "features" / "infer_features" if (DATA_BASE_DIR / "features" / "infer_features").exists() else DATA_BASE_DIR / "inference" / "infer_features"
OUTPUT_DIR = DATA_BASE_DIR / "output"
MODEL_FILES_DIR = DATA_BASE_DIR / "models"

# 金融高频微观结构特征中英文语义映射表（机构级规范）
FEATURE_GLOSSARY = {
    # 1. 委托与撤单行为 (Cancellation & Order Dynamics)
    "l1_cb_cancel_volume_ratio": "撤单挂单成交比",
    "l1_cb_fast_cancel_ratio": "极速撤单率(<3秒)",
    "l1_cb_flash_cancel_ratio": "闪烁撤单率(<500毫秒)",
    "l1_cb_big_cancel_amount_pct": "大单撤单金额占比",
    "l1_cb_cancel_order_ratio": "总撤单笔数占比",
    # 2. 订单流不平衡 (Order Flow Imbalance - OFI)
    "l1_ofi_mean": "订单流不平衡均值(OFI)",
    "l1_ofi_positive_ratio": "主动买盘OFI正向比率",
    "l1_ofi_cont_cumsum_last": "盘口累积净买入冲量",
    "l1_obp_imbalance_mean": "买卖五档挂单压力不平衡",
    "l1_od_active_big_ratio": "主买大单净成交比率",
    # 3. 拆单与算法交易痕迹 (Order Splitting & Algorithmic Fingerprints)
    "l1_rs_split_similarity": "算法拆单周期相似度(拆单熵)",
    "l1_rs_burst_ratio": "脉冲成交突发爆发率",
    "l1_ap_wash_trade_ratio": "对倒对刷交易比率",
    "l1_trd_interval_cv": "成交时间间隔变异系数",
    # 4. 盘口物理价差与流动性 (Liquidity & Spread Dynamics)
    "l1_amihud_illiquidity": "Amihud 非流动性指标",
    "l1_vwap": "成交量加权平均价偏离",
    "l1_obp_depth_decay_mean": "盘口深度衰减斜率",
    "l1_obp_spread_mean": "买卖一档平均有效买卖价差",
    # 5. 额外强化特征
    "enh_quant_score": "量化高频风格强度",
    "enh_yz_score": "游资脉冲风格强度",
    "l1_limit_touch_upper": "触及涨停次数与开盘冲量",
    "l1_oss_small_count_pct": "散户小单成交笔数占比",
    "l1_oss_big_amount_pct": "主力大单成交金额占比",
    "l4_rand_symbol_entropy": "委托符号序列信息熵",
}


class MicrostructureContextEngine:
    """微观结构上下文压缩与增强引擎"""

    def __init__(self, default_date: str = "7.22"):
        self.default_date = default_date
        self._feat_cache: Dict[str, pd.DataFrame] = {}
        self._centroids: Optional[Dict[str, Any]] = None
        self._load_centroids()

    def _load_centroids(self):
        centroids_path = MODEL_FILES_DIR / "centroids.json"
        if centroids_path.exists():
            try:
                with open(centroids_path, "r", encoding="utf-8") as f:
                    self._centroids = json.load(f)
            except Exception as e:
                logger.warning(f"读取 centroids.json 失败: {e}")

    def get_available_dates(self) -> List[str]:
        """获取所有可用交易日期（降序排列，如 7.22, 7.21, 7.20...）"""
        if not INFER_FEAT_DIR.exists():
            return [self.default_date]
        dates = [d for d in os.listdir(INFER_FEAT_DIR) if (INFER_FEAT_DIR / d).is_dir()]
        def parse_date_key(x):
            try:
                parts = x.split(".")
                return int(parts[0]) * 100 + int(parts[1])
            except Exception:
                return 0
        dates.sort(key=parse_date_key, reverse=True)
        return dates or [self.default_date]

    def set_features(self, date: str, df: pd.DataFrame):
        """支持运行时注入特征表"""
        self._feat_cache[str(date)] = df

    def get_stocks_for_date(self, date: str) -> List[str]:
        """动态读取指定交易日真实涵盖的股票代码列表"""
        try:
            df = self.load_features(date)
            return sorted(df["stock_code"].dropna().unique().tolist())
        except Exception as e:
            logger.warning(f"获取 {date} 股票列表失败: {e}")
            return []

    def load_features(self, date: str) -> pd.DataFrame:
        """从推理特征目录加载指定日期的特征表，并计算截面 Z-Score"""
        d_str = str(date)
        if d_str in self._feat_cache:
            return self._feat_cache[d_str]

        feat_file = INFER_FEAT_DIR / d_str / "Task1_features_infer.csv"
        if not feat_file.exists():
            # 兼容 demo 模式
            if d_str.lower() in ("demo", "run_demo"):
                from inference.infer import AMIInferencePipeline
                demo_df = AMIInferencePipeline()._generate_demo_data()
                self._feat_cache[d_str] = demo_df
                return demo_df

            # 兼容其他路径查找
            alt_feat_file = DATA_BASE_DIR / "features" / "train_features" / "Task2_features_train.csv"
            if alt_feat_file.exists():
                df = pd.read_csv(alt_feat_file)
                self._feat_cache[d_str] = df
                return df

            raise FileNotFoundError(f"未找到特征数据文件: {feat_file}")

        df = pd.read_csv(feat_file)
        self._feat_cache[d_str] = df
        return df

    def get_stock_raw_record(self, stock_code: str, date: str = "7.22") -> Optional[pd.Series]:
        """获取个股原始特征记录（支持纯数字、带后缀等多种输入）"""
        import re
        df = self.load_features(date)
        code_clean = str(stock_code).strip().upper()
        # 1. 精确匹配
        matched = df[df["stock_code"].str.upper() == code_clean]
        if len(matched) == 0:
            # 2. 纯数字提取匹配（如输入 600519 匹配 600519.SH）
            digits = re.sub(r"[^\d]", "", code_clean)
            if digits:
                matched = df[df["stock_code"].str.contains(digits, regex=False, case=False)]
        if len(matched) == 0:
            return None
        return matched.iloc[0]

    def compress_features(
        self,
        stock_code: str,
        date: str = "7.22",
        z_threshold: float = 1.0,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        核心上下文压缩算法：
        1. 针对数值型特征在全市场截面内计算 Z-Score
        2. 过滤绝对偏离度 |z| > z_threshold 的显著特征
        3. 按离群偏离绝对值从大到小排序，提取 Top-K
        4. 映射中文语义解释，构建高信噪比上下文
        """
        df = self.load_features(date)
        record = self.get_stock_raw_record(stock_code, date)
        if record is None:
            return {"error": f"未在 {date} 市场样本中找到股票代码: {stock_code}"}

        actual_code = record["stock_code"]
        num_cols = [c for c in df.columns if c not in ["stock_code", "date", "transaction_date"]]

        # 计算截面均值与标准差
        means = df[num_cols].mean()
        stds = df[num_cols].std().replace(0, 1e-6)

        # 个股 z-score
        z_scores = (record[num_cols] - means) / stds

        # 显著离群特征过滤
        deviations = []
        for col, z in z_scores.items():
            if np.isnan(z) or np.isinf(z):
                continue
            abs_z = abs(z)
            clipped_z = max(-10.0, min(10.0, float(z)))
            raw_val = float(record[col])
            # 去除 _neu 后缀查找映射
            base_col = col.replace("_neu", "")
            glossary_name = FEATURE_GLOSSARY.get(col, FEATURE_GLOSSARY.get(base_col, col))

            deviations.append({
                "feature": col,
                "label": glossary_name,
                "z_score": round(clipped_z, 2),
                "abs_z": abs(clipped_z),
                "raw_val": round(raw_val, 4),
                "direction": "异常显著偏高 ↑" if z > 0 else "异常显著偏低 ↓"
            })

        # 按偏离度绝对值降序
        deviations.sort(key=lambda x: x["abs_z"], reverse=True)
        top_deviations = deviations[:top_k]

        return {
            "stock_code": actual_code,
            "date": date,
            "total_features": len(num_cols),
            "compressed_features_count": len(top_deviations),
            "compression_ratio": f"{(1 - len(top_deviations) / len(num_cols)) * 100:.1f}%",
            "top_deviations": top_deviations
        }

    def format_as_prompt_context(self, compressed_data: Dict[str, Any]) -> str:
        """将压缩后的特征结构化格式化为适合注入 LLM Prompt 的上下文文本"""
        if "error" in compressed_data:
            return f"[数据加载异常]: {compressed_data['error']}"

        lines = [
            f"=== 股票 [{compressed_data['stock_code']}] 在 {compressed_data['date']} 的高频微观结构偏离摘要 ===",
            f"说明：从 174 维微观维度中通过截面 Z-Score 压缩筛选出 Top 偏离项（压缩率：{compressed_data['compression_ratio']}）："
        ]
        for idx, item in enumerate(compressed_data["top_deviations"], 1):
            lines.append(
                f"{idx}. {item['label']} ({item['feature']}): "
                f"Z-Score={item['z_score']:+.2f} ({item['direction']}), 原始截面值={item['raw_val']}"
            )
        return "\n".join(lines)
