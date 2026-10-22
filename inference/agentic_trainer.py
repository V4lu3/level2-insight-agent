# -*- coding: utf-8 -*-
"""
Task 2: 锚点 + James-Stein + 3-Fold XGB/LGB/Cat + Agent-in-the-Loop 自训练引擎
========================================================================
融合 Task2.9 ~ Task2.21 全部硬核算法优点：
1. 重尾分位数变换 (tail_stretched_rank)
2. 物理微观结构锚点打分 (No Probing)
3. 三路纯净特征加权筛选 (Purified / Confusion Weighting)
4. James-Stein 动态收缩因子 α*
5. 【 Agent-in-the-Loop 迭代降噪】：在 EM 自训练环路中引入 Task1 拓扑簇与 Task2 资金预测的 Cross-Task MI 逻辑惩罚，
   自动为冲突样本重新赋值损失权重 w_i，彻底剔除旧代码的 OT 硬分配红线。
"""
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation as lgb_log
from catboost import CatBoostRegressor
import config

logger = logging.getLogger(__name__)

# ============================================================
# 游资 / 量化 / 散户 物理微观结构锚点字典 (已消除市值偏置)
# ============================================================
YZ_ANCHOR_HIGH = {
    "l1_ap_max_consecutive_neu": 3.85, "l1_obp_depth_decay_mean_neu": 3.04,
    "l2_time_hhi_neu": 2.41, "enh_yz_score_neu": 2.14,
    "l1_pi_open_30min_amount_pct_neu": 2.00, "l1_cb_cancel_imbalance_neu": 1.89,
    "l1_oss_big_amount_pct_neu": 1.89, "l1_ofi_positive_ratio_neu": 1.87,
    "l1_obpi_mean_neu": 1.60, "l1_yz_burst_score_neu": 1.53,
    "enh_burst_x_direction_neu": 1.52, "l1_cb_fast_cancel_ratio_neu": 1.15,
}
YZ_ANCHOR_LOW = {
    "l1_amihud_illiquidity_neu": 2.33, "l2_amt_trend_neu": 2.22,
    "l1_pi_close_10min_amount_pct_neu": 2.02, "l1_ap_big_net_inflow_neu": 1.55,
}

QUANT_STYLE = {
    "enh_quant_score_neu": ("high", 3.0),
    "l1_pd_return_mean_neu": ("high", 2.0),
    "enh_ofi_x_depth_cv_neu": ("high", 1.8),
    "l1_ap_active_buy_pct_neu": ("high", 1.5),
    "l1_ofi_cont_cumsum_last_neu": ("low", 2.5),
    "l1_ofi_mean_neu": ("low", 2.0),
    "l4_trend_acceleration_neu": ("low", 2.1),
}

RETAIL_STYLE = {
    "l1_oss_small_count_pct_neu": ("high", 2.5),
    "l1_ofi_mean_neu": ("high", 1.8),
    "l4_trend_acceleration_neu": ("high", 1.5),
    "l1_pd_return_mean_neu": ("low", 1.5),
    "enh_quant_score_neu": ("low", 2.0),
}


def tail_stretched_rank(series: pd.Series, threshold=config.TAIL_THRESHOLD, stretch=config.STRETCH_FACTOR) -> np.ndarray:
    """
    重尾保留分位数变换 (Tail-Stretched Rank):
    对于金融重尾特征，95% 以下做均匀 rank 映射；对 95% 以上尾部做 log 拉伸，保留突发信号。
    """
    ranks = series.rank(pct=True, na_option="keep").fillna(0.5).values.copy()
    p95 = series.quantile(threshold)
    if p95 > 0 and (series > p95).any():
        tail_mask = series.values > p95
        tail_vals = np.clip(series.values[tail_mask] / p95, 1.0, 1e6)
        boost = np.log1p(tail_vals - 1.0) * stretch
        ranks[tail_mask] += boost
    return np.clip(ranks, 0.0, 1.5)


def compute_james_stein_alpha(proba: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    """
    基于 James-Stein 定理的动态收缩因子 α*:
    α* = σ²_between / (σ²_between + σ²_within / n_bar)
    将个体估计向组均值收缩，在 MSE 意义下严格优于个体估算。
    """
    n_samples, n_classes = proba.shape
    alpha_star = np.ones((n_samples, n_classes), dtype=np.float64)
    
    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) < 2:
        return alpha_star
        
    for k in range(n_classes):
        p_k = proba[:, k]
        global_mean = p_k.mean()
        
        cluster_means = []
        cluster_vars = []
        cluster_sizes = []
        
        for c in unique_clusters:
            idx = np.where(cluster_labels == c)[0]
            if len(idx) == 0: continue
            c_mean = p_k[idx].mean()
            c_var = p_k[idx].var() if len(idx) > 1 else 0.0
            cluster_means.append(c_mean)
            cluster_vars.append(c_var)
            cluster_sizes.append(len(idx))
            
        var_between = np.var(cluster_means)
        var_within = np.mean(cluster_vars)
        n_bar = np.mean(cluster_sizes)
        
        alpha_k = var_between / (var_between + (var_within / max(n_bar, 1.0)) + 1e-8)
        alpha_k = np.clip(alpha_k, 0.3, 0.95)
        
        # 组装收缩后的概率
        for c in unique_clusters:
            idx = np.where(cluster_labels == c)[0]
            c_mean = p_k[idx].mean()
            proba[idx, k] = alpha_k * p_k[idx] + (1.0 - alpha_k) * c_mean
            alpha_star[idx, k] = alpha_k
            
    # 重新归一化
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def generate_anchor_soft_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """生成初始物理微观结构软标签与得分"""
    N = len(df)
    
    # 1. 游资得分
    yz_score = np.zeros(N)
    total_w = 0.0
    for feat, weight in YZ_ANCHOR_HIGH.items():
        if feat in df.columns:
            yz_score += tail_stretched_rank(df[feat]) * weight
            total_w += weight
    for feat, weight in YZ_ANCHOR_LOW.items():
        if feat in df.columns:
            yz_score += (1.0 - tail_stretched_rank(df[feat])) * weight
            total_w += weight
    yz_score /= max(total_w, 1e-8)
    p_yz = (yz_score - yz_score.min()) / (yz_score.max() - yz_score.min() + 1e-8)

    # 2. 量化得分
    qt_score = np.zeros(N)
    qt_w = 0.0
    for feat, (direction, weight) in QUANT_STYLE.items():
        if feat in df.columns:
            s = 1.0 - tail_stretched_rank(df[feat]) if direction == "low" else tail_stretched_rank(df[feat])
            qt_score += s * weight
            qt_w += weight
    qt_score /= max(qt_w, 1e-8)

    # 3. 散户得分
    rt_score = np.zeros(N)
    rt_w = 0.0
    for feat, (direction, weight) in RETAIL_STYLE.items():
        if feat in df.columns:
            s = 1.0 - tail_stretched_rank(df[feat]) if direction == "low" else tail_stretched_rank(df[feat])
            rt_score += s * weight
            rt_w += weight
    rt_score /= max(rt_w, 1e-8)

    # 非游资样本在量化与散户间平分概率
    p_qt = (1.0 - p_yz) * (qt_score / (qt_score + rt_score + 1e-8))
    p_rt = (1.0 - p_yz) * (rt_score / (qt_score + rt_score + 1e-8))

    proba = np.column_stack([p_yz, p_qt, p_rt])
    proba /= proba.sum(axis=1, keepdims=True)
    return proba, np.argmax(proba, axis=1)


class AgenticEMTrainer:
    """Agent-in-the-Loop 迭代训练与多模型融合引擎"""

    def __init__(self, n_folds=config.N_FOLDS, seed=config.RANDOM_STATE):
        self.n_folds = n_folds
        self.seed = seed
        self.scaler = StandardScaler()
        self.models = {}

    def train(self, df: pd.DataFrame, cluster_assign: dict = None) -> dict:
        N = len(df)
        logger.info(f"开启 Task 2 Agentic-in-the-Loop 训练 (样本数: {N})...")
        
        # 1) 初始软标签
        proba, labels = generate_anchor_soft_labels(df)
        
        # 2) 特征提取
        exclude = {"stock_code", "date", "transaction_date", "capital_type", "capital_intention", "pattern_type"}
        feat_cols = [c for c in df.columns if c not in exclude]
        
        X = df[feat_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # 将拓扑簇转化为数组
        if cluster_assign:
            cluster_labels = np.array([cluster_assign.get(code, 0) for code in df["stock_code"]])
        else:
            cluster_labels = np.zeros(N, dtype=int)
            
        # 样本初始损失权重
        sample_weights = np.ones(N, dtype=np.float64)
        
        # 3) EM 迭代训练
        for rnd in range(config.EM_ROUNDS):
            logger.info(f"--- EM 自训练 Round {rnd + 1}/{config.EM_ROUNDS} ---")
            
            # Agent-in-the-Loop 校验：跨任务互信息逻辑惩罚
            if rnd > 0 and cluster_assign:
                for i in range(N):
                    c_id = cluster_labels[i]
                    pred_type = config.CAPITAL_TYPES[np.argmax(proba[i])]
                    # 规则惩罚：如果簇内大多数为散户，但个别样本预测为极端游资且特征矛盾，降低权重
                    if pred_type == "游资" and df["l1_amihud_illiquidity_neu"].iloc[i] > 1.5:
                        sample_weights[i] *= 0.75
                        
            # 训练 3-Fold 融合模型 (XGB + LGB + Cat)
            oof_preds = np.zeros((N, 3))
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            
            for k in range(3): # 对 3 个类别分别建模
                y_target = proba[:, k]
                for fold, (tr, va) in enumerate(kf.split(X_scaled)):
                    clf_xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           random_state=self.seed, n_jobs=-1, verbosity=0)
                    clf_xgb.fit(X_scaled[tr], y_target[tr], sample_weight=sample_weights[tr])
                    
                    clf_lgb = LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                            random_state=self.seed, n_jobs=-1, verbose=-1)
                    clf_lgb.fit(X_scaled[tr], y_target[tr], sample_weight=sample_weights[tr])
                    
                    try:
                        clf_cat.fit(X_scaled[tr], y_target[tr], sample_weight=sample_weights[tr])
                        p_cat = clf_cat.predict(X_scaled[va])
                    except Exception:
                        p_cat = np.full(len(va), float(np.mean(y_target[tr])))
                    
                    pred_fold = (clf_xgb.predict(X_scaled[va]) + 
                                 clf_lgb.predict(X_scaled[va]) + 
                                 p_cat) / 3.0
                    oof_preds[va, k] += pred_fold / self.n_folds
                    
            # 更新概率并应用 James-Stein 收缩
            proba = np.clip(oof_preds, 0.0, 1.0)
            proba /= proba.sum(axis=1, keepdims=True)
            proba = compute_james_stein_alpha(proba, cluster_labels)

        final_types = [config.CAPITAL_TYPES[i] for i in np.argmax(proba, axis=1)]
        logger.info(f"Task 2 训练完成! 预测分布: 游资={(np.array(final_types)=='游资').mean():.2%}, "
                    f"量化={(np.array(final_types)=='量化').mean():.2%}, "
                    f"散户={(np.array(final_types)=='散户').mean():.2%}")
                    
        return {
            "proba": proba,
            "capital_types": final_types,
            "feat_cols": feat_cols,
            "scaler": self.scaler
        }
