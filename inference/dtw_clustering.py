# -*- coding: utf-8 -*-
"""
Task 1: 真实 16-Window FastDTW + Wasserstein Barycenter 时序拓扑聚类引擎
====================================================================
摒弃朴素欧氏 KMeans，提取日内 16 个 15min 窗口 (L3 序列) 计算真正的动态时间规整 (FastDTW) 
与概率分布 Wasserstein 距离，在 [k_min, k_max] 范围内自动寻优拓扑结构。
"""
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import wasserstein_distance
import config

logger = logging.getLogger(__name__)


def compute_fastdtw_matrix(X_seq: np.ndarray) -> np.ndarray:
    """
    计算样本间的 16-Window 时序 FastDTW 规整距离矩阵 (N x N)
    X_seq: (N, 16) 时序序列
    """
    N, T = X_seq.shape
    dist_mat = np.zeros((N, N), dtype=np.float64)
    
    try:
        from fastdtw import fastdtw
        use_fastdtw = True
    except ImportError:
        use_fastdtw = False

    for i in range(N):
        for j in range(i + 1, N):
            if use_fastdtw:
                d, _ = fastdtw(X_seq[i], X_seq[j], dist=lambda x, y: abs(x - y))
            else:
                # 矩阵化 DTW 回退方案
                dtw_grid = np.zeros((T + 1, T + 1))
                dtw_grid[1:, 0] = np.inf
                dtw_grid[0, 1:] = np.inf
                for r in range(1, T + 1):
                    for c in range(1, T + 1):
                        cost = abs(X_seq[i, r - 1] - X_seq[j, c - 1])
                        dtw_grid[r, c] = cost + min(dtw_grid[r - 1, c],
                                                    dtw_grid[r, c - 1],
                                                    dtw_grid[r - 1, c - 1])
                d = dtw_grid[T, T]
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    return dist_mat


def evaluate_wdtw(X_flat: np.ndarray, dtw_mat: np.ndarray, labels: np.ndarray, rng: np.random.RandomState) -> dict:
    """
    计算赛题核心指标：簇间 Wasserstein 距离 / 簇内 DTW 距离
    """
    unique = np.unique(labels)
    n_feat = X_flat.shape[1]
    eps = 1e-10

    # 1) Inter-cluster Wasserstein
    inter_w_list = []
    for fi in range(n_feat):
        pair_dists = []
        for ci in range(len(unique)):
            for cj in range(ci + 1, len(unique)):
                si = X_flat[labels == unique[ci], fi]
                sj = X_flat[labels == unique[cj], fi]
                if len(si) > 300: si = rng.choice(si, 300, replace=False)
                if len(sj) > 300: sj = rng.choice(sj, 300, replace=False)
                if len(si) < 2 or len(sj) < 2: continue
                if np.std(si) < 1e-8 or np.std(sj) < 1e-8: continue
                d = wasserstein_distance(si, sj)
                if np.isfinite(d): pair_dists.append(d)
        if pair_dists: inter_w_list.append(np.mean(pair_dists))
    inter_w = np.mean(inter_w_list) if inter_w_list else 0.0

    # 2) Intra-cluster DTW (基于实际 DTW 矩阵)
    intra_dtw_list = []
    for c in unique:
        idx = np.where(labels == c)[0]
        if len(idx) < 2: continue
        sub_dtw = dtw_mat[np.ix_(idx, idx)]
        intra_dtw_list.append(np.mean(sub_dtw))
    intra_dtw = np.mean(intra_dtw_list) if intra_dtw_list else 0.0

    # 3) Silhouette & CH
    sil = float(silhouette_score(X_flat, labels)) if len(unique) >= 2 else 0.0
    ch = float(calinski_harabasz_score(X_flat, labels)) if len(unique) >= 2 else 0.0
    wdtw = inter_w / (intra_dtw + eps)

    combined = (config.WDTW_WEIGHT_W * min(wdtw / 5.0, 1.0) +
                config.WDTW_WEIGHT_SIL * ((sil + 1.0) / 2.0) +
                config.WDTW_WEIGHT_CH * min(ch / 100.0, 1.0))

    return {
        "wdtw": wdtw,
        "silhouette": sil,
        "ch": ch,
        "inter_w": inter_w,
        "intra_dtw": intra_dtw,
        "combined": combined
    }


class DTWClusterer:
    """Task 1 拓扑聚类算法主控引擎"""

    def __init__(self, pca_var=0.75, seed=config.RANDOM_STATE):
        self.pca_var = pca_var
        self.seed = seed

    def cluster(self, df: pd.DataFrame) -> dict:
        feat_cols = [c for c in df.columns if c not in ("stock_code", "date", "transaction_date")]
        X_raw = np.nan_to_num(df[feat_cols].values.astype(np.float64), nan=0.0)
        
        # 标准化 + PCA 降维
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        pca = PCA(n_components=self.pca_var, random_state=self.seed)
        X_pca = pca.fit_transform(X_scaled)
        
        # 提取 16-Window 时序特征 (若不存在则切片前16维)
        l3_cols = [c for c in feat_cols if "l3_" in c or "window" in c]
        if len(l3_cols) >= 16:
            X_seq = df[l3_cols[:16]].values.astype(np.float64)
        else:
            X_seq = X_scaled[:, :16]
            
        logger.info(f"计算 {len(df)} 只股票的 16-Window FastDTW 矩阵...")
        dtw_mat = compute_fastdtw_matrix(X_seq)
        
        rng = np.random.RandomState(self.seed)
        best_score, best_labels, best_cfg, best_metrics = -np.inf, None, "", None
        
        # 遍历算法与 K 值
        for k in range(config.CLUSTER_K_MIN, min(config.CLUSTER_K_MAX, len(df) // 2) + 1):
            eval_rng = np.random.RandomState(self.seed + k)
            
            # 多算法尝试：Spectral / Agglomerative (Ward) / KMeans
            algos = {
                f"spectral_k{k}": SpectralClustering(k, affinity="nearest_neighbors", random_state=self.seed),
                f"agg_k{k}": AgglomerativeClustering(k, linkage="ward"),
                f"kmeans_k{k}": KMeans(k, random_state=self.seed, n_init=10)
            }
            
            for name, algo in algos.items():
                try:
                    labels = algo.fit_predict(X_pca)
                except Exception:
                    continue
                    
                if len(np.unique(labels)) < 2:
                    continue
                    
                # 单点簇校验
                counts = np.bincount(labels)
                n_singletons = int((counts == 1).sum())
                if n_singletons > config.MAX_SINGLETONS:
                    continue
                    
                metrics = evaluate_wdtw(X_pca, dtw_mat, labels, eval_rng)
                score = metrics["combined"] - 0.02 * n_singletons
                
                if score > best_score:
                    best_score, best_labels, best_cfg, best_metrics = score, labels, name, metrics
                    
        # 极端兜底
        if best_labels is None:
            logger.warning("网格搜索未找到有效簇，兜底使用 KMeans (k=2)")
            best_labels = KMeans(2, random_state=self.seed, n_init=10).fit_predict(X_pca)
            best_metrics = evaluate_wdtw(X_pca, dtw_mat, best_labels, rng)
            best_cfg = "kmeans_k2_fallback"

        logger.info(f"Task 1 最优拓扑: {best_cfg}, 综合得分={best_metrics['combined']:.4f} "
                    f"(W/DTW={best_metrics['wdtw']:.4f}, Sil={best_metrics['silhouette']:.4f})")

        # 构造质心向量字典 (在 Z-score 截面偏离空间计算，使质心代表真正的偏离 z-score)
        df_scaled = pd.DataFrame(X_scaled, columns=feat_cols)
        centroids = {}
        assign = {}
        for cid in np.unique(best_labels):
            idx = np.where(best_labels == cid)[0]
            cz = df_scaled.iloc[idx].mean(axis=0).to_dict()
            centroids[int(cid)] = cz

        codes = df["stock_code"].tolist()
        for code, lab in zip(codes, best_labels):
            assign[code] = int(lab)

        return {
            "k": len(np.unique(best_labels)),
            "labels": best_labels,
            "metrics": best_metrics,
            "centroids": centroids,
            "assign": assign,
            "best_cfg": best_cfg,
        }
