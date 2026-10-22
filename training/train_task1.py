# -*- coding: utf-8 -*-
"""
模型训练 - Task 1: 16-Window FastDTW + Wasserstein 拓扑聚类训练
============================================================
从 特征工程/训练特征 读取多日特征数据集，提取 16 窗口 L3 序列，
计算 FastDTW 规整矩阵与 Wasserstein 距离，寻优最优拓扑 K 值与质心保存至 模型文件/centroids.json。
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from inference.loader import load_features, clean_and_winsorize
from inference.dtw_clustering import DTWClusterer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train_Task1")


def train_task1_clustering(date: str = "7.22"):
    logger.info(f"=== 开始 Task 1 FastDTW 拓扑聚类训练 (日期: {date}) ===")
    df = load_features(date)
    clusterer = DTWClusterer()
    cl_res = clusterer.cluster(df)

    # 保存质心与拓扑结果至 模型文件/
    os.makedirs(config.MODEL_FILES_DIR, exist_ok=True)
    out_json = config.MODEL_FILES_DIR / "centroids.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "k": cl_res["k"],
            "best_cfg": cl_res["best_cfg"],
            "metrics": cl_res["metrics"],
            "centroids": cl_res["centroids"]
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Task 1 聚类训练完成! K={cl_res['k']}, 得分={cl_res['metrics']['combined']:.4f}")
    logger.info(f"模型质心参数已保存至: {out_json}")
    return cl_res


if __name__ == "__main__":
    train_task1_clustering()
