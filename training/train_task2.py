# -*- coding: utf-8 -*-
"""
模型训练 - Task 2: 锚点 + James-Stein + 3-Fold Ensemble Agentic 自训练
======================================================================
从 特征工程/训练特征 加载历史数据，执行重尾分位数变换、物理锚点概率初始化、
3 折 XGBoost+LightGBM+CatBoost 融合自训练，并将模型 Scaler 保存至 模型文件/task2_scaler.pkl。
"""
import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from inference.loader import load_features
from inference.agentic_trainer import AgenticEMTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train_Task2")


def train_task2_models(date: str = "7.22", cluster_assign: dict = None):
    logger.info(f"=== 开始 Task 2 Agentic-in-the-Loop 模型自训练 (日期: {date}) ===")
    df = load_features(date)
    trainer = AgenticEMTrainer()
    train_res = trainer.train(df, cluster_assign=cluster_assign)

    os.makedirs(config.MODEL_FILES_DIR, exist_ok=True)
    scaler_path = config.MODEL_FILES_DIR / "task2_scaler.pkl"
    joblib.dump(train_res["scaler"], scaler_path)

    logger.info(f"Task 2 模型训练完成! 比例分布: {pd.Series(train_res['capital_types']).value_counts(normalize=True).to_dict()}")
    logger.info(f"模型文件 Scaler 已保存至: {scaler_path}")
    return train_res


if __name__ == "__main__":
    train_task2_models()
