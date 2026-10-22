# -*- coding: utf-8 -*-
"""
AFAC2026 统一推理与 Agent 因果质证管道 (Inference & Cross-Examination Pipeline)
=============================================================================
标准两阶段架构：
1. 数据加载层：载入日内高频 Level-2 微观特征；
2. Stage 1 (固定机器学习初筛)：
   - Task 1: 16-Window FastDTW + 谱聚类获取初步簇号与质心；
   - Task 2: 锚点 + James-Stein + 3折集成树模型 (XGB/LGB/Cat) 获取初步类别概率矩阵与 Margin 置信度；
3. Stage 2 (真正 Agent 因果质证与反思纠错)：
   - 提取 Tier-1 核心微观物理特征包；
   - 物理守恒硬冲突 -> 直接一票否决并修正；
   - 预测置信度存疑 (Margin < 0.15) -> 主动触发 Tier-2 全量 174 维微观特征调取仲裁；
   - 一致性自洽确认；
4. Stage 3 (文本同步与最终修改后 CSV 导出)：
   - 严格依据终审裁定，同步生成自洽的 pattern_explanation 与 capital_intention；
   - 导出修改后的 pattern_reco.csv 与 predict_result.csv；
   - 导出全流程质证审计记录 cross_examination_audit.json 与资金语义拓扑图。
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from inference import loader, dtw_clustering, agentic_trainer, capital_graph
from agent.judge import CrossExaminationJudge

logger = logging.getLogger("InferencePipeline")


def format_transaction_date(date) -> int:
    s = str(date).strip().replace(".", "")
    if len(s) == 8 and s.isdigit():
        return int(s)
    if len(s) == 4 and s.isdigit():
        return int(f"2026{s}")
    if len(s) == 3 and s.isdigit():
        return int(f"20260{s}")
    return 20260722


class AMIInferencePipeline:
    """两阶段推理与 Agent 仲裁全流程管道"""

    def __init__(self):
        self.trace = []

    def run(
        self,
        date: str = "7.22",
        is_demo: bool = False,
        use_llm: bool = True,
        max_workers: int = 2,
        all_stocks: bool = True,
        max_llm_stocks: Optional[int] = None
    ) -> dict:
        out_dir = os.path.join(config.OUTPUT_DIR, f"run_{date}")
        os.makedirs(out_dir, exist_ok=True)
        self.trace.append(f"[Pipeline] 启动日期: {date}")

        # =========================================================================
        # 阶段 1: 数据输入与预处理
        # =========================================================================
        logger.info(f"--- [阶段 1: 数据层] 正在加载交易日 {date} 特征数据 ---")
        if is_demo:
            df = self._generate_demo_data()
        else:
            df = loader.load_features(date)
        self.trace.append(f"[数据层] 成功加载 {len(df)} 只标的微观结构特征")

        # =========================================================================
        # 阶段 2: Task 1 - 固定机器学习聚类初筛 (FastDTW + 谱聚类)
        # =========================================================================
        logger.info("--- [阶段 2: Task 1 初筛] 启动 16-Window FastDTW 拓扑聚类 ---")
        clusterer = dtw_clustering.DTWClusterer()
        cl_res = clusterer.cluster(df)
        self.trace.append(f"[Task 1 FastDTW] 初步聚类完成: K={cl_res['k']}, 得分={cl_res['metrics']['combined']:.4f}")

        # =========================================================================
        # 阶段 3: Task 2 - 固定机器学习集成树模型初筛 (锚点 + James-Stein + 3折集成)
        # =========================================================================
        logger.info("--- [阶段 3: Task 2 初筛] 启动 3折 XGB/LGB/CatBoost 集成模型预测 ---")
        trainer = agentic_trainer.AgenticEMTrainer()
        train_res = trainer.train(df, cluster_assign=cl_res["assign"])
        self.trace.append("[Task 2 集成模型] 初步概率分布与类别生成完成")

        # =========================================================================
        # 阶段 4: 真正的 Agent 因果质证、反思与动态修正 (Cross-Examination Judge)
        # =========================================================================
        logger.info("--- [阶段 4: Agent 质证] 启动 CrossExaminationJudge 执行因果审查与反思仲裁 ---")
        judge = CrossExaminationJudge(
            df=df,
            cl_res=cl_res,
            train_res=train_res,
            date=str(date),
            margin_threshold=0.15
        )
        exam_res = judge.examine_and_reflect(
            use_llm=use_llm,
            max_workers=max_workers,
            all_stocks=all_stocks,
            max_llm_stocks=max_llm_stocks
        )
        pat_df = exam_res["df_task1"]
        pred_df = exam_res["df_task2"]
        audit_records = exam_res["audit_records"]
        stats = exam_res["stats"]

        self.trace.append(f"[Agent 质证] 审定完成: 自洽确认 {stats['confirmed']} 只, 纠偏改判 {stats['overridden']} 只, 全量扩展 {stats['tier2_expanded']} 只")

        # =========================================================================
        # 阶段 5: 交付物同步与报告导出
        # =========================================================================
        logger.info("--- [阶段 5: 结果导出] 导出修改后 CSV、因果质证审计报告与资金图谱 ---")

        # 1. 导出 Task 1 最终修改后表格 pattern_reco.csv
        pat_path = os.path.join(out_dir, "pattern_reco.csv")
        pat_df.to_csv(pat_path, index=False, encoding="utf-8-sig")
        pat_df.to_csv(config.BASE_DIR / "pattern_reco.csv", index=False, encoding="utf-8-sig")
        pat_df.to_csv(config.SUBMISSION_DIR / "pattern_reco.csv", index=False, encoding="utf-8-sig")

        # 2. 导出 Task 2 最终修改后表格 predict_result.csv
        pred_path = os.path.join(out_dir, "predict_result.csv")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        pred_df.to_csv(config.BASE_DIR / "predict_result.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(config.SUBMISSION_DIR / "predict_result.csv", index=False, encoding="utf-8-sig")

        # 3. 导出因果质证与反思详细审计报告 cross_examination_audit.json
        audit_path = os.path.join(out_dir, "cross_examination_audit.json")
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": stats,
                "records": audit_records
            }, f, ensure_ascii=False, indent=2)

        # 4. 资金拓扑语义图
        labels = {cid: f"Cluster_{cid}" for cid in cl_res["centroids"]}
        for _, r in pat_df.iterrows():
            cid_str = str(r["pattern_type"]).replace("cluster_", "")
            if cid_str.isdigit():
                cid_int = int(cid_str)
                # 截取前4个字作为拓扑标签
                labels[cid_int] = f"{cid_int}:{pred_df.loc[pred_df['stock_code']==r['stock_code'], 'capital_type'].values[0]}"

        _, graph_img = capital_graph.build(cl_res["centroids"], labels, out_dir)

        logger.info(f"推理链路与 Agent 终审执行完毕，修改后文件已保存至:\n  - {config.BASE_DIR / 'pattern_reco.csv'}\n  - {config.BASE_DIR / 'predict_result.csv'}\n  - {audit_path}")

        return {
            "out_dir": out_dir,
            "pattern_reco": pat_path,
            "predict_result": pred_path,
            "k": cl_res["k"],
            "silhouette": cl_res["metrics"]["silhouette"],
            "wdtw": cl_res["metrics"]["wdtw"],
            "graph_img": graph_img,
            "stats": stats
        }

    def _generate_demo_data(self, n_stocks: int = 60) -> pd.DataFrame:
        """生成用于端到端离线测试的稳健样本数据"""
        import numpy as np
        rng_num = np.random.default_rng(config.RANDOM_STATE)
        cols = [f"f_{i}" for i in range(20)] + [
            "l1_oss_big_amount_pct_neu", "l1_oss_small_count_pct_neu",
            "l1_ap_active_buy_pct_neu", "l1_ap_max_consecutive_neu",
            "l1_ap_sweep_ratio_neu", "l1_cb_fast_cancel_ratio_neu",
            "l1_cb_flash_cancel_ratio_neu", "l1_rs_split_similarity_neu",
            "l1_ofi_mean_neu", "l1_pi_close_10min_amount_pct_neu",
            "l1_limit_touch_upper_neu", "l4_rand_symbol_entropy_neu",
            "enh_quant_score_neu", "enh_yz_score_neu", "l1_amihud_illiquidity_neu",
            "l1_od_active_big_ratio_neu", "l1_rs_burst_ratio_neu",
            "l1_ap_wash_trade_ratio_neu", "l1_obp_depth_decay_mean_neu",
            "l1_obp_spread_mean_neu"
        ]
        cols += [f"l3_w_{i}" for i in range(16)]

        data = []
        for i in range(n_stocks):
            row = {c: float(rng_num.normal(0, 0.3)) for c in cols}
            row["stock_code"] = f"{600000 + i}.SH"
            row["date"] = "20260722"
            if i % 3 == 0:
                row["enh_yz_score_neu"] = 1.2
                row["l1_limit_touch_upper_neu"] = 1.0
                row["l1_od_active_big_ratio_neu"] = 1.1
                row["l1_rs_burst_ratio_neu"] = 1.3
            elif i % 3 == 1:
                row["enh_quant_score_neu"] = 1.3
                row["l1_cb_fast_cancel_ratio_neu"] = 1.1
                row["l1_rs_split_similarity_neu"] = 1.0
            else:
                row["l1_oss_small_count_pct_neu"] = 1.2
            data.append(row)
        return pd.DataFrame(data)
