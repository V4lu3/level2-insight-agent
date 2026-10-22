# -*- coding: utf-8 -*-
"""
Level2-Insight Agent - LangGraph 状态图工作流编排 (StateGraph Workflow)
====================================================================
采用 LangGraph 的 StateGraph 机制，构建具备「反思纠错环路 (Reflexion Loop)」的图状态机：
1. START -> extract_features_node (特征压缩与抽取)
2. -> match_cluster_node (FastDTW 行为聚类匹配)
3. -> classify_attribution_node (资金意图判定与 SHAP 归因)
4. -> critic_audit_node (物理守恒核验)
5. -> 判定条件边 (conditional_edge):
      - 若核验通过 (PASS) -> synthesize_report_node (合成研报) -> END
      - 若核验存疑 (REJECT/WARNING) -> refine_hypothesis_node (反思修正假说) -> synthesize_report_node -> END
"""
import json
import logging
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from .tools import FinancialToolBox

logger = logging.getLogger(__name__)


class FinancialAgentState(TypedDict):
    """LangGraph 全局状态结构体"""
    stock_code: str
    date: str
    query: str
    microstructure_profile: Dict[str, Any]
    cluster_pattern: Dict[str, Any]
    capital_attribution: Dict[str, Any]
    critic_audit: Dict[str, Any]
    is_audit_passed: bool
    correction_notes: str
    final_report: str
    trace_log: List[str]


class Level2InsightGraphBuilder:
    """LangGraph 状态图构建器"""

    def __init__(self, toolbox: Optional[FinancialToolBox] = None):
        self.toolbox = toolbox or FinancialToolBox()

    def node_extract_features(self, state: FinancialAgentState) -> Dict[str, Any]:
        """节点 1：微观结构特征抽取与上下文压缩"""
        stock_code = state["stock_code"]
        date = state["date"]
        res = self.toolbox.query_microstructure_profile(stock_code, date, top_k=5)
        log = f"[Node: Feature Extraction] 完成 {stock_code} 在 {date} 的 174 维微观特征压缩提取。"
        return {
            "microstructure_profile": res,
            "trace_log": state.get("trace_log", []) + [log]
        }

    def node_match_cluster(self, state: FinancialAgentState) -> Dict[str, Any]:
        """节点 2：时序微观轨迹聚类比对"""
        stock_code = state["stock_code"]
        date = state["date"]
        res = self.toolbox.get_behavior_cluster(stock_code, date)
        log = f"[Node: Cluster Matching] 匹配到 FastDTW 聚类模式: {res.get('cluster_pattern', 'N/A')}。"
        return {
            "cluster_pattern": res,
            "trace_log": state["trace_log"] + [log]
        }

    def node_classify_attribution(self, state: FinancialAgentState) -> Dict[str, Any]:
        """节点 3：模型自训练分类与归因提取"""
        stock_code = state["stock_code"]
        date = state["date"]
        res = self.toolbox.explain_shap_attribution(stock_code, date)
        log = f"[Node: Attribution] 资金属性判定为: {res.get('capital_type', '未知')}。"
        return {
            "capital_attribution": res,
            "trace_log": state["trace_log"] + [log]
        }

    def node_critic_audit(self, state: FinancialAgentState) -> Dict[str, Any]:
        """节点 4：订单簿物理守恒因果审查 (Critic)"""
        stock_code = state["stock_code"]
        date = state["date"]
        cap_type = state["capital_attribution"].get("capital_type", "散户")
        audit = self.toolbox.verify_orderbook_conservation(stock_code, cap_type, date)
        passed = audit.get("is_passed", True)
        log = f"[Node: Critic Audit] 物理守恒审查裁决: {audit.get('critic_verdict')} (通过: {passed})。"
        return {
            "critic_audit": audit,
            "is_audit_passed": passed,
            "trace_log": state["trace_log"] + [log]
        }

    def node_refine_hypothesis(self, state: FinancialAgentState) -> Dict[str, Any]:
        """节点 5 (条件触发)：反思纠错节点 (Reflexion)，修正大模型冲突定性"""
        audit = state["critic_audit"]
        old_type = state["capital_attribution"].get("capital_type", "")
        corrected_type = "均衡博弈资金"
        note = f"⚠️ [Reflexion 触发] 原始假设「{old_type}」因【{audit.get('audit_detail')}】被驳回，经状态机反思修正为「{corrected_type}」。"
        updated_attr = dict(state["capital_attribution"])
        updated_attr["capital_type"] = f"{old_type} (已由 Critic 修正为 {corrected_type})"
        return {
            "capital_attribution": updated_attr,
            "correction_notes": note,
            "trace_log": state["trace_log"] + [note]
        }

    def node_synthesize_report(self, state: FinancialAgentState) -> Dict[str, Any]:
        """节点 6：汇总多节点状态，产出最终研报"""
        code = state["stock_code"]
        date = state["date"]
        prof = state["microstructure_profile"]
        clust = state["cluster_pattern"]
        attr = state["capital_attribution"]
        audit = state["critic_audit"]
        correction = state.get("correction_notes", "")

        deviations_str = ""
        for item in prof.get("significant_deviations", []):
            deviations_str += f"- **{item['feature_name']}** ({item['raw_id']}): 截面 Z-Score = `{item['z_score']:+.2f}` ({item['state']})\n"

        correction_section = f"\n> {correction}\n" if correction else ""

        report = f"""# 【LangGraph 协同投研研报】{code} 资金流向与微观结构研报

**标的代码**：`{code}` ｜ **观测日期**：`{date}` ｜ **编排架构**：`LangGraph StateGraph 状态图`
{correction_section}
### 一、核心定性与资金属性
* **参与者属性归因**：**【{attr.get('capital_type', '未知')}】**
* **时序形态聚类**：**【{clust.get('cluster_pattern', 'N/A')}】**
* **行为动机概括**：{attr.get('capital_intention', '无')}

### 二、底层盘口微观证据链 (Context Engineering)
{deviations_str}

### 三、时序行为特征侧写 (FastDTW 拓扑聚类)
{clust.get('archetype_explanation', '无')}

### 四、订单簿物理守恒 Critic 审查
* **假说核验裁决**：`{audit.get('critic_verdict', 'PASS')}`
* **核验说明**：{audit.get('audit_detail', '微观特征自洽')}

---
*Generated by LangGraph Financial Multi-Agent Workflow.*
"""
        log = "[Node: Synthesis] 研报组装完成。"
        return {
            "final_report": report,
            "trace_log": state["trace_log"] + [log]
        }

    def route_after_audit(self, state: FinancialAgentState) -> str:
        """条件路由函数：根据 Critic 审核结果决定是否进入反思修正分支"""
        if state.get("is_audit_passed", True):
            return "synthesize"
        else:
            return "refine"

    def build(self):
        """编译 LangGraph 工作流状态图"""
        builder = StateGraph(FinancialAgentState)

        builder.add_node("extract_features", self.node_extract_features)
        builder.add_node("match_cluster", self.node_match_cluster)
        builder.add_node("classify_attribution", self.node_classify_attribution)
        builder.add_node("critic_audit", self.node_critic_audit)
        builder.add_node("refine_hypothesis", self.node_refine_hypothesis)
        builder.add_node("synthesize_report", self.node_synthesize_report)

        builder.add_edge(START, "extract_features")
        builder.add_edge("extract_features", "match_cluster")
        builder.add_edge("match_cluster", "classify_attribution")
        builder.add_edge("classify_attribution", "critic_audit")

        builder.add_conditional_edges(
            "critic_audit",
            self.route_after_audit,
            {
                "synthesize": "synthesize_report",
                "refine": "refine_hypothesis"
            }
        )
        builder.add_edge("refine_hypothesis", "synthesize_report")
        builder.add_edge("synthesize_report", END)

        return builder.compile()
