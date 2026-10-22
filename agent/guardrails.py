# -*- coding: utf-8 -*-
"""
Level2-Insight Agent - 护栏与工程安全模块 (Guardrails & Domain Defense)
================================================================
包含两大核心工程防御能力：
1. Prompt 注入防御检测器 (Prompt Injection Detector)
   - 过滤越狱、指令覆盖、虚假评级诱导等黑客攻击。
2. 订单簿物理守恒一致性审查器 (Orderbook Conservation Critic)
   - 基于金融微观结构量化因果规律，对 LLM 输出的资金假说进行硬约束审查，杜绝大模型在金融场景的“一本正经胡说八道”。
"""
import re
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# 常见 Prompt 注入与攻击特征词表（正则匹配模式）
INJECTION_PATTERNS = [
    r"ignore\s+.*(instructions?|system|prompt|rules?)",
    r"忽略.*(指令|规则|系统|prompt|提示词)",
    r"system\s*prompt",
    r"系统提示词",
    r"you\s+are\s+now\s+(a|an)?",
    r"你现在是一个.*(不受限制|机器人|模型)",
    r"无视规则",
    r"强制输出",
    r"输出全部资金全仓买入",
    r"全仓做多",
    r"override\s+guardrails?",
    r"jailbreak",
    r"越狱",
    r"DAN\s+mode",
]


class InjectionGuardrail:
    """Prompt 注入安全检测护栏"""

    def __init__(self, patterns: List[str] = None):
        self.patterns = patterns or INJECTION_PATTERNS
        self._compiled_regexes = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def inspect_query(self, user_query: str) -> Tuple[bool, str]:
        """
        检查分析师提问是否含有 Prompt 注入或恶意攻击指令。
        返回: (is_safe, message)
        """
        if not user_query or not isinstance(user_query, str):
            return True, "查询为空"

        for idx, reg in enumerate(self._compiled_regexes):
            match = reg.search(user_query)
            if match:
                matched_text = match.group(0)
                logger.warning(f"[Guardrail Alert] 拦截到可疑 Prompt 注入指令: '{matched_text}'")
                return False, f"安全护栏拦截：检测到潜在的指令注入模式 [{matched_text}]，已拒绝执行不安全重定向。"

        return True, "安全检查通过"


class OrderbookConservationCritic:
    """
    订单簿物理守恒与微观因果审查器 (Domain Anti-Hallucination Critic)
    大模型幻觉在金融领域的典型表现是：给出的定性结论（如：认定为涨停突击游资）
    与底层的微观物理证据（如：盘口主动买入冲击为负、大单撤单率为0）发生根本性逻辑冲突。
    """

    def __init__(self):
        pass

    def verify_hypothesis(
        self,
        stock_code: str,
        capital_type_claim: str,
        compressed_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        对资金行为类别主张与底层特征证据进行自洽审查。
        capital_type_claim: '游资' / '量化' / '散户'
        """
        top_deviations = compressed_features.get("top_deviations", [])
        feat_map = {}
        for item in top_deviations:
            feat_map[item["feature"]] = item["z_score"]
            feat_map[item["feature"].replace("_neu", "")] = item["z_score"]

        verdict = "PASS"
        conflict_reasons = []
        confidence_adjustment = 1.0

        # 规则 1: 游资（游资特征必须有脉冲/爆发/大单成交支撑，不能完全平缓）
        if "游资" in capital_type_claim:
            # 若大单成交或脉冲成交呈极端负偏离，游资假说存疑
            active_big = feat_map.get("l1_od_active_big_ratio", 0.0)
            burst = feat_map.get("l1_rs_burst_ratio", 0.0)
            yz_score = feat_map.get("enh_yz_score", 0.0)
            if (active_big < -1.5 and burst < -1.0) or (yz_score < -1.5 and burst < -1.0):
                verdict = "REJECT"
                conflict_reasons.append("游资主张与盘口冲突：大单净成交比率与突发脉冲均呈现显著负偏离(Z < -1.0)，缺乏游资特征。")
                confidence_adjustment = 0.3

        # 规则 2: 量化高频/做市（必须有算法拆单周期或极速撤单痕迹）
        elif "量化" in capital_type_claim:
            cancel_ratio = feat_map.get("l1_cb_fast_cancel_ratio", 0.0)
            split_sim = feat_map.get("l1_rs_split_similarity", 0.0)
            quant_score = feat_map.get("enh_quant_score", 0.0)
            if cancel_ratio < -2.0 and split_sim < -2.0:
                verdict = "WARNING"
                conflict_reasons.append("量化主张存疑：高频极速撤单率与拆单自相关性均低于均值 2 个标准差，算法特征微弱。")
                confidence_adjustment = 0.6

        # 规则 3: 散户资金（若出现极显著的机构级对倒或主买扫盘，则非纯粹散户）
        elif "散户" in capital_type_claim:
            sweep_ratio = feat_map.get("l1_ap_sweep_ratio", 0.0)
            wash_ratio = feat_map.get("l1_ap_wash_trade_ratio", 0.0)
            if sweep_ratio > 3.0 or wash_ratio > 3.0:
                verdict = "REJECT"
                conflict_reasons.append("散户主张被否决：检测到极度异常的主动扫盘或对倒对刷痕迹(Z > 3.0)，属于典型主力行为。")
                confidence_adjustment = 0.2

        return {
            "stock_code": stock_code,
            "claim": capital_type_claim,
            "verdict": verdict,
            "passed": verdict == "PASS",
            "confidence_adjustment": confidence_adjustment,
            "conflict_reasons": conflict_reasons,
            "detail": "物理守恒与微观因果审查通过" if verdict == "PASS" else "；".join(conflict_reasons)
        }
