# -*- coding: utf-8 -*-
"""
Level2-Insight Agent - ReAct 决策引擎 (Reasoning & Acting Engine)
================================================================
纯 Python 源码实现从第一性原理构建的 ReAct 决策循环：
1. 任务意图解析与规划 (Thought)
2. 标准化工具调用与执行 (Action & Observation)
3. 循环迭代直到达成目标 (Iteration Loop, 带 max_steps 熔断)
4. 双后端容灾：LLM API 异常/无 Key 时，自动降级至确定性规则状态机 (Deterministic Fallback)
5. 全流程 Trace 结构化记录，方便前端可视化与审计回放
"""
import os
import re
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .tools import TOOL_SCHEMAS, execute_tool, FinancialToolBox
from .guardrails import InjectionGuardrail, OrderbookConservationCritic

logger = logging.getLogger(__name__)

# 系统人设提示词（机构级金融分析师）
SYSTEM_PROMPT = """你是一个服务于顶级量化对冲基金与券商投研团队的「Level-2 微观结构与资金行为分析 Agent」。
你的职责是：接收专业分析师的自然语言指令，通过自主规划并调用工具，完成对 A 股个股的高频盘口微观结构审查、行为聚类比对、资金属性归因及抗幻觉核验，最终产出严密、可信、具备因果证据链的机构级研究报告。

【可用工具体系】
1. query_microstructure_profile: 查询 174 维微观特征中偏离度最显著的 Top 离群指标（OFI、撤单率、盘口压力等）。
2. get_behavior_cluster: 查询个股在 FastDTW 16 窗时序聚类中的行为模式侧写（如：高频撤单博弈型、打板封板型）。
3. explain_shap_attribution: 查询机器学习分类器（集成模型）输出的资金类别（游资/量化/散户）及特征归因支持。
4. verify_orderbook_conservation: 抗幻觉质检工具，输入定性假说，核查底层盘口微观结构是否存在物理矛盾。

【分析与作业规范】
- 每一步推理前，先明确输出你的思考（Thought）；
- 必须基于工具返回的客观数值（Z-Score、撤单比率、OFI 等）给出结论，严禁脱离证据凭空臆测；
- 得出初步资金属性定性后，建议调用 verify_orderbook_conservation 进行自洽性审查；
- 最终输出结构化的专业分析报告（包含：核心定性结论、微观物理证据链、行为意图侧写与博弈风险提示）。
"""


@dataclass
class AgentStep:
    step_num: int
    thought: str
    action: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class AgentTrace:
    query: str
    stock_code: str
    date: str
    steps: List[AgentStep] = field(default_factory=list)
    final_report: str = ""
    execution_mode: str = "LLM_LIVE"  # LLM_LIVE 或 DETERMINISTIC_FALLBACK
    is_safe: bool = True
    guardrail_note: str = ""
    total_time_ms: float = 0.0


class DeterministicFallbackPlanner:
    """
    确定性规则后备计划器 (Deterministic Fallback Engine)
    当无大模型 API Key 或网络通信异常时，由确定性逻辑完全接管，保证 100% 零崩溃运行。
    """

    def __init__(self, toolbox: FinancialToolBox):
        self.toolbox = toolbox

    def execute_plan(self, stock_code: str, date: str) -> AgentTrace:
        trace = AgentTrace(
            query=f"分析 {stock_code} 在 {date} 的盘口微观结构与资金行为",
            stock_code=stock_code,
            date=date,
            execution_mode="DETERMINISTIC_FALLBACK"
        )
        t0 = time.time()

        # Step 1: 提取微观特征
        s1 = AgentStep(
            step_num=1,
            thought=f"[规则兜底] 分析个股 {stock_code}，首先调用微观结构工具提取截面显著偏离特征（Top-5）。",
            action="query_microstructure_profile",
            arguments={"stock_code": stock_code, "date": date, "top_k": 5}
        )
        t_sub = time.time()
        s1.observation = execute_tool(s1.action, s1.arguments)
        s1.duration_ms = (time.time() - t_sub) * 1000
        trace.steps.append(s1)

        # Step 2: 提取行为聚类
        s2 = AgentStep(
            step_num=2,
            thought="[规则兜底] 盘口特征已提取，接下来查询该股在日内 16 窗 FastDTW 时序聚类中的行为模式侧写。",
            action="get_behavior_cluster",
            arguments={"stock_code": stock_code, "date": date}
        )
        t_sub = time.time()
        s2.observation = execute_tool(s2.action, s2.arguments)
        s2.duration_ms = (time.time() - t_sub) * 1000
        trace.steps.append(s2)

        # Step 3: 提取模型分类与归因
        s3 = AgentStep(
            step_num=3,
            thought="[规则兜底] 进一步获取自训练集成模型的资金属性判定（游资/量化/散户）及特征支撑。",
            action="explain_shap_attribution",
            arguments={"stock_code": stock_code, "date": date}
        )
        t_sub = time.time()
        s3.observation = execute_tool(s3.action, s3.arguments)
        s3.duration_ms = (time.time() - t_sub) * 1000
        trace.steps.append(s3)

        # 解析 Step 3 判定的资金类型，以进入 Step 4 质检
        cap_type = "散户"
        try:
            obs3_data = json.loads(s3.observation)
            cap_type = obs3_data.get("capital_type", "散户")
        except Exception:
            pass

        # Step 4: 物理守恒质检
        s4 = AgentStep(
            step_num=4,
            thought=f"[规则兜底] 针对判定结论「{cap_type}」，调用 Critic 工具执行订单簿物理守恒一致性审查。",
            action="verify_orderbook_conservation",
            arguments={"stock_code": stock_code, "hypothesis": cap_type, "date": date}
        )
        t_sub = time.time()
        s4.observation = execute_tool(s4.action, s4.arguments)
        s4.duration_ms = (time.time() - t_sub) * 1000
        trace.steps.append(s4)

        # 合成最终机构级研报
        trace.final_report = self._synthesize_report(stock_code, date, trace.steps)
        trace.total_time_ms = (time.time() - t0) * 1000
        return trace

    def _synthesize_report(self, stock_code: str, date: str, steps: List[AgentStep]) -> str:
        try:
            d_micro = json.loads(steps[0].observation)
            d_clust = json.loads(steps[1].observation)
            d_attr = json.loads(steps[2].observation)
            d_critic = json.loads(steps[3].observation)
        except Exception as e:
            return f"研报合成异常: {e}"

        deviations_str = ""
        for item in d_micro.get("significant_deviations", []):
            deviations_str += f"- **{item['feature_name']}** ({item['raw_id']}): 截面 Z-Score = `{item['z_score']:+.2f}` ({item['state']})\n"

        report = f"""# 【Level2-Insight 投研报告】{stock_code} 资金行为与微观结构全景分析

**分析标的**：`{stock_code}` ｜ **观测日期**：`{date}` ｜ **执行引擎**：`Deterministic Fallback (100% 规则兜底保证)`

---

### 一、核心定性结论
* **参与者属性归因**：**【{d_attr.get('capital_type', '未知')}】**
* **微观行为聚类侧写**：**【{d_clust.get('cluster_pattern', 'N/A')}】**
* **行为动机概括**：{d_attr.get('capital_intention', '暂无详细意图说明')}

---

### 二、底层盘口微观证据链 (Context Engineering 提取)
从 174 维微观维度中自适应压缩（压缩率：`{d_micro.get('compression_ratio', '97%')}`），以下高频特征呈现显著截面离群：
{deviations_str}

---

### 三、时序轨迹与形态解构 (FastDTW 拓扑聚类)
{d_clust.get('archetype_explanation', '无详细聚类阐述')}

---

### 四、抗幻觉与订单簿物理守恒审查 (Critic Audit)
* **审查定性假说**：`{d_critic.get('hypothesis', '')}`
* **审查裁决结论**：`{d_critic.get('critic_verdict', '')}`（通过状态：`{d_critic.get('is_passed', False)}`）
* **因果自洽明细**：{d_critic.get('audit_detail', '通过自洽性核验')}

---
> 💡 *本研报由 Level2-Insight Agent 自动化分析工作台生成，微观特征计算依据沪深交易所 Level-2 高频逐笔与十档订单簿数据。*
"""
        return report


class ReActAgentEngine:
    """ReAct 核心调度引擎"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        try:
            import config
            cfg_key = getattr(config, "LLM_API_KEY", "")
            cfg_url = getattr(config, "LLM_BASE_URL", "")
            cfg_model = getattr(config, "LLM_MODEL", "")
        except Exception:
            cfg_key = ""
            cfg_url = ""
            cfg_model = ""

        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("AMI_LLM_API_KEY", "") or cfg_key
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("AMI_LLM_BASE_URL", "") or cfg_url or "https://api.openai.com/v1"
        self.model_name = model_name or os.getenv("LLM_MODEL") or os.getenv("AMI_LLM_MODEL", "") or cfg_model or "gpt-4o-mini"

        self.guardrail = InjectionGuardrail()
        self.toolbox = FinancialToolBox()
        self.context_engine = self.toolbox.context_engine
        self.fallback_planner = DeterministicFallbackPlanner(self.toolbox)

    def run(
        self,
        user_query: str,
        stock_code: str = "600519.SH",
        date: str = "7.22",
        max_steps: int = 5,
        force_fallback: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> AgentTrace:
        """运行 Agent 主调度逻辑"""
        t0 = time.time()

        active_api_key = api_key or self.api_key
        active_base_url = base_url or self.base_url
        active_model = model_name or self.model_name

        # 1. 安全护栏检测（防 Prompt 注入）
        is_safe, guard_msg = self.guardrail.inspect_query(user_query)
        if not is_safe:
            return AgentTrace(
                query=user_query,
                stock_code=stock_code,
                date=date,
                is_safe=False,
                guardrail_note=guard_msg,
                final_report=f"⛔ **【安全警告】执行已终止**：{guard_msg}",
                total_time_ms=(time.time() - t0) * 1000
            )

        # 2. 检查是否有 API Key 以及是否强制启用兜底
        if force_fallback or not active_api_key or active_api_key.strip() == "":
            logger.info("未检测到有效 API Key 或指定了强制降级，切换到确定性兜底计划器。")
            trace = self.fallback_planner.execute_plan(stock_code, date)
            trace.query = user_query
            return trace

        # 3. 尝试调用真实的 OpenAI 兼容大模型进行 ReAct 自主规划
        try:
            import openai
            client = openai.OpenAI(api_key=active_api_key, base_url=active_base_url)

            trace = AgentTrace(
                query=user_query,
                stock_code=stock_code,
                date=date,
                execution_mode=f"LLM_LIVE ({active_model})"
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{user_query}\n(目标标的: {stock_code}, 交易日期: {date})"}
            ]

            step_count = 0
            while step_count < max_steps:
                step_count += 1
                t_step_start = time.time()

                response = client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.2
                )

                resp_msg = response.choices[0].message
                content = resp_msg.content or ""
                tool_calls = resp_msg.tool_calls

                if not tool_calls:
                    trace.steps.append(AgentStep(
                        step_num=step_count,
                        thought="已收集完整特征与质检证据，生成最终研报。",
                        duration_ms=(time.time() - t_step_start) * 1000
                    ))
                    trace.final_report = content
                    break

                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                for tc in tool_calls:
                    fn_name = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    s = raw_args.strip()
                    if s.startswith("```"):
                        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
                        s = re.sub(r"\n?```$", "", s).strip()
                    try:
                        fn_args = json.loads(s)
                    except Exception:
                        try:
                            fn_args = json.loads(s.replace("'", '"'))
                        except Exception:
                            fn_args = {}

                    if "stock_code" not in fn_args:
                        fn_args["stock_code"] = stock_code
                    if "date" not in fn_args:
                        fn_args["date"] = date

                    thought_text = content if content else f"自主决策：调用工具 [{fn_name}] 获取必要量化指标。"
                    tool_obs = execute_tool(fn_name, fn_args)

                    step_record = AgentStep(
                        step_num=step_count,
                        thought=thought_text,
                        action=fn_name,
                        arguments=fn_args,
                        observation=tool_obs,
                        duration_ms=(time.time() - t_step_start) * 1000
                    )
                    trace.steps.append(step_record)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_obs
                    })

            if not trace.final_report:
                final_resp = client.chat.completions.create(
                    model=active_model,
                    messages=messages + [{"role": "user", "content": "请综合上述所有工具调用结果，输出完整的终审投研分析报告。"}],
                    temperature=0.2
                )
                trace.final_report = final_resp.choices[0].message.content

            trace.total_time_ms = (time.time() - t0) * 1000
            return trace

        except Exception as e:
            logger.warning(f"大模型 ReAct 调用异常 ({e})，立刻自动无缝切换到确定性逻辑兜底。")
            trace = self.fallback_planner.execute_plan(stock_code, date)
            trace.query = user_query
            trace.guardrail_note = f"LLM 接口故障已自动降级: {str(e)}"
            return trace
