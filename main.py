# -*- coding: utf-8 -*-
"""
AFAC2026 挑战组复现主入口 (main.py)
"""
import sys
import os
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import io
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import config
from inference.infer import AMIInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Main")


def main():
    parser = argparse.ArgumentParser(description="AFAC2026 AMI Master 复现入口")
    parser.add_argument("--date", type=str, default="7.22", help="推理数据日期")
    parser.add_argument("--demo", action="store_true", help="运行 Demo 端到端测试")
    parser.add_argument("--web", action="store_true", help="启动 Level2-Insight 交互式投研工作台")
    parser.add_argument("--test", action="store_true", help="运行 Level2-Insight Agent 自动化评测测试套件")
    parser.add_argument("--agent", type=str, default="", help="针对单只股票代码启动 ReAct Agent 深度分析 (如 600519.SH)")
    parser.add_argument("--fast", action="store_true", help="跳过大模型网络调用，使用本地确定性因果规则秒级完成")
    parser.add_argument("--workers", type=int, default=2, help="大模型质证并发线程数 (默认: 2)")
    parser.add_argument("--max-llm-stocks", type=int, default=None, help="大模型审查标的上限数 (默认: None，全量覆盖所有股票)")
    args = parser.parse_args()

    if args.web:
        logger.info("启动 Level2-Insight 交互式 Web 工作台 (Streamlit)...")
        import subprocess
        web_path = BASE_DIR / "web" / "app.py"
        subprocess.run(["streamlit", "run", str(web_path)])
        return

    if args.test:
        logger.info("运行 Level2-Insight Agent 单元测试套件...")
        import unittest
        suite = unittest.defaultTestLoader.discover(str(BASE_DIR / "tests"), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        sys.exit(0 if res.wasSuccessful() else 1)

    if args.agent:
        stock = args.agent.strip().upper()
        logger.info(f"=== 启动 Level2-Insight ReAct Agent 单标的深度审查: {stock} ({args.date}) ===")
        from agent.engine import ReActAgentEngine
        engine = ReActAgentEngine()
        trace = engine.run(
            user_query=f"请对标的 {stock} 在 {args.date} 的高频订单簿、微观结构及资金行为展开全景审查，并核查物理守恒。",
            stock_code=stock,
            date=args.date
        )
        print("\n" + "=" * 65)
        print(trace.final_report)
        print("=" * 65 + "\n")
        return

    pipeline = AMIInferencePipeline()

    use_llm = not args.fast
    if args.demo:
        logger.info(f"=== 执行 Demo 端到端测试 (大模型质证: {'启用' if use_llm else '跳过'}) ===")
        res = pipeline.run(
            date="demo",
            is_demo=True,
            use_llm=use_llm,
            max_workers=args.workers,
            all_stocks=(args.max_llm_stocks is None),
            max_llm_stocks=args.max_llm_stocks
        )
    else:
        logger.info(f"=== 执行日期 {args.date} 复现流水线 (大模型质证: {'启用' if use_llm else '跳过'}) ===")
        res = pipeline.run(
            date=args.date,
            is_demo=False,
            use_llm=use_llm,
            max_workers=args.workers,
            all_stocks=(args.max_llm_stocks is None),
            max_llm_stocks=args.max_llm_stocks
        )

    logger.info("=" * 65)
    logger.info("AFAC2026 AMI Master 执行完成:")
    logger.info(f"  Task 1 聚类簇数: {res['k']}, 综合拓扑得分: {res['wdtw']:.4f}")
    if "stats" in res:
        st = res["stats"]
        logger.info(f"  Agent 因果质证结果: 自洽确认 {st['confirmed']} 只, 纠偏修正 {st['overridden']} 只, 全量特征扩展 {st['tier2_expanded']} 只")
    logger.info(f"  Task 1 修改后结果: {config.BASE_DIR / 'pattern_reco.csv'}")
    logger.info(f"  Task 2 修改后结果: {config.BASE_DIR / 'predict_result.csv'}")
    logger.info(f"  质证详细审计报告: {Path(res['out_dir']) / 'cross_examination_audit.json'}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
