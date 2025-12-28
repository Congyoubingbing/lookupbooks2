from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config_loader import load_config, ensure_dirs
from .knowledge.knowledge_store import KnowledgeStore
from .agents.question_agent import QuestionAgent
from .agents.code_agent import CodeAgent
from .utils.logger import get_logger

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASAPlus LookupBooks CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="向系统提问并生成解题方案")
    ask.add_argument("question", type=str, help="问题文本")
    ask.add_argument(
        "--confirm",
        action="store_true",
        help="占位：是否允许执行代码（目前由 main 层处理）",
    )
    return parser


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    # 加载知识库
    ks = KnowledgeStore(cfg)
    ks.load()

    # 调用 QuestionAgent（注意：QuestionAgent.solve 返回 SolveResult）
    q_agent = QuestionAgent(cfg, ks)
    # QuestionAgent.solve 的签名可能是 solve(question) — 我们仅传入 question
    result = q_agent.solve(question=args.question)

    # 输出解题思路（人类可读）
    print("\n========== 解题思路 ==========")
    print(result.plan_text)

    # 输出证据摘要（展示用）
    if result.evidence_notes:
        print("\n========== 使用到的章节（证据链） ==========")
        for ev in result.evidence_notes:
            print(f"- {ev}")

    # Code generation: 传入结构化 final_plan 与 used_sources
    c_agent = CodeAgent(cfg)
    # final_plan 供 CodeAgent 使用；这里我们使用 assessment.solution_outline（若 agent 将 final_plan 放在其它字段，也可直接使用 result.final_plan）
    final_plan = result.final_plan if result.final_plan is not None else result.assessment.solution_outline
    used_sources = result.used_sources

    code_out = c_agent.generate(
        q0=args.question,
        final_plan=final_plan,
        used_sources=used_sources,
        session_id=result.session_id or "session",
    )

    # 展示代码生成信息
    print("\n========== 代码生成完成 ==========")
    print(f"使用引擎: {code_out.engine_choice}")
    print(f"生成文件数: {len(code_out.artifacts)}")
    if code_out.artifacts:
        print("\n生成的文件：")
        for art in code_out.artifacts:
            print(f"- {art.path}")

    if code_out.run_instructions:
        print("\n运行说明：")
        for line in code_out.run_instructions:
            print(f"- {line}")

    print("\n（完整生成结果已保存到 outputs/generated_code/ 对应 session 目录）")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ask":
        cmd_ask(args)
    else:
        parser.error(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
