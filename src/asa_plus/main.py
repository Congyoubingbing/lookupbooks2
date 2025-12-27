from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer

from .config_loader import load_config, ensure_dirs, resolve_path
from .knowledge.knowledge_builder import KnowledgeBuilder
from .knowledge.knowledge_store import KnowledgeStore
from .agents.question_agent import QuestionAgent
from .agents.code_agent import CodeAgent
from .execution.executor import select_executor
from .utils.report_generator import ReportGenerator
from .utils.logger import setup_logging, get_logger

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


@app.command("init")
def cmd_init():
    """
    初始化项目（创建必要目录、复制示例文件等）
    """
    setup_logging()
    cfg = load_config()
    ensure_dirs(cfg)

    root = Path(cfg.project.root_path).resolve()
    # 如果 books_dir 为空，可以复制示例书籍（可选）
    books_dir = resolve_path(cfg, cfg.paths.books_dir)
    if not any(books_dir.glob("*.txt")):
        log.warning("books_dir 为空：你需要把清洗后的 .txt 书籍放到 data/books 下。")

    log.info("初始化完成。")
    log.info(f"项目根目录: {root}")
    log.info(f"books_dir: {books_dir}")


@app.command("build-s0")
def cmd_build_s0():
    """
    构建 S0：逐书解析 -> 逐章/节摘要 -> 写入 s0_knowledge.json 与 content_index.json
    """
    setup_logging()
    cfg = load_config()
    ensure_dirs(cfg)

    kb = KnowledgeBuilder(cfg)
    kb.build_s0()


@app.command("show-s0")
def cmd_show_s0(limit: int = 20):
    """
    展示 S0 中的部分节点（用于快速检查）
    """
    setup_logging()
    cfg = load_config()
    ks = KnowledgeStore(cfg)
    ks.load()

    nodes = list(ks.iter_nodes())
    log.info(f"S0 nodes: {len(nodes)}")
    for n in nodes[:limit]:
        print(f"[{n.level}] {n.node_id} | {n.title}")


@app.command("ask")
def cmd_ask(
    question: str = typer.Argument(..., help="复杂问题 Q0（自然语言）"),
    confirm: bool = typer.Option(False, "--confirm", help="是否允许执行代码（需要你确认后才会执行）"),
):
    """
    输入复杂问题 Q0：
    - 基于 S0 做逐层分类与推理，得到解题思路
    - 生成可执行代码（默认不执行）
    - 你使用 --confirm 才会执行（本机或远程）
    """
    setup_logging()
    cfg = load_config()
    ensure_dirs(cfg)

    ks = KnowledgeStore(cfg)
    ks.load()

    router = None
    # QuestionAgent 内部会创建 LLMRouter；这里保持结构清晰
    q_agent = QuestionAgent(cfg, ks)
    result = q_agent.solve(question)

    # 输出解题思路与引用
    print("\n========== 解题思路 ==========")
    print(result.plan_text)

    print("\n========== 使用到的章节（证据链） ==========")
    for ev in result.evidence_notes:
        print(f"- {ev}")

    # 代码生成
    c_agent = CodeAgent(cfg)
    code_artifact = c_agent.generate_code(question=question, plan=result.plan_text, evidence=result.evidence_notes)

    print("\n========== 生成代码路径 ==========")
    print(code_artifact.code_path)

    # 报告
    reporter = ReportGenerator(cfg)
    report_path = reporter.generate(
        question=question,
        plan_text=result.plan_text,
        evidence_notes=result.evidence_notes,
        code_path=code_artifact.code_path,
    )
    print("\n========== 报告路径 ==========")
    print(report_path)

    if not confirm:
        print("\n[提示] 你尚未允许执行代码。若确认要执行，请使用：")
        print(f'  python main.py ask "{question}" --confirm')
        return

    # 执行（本机/远程）
    executor = select_executor(cfg)
    exec_result = executor.run_python(code_artifact.code_path)
    print("\n========== 执行结果 ==========")
    print(exec_result.stdout)
    if exec_result.stderr:
        print("\n========== 执行错误输出 ==========")
        print(exec_result.stderr)


@app.command("run-code")
def cmd_run_code(
    code_path: str = typer.Argument(..., help="要执行的代码文件路径（通常来自 outputs/generated_code）"),
    confirm: bool = typer.Option(False, "--confirm", help="是否允许执行（安全开关）"),
):
    """
    单独执行某个生成的代码文件（需要 --confirm）
    """
    setup_logging()
    cfg = load_config()
    ensure_dirs(cfg)

    if not confirm:
        print("\n[提示] 未允许执行。请添加 --confirm")
        return

    executor = select_executor(cfg)
    exec_result = executor.run_python(code_path)
    print(exec_result.stdout)
    if exec_result.stderr:
        print(exec_result.stderr, file=sys.stderr)


def main():
    app()


if __name__ == "__main__":
    main()
