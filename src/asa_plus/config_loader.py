from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator


# -----------------------------
# Pydantic 配置模型
# -----------------------------

class PathsConfig(BaseModel):
    books_dir: str
    processed_dir: str
    cache_dir: str
    outputs_dir: str
    generated_code_dir: str
    reports_dir: str


class KnowledgeConfig(BaseModel):
    s0_file: str
    content_index_file: str
    split_books_dir: str

    # S0 摘要生成的层级：默认对章(level=1)和节(level=2)做摘要。需要更细可加入 3。
    summary_levels: List[int] = Field(default_factory=lambda: [1, 2])

    # 单次摘要调用允许传入的最大字符数（过长会分块后做 map-reduce）
    max_chars_per_summary_call: int = 20000


class ProviderConfig(BaseModel):
    """
    type:
      - openai: OpenAI 官方 API
      - openai_compatible: DeepSeek / 其他 OpenAI 兼容 API
      - dashscope: 阿里 DashScope（Qwen）
    """
    type: Literal["openai", "openai_compatible", "dashscope"]
    api_key_env: str
    base_url_env: Optional[str] = None
    default_temperature: float = 0.1
    default_max_tokens: int = 4096
    timeout_s: int = 120
    models: Dict[str, str]  # outline/reasoning/coding 等


class RoutingRule(BaseModel):
    # task 对应的 provider 优先级。允许为空：此时使用 llm.default_provider_priority。
    provider_priority: List[str] = Field(default_factory=list)


class LLMConfig(BaseModel):
    use_cache: bool = True
    cache_ttl_days: int = 365

    # ★新增：全局默认 provider 优先级（当 routing.<task>.provider_priority 为空时使用）
    # 例如：["qwen","openai","deepseek"]
    default_provider_priority: List[str] = Field(default_factory=list)

    providers: Dict[str, ProviderConfig]
    routing: Dict[str, RoutingRule]


class AgentConfig(BaseModel):
    max_depth: int = 6
    max_selected_nodes: int = 8
    max_subquestions: int = 8
    stop_if_confidence_ge: float = 0.85
    chunk_size_chars: int = 12000
    chunk_overlap_chars: int = 400
    max_chunks_per_node: int = 200
    require_user_confirm_if_total_chunks_ge: int = 80


class ExecutionLocalConfig(BaseModel):
    python_bin: str = "python"
    workdir: str = "outputs/runtime"


class ExecutionSSHConfig(BaseModel):
    host: str = ""
    port: int = 22
    username: str = ""
    password: str = ""
    key_path: str = ""
    workdir: str = "/tmp/asa_plus_runtime"
    python_bin: str = "python3"


class ExecutionHTTPConfig(BaseModel):
    endpoint: str = ""
    token: str = ""


class ExecutionConfig(BaseModel):
    mode: Literal["local", "remote_ssh", "remote_http"] = "local"
    local: ExecutionLocalConfig = Field(default_factory=ExecutionLocalConfig)
    remote_ssh: ExecutionSSHConfig = Field(default_factory=ExecutionSSHConfig)
    remote_http: ExecutionHTTPConfig = Field(default_factory=ExecutionHTTPConfig)


class ReportConfig(BaseModel):
    include_full_code_in_report: bool = True
    include_evidence_notes: bool = True
    max_evidence_chars_per_note: int = 1200


class ProjectConfig(BaseModel):
    name: str = "ASAPlus"
    root_path: str = "."


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    knowledge: KnowledgeConfig
    llm: LLMConfig
    agent: AgentConfig
    execution: ExecutionConfig
    report: ReportConfig

    @model_validator(mode="after")
    def _validate_llm_refs(self):
        # 校验 routing/default_provider_priority 引用的 provider 名称必须存在
        provider_names = set(self.llm.providers.keys())

        for p in self.llm.default_provider_priority:
            if p not in provider_names:
                raise ValueError(f"llm.default_provider_priority 引用了不存在的 provider: {p}")

        for task, rule in self.llm.routing.items():
            for p in rule.provider_priority:
                if p not in provider_names:
                    raise ValueError(f"llm.routing[{task}] 引用了不存在的 provider: {p}")
        return self


# -----------------------------
# 加载配置
# -----------------------------

def _find_project_root(start: Optional[Path] = None) -> Path:
    """向上寻找包含 config/config.yaml 的目录作为项目根。"""
    start = start or Path.cwd()
    cur = start.resolve()
    for _ in range(10):
        cfg = cur / "config" / "config.yaml"
        if cfg.exists():
            return cur
        cur = cur.parent
    return start.resolve()


def load_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> AppConfig:
    """
    加载 config/config.yaml 并结合 .env 环境变量。
    - config_path: 默认自动寻找项目根下的 config/config.yaml
    - env_path: 默认自动寻找项目根下的 .env
    """
    project_root = _find_project_root()

    # 优先加载显式 env_path，否则加载项目根的 .env（若存在）
    if env_path:
        load_dotenv(env_path)
    else:
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    # 读取 YAML
    if config_path:
        cfg_file = Path(config_path).expanduser().resolve()
    else:
        cfg_file = project_root / "config" / "config.yaml"

    if not cfg_file.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_file}")

    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"配置文件为空: {cfg_file}")

    # 注入 root_path（如果未显式给出）
    raw.setdefault("project", {})
    raw["project"].setdefault("root_path", str(project_root))

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"配置校验失败: {e}") from e

    return config


def resolve_path(config: AppConfig, path_str: str) -> Path:
    """把 config 中的相对路径解析为绝对路径。"""
    root = Path(config.project.root_path).expanduser().resolve()
    p = Path(path_str)
    return p if p.is_absolute() else (root / p).resolve()


def ensure_dirs(config: AppConfig) -> None:
    """确保关键目录存在。"""
    for p in [
        config.paths.books_dir,
        config.paths.processed_dir,
        config.paths.cache_dir,
        config.paths.outputs_dir,
        config.paths.generated_code_dir,
        config.paths.reports_dir,
        config.knowledge.split_books_dir,
        config.agent and config.execution.local.workdir,
    ]:
        abs_p = resolve_path(config, p)
        abs_p.mkdir(parents=True, exist_ok=True)
