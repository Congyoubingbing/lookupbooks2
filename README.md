# ASA Plus（lookupbooks）——基于 API 的高分子/流变学智能体系统

> **重要原则**：本项目**不在本地部署任何大模型**；所有与 GPT-4o / Qwen / DeepSeek 的交互均通过各自的 **API** 完成。
> 本项目严格贯彻你的系统设计构想：**S0 知识框架构建 → Q0 多层分类 S1..Sn → 基于选中章节原文推理 → 公式推导与编程 → 用户确认后执行（本地/服务器）→ 返回模拟结果与报告**。

## 1. 你将获得什么

- **S0 构建**：对多本书的 `.txt` 原文进行“按章/节分块”，并由 LLM 生成**层级知识框架 S0**（章节标题 + 摘要 + 关键概念/公式索引 + 可追溯 node_id）。
- **S1..Sn 多层分类与推理**：
  1) 将 Q0 + S0 发送给 LLM 得到 S1（选择相关章节/子类）。
  2) 系统自动读取这些章节/小节的**全部原文**（分块后逐块发送给 LLM），要求 LLM 输出**引用了哪本书哪一章哪一节**，并给出可得结论。
  3) 若仍不能完全解决 Q0，则把 S1 + Q0 + S0 再次发给 LLM 产生更细的分类 S2...，直到 Sn 可解决或达到深度上限。
- **代码生成**：LLM 基于最终推理链给出**公式推导**与**可运行代码（Python 优先）**。
- **执行门控**：代码生成后**必须由用户确认**，确认后才会执行（本地/远程服务器均支持）。
- **报告**：自动生成包含推理链、引用章节、代码与结果的 Markdown 报告。

## 2. 项目目录结构

```
lookupbooks/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── config.yaml
├── data/
│   ├── books/              # 放你的书籍 .txt（由 .tex 转 .txt 后放这里）
│   ├── processed/          # S0、章节切分产物等
│   └── cache/              # LLM 调用缓存（强烈建议开启）
├── outputs/
│   ├── generated_code/     # 生成的代码
│   └── reports/            # 报告
├── docs/
│   ├── architecture.dot
│   └── architecture.png
└── src/
    └── asa_plus/           # 主包
        ├── main.py
        ├── config_loader.py
        ├── llm/
        ├── knowledge/
        ├── agents/
        ├── execution/
        └── utils/
```

## 3. 安装与运行

### 3.1 安装依赖
建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

### 3.2 配置环境变量
复制 `.env.example` 为 `.env`，填入你的密钥：

```bash
cp .env.example .env
```

### 3.3 修改配置文件
编辑 `config/config.yaml`：
- 指定三类模型（GPT-4o / Qwen / DeepSeek）的 API 配置与路由策略
- 指定 data/books 路径与 S0 输出路径
- 指定执行模式（local / remote_ssh / remote_http）

### 3.4 构建 S0
把书籍 txt 放入 `data/books/`，然后：

```bash
python -m asa_plus.main build-s0
```

### 3.5 提问（生成 S1..Sn、推理、产出代码、确认执行）
```bash
python -m asa_plus.main ask "这里输入你的复杂问题Q0"
```

> 你也可以不传参数，进入交互式输入。

## 4. 重要提醒（你需要修改的地方）

- `.env`：填入 `OPENAI_API_KEY / QWEN_API_KEY / DEEPSEEK_API_KEY`
- `config/config.yaml`：
  - `llm.providers.*.models.*`：按你的账号可用模型填写
  - `execution.mode` 与服务器信息（如用远程执行）
- `data/books/`：放你的 txt 书籍文件（文件名可任意，系统会生成 book_id）

## 5. 系统架构图
见 `docs/architecture.png`（由 `docs/architecture.dot` 生成）。

