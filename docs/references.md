# 技术选型与参考来源（GitHub / 论文）

> 说明：以下均为你要求的“借鉴来源”。为避免在正文里直接贴 URL（渲染限制），这里集中放到文档中。

## 编排与多智能体/流程编排
- LangGraph（LangChain 生态的有向图状态机，适合 Sn 迭代循环）
  - GitHub: https://github.com/langchain-ai/langgraph
- LangChain（消息/提示词/模型接口抽象）
  - GitHub: https://github.com/langchain-ai/langchain

## 分解与推理范式（论文）
- ReAct: Synergizing Reasoning and Acting in Language Models
  - 论文: https://arxiv.org/abs/2210.03629
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models
  - 论文: https://arxiv.org/abs/2305.10601
- Self-Ask: Measuring and Narrowing the Compositionality Gap in Language Models
  - 论文: https://arxiv.org/abs/2210.03350

## 长文档分块与摘要（工程实践）
- Map-Reduce Summarization（LangChain 示例链路）
  - 说明/代码： https://python.langchain.com/docs/how_to/summarize_map_reduce/
  - GitHub（LangChain）： https://github.com/langchain-ai/langchain

## LLM API
- OpenAI Python SDK（调用 GPT-4o 等）
  - GitHub: https://github.com/openai/openai-python
- DashScope SDK（调用 Qwen）
  - GitHub: https://github.com/dashscope/dashscope-sdk-python
- DeepSeek API（OpenAI-compatible 接口）
  - 官方文档（示例）：https://platform.deepseek.com/

## 远程执行
- Paramiko（SSH 上传与执行）
  - GitHub: https://github.com/paramiko/paramiko

## 图示
- Graphviz（本项目 docs/architecture.dot -> architecture.png）
  - GitLab 镜像：https://gitlab.com/graphviz/graphviz
