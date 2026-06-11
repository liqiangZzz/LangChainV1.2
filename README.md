# LangChainV1.2

## 项目简介

这是一个 LangChain 学习与示例项目，主要使用 DeepSeek 聊天模型演示以下能力：

- 基础模型调用、流式输出、批处理和异步调用
- 模型初始化、运行时配置、限流与回调
- 工具调用和 Agent
- Pydantic、TypedDict、JSON Schema 等结构化输出
- 静态提示词、动态提示词和 Agent middleware

项目中的示例以可直接运行的 Python 脚本为主，公共 DeepSeek 模型实例统一定义在
`my_llm.py` 中。

## 目录结构

```text
.
├── agent_part/       # LangChain Agent 示例
├── models/           # 聊天模型能力示例
├── docs/skills/      # 项目文档维护 skill
├── scripts/          # 文档审计与维护辅助脚本
├── env_utils.py      # 加载 DeepSeek 环境变量
├── my_llm.py         # 公共 DeepSeek Chat 模型实例
└── quick_start.py    # Agent 快速开始示例
```

## 环境配置

建议先创建并启用虚拟环境，然后安装当前示例使用的依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain langchain-deepseek langchain-openai python-dotenv pydantic
```

在项目根目录创建 `.env` 文件，并配置以下变量：

```dotenv
DEEPSEEK_API_KEY=你的 API Key
DEEPSEEK_BASE_URL=DeepSeek API Base URL
```

请勿提交包含真实密钥的 `.env` 文件。

## 快速开始

在项目根目录运行 Agent 天气查询示例：

```bash
python quick_start.py
```

也可以按模块运行其他示例，例如：

```bash
python -m models.block_call
python -m models.stream_output
python -m agent_part.05_agent_dynamic_prompt
```

## 学习模块

### 模型基础能力

`models/` 包含聊天模型的基础调用示例：

- `block_call.py`：同步阻塞调用
- `stream_output.py`：同步流式输出
- `batch_process.py`：同步批处理
- `async_call.py`：异步调用
- `async_batch_process.py`：异步批处理
- `async_stream_output.py`：异步流式输出

### 模型初始化

- `models/initchatmodel/`：使用 `init_chat_model` 统一入口初始化模型
- `models/modelclass/`：使用 `ChatDeepSeek`、`ChatOpenAI` 等具体模型类初始化模型

### 工具调用

`models/model_tool_calling/` 演示 `bind_tools`、`tool_calls` 和 `ToolMessage` 的使用，
以及如何手动处理单工具和多工具调用流程。

### 结构化输出

`models/model_strcutured_optput/` 演示以下结构化输出方式：

- Pydantic 模型
- TypedDict
- JSON Schema
- `JsonOutputParser`

目录名 `model_strcutured_optput` 保留项目现有命名，实际含义为
`model_structured_output`。

### 模型高级能力

`models/model_other/` 包含推理模型、内存限流、回调、运行时配置和
`configurable_fields` 等示例。

### Agent

`agent_part/` 中的示例按文件编号组织，主要包括：

- 创建静态模型 Agent
- 使用 middleware 包装模型调用
- 查看 `agent.invoke` 的输入与消息轨迹
- 配置字符串或 `SystemMessage` 系统提示词
- 根据运行时 context 动态生成系统提示词

## 运行注意事项

- 多数示例会调用真实 DeepSeek 模型并消耗 API 额度。
- Agent 和工具调用可能触发多轮模型请求，调用次数与 token 消耗会相应增加。
- `models/model_other/01_model_reasoner.py` 默认调用 `deepseek-reasoner`，推理模型通常比
  `deepseek-chat` 成本更高。
- 请从项目根目录运行脚本或使用 `python -m <模块路径>`，以确保可以正确导入
  `my_llm.py`。
- 示例中的天气、股票价格和新闻等工具返回模拟数据，不代表真实外部查询结果。

## 文档维护

检查 README 与当前项目结构是否一致：

```bash
python scripts/audit_project_readme.py
```

README 的维护规则位于：

```text
docs/skills/project-readme-maintainer/
```
