# LangChainV1.2

## 项目简介

这是一个 LangChain 学习与示例项目，主要使用 DeepSeek 聊天模型演示以下能力：

- 基础模型调用、流式输出、批处理和异步调用
- 模型初始化、运行时配置、限流与回调
- 工具调用和 Agent
- Pydantic、TypedDict、JSON Schema 等结构化输出
- 静态提示词、动态提示词和 Agent middleware

项目中的示例以可直接运行的 Python 脚本为主。公共 DeepSeek 模型实例通过
LangChain 的 `init_chat_model` 统一入口创建，并集中定义在
`models/init_chat_model/init_chat_model_llm.py` 中。

## 目录结构

```text
.
├── agents/           # LangChain Agent 与工具创建示例
│   ├── basics/       # Agent 创建、调用、提示词与 middleware 示例
│   ├── async_invocation/  # Agent 的 ainvoke 异步调用示例
│   ├── streaming/    # Agent 流式执行与 stream_mode 示例
│   ├── agent_structured_output/  # Agent 结构化输出与错误处理示例
│   ├── tool_creation/  # @tool、Pydantic 和 JSON Schema 工具示例
│   └── tool_call_error_handling/  # Agent 工具调用异常处理示例
├── models/           # 聊天模型能力示例
│   ├── basics/       # 同步、流式、批处理和异步调用示例
│   ├── init_chat_model/  # 公共模型实例与统一初始化入口示例
├── short_memory/     # Agent 短期记忆与 checkpoint 示例
│   └── llm_content/  # LLM 上下文消息截断、删除、摘要和自定义策略示例
├── docs/skills/      # 项目文档维护 skill
├── scripts/          # 文档审计与维护辅助脚本
├── env_utils.py      # 加载 DeepSeek 和 MySQL 环境变量
└── quick_start.py    # Agent 快速开始示例
```

## 环境配置

建议先创建并启用虚拟环境，然后安装当前示例使用的依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain langchain-deepseek langchain-openai python-dotenv pydantic
```

如果要运行 MySQL checkpoint 示例，还需要安装：

```bash
python -m pip install \
  "langgraph-checkpoint-mysql[pymysql]==3.0.0" \
  "PyMySQL[rsa]==1.1.2" \
  "cryptography==46.0.3"
```

在项目根目录创建 `.env` 文件，并配置以下变量：

```dotenv
DEEPSEEK_API_KEY=你的 API Key
DEEPSEEK_BASE_URL=DeepSeek API Base URL

# 仅运行 short_memory/03_short_memory_indb.py 时需要
MYSQL_DATABASE_URL=mysql://langchain_user:你的密码@localhost:3306/langchain_db
```

也可以复制 `.env.example` 后再填入真实值。
请勿提交包含真实密钥的 `.env` 文件。

## 快速开始

在项目根目录运行 Agent 天气查询示例：

```bash
python quick_start.py
```

也可以按模块运行其他示例，例如：

```bash
python -m models.basics.blocking_call
python -m models.basics.stream_output
python -m agents.basics.05_agent_dynamic_prompt
python -m agents.async_invocation.01_basic_ainvoke
python -m agents.streaming.01_stream_updates
python -m agents.agent_structured_output.01_pydantic_schema
python -m agents.tool_creation.01_create_tool
python -m agents.tool_call_error_handling.01_generic_tool_error_handler
```

## 学习模块

### 模型基础能力

`models/basics/` 包含聊天模型的基础调用示例：

- `blocking_call.py`：同步阻塞调用
- `stream_output.py`：同步流式输出
- `batch_process.py`：同步批处理
- `async_call.py`：异步调用
- `async_batch_process.py`：异步批处理
- `async_stream_output.py`：异步流式输出

### 模型初始化

- `models/init_chat_model/`：使用 `init_chat_model` 统一入口创建项目共享模型，
  其他普通示例复用该目录中的 `deepseek_llm`
- `models/model_classes/`：使用 `ChatDeepSeek`、`ChatOpenAI` 等具体模型类初始化模型

### 工具调用

`models/tool_calling/` 演示 `bind_tools`、`tool_calls` 和 `ToolMessage` 的使用，
以及如何手动处理单工具和多工具调用流程。

### 结构化输出

`models/structured_output/` 演示以下结构化输出方式：

- Pydantic 模型
- TypedDict
- JSON Schema
- `JsonOutputParser`

### 模型高级能力

`models/advanced_features/` 包含推理模型、内存限流、回调、运行时配置和
`configurable_fields` 等示例。

### Agent

`agents/basics/` 中的示例按文件编号组织，主要包括：

- 创建静态模型 Agent
- 使用 middleware 包装模型调用
- 查看 `agent.invoke` 的输入与消息轨迹
- 配置字符串或 `SystemMessage` 系统提示词
- 根据运行时 context 动态生成系统提示词

### Agent 结构化输出

`agents/agent_structured_output/` 演示通过 `response_format` 和 `ToolStrategy`
约束 Agent 的最终输出，主要包括：

- 使用 Pydantic、dataclass、TypedDict 和 JSON Schema 定义输出结构
- 通过 `tool_message_content` 自定义结构化输出成功消息
- 使用 `handle_errors` 配置默认重试、固定错误提示或关闭重试
- 对比 `Union` 二选一与嵌套组合模型同时返回多类信息
- 使用自定义错误处理函数区分校验错误和多结构输出错误

可以从项目根目录按模块运行：

```bash
python -m agents.agent_structured_output.01_pydantic_schema
python -m agents.agent_structured_output.06_tool_strategy_error_handling
python -m agents.agent_structured_output.07_combined_structured_output
python -m agents.agent_structured_output.08_tool_strategy_custom_error_handler
```

### Agent 异步调用

`agents/async_invocation/` 演示在异步函数中使用 `await agent.ainvoke(...)`
调用 Agent。当前示例通过旅行规划场景组合天气、交通和景点工具，并返回完整消息历史。

```bash
python -m agents.async_invocation.01_basic_ainvoke
```

### Agent 流式执行

`agents/streaming/` 演示使用 `agent.stream()` 查看 Agent 执行过程。当前示例通过
`stream_mode="updates"` 逐步输出模型节点和工具节点写入的消息，并说明流模式与
检查点持久化的职责区别；同时演示 `stream_mode="checkpoints"` 输出状态快照，
使用固定 `thread_id` 延续同一会话，以及通过 `stream_mode="messages"`
实时接收模型消息片段。其他示例分别展示完整状态、任务生命周期、调试事件，
以及工具内部主动发送的自定义进度事件。

```bash
python -m agents.streaming.01_stream_updates
python -m agents.streaming.02_stream_checkpoints
python -m agents.streaming.03_stream_messages
python -m agents.streaming.04_stream_values
python -m agents.streaming.05_stream_tasks
python -m agents.streaming.06_stream_debug
python -m agents.streaming.07_stream_custom
```

### Agent 工具创建

`agents/tool_creation/` 演示三种 Agent 工具创建方式：

- `01_create_tool.py`：使用 `@tool` 装饰器和函数签名创建工具
- `02_create_pydantic_tool.py`：使用 Pydantic 模型定义强类型参数和字段校验
- `03_create_schema_tool.py`：直接使用 JSON Schema 定义参数、枚举和必填规则

可以从项目根目录按模块运行：

```bash
python -m agents.tool_creation.01_create_tool
python -m agents.tool_creation.02_create_pydantic_tool
python -m agents.tool_creation.03_create_schema_tool
```

### Agent 工具异常处理

`agents/tool_call_error_handling/` 演示使用 `wrap_tool_call` middleware
统一处理 Agent 工具执行异常：

- `01_generic_tool_error_handler.py`：捕获外部股票服务异常并返回统一的 `ToolMessage`
- `02_exception_specific_tool_error_handler.py`：分别处理连接失败、权限不足、
  业务校验和未知异常

可以从项目根目录按模块运行：

```bash
python -m agents.tool_call_error_handling.01_generic_tool_error_handler
python -m agents.tool_call_error_handling.02_exception_specific_tool_error_handler
```

### Agent 短期记忆

`short_memory/` 演示 Agent 如何通过 checkpointer 保存同一会话中的消息状态：

- `01_memory_demo.py`：使用 `InMemorySaver` 演示基础短期记忆
- `02_short_memory_inmemory.py`：加入工具调用，并通过 `get_state()` 查看会话状态
- `03_short_memory_indb.py`：使用 `PyMySQLSaver` 把 checkpoint 保存到 MySQL
- `04_custom_state.py`：使用 `state_schema` 扩展 Agent 状态，并通过动态提示词读取状态
- `05_tool_modify_state.py`：演示工具返回 `Command(update=...)` 修改自定义状态
- `06_middleware_modify_state.py`：演示 `before_model` 和 `after_model` 在模型调用前后更新状态
- `07_middleware_modify_state.py`：演示 `after_model` 读取结构化输出并保存订单商品名
- `08_context_state.py`：演示 runtime context 与 Agent state 的区别
- `llm_content/`：演示 LLM 上下文变长后的消息管理方案，包括 `trim_messages` 截断、
  `RemoveMessage` 删除、手写摘要、内置 `SummarizationMiddleware` 和自定义保留策略

运行 MySQL 示例前，需要先创建 `langchain_db` 数据库，并在 `.env` 中配置
`MYSQL_DATABASE_URL`。

## 运行注意事项

- 多数示例会调用真实 DeepSeek 模型并消耗 API 额度。
- Agent 和工具调用可能触发多轮模型请求，调用次数与 token 消耗会相应增加。
- Agent 结构化输出发生 Schema 校验错误并自动重试时，也会增加模型调用次数。
- `tool_call_error_handling/01_generic_tool_error_handler.py` 会访问模拟失败的外部接口；
  `02_exception_specific_tool_error_handler.py` 包含随机异常，因此重复运行的结果可能不同。
- `models/advanced_features/01_model_reasoner.py` 默认调用 `deepseek-reasoner`，推理模型通常比
  `deepseek-chat` 成本更高。
- 请从项目根目录运行脚本或使用 `python -m <模块路径>`，以确保可以正确导入
  `models.init_chat_model.init_chat_model_llm`。
- `short_memory/03_short_memory_indb.py` 会连接 MySQL，并把同一 `thread_id` 的 checkpoint
  持久化到数据库；重复测试时可以更换 `thread_id` 避免读取旧会话。
- `short_memory/llm_content/` 下的摘要示例可能在正式回答前额外调用模型生成摘要，
  会增加 API 调用次数和 token 消耗。
- 清空 MySQL checkpoint 时，不要只删除 `checkpoint_migrations` 的数据；如果要完全重置，
  请删除 checkpoint 相关表后让示例重新创建表结构。
- 示例中的天气、股票价格和新闻等工具返回模拟数据，不代表真实外部查询结果。

## 文档维护

检查 README 与当前项目结构是否一致：

```bash
python scripts/audit_project_readme.py
```

扫描各 Python 包的 `__init__.py` 说明状态：

```bash
python scripts/audit_init_docs.py
```

该脚本会列出项目中的 Python 包、包内示例文件，以及每个 `__init__.py`
是否为空或包含包级说明。脚本只负责扫描，文档内容由 Codex 根据 skill 维护。

项目内文档维护 skill 位于：

```text
docs/skills/package-init-doc-maintainer/
docs/skills/project-readme-maintainer/
```

在 Codex 中可以直接提出：

```text
使用 package-init-doc-maintainer 更新 tool_creation 的 __init__.py
使用 project-readme-maintainer 根据当前代码更新 README.md
```
