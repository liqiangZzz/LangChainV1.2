"""
Agent 短期记忆示例包。

本包围绕 LangChain Agent 的短期记忆和状态管理展开，重点演示同一个
`thread_id` 下，Agent 如何通过 checkpointer 保存消息历史，以及如何扩展、
读取和更新自定义 state。

学习主线：

1. 使用 InMemorySaver 观察最基础的会话记忆。
2. 将工具调用加入短期记忆流程，查看消息和工具结果如何进入 state。
3. 使用 MySQL checkpointer 持久化 checkpoint，让会话状态跨进程保留。
4. 使用 state_schema 扩展业务字段，并通过 dynamic_prompt 让模型看到 state。
5. 使用工具返回 Command(update=...) 修改 state。
6. 使用 before_model / after_model middleware 在模型调用前后统计和更新状态。
7. 区分 runtime context 和 Agent state，理解“本次调用参数”和“持久化会话状态”的边界。
8. 在 LLM 上下文变长时，通过截断、删除、摘要和自定义策略管理 messages。

state 与 context：

- state 是 Agent 的会话状态，会被 checkpointer 按 thread_id 保存。默认包含
  messages，也可以通过 state_schema 扩展 user_id、订单号、商品名、统计次数等
  业务字段。适合保存后续多轮对话还要继续使用的信息。
- context 是本次 invoke/stream 传入的运行时上下文，只在当前调用中有效，不会自动
  写入短期记忆。适合传递当前登录用户、渠道、租户、权限、语言等本次请求参数。
- 工具和 middleware 可以通过 runtime.state 读取 state，通过 runtime.context 读取
  context；二者都不会自动暴露给模型。若希望模型看到 state，需要用 dynamic_prompt
  或 middleware 显式注入提示词。
- 当 context 中的信息需要跨轮保留时，可以在 middleware 中返回 dict，将本次
  context 合并进 state；否则应保持为临时运行参数。

常见 state 更新方式：

- invoke 输入中传入 state_schema 声明过的字段，作为本轮初始状态的一部分。
- before_model / after_model middleware 返回 dict，让 LangGraph 合并进 state。
- 工具返回 Command(update=...)，在工具执行后同时更新 messages 和业务状态字段。
- 返回新的 messages、RemoveMessage 或摘要消息，管理 state["messages"] 的长度和内容。

主要文件：

- 01_memory_demo.py
  使用 InMemorySaver 和相同 thread_id 演示最基础的两轮短期记忆效果，适合先观察
  messages 如何被 checkpointer 保存。

- 02_short_memory_inmemory.py
  在短期记忆示例中加入工具调用，并通过 get_state() 查看同一会话的最终状态。
  重点观察 ToolMessage 如何进入消息历史。

- 03_short_memory_indb.py
  使用 PyMySQLSaver 把 Agent checkpoint 保存到 MySQL，演示跨进程保留会话状态。
  运行前需要配置 MySQL 连接，并注意 checkpoint 迁移表不要被单独清空。

- 04_custom_state.py
  演示通过 state_schema 扩展 Agent 状态，并使用 dynamic_prompt 让模型读取自定义状态。
  重点区分“把字段保存进 state”和“把 state 暴露给模型”。

- 05_tool_modify_state.py
  演示工具返回 Command(update=...)，在工具调用过程中修改 Agent 自定义状态。
  重点是 Command(update=...) 如何同时写入 ToolMessage 和业务字段。

- 06_middleware_modify_state.py
  演示通过 before_model 和 after_model middleware，在模型调用前后自动更新统计状态。
  重点观察模型调用次数和工具结果数量如何变化。

- 07_middleware_modify_state.py
  演示 after_model 读取结构化输出，并把订单商品名保存到 state 供后续工具使用。
  重点是把结构化结果转成后续工具可复用的 state。

- 08_context_state.py
  演示 runtime context 和 Agent state 的区别，以及如何把本次 context 合并进持久化 state。
  重点理解 context 只在本次 invoke 生效，而 state 会按 thread_id 保存。

- llm_content/
  演示短期记忆消息进入 LLM 上下文前的管理方式，包括 trim_messages 截断、
  RemoveMessage 删除、手写摘要、内置 SummarizationMiddleware 和自定义保留策略。

运行注意事项：

- InMemorySaver 只把状态保存在当前进程内存中，程序退出后记忆会丢失。
- MySQL 示例需要提前安装 langgraph-checkpoint-mysql 和 PyMySQL，并配置 MYSQL_DATABASE_URL。
- before_model 和 after_model 围绕“每次模型调用”执行；一轮用户请求如果触发工具调用，
  通常会经历两次模型调用。
- llm_content 下的摘要类示例可能在正式回答前额外调用一次模型生成摘要，会增加 API 调用次数。
- 示例会调用真实 LLM，运行前需要确认 .env 中配置了 DeepSeek 相关环境变量。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import short_memory 时触发真实模型调用。
"""
