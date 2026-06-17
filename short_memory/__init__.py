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

运行注意事项：

- InMemorySaver 只把状态保存在当前进程内存中，程序退出后记忆会丢失。
- MySQL 示例需要提前安装 langgraph-checkpoint-mysql 和 PyMySQL，并配置 MYSQL_DATABASE_URL。
- before_model 和 after_model 围绕“每次模型调用”执行；一轮用户请求如果触发工具调用，
  通常会经历两次模型调用。
- 示例会调用真实 LLM，运行前需要确认 .env 中配置了 DeepSeek 相关环境变量。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import short_memory 时触发真实模型调用。
"""
