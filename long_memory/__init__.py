"""
Agent 长期记忆示例包。

本包围绕 LangGraph Store 展开，演示如何把用户资料、偏好等信息保存为长期记忆，
并结合 checkpointer 的短期记忆、runtime context、工具调用和 MySQL 持久化使用。

学习主线：

1. 使用 InMemoryStore 理解 namespace、key、value 的基础读写。
2. 在 Agent 工具中通过 runtime.store 查询长期记忆。
3. 使用 MySQL 同时持久化短期记忆 checkpointer 和长期记忆 store。
4. 在工具中修改长期记忆，并跨 thread_id 读取同一用户偏好。
5. 综合短期记忆、长期记忆、自定义 state、消息摘要和流式输出。

主要文件：

- 01_long_memory_demo.py
  使用 InMemoryStore 演示长期记忆基础操作，包括 put、get、search、delete 和更新。

- 02_long_memory_in_memory.py
  在 Agent 中使用 InMemoryStore 查询用户资料，同时观察相同 thread_id 下短期记忆
  对后续回答的影响。

- 03_long_memory_in_db.py
  使用 PyMySQLSaver 和 PyMySQLStore，把短期记忆和长期记忆都保存到 MySQL。
  运行前需要配置 MYSQL_DATABASE_URL，并安装 MySQL 相关依赖。

- 04_modify_long_memory_in_tool.py
  演示在工具中通过 runtime.store 写入和读取用户偏好，说明 user_id 与 thread_id
  分别控制长期记忆命名空间和短期记忆会话。

- 05_short_and_long_memory_demo.py
  综合示例：电商客服场景中同时使用 MySQL checkpointer、MySQL store、自定义 state、
  runtime context、SummarizationMiddleware、工具异常处理和流式输出。

运行注意事项：

- InMemoryStore 只在当前进程内保存数据，程序退出后长期记忆会丢失。
- MySQL 示例需要提前配置 MYSQL_DATABASE_URL，并确认数据库和依赖可用。
- 多数示例会调用真实 DeepSeek 模型，运行前需要确认 .env 中配置了 DeepSeek 信息。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import long_memory 时触发真实模型调用。
"""
