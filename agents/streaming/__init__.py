"""
Agent 流式执行示例包。

本包演示如何通过 ``agent.stream()`` 观察 Agent 执行过程中产生的状态更新。
当前示例使用默认的 ``updates`` 模式区分模型节点和工具节点，并逐步打印各节点写入的消息。

主要文件：

- 01_stream_updates.py
  使用客户服务场景演示 ``stream_mode="updates"``，同时说明流式输出模式与
  InMemorySaver 检查点持久化之间的区别。

- 02_stream_checkpoints.py
  使用 InMemorySaver 和固定的 thread_id 保存短期会话状态，
  并通过 ``stream_mode="checkpoints"`` 查看检查点。

- 03_stream_messages.py
  使用 ``stream_mode="messages"`` 接收模型消息片段和运行元数据，
  并实时打印模型生成的文本。

- 04_stream_values.py
  使用 ``stream_mode="values"`` 查看每一步执行后的完整 Agent 状态。

- 05_stream_tasks.py
  使用 ``stream_mode="tasks"`` 查看节点任务的开始、结束、结果和错误。

- 06_stream_debug.py
  使用 ``stream_mode="debug"`` 同时查看检查点和任务调试事件。

- 07_stream_custom.py
  使用 ``stream_mode="custom"`` 和 ``get_stream_writer()``，从工具内部发送
  自定义报告进度事件。

运行方式：

- 请从项目根目录使用模块方式运行：
  python -m agents.streaming.01_stream_updates
  python -m agents.streaming.02_stream_checkpoints
  python -m agents.streaming.03_stream_messages
  python -m agents.streaming.04_stream_values
  python -m agents.streaming.05_stream_tasks
  python -m agents.streaming.06_stream_debug
  python -m agents.streaming.07_stream_custom

运行注意事项：

- 七个示例都会调用真实 DeepSeek 模型并消耗 API 额度。
- Agent 可能调用多个工具并产生多轮模型请求。
- 01 示例中的客户资料、订单历史和促销活动均为本地模拟数据。
- InMemorySaver 只在当前 Python 进程中保存状态，进程退出后检查点会丢失。
- 本包的 __init__.py 只提供说明，不导入示例模块。
"""
