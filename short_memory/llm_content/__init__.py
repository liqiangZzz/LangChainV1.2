"""
LLM 上下文消息管理示例包。

本包聚焦短期记忆变长后，如何在调用模型前压缩、删除或筛选
state["messages"]，避免完整历史持续进入 LLM 上下文。

主要文件：

- 01_trim_message_code.py
  使用 trim_messages 计算保留结果，并通过 REMOVE_ALL_MESSAGES 清空旧消息后重写
  截断后的 messages。示例使用消息条数作为计数策略，避免 DeepSeek 模型 token
  计数接口不可用导致报错。

- 02_delete_message_code.py
  使用 RemoveMessage(id=旧消息.id) 精准删除较早消息，和“清空全部后重写”的 trim
  方案形成对比。

- 03_summarize_message_code.py
  手写摘要流程：在 before_model 中把较早消息摘要成 SystemMessage，再保留最近消息。
  触发摘要时会额外调用一次 DeepSeek。

- 04_summarization_middleware_code.py
  使用 LangChain 内置 SummarizationMiddleware 自动摘要较早消息，并保留最近上下文。

- 05_custom_strategy_code.py
  演示自定义消息保留策略：根据订单号、商品、故障和诉求等业务关键词保留关键消息，
  同时保留最近上下文。

- list_demo.py
  辅助观察 Python 列表切片，例如 list[-n:] 和 list[:-n] 在消息保留策略中的含义。

运行注意事项：

- 示例会调用真实 DeepSeek 模型，运行前需要确认 .env 中配置了 DeepSeek 相关环境变量。
- 摘要类示例可能在正式回答前额外调用一次模型生成摘要，会增加 API 调用次数和额度消耗。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 时触发真实模型调用。
"""
