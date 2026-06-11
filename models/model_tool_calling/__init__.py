"""
模型工具调用示例包。

本包演示的是“聊天模型绑定工具”这一层能力，而不是完整 Agent。
核心流程是：模型先根据用户问题生成 tool_calls，代码再手动执行对应工具，
最后把 ToolMessage 放回 messages，让模型基于工具结果生成最终回答。

主要文件：

- 01_model_single_tool_code.py
  单工具调用示例。模型绑定 get_weather 后，会根据用户问题生成天气工具调用；
  代码读取 response.tool_calls，执行工具，再把工具结果追加回消息列表。

- 02_model_multi_tool_code.py
  多工具调用示例。模型同时绑定股票价格和新闻搜索两个工具；
  代码通过 tools_map 根据工具名选择要执行的工具，并用循环处理多轮工具调用，
  直到模型不再返回 tool_calls，输出最终答案。

学习重点：

- @tool 如何把普通 Python 函数包装为模型可识别的工具。
- bind_tools 只让模型“知道有哪些工具”，不会自动执行工具。
- response.tool_calls 中包含工具名、参数和调用 ID。
- ToolMessage 的 tool_call_id 必须和模型生成的工具调用 ID 对应。
- 多工具场景下通常需要工具映射表和循环执行逻辑。

和 Agent 的区别：

- 本包示例手动处理工具调用，更适合理解底层消息流。
- agent_part 包中的 create_agent 会帮你封装工具执行流程，更接近实际应用写法。
"""
