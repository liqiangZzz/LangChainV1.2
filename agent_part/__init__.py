"""
Agent 示例模块。

这个包用于集中演示 LangChain Agent 的基础用法，学习顺序建议按照文件编号从小到大阅读。
每个示例都尽量保持为可直接运行的脚本，便于观察 Agent 创建、消息输入、工具调用、
系统提示词和 middleware 的实际执行效果。

模块说明：

1. 01_create_agent_staticmodel.py
   演示最基础的静态 Agent 创建方式：
   - 使用固定模型创建 Agent。
   - 使用 @tool 把普通 Python 函数注册为工具。
   - 通过 system_prompt 约束 Agent 的角色和能力范围。
   - 使用 agent.invoke(...) 发起一次同步调用。

2. 02_create_agent_dynamicmodel.py
   演示模型调用 middleware 的使用方式：
   - 使用 wrap_model_call 包装模型调用流程。
   - 根据对话消息数量判断当前请求是否更复杂。
   - 当前项目只有 deepseek-chat，因此示例不会切换到 deepseek-reasoner，
     也不会切换到其他更贵模型，只保留“动态选择调用策略”的结构。
   - Agent 调用工具时可能会多次进入模型调用流程，所以日志可能打印多次。

3. 03_agent_invoke.py
   演示 agent.invoke 的输入和输出结构：
   - invoke 入参以 messages 列表为核心。
   - messages 可以包含 system、user 等角色消息。
   - Agent 返回结果中会保留完整消息轨迹，包括 HumanMessage、AIMessage、
     ToolMessage 等，适合用 pretty_print() 观察执行链路。

4. 04_agent_prompt.py
   演示系统提示词的写法：
   - create_agent 的 system_prompt 可以传字符串，也可以传 SystemMessage。
   - system_prompt 用于设定 Agent 的身份、职责、回答边界和工具使用倾向。
   - 工具函数的函数名、参数类型标注和 docstring 会影响模型判断是否调用工具。

5. 05_agent_dynamic_prompt.py
   演示动态系统提示词：
   - 使用 @dynamic_prompt 根据运行时 context 动态生成系统提示词。
   - dynamic_prompt 包装后的函数需要传给 create_agent 的 middleware 参数，
     不能传给 system_prompt。
   - 示例通过 context={"query_type": "..."} 在普通客服和 VIP 客服提示词之间切换。
   - 动态提示词会在每次模型调用前执行；如果 Agent 中途调用工具，可能触发多次。

子包说明：

- create_tool
  演示使用 @tool、Pydantic 模型和 JSON Schema 创建 Agent 工具。

运行注意事项：

- 这些示例会调用真实大模型，运行前需要确认 .env 中配置了 DeepSeek 相关环境变量。
- 真实模型调用会消耗 API 额度；对比多个场景时，调用次数和 token 消耗也会增加。
- 建议从项目根目录运行脚本或使用模块方式运行，避免出现无法导入 my_llm 的问题。
  例如：
  python -m agent_part.05_agent_dynamic_prompt

设计约定：

- 公共模型实例优先复用 my_llm.py 中的 deepseek_llm。
- 本包的 __init__.py 只提供说明，不导入各示例模块，避免 import agent_part 时触发真实模型调用。
"""
