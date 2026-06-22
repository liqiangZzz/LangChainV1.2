"""
Agent 基础能力示例包。

本包按照编号演示 LangChain Agent 的创建、调用、系统提示词和 middleware。
建议从 01 开始顺序阅读，以逐步理解 Agent 的基本执行流程。

主要文件：

1. 01_create_agent_static_model.py
   使用固定的 DeepSeek 模型、天气工具和 system_prompt 创建基础 Agent。

2. 02_create_agent_dynamic_model.py
   使用 wrap_model_call middleware，根据消息数量、问题长度和关键词，
   在同一个 DeepSeek 模型的两组生成参数之间动态选择。

3. 03_agent_invoke.py
   演示 invoke 的 messages 输入格式，以及 HumanMessage、AIMessage、
   ToolMessage 等完整消息轨迹。

4. 04_agent_prompt.py
   演示字符串和 SystemMessage 系统提示词，以及工具签名和 docstring
   对模型选择工具的影响。

5. 05_agent_dynamic_prompt.py
   使用 dynamic_prompt 和运行时 context，在普通客服与 VIP 客服提示词之间切换。

运行方式：

- 请从项目根目录使用模块方式运行，例如：
  python -m agents.basics.01_create_agent_static_model
  python -m agents.basics.02_create_agent_dynamic_model
  python -m agents.basics.05_agent_dynamic_prompt

运行注意事项：

- 所有示例都会调用真实 DeepSeek 模型并消耗 API 额度。
- Agent 调用工具时可能产生多轮模型请求。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 包时触发模型调用。
"""
