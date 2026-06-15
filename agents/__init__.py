"""
LangChain Agent 示例包。

本包按主题组织 Agent 的基础使用、工具创建、工具异常处理和结构化输出示例。
各子包均提供独立的包级说明，建议先学习 basics，再按需要阅读其他专题。

子包说明：

- basics
  演示 Agent 创建、invoke、系统提示词、动态提示词和模型调用 middleware。

- tool_creation
  演示使用 @tool、Pydantic 模型和 JSON Schema 创建 Agent 工具。

- tool_call_error_handling
  演示使用 wrap_tool_call middleware 捕获工具异常，并转换为 ToolMessage。

- agent_structured_output
  演示 Pydantic、dataclass、TypedDict、JSON Schema 结构化输出，
  以及 ToolStrategy 的消息定制、错误重试、组合模型和自定义错误处理。

运行注意事项：

- 示例会调用真实 DeepSeek 模型并消耗 API 额度。
- Agent 工具调用和错误重试可能产生多轮模型请求。
- 本包的 __init__.py 只提供说明，不导入子包中的示例模块。
"""
