"""
LangChain 模型能力示例包。

本包集中演示 LangChain 中和聊天模型直接相关的能力，主要围绕 DeepSeek 模型调用展开。
内容覆盖基础调用、流式输出、批处理、异步调用、工具调用、结构化输出、模型初始化和调用配置。

子包说明：

- basics
  演示 invoke、stream、batch、ainvoke、abatch 和 astream 等基础调用方式。
- init_chat_model
  使用 LangChain 的 init_chat_model 统一入口初始化聊天模型。
- model_classes
  使用 ChatDeepSeek、ChatOpenAI 等具体模型类初始化模型。
- tool_calling
  演示 bind_tools、tool_calls、ToolMessage 和多工具调用流程。
- structured_output
  演示 Pydantic、TypedDict、JSON Schema、JsonOutputParser 等结构化输出方式。
- advanced_features
  演示推理模型、速率限制、回调、configurable_fields 等模型调用配置。

运行注意事项：

- 多数脚本会触发真实 LLM 调用，运行前需要确认 .env 已配置 DeepSeek API 信息。
- 本项目统一使用 deepseek-v4-pro。普通示例通常关闭思考模式；推理示例会显式开启
  思考模式，运行前注意额度消耗。
- 多数示例在模块顶层执行调用，导入具体示例模块也可能立即发起模型请求。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import models 时触发真实模型请求。
"""
