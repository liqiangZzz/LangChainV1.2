"""
LangChain 模型能力示例包。

本包集中演示 LangChain 中和聊天模型直接相关的能力，主要围绕 DeepSeek 模型调用展开。
内容覆盖基础调用、流式输出、批处理、异步调用、工具调用、结构化输出、模型初始化和调用配置。

基础调用示例：

- block_call.py
  演示同步阻塞调用，适合观察 invoke 的基本输入输出。
- stream_output.py
  演示同步流式输出，适合边生成边打印内容。
- batch_process.py
  演示同步批量调用和 batch_as_completed 的使用方式。
- async_call.py
  演示异步 ainvoke，适合学习并发请求模型。
- async_batch_process.py
  演示异步批量调用。
- async_stream_output.py
  演示异步流式输出。

子包说明：

- initchatmodel
  使用 LangChain 的 init_chat_model 统一入口初始化聊天模型。
- modelclass
  使用 ChatDeepSeek、ChatOpenAI 等具体模型类初始化模型。
- model_tool_calling
  演示 bind_tools、tool_calls、ToolMessage 和多工具调用流程。
- model_strcutured_optput
  演示 Pydantic、TypedDict、JSON Schema、JsonOutputParser 等结构化输出方式。
- model_other
  演示推理模型、速率限制、回调、configurable_fields 等模型调用配置。

运行注意事项：

- 多数脚本会触发真实 LLM 调用，运行前需要确认 .env 已配置 DeepSeek API 信息。
- 推理模型示例 deepseek-reasoner 可能比普通 deepseek-chat 更贵，运行前注意额度消耗。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import models 时触发真实模型请求。
"""
