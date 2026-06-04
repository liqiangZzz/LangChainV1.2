"""
LangChain 模型调用模块

本包包含了使用 LangChain 框架调用 DeepSeek 大语言模型的各种示例和工具，包括：
- 基础调用方式：阻塞式调用、流式输出
- 批量处理：同步批量、异步批量、并发处理
- 异步调用：单任务异步、高并发异步请求
- 模型初始化：通过 init_chat_model 和 ChatDeepSeek/ChatOpenAI 类初始化

主要功能模块：
- block_call: 阻塞式调用，等待完整结果返回
- stream_output: 流式输出，逐字显示 AI 回答
- batch_process: 批量处理多个输入
- async_call: 异步调用，适合高并发场景
- async_batch_process: 异步批量处理
- async_stream_output: 异步流式输出
- initchatmodel: 使用 init_chat_model 函数初始化模型
- modelclass: 使用 ChatDeepSeek/ChatOpenAI 类初始化模型
"""
