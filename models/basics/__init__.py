"""
聊天模型基础调用示例包。

本包演示 DeepSeek 聊天模型的同步调用、流式输出、批处理和异步调用方式。
这些示例直接调用聊天模型，不涉及 Agent、工具调用或结构化输出。

主要文件：

- 01_blocking_invoke.py
  使用 invoke 发起同步阻塞调用，并演示字典消息和 LangChain 消息对象。

- 02_stream_output.py
  使用 stream 逐块读取并打印模型输出。

- 03_batch_process.py
  使用 batch_as_completed 并发处理多个输入，并按完成顺序读取结果。

- 04_async_high_concurrency_ainvoke.py
  使用 ainvoke 和 asyncio.gather 并发发起多个异步请求。

- 05_async_abatch_process.py
  使用 abatch 批量异步处理多个输入。

- 06_async_astream_output.py
  使用 astream 异步迭代并实时打印模型输出。

运行方式：

- 请从项目根目录使用模块方式运行，例如：
  python -m models.basics.01_blocking_invoke
  python -m models.basics.02_stream_output
  python -m models.basics.06_async_astream_output

运行注意事项：

- 所有示例都会调用真实 DeepSeek 模型并消耗 API 额度。
- 批处理和并发示例会在一次运行中发起多个模型请求。
- 示例在模块顶层执行调用，导入具体示例模块也可能立即触发模型请求。
- 本包的 __init__.py 只提供说明，不导入示例模块。
"""
