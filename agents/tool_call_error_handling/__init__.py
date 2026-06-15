"""
Agent 工具调用异常处理示例包。

本包演示如何使用 wrap_tool_call middleware 包裹 Agent 的工具执行过程，
把工具抛出的异常转换为 ToolMessage，让模型能够根据错误信息生成最终回答，
避免工具异常直接中断整个 Agent 流程。

主要文件：

1. 01_generic_tool_error_handler.py
   演示统一兜底处理工具异常：
   - 使用模拟股票接口触发外部服务请求失败。
   - 在 middleware 中捕获任意工具异常。
   - 将异常转换为与原工具调用 ID 对应的 ToolMessage。

2. 02_exception_specific_tool_error_handler.py
   演示按照异常类型生成不同的工具错误消息：
   - ConnectionError 用于模拟数据库连接失败。
   - PermissionError 用于模拟权限不足。
   - ToolException 用于表示工单不存在、状态非法等业务校验错误。
   - Exception 分支负责处理未预料到的其他异常。
   - 查询和更新工具通过随机概率模拟不同失败场景。

运行方式：

- 请从项目根目录使用模块方式运行：
  python -m agents.tool_call_error_handling.01_generic_tool_error_handler
  python -m agents.tool_call_error_handling.02_exception_specific_tool_error_handler

运行注意事项：

- 两个示例都会调用真实 DeepSeek 模型并消耗 API 额度。
- 一次工具调用通常包含模型选择工具和模型读取工具结果等多次模型请求。
- 01_generic_tool_error_handler.py 会访问用于模拟失败的外部接口，需要网络连接。
- 02_exception_specific_tool_error_handler.py 包含随机异常，
  同一测试用例多次运行时结果可能不同。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 包时触发模型调用。
"""
