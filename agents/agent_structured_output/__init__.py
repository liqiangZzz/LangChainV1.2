"""
Agent 结构化输出示例包。

本包演示如何通过 create_agent 的 response_format 和 ToolStrategy，让 Agent 在调用工具、
分析消息后返回符合指定 Schema 的 structured_response。示例同时覆盖多种 Schema
定义方式、成功消息定制、错误自动重试、组合模型和自定义错误处理。

主要文件：

1. 01_pydantic_tool_strategy.py
   使用 ToolStrategy 结合 Pydantic BaseModel 和 Field 定义客户分析结果，
   并通过查询、邮件工具生成经过运行时校验的结构化对象。

2. 02_dataclass_tool_strategy.py
   使用 ToolStrategy 结合 Python dataclass、类型标注和 field(metadata=...)
   定义结构化输出，并通过 dataclasses.asdict() 将结果转换为字典。

3. 03_typed_dict_tool_strategy.py
   使用 ToolStrategy 结合 TypedDict、Annotated 和 Literal 描述字典结构；
   通过 total=False 允许字段省略，适合返回普通 dict 的场景。

4. 04_json_schema_tool_strategy.py
   使用 ToolStrategy 直接接收 JSON Schema 字典，配置 properties、enum、
   required 和 additionalProperties，不依赖 Python 数据模型类。

5. 05_tool_strategy_custom_message.py
   使用 ToolStrategy 的 tool_message_content 自定义结构化输出成功后写入消息历史的
   ToolMessage，同时配置 handle_errors=False，说明如何关闭结构化输出错误的自动重试。

6. 06_tool_strategy_error_handling.py
   使用 Union 提供多个候选 Schema，演示模型一次返回多个结构化结果时，
   handle_errors=True、固定字符串和 False 的处理差异。

7. 07_combined_structured_output.py
   使用唯一的组合模型同时承载联系人和活动信息，说明需要同时返回多类数据时，
   应使用嵌套模型而不是 Union 二选一。

8. 08_tool_strategy_custom_error_handler.py
   将自定义函数传给 handle_errors，根据 StructuredOutputValidationError、
   MultipleStructuredOutputsError 等异常返回不同的模型重试提示。

9. 09_tool_strategy_callable_handle_errors.py
   显式使用 Callable[[Exception], str] 定义 handle_errors，自定义结构化输出
   失败后的重试提示。

运行方式：

- 请从项目根目录使用模块方式运行，例如：
  python -m agents.agent_structured_output.01_pydantic_tool_strategy
  python -m agents.agent_structured_output.06_tool_strategy_error_handling
  python -m agents.agent_structured_output.08_tool_strategy_custom_error_handler
  python -m agents.agent_structured_output.09_tool_strategy_callable_handle_errors

运行注意事项：

- 所有示例都会调用真实 DeepSeek 模型并消耗 API 额度。
- ToolStrategy 通常需要模型生成结构化工具调用；出现校验错误并自动重试时，
  模型请求次数和 token 消耗会增加。
- agent.invoke(...) 的返回结果中，messages 通常用于查看完整消息历史和调试过程；
  structured_response 只有在配置 response_format 且结构化输出成功时才会出现，
  适合业务代码读取最终结构化数据。
- 客户查询和邮件发送均为本地模拟工具，不会访问真实客户数据库或发送真实邮件。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 包时触发模型调用。
"""
