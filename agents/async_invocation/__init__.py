"""
Agent 异步调用示例包。

本包演示如何使用 ``agent.ainvoke()`` 在异步函数中调用 LangChain Agent。
当前旅行规划示例会让模型根据用户问题选择天气、交通和景点工具，并异步等待
包含完整消息历史的 Agent 状态。

主要文件：

- 01_basic_ainvoke.py
  创建旅行规划 Agent，并通过 ``await agent.ainvoke(...)`` 执行一次异步调用。

运行方式：

- 请从项目根目录使用模块方式运行：
  python -m agents.async_invocation.01_basic_ainvoke

运行注意事项：

- 示例会调用真实 DeepSeek 模型并消耗 API 额度。
- 一次 Agent 执行可能包含工具调用前后的多轮模型请求。
- 天气、交通和景点结果均为本地模拟数据，不代表真实查询结果。
- 本包的 __init__.py 只提供说明，不导入示例模块。
"""
