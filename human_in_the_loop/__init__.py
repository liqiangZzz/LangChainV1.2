"""
Human-in-the-loop 人工介入示例包。

本包用于演示 LangChain Agent 中 `HumanInTheLoopMiddleware` 的常见用法，
包括工具调用前审批、拒绝、参数编辑、人工响应以及一次中断处理多个工具调用。

主要文件：

- 01_human_in_the_loop_middleware.py
  演示 middleware 基础用法：普通工具直接放行，敏感工具执行前触发人工审批。

- 02_hitl_approve_reject_demo.py
  演示 approve / reject 两种人工决策。

- 03_hitl_approve_reject_edit_demo.py
  演示 approve / reject / edit，人工可以在工具执行前修改参数。

- 04_hitl_approve_reject_respond_demo.py
  演示 respond，人工直接模拟工具返回值并恢复 Agent。

- 05_hitl_approve_reject_edit_respond_demo.py
  综合演示 approve / reject / edit / respond 四种决策。

- 06_hitl_multi_descisions_demo.py
  演示一次中断中包含多个工具调用时，按顺序提交多个决策。

- 07_hitl_comprehensive_demo.py
  文件管理助手综合示例，用虚拟文件系统串联 approve / reject / edit / respond 四种决策。

运行注意事项：

- 多数示例会在模块顶层调用真实模型，运行前注意 API 额度、网络和环境变量配置。
- 本包不在 `__init__.py` 中导入示例模块，避免导入包时触发真实 LLM 请求。
"""
