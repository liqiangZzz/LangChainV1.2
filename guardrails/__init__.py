"""
Guardrails 安全护栏示例包。

本包用于演示 LangChain Agent 中 PII（个人敏感信息）检测与处理机制，
包括多种策略：脱敏（redact）、遮蔽（mask）、哈希（hash）、阻断（block），
以及自定义中间件实现业务规则拦截。

主要文件：

- 01_redact.py
  演示 PIIMiddleware 使用 redact 策略，将检测到的 PII 替换为 [REDACTED_TYPE] 占位符。

- 02_mask.py
  演示 PIIMiddleware 使用 mask 策略，部分遮蔽 PII（如邮箱显示首尾字符）。

- 03_hash.py
  演示 PIIMiddleware 使用 hash 策略，将 PII 替换为可追溯的哈希值，便于日志审计但不可逆推原文。

- 04_block.py
  演示 PIIMiddleware 使用 block 策略，当检测到 PII 时直接抛出异常阻止执行。

- 05_hitl.py
  演示将 HumanInTheLoopMiddleware 作为 guardrail 使用，实现订单金额 >500 时人工审批。

- 06_custom_guardrails_before_agent.py
  演示自定义 before_agent 中间件，在 Agent 执行前检查输入内容并拦截敏感话题。

- 07_custom_guardrails_after_agent.py
  演示自定义 after_agent 中间件，在 Agent 执行后检查输出内容并过滤敏感信息。

运行方式：

- 请从项目根目录使用模块方式运行，例如：
  python -m guardrails.01_redact
  python -m guardrails.03_hash
  python -m guardrails.05_hitl

运行注意事项：

- 多数示例会在模块顶层调用真实模型，运行前注意 API 额度、网络和环境变量配置。
- 本包不在 `__init__.py` 中导入示例模块，避免导入包时触发真实 LLM 请求。
- guardrails 模块主要处理 PII 检测，实际生产环境需结合业务需求选择合适策略。
"""