"""
模型高级能力与调用配置示例包。

本包放置一些不属于基础 invoke/stream/batch 的模型能力示例，主要用于观察模型初始化参数、
运行时配置、速率限制和回调机制。

主要文件：

- 01_model_reasoner.py
  演示当前 DeepSeek 模型的思考/推理模式，使用 DeepSeek-V4-Flash 并显式开启
  thinking 与 high reasoning_effort。

- 02_model_rate_limiter.py
  演示 InMemoryRateLimiter 的使用方式，通过 requests_per_second、check_every_n_seconds、
  max_bucket_size 控制请求频率，适合学习本地限流。

- 03_model_invocation_config.py
  演示运行时 config 的使用，包括 run_name、tags、metadata、callbacks，
  以及 configurable_fields 动态调整 model、temperature、max_tokens。

学习重点：

- 模型初始化参数和运行时 config 的区别。
- callback 如何观察模型开始、结束、token 使用等信息。
- configurable_fields 如何把模型内部参数暴露给 invoke(config=...) 动态配置。
- 限流器适合控制本地调用节奏，但不会减少单次请求的 token 消耗。

运行注意事项：

- 三个示例都会调用真实 DeepSeek API；速率限制示例会连续发起三次请求。
- 本项目统一使用 DeepSeek-V4-Flash。普通示例通常会显式关闭思考模式；
  推理示例会显式开启思考模式。
- callback 只用于观察运行信息，不要在日志中记录 API Key 等敏感配置。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 包时触发模型调用。
"""
