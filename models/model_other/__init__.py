"""
模型高级能力与调用配置示例包。

本包放置一些不属于基础 invoke/stream/batch 的模型能力示例，主要用于观察模型初始化参数、
运行时配置、速率限制和回调机制。

主要文件：

- 01_model_reasoner.py
  对比普通 deepseek-chat 和推理模型 deepseek-reasoner 的初始化方式。
  当前脚本默认调用 deepseek-reasoner，运行会触发真实 API 请求，并且推理模型通常更贵。

- 02_model_rate_limiter.py
  演示 InMemoryRateLimiter 的使用方式，通过 requests_per_second、check_every_n_seconds、
  max_bucket_size 控制请求频率，适合学习本地限流。

- 03_model_lnvocation_config.py
  演示运行时 config 的使用，包括 run_name、tags、metadata、callbacks，
  以及 configurable_fields 动态调整 model、temperature、max_tokens。
  文件名中的 lnvocation 是当前项目已有命名，实际含义是 invocation/config。

学习重点：

- 模型初始化参数和运行时 config 的区别。
- callback 如何观察模型开始、结束、token 使用等信息。
- configurable_fields 如何把模型内部参数暴露给 invoke(config=...) 动态配置。
- 限流器适合控制本地调用节奏，但不会减少单次请求的 token 消耗。
"""
