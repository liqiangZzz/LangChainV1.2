"""
模型类初始化示例包。

本包演示不通过 init_chat_model，而是直接使用具体模型类创建聊天模型实例。
当前示例使用 ChatDeepSeek 和 ChatOpenAI 两种类来连接 DeepSeek 兼容接口。

主要文件：

- model_instances.py
  创建两个模型实例：
  deepseek_llm 使用 ChatDeepSeek；
  deepseek_llm2 使用 ChatOpenAI，并通过 DeepSeek 的 base_url 访问 DeepSeek 模型。

- invoke_model.py
  导入已创建好的模型实例并发起一次 invoke 调用，用于验证模型连接和基础对话。

特点：

- 直接使用具体模型类，参数更直观。
- ChatOpenAI 可用于调用兼容 OpenAI API 格式的 DeepSeek 服务。
- 适合理解不同模型类的初始化差异。

和 init_chat_model 包的区别：

- init_chat_model 是统一入口，适合快速切换不同 provider。
- 具体模型类写法更显式，适合需要了解底层类和参数的学习场景。

运行注意事项：

- invoke_model.py 会调用真实 DeepSeek 模型并消耗 API 额度。
- ChatOpenAI 示例使用 DeepSeek 的 OpenAI 兼容接口，并不代表调用 OpenAI 模型。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 包时触发模型调用。
"""
