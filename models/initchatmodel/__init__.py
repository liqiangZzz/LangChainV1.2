"""
使用 init_chat_model 函数初始化聊天模型

本模块演示了如何使用 LangChain 的 init_chat_model 通用函数来初始化 DeepSeek 模型。
这种方式提供了一种统一的接口来初始化不同提供商的聊天模型。

主要文件：
- init_chat_model_llm.py: 定义并初始化 deepseek_llm 实例，配置 API Key 和 Base URL
- init_mode.py: 测试和使用初始化后的模型进行对话

特点：
- 使用 langchain.chat_models.init_chat_model 统一接口
- 支持多模型提供商（通过 model_provider 参数指定）
- 配置简洁，适合快速切换不同模型提供商
"""
