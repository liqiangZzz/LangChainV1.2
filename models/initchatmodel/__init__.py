"""
init_chat_model 初始化示例包。

本包演示如何使用 LangChain 的 init_chat_model 通用入口初始化 DeepSeek 聊天模型。
这种方式通过 model 和 model_provider 描述模型来源，比直接实例化具体模型类更统一。

主要文件：

- init_chat_model_llm.py
  使用 init_chat_model 创建 deepseek_llm，并配置 API Key、Base URL 和 model_provider。

- init_mode.py
  导入 deepseek_llm 并发起一次 invoke 调用，用于验证模型是否可用。

特点：

- 入口统一，适合快速切换不同模型提供商。
- 配置较简洁，适合学习和原型示例。
- 和 modelclass 包形成对比：一个使用统一工厂函数，一个使用具体模型类。
"""
