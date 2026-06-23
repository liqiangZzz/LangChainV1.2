"""
init_chat_model 初始化示例包。

本包演示如何使用 LangChain 的 init_chat_model 通用入口初始化 DeepSeek 聊天模型。
这种方式通过 model 和 model_provider 描述模型来源，比直接实例化具体模型类更统一。
该包同时提供项目公共模型实例，普通示例统一从这里导入，避免重复初始化。

主要文件：

- init_chat_model_llm.py
  使用 init_chat_model 创建公共 deepseek_llm，统一模型为 DeepSeek-V4-Flash，
  并配置 API Key、api_base、model_provider 和关闭思考模式参数。

- invoke_model.py
  导入 deepseek_llm 并发起一次 invoke 调用，用于验证模型是否可用。

特点：

- 入口统一，适合快速切换不同模型提供商。
- 配置较简洁，适合学习和原型示例。
- 和 model_classes 包形成对比：一个使用统一工厂函数，一个使用具体模型类。

运行注意事项：

- invoke_model.py 会调用真实 DeepSeek 模型并消耗 API 额度。
- 运行前需要确认 .env 中已配置 DeepSeek API 信息。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 包时触发模型调用。
"""
