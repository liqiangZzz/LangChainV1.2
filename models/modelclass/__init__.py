"""
使用模型类直接初始化聊天模型

本模块演示了如何通过直接使用 ChatDeepSeek 和 ChatOpenAI 类来初始化 DeepSeek 模型。
这种方式提供了更细粒度的控制和更多的配置选项。

主要文件：
- class_llm.py: 创建 deepseek_llm (ChatDeepSeek) 和 deepseek_llm2 (ChatOpenAI) 两个实例
- init_model.py: 测试和使用初始化后的模型进行对话

特点：
- 直接使用具体的模型类（ChatDeepSeek、ChatOpenAI）
- 可以访问更多高级配置选项
- ChatOpenAI 可以兼容调用 DeepSeek（因为 DeepSeek 兼容 OpenAI API）
- 适合需要精细控制模型行为的场景

两种初始化方式对比：
- init_chat_model: 统一接口，简单易用，适合快速原型开发
- 模型类初始化: 功能更强大，配置更灵活，适合生产环境
"""
