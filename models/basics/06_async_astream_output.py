"""
异步流式输出

astream 是 stream 的异步版本，适合在 async 函数中边生成边展示模型输出。
它不会等完整回答生成完再返回，而是持续返回一个个消息片段 chunk。
"""

import asyncio

from models.init_chat_model.init_chat_model_llm import deepseek_llm


async def async_call():
    print("AI 回答：", end="", flush=True)

    # async for 用来遍历异步迭代器。
    # deepseek_llm.astream(...) 会不断产出模型生成过程中的片段，而不是一次性返回完整结果。
    async for chunk in deepseek_llm.astream("用三句话介绍北京"):
        # chunk.content 通常是当前生成的文本片段。
        # end="" 表示不自动换行，flush=True 让终端尽快显示每个流式片段。
        print(chunk.content, end="", flush=True)

    # 流式输出结束后补一个换行，避免终端提示符和最后一段文本挤在一起。
    print()


# 启动事件循环，运行异步流式输出示例。
asyncio.run(async_call())
