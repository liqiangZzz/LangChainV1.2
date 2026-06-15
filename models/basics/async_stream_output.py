"""
异步流式输出

"""

import asyncio

from models.init_chat_model.init_chat_model_llm import deepseek_llm


async def async_call():
    print("AI 回答：", end="", flush=True)
    async for chunk in deepseek_llm.astream("用三句话介绍北京"):
        print(chunk.content, end="", flush=True)
    print()


asyncio.run(async_call())
