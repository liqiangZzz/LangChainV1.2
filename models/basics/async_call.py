import asyncio

from models.init_chat_model.init_chat_model_llm import deepseek_llm

"""
异步调用
非阻塞，适合高并发场景。
"""
# async def async_call():
#     response = await deepseek_llm.ainvoke("请用一句话介绍什么是机器学习")
#     return  response
#
#
# # 运行异步函数
# result = asyncio.run(async_call())
# print(result)


"""
高并发：同时发多个 ainvoke
"""


async def process_one(question: str, index: int):
    """处理单个请求"""
    print(f"请求 {index}: 开始发送...")
    response = await deepseek_llm.ainvoke(question)
    print(f"请求 {index}: 完成 -> {response.content[:30]}...")
    return response


async def high_concurrency():
    questions = [
        "什么是深度学习？",
        "什么是强化学习？",
        "什么是迁移学习？"
    ]

    # 创建多个任务(高并发)
    tasks = [process_one(q, i) for i, q in enumerate(questions)]
    responses = await asyncio.gather(*tasks)

    # 处理所有结果
    for i, r in enumerate(responses):
        print(f"\n最终结果 {i}: {r.content[:50]}...")


asyncio.run(high_concurrency())
