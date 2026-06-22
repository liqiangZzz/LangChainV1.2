import asyncio

from models.init_chat_model.init_chat_model_llm import deepseek_llm

"""
异步调用
非阻塞，适合高并发场景。

这里选择 asyncio 协程，而不是线程：
1. 模型调用主要是在等待网络响应，属于 IO 密集型任务。
2. deepseek_llm 已经提供 ainvoke 异步接口，可以直接交给事件循环调度。
3. 协程在单线程内切换，创建和调度成本通常比线程更低。
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
    """处理单个请求

    Args:
        question: 用户问题文本。
        index: 当前问题在批处理列表中的序号。
    """
    # 这里的打印顺序受网络、模型响应耗时影响，多个请求并发执行时不保证按 index 顺序完成。
    print(f"请求 {index}: 开始发送...")

    # ainvoke 是异步调用，需要在 async 函数中使用 await。
    # await 会让出当前协程的执行权，事件循环可以继续调度其他请求，不会阻塞整个程序。
    # 如果这里只有同步 invoke 接口，才更适合考虑线程池来避免阻塞主流程。
    response = await deepseek_llm.ainvoke(question)

    # 谁先拿到模型响应，谁就会先打印完成日志。
    print(f"请求 {index}: 完成 -> {response.content[:30]}...")
    return response


async def high_concurrency():
    questions = [
        "什么是深度学习？",
        "什么是强化学习？",
        "什么是迁移学习？"
    ]

    # 创建多个协程对象，每个协程都代表一次独立的 ainvoke 请求。
    # 此时只是准备任务列表，真正并发调度发生在下面的 asyncio.gather。
    tasks = [process_one(q, i) for i, q in enumerate(questions)]

    # gather 接收的是可 await 对象；这里传入的是 process_one(...) 创建的协程对象。
    # 它会并发执行多个协程，并等待它们全部完成。
    # 注意：中间的“完成日志”可能乱序，但 gather 返回的 responses 会保持 tasks 的顺序。
    responses = await asyncio.gather(*tasks)

    # 这里按 responses 的顺序处理最终结果，因此最终结果编号仍然对应 questions 的输入顺序。
    for i, r in enumerate(responses):
        print(f"\n最终结果 {i}: {r.content[:50]}...")


# asyncio.run 用来启动事件循环，并运行最外层的异步入口函数。
asyncio.run(high_concurrency())
