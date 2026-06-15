
"""
 异步批量
"""

import asyncio
import time
from my_llm import deepseek_llm


async def async_batch_process():
    questions = [
        "什么是深度学习？",
        "什么是强化学习？",
        "什么是迁移学习？"
    ]

    print(f"[{time.strftime('%H:%M:%S')}] 开始批量请求（3个请求并发）")
    start = time.time()

    # abatch 内部自动并发发送所有请求
    responses = await deepseek_llm.abatch(questions)

    elapsed = time.time() - start
    print(f"[{time.strftime('%H:%M:%S')}] 全部完成，耗时: {elapsed:.2f}秒")
    print(f"说明: {len(questions)}个请求是并发执行的（如果串行需要约{elapsed * 3:.2f}秒）\n")

    for q, r in zip(questions, responses):
        print(f"问：{q}")
        print(f"答：{r.content[:50]}....")
        print("-" * 30)


asyncio.run(async_batch_process())


