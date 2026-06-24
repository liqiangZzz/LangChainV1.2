
"""
 异步批量

 abatch 是 batch 的异步版本，适合一次性提交多个输入并等待全部结果。
 它内部会并发处理请求，但最终 responses 的顺序仍然和 questions 的顺序一致。
"""

import asyncio
import time
from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 异步批量入口 —— 准备多个问题并统计耗时
# =====================================================================

async def async_batch_process():
    # 多个问题会作为一个列表传给 abatch，表示本次需要批量处理的输入。
    questions = [
        "什么是深度学习？",
        "什么是强化学习？",
        "什么是迁移学习？"
    ]

    print(f"[{time.strftime('%H:%M:%S')}] 开始批量请求（3个请求并发）")
    start = time.time()

    # abatch 内部会自动并发发送所有请求。
    # 与 batch_as_completed 不同，abatch 会等全部请求完成后一次性返回结果列表。
    # 返回的 responses 会保持输入顺序，responses[0] 对应 questions[0]。
    responses = await deepseek_llm.abatch(questions)

    elapsed = time.time() - start
    print(f"[{time.strftime('%H:%M:%S')}] 全部完成，耗时: {elapsed:.2f}秒")
    print(f"说明: {len(questions)}个请求是并发执行的（如果串行需要约{elapsed * 3:.2f}秒）\n")

    # zip 会按顺序把原始问题和对应回答配对，方便观察每个输入对应的输出。
    for q, r in zip(questions, responses):
        print(f"问：{q}")
        print(f"答：{r.content[:50]}....")
        print("-" * 30)


# =====================================================================
# 2. 运行示例 —— 启动事件循环执行 abatch
# =====================================================================

# 启动事件循环，运行异步批量处理入口。
asyncio.run(async_batch_process())
