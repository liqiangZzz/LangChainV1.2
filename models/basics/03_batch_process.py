"""
批量处理
一次性处理多个输入，内部可能会并行处理。
"""
from langchain_core.runnables.utils import Output

from models.init_chat_model.init_chat_model_llm import deepseek_llm

# =====================================================================
# 1. batch 示例 —— 等全部请求完成后按输入顺序返回
# =====================================================================

# batch 会批量处理多个问题，并按输入列表的顺序返回结果。
# 即使内部可能并发执行，最终 responses 的顺序仍然对应下面三个问题的顺序。
# responses:list[Output] = deepseek_llm.batch([
#     "为什么鹦鹉的羽毛是彩色的？",
#     "飞机是如何飞行的？",
#     "什么是量子计算？"
# ])
#
# for response in responses:
#     print(response.content)


# =====================================================================
# 2. batch_as_completed 示例 —— 谁先完成就先返回谁
# =====================================================================

# batch_as_completed 会在每个请求完成后立即 yield 结果。
# 因为不同请求耗时不同，所以输出顺序可能和输入顺序不一致。
# 返回的 index 表示该结果对应原始输入列表中的第几个问题。
responses: list[Output] = deepseek_llm.batch_as_completed([
    "为什么鹦鹉的羽毛是彩色的？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
], config={'max_concurrency': 2})


# =====================================================================
# 3. 读取结果 —— 使用 index 对应原始输入位置
# =====================================================================

# max_concurrency=2 表示最多同时处理 2 个请求，不代表输出一定按前两个、后一个的顺序出现。
for index, response in responses:
    print(f"索引 {index} : {response.content[:50]}...")
