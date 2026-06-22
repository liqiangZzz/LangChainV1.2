"""
批量处理
一次性处理多个输入，内部可能会并行处理。
"""
from langchain_core.runnables.utils import Output

from models.init_chat_model.init_chat_model_llm import deepseek_llm

#  批量处理多个问题
# responses:list[Output] = deepseek_llm.batch([
#     "为什么鹦鹉的羽毛是彩色的？",
#     "飞机是如何飞行的？",
#     "什么是量子计算？"
# ])
#
# for response in responses:
#     print(response.content)


# 每个请求完成后立即 yield 结果，结果可能乱序，但包含索引信息。
responses: list[Output] = deepseek_llm.batch_as_completed([
    "为什么鹦鹉的羽毛是彩色的？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
], config={'max_concurrency': 2})

for index, response in responses:
    print(f"索引 {index} : {response.content[:50]}...")
