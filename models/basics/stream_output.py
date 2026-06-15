"""
流式输出
"""
from models.init_chat_model.init_chat_model_llm import deepseek_llm

# 流式输出
print("AI 回答：", end="", flush=True)
for chunk in deepseek_llm.stream("写一首关于夏天的五言绝句"):
    print(chunk.content, end="", flush=True)
print()  # 换行
