"""
    阻塞式调用
    一次性发送请求，等待完整结果返回。
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from my_llm import deepseek_llm

# response = deepseek_llm.invoke("请用一句话介绍什么是人工智能")
# print(response)


# 消息列表(字典格式)
# conversations = [
#     {"role": "system", "content": "你是一个有帮助的助手，可以将汉语翻译成英语。"},
#     {"role": "user", "content": "翻译: 我喜欢编程"},
#     {"role": "assistant", "content": "I love programming."},
#     {"role": "user", "content": "翻译: 我喜欢大模型"}
# ]
#
# resp= deepseek_llm.invoke(conversations)
# print(type(resp))
# print(resp.content)  # 输出: I like large models.



# 消息列表(消息对象格式)
conversation = [
    SystemMessage("你是一个有帮助的助手，可以将汉语翻译成英语。"),
    HumanMessage("翻译: 我喜欢编程"),
    AIMessage("I love programming."),
    HumanMessage("翻译: 我喜欢大模型")
]

resp = deepseek_llm.invoke(conversation)
print(type(resp))
print(resp)
print(resp.content)  # 输出: I like large models.
