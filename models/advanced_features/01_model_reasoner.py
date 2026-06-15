"""
推理模型
"""

from langchain.chat_models import init_chat_model

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
)

deepseek_llm2 = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
)

# print("-------无推理----------")
# print(deepseek_llm.invoke("我有5个苹果，吃了3个，还剩几个？"))

print("-----------------")
print(deepseek_llm2.invoke("我有5个苹果，吃了3个，还剩几个？"))
