"""
推理模型

本项目统一使用 deepseek-v4-pro。
普通模型调用会关闭思考模式；推理示例使用同一个模型并显式开启思考模式。
"""

from langchain.chat_models import init_chat_model

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_llm = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}},
)

deepseek_reasoner_llm = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

# print("-------无推理----------")
# print(deepseek_llm.invoke("我有5个苹果，吃了3个，还剩几个？"))

print("-----------------")
print(deepseek_reasoner_llm.invoke("我有5个苹果，吃了3个，还剩几个？"))
