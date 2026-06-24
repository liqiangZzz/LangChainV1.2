"""
推理模型

本项目统一使用 DeepSeek-V4-Flash。
普通模型调用会关闭思考模式；推理示例使用同一个模型并显式开启思考模式。
"""

from langchain.chat_models import init_chat_model

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# =====================================================================
# 1. 初始化普通模型 —— 关闭思考模式作为对照
# =====================================================================

deepseek_llm = init_chat_model(
    model="DeepSeek-V4-Flash",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}},
)


# =====================================================================
# 2. 初始化推理模型 —— 开启 thinking 并提高 reasoning_effort
# =====================================================================

deepseek_reasoner_llm = init_chat_model(
    model="DeepSeek-V4-Flash",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)


# =====================================================================
# 3. 发起调用 —— 对比普通回答和推理回答
# =====================================================================

# print("-------无推理----------")
# print(deepseek_llm.invoke("我有5个苹果，吃了3个，还剩几个？"))

print("-----------------")
print(deepseek_reasoner_llm.invoke("我有5个苹果，吃了3个，还剩几个？"))
