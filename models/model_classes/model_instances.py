from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 创建 deepseek llm
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    model="DeepSeek-V4-Flash",
    # 关闭思考模式，使基础示例的响应更直接，并保持与原公共模型配置一致。
    extra_body={"thinking": {"type": "disabled"}},
)


# deepseek_llm2 = ChatOpenAI(
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
#     model="DeepSeek-V4-Flash",
#     关闭思考模式，使基础示例的响应更直接，并保持与原公共模型配置一致。
#     extra_body = {"thinking": {"type": "disabled"}},
# )
