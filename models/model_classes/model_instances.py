from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, GLM_API_KEY, GLM_BASE_URL

# =====================================================================
# 1. 使用具体模型类创建 DeepSeek 模型实例
# =====================================================================

deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    model="DeepSeek-V4-Flash",
    # 关闭思考模式，使基础示例的响应更直接，并保持与原公共模型配置一致。
    extra_body={"thinking": {"type": "disabled"}},
)


# =====================================================================
# 2. 保留 OpenAI 兼容写法示例 —— 按需切换时取消注释
# =====================================================================

# deepseek_llm2 = ChatOpenAI(
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
#     model="DeepSeek-V4-Flash",
#     关闭思考模式，使基础示例的响应更直接，并保持与原公共模型配置一致。
#     extra_body = {"thinking": {"type": "disabled"}},
# )


# =====================================================================
# 3. 使用具体模型类创建 GLM 模型实例
# =====================================================================

glm_llm: BaseChatModel = ChatOpenAI(
    model="glm-5.1",
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
)
