"""
创建一个静态模型的智能体
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


@tool
def get_weather(city: str):
    # 模拟天气查询
    """获取给定城市的天气。

    Args:
        city: 城市名称。
    """
    return f"{city} 天气晴朗！"


deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
)

agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    system_prompt='你是一个助手，你可以查询城市的天气。'
)


result = agent.invoke({"messages":[{"role":"user","content":"查询北京的天气"}]})
print(type(result))
print(result)
