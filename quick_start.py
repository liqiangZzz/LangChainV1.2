"""
使用 Agent查询天气
"""
from langchain.tools import tool
from langchain.agents import create_agent

from models.init_chat_model.init_chat_model_llm import deepseek_llm

@tool
def get_weather(city: str):
    # 模拟天气查询
    """获取给定城市的天气。"""
    return f"{city} 天气晴朗！"


# 创建agent
agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    system_prompt='你是一个助手，你可以查询城市的天气。'
)

# 调用agent
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "查询上海的天气"}]}
)


print(resp)