"""
演示 Agent 的 invoke 调用方式。

重点观察：
1. invoke 入参需要传入 messages 列表。
2. messages 中可以包含 system/user 等角色消息。
3. Agent 返回的 response 里会保留完整消息轨迹，包括工具调用过程。
4. create_agent 的 system_prompt 和 invoke 里的 system 消息会共同影响模型回答。
"""
from langchain.agents import create_agent
from langchain.tools import tool

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义工具 —— 为 Agent 提供天气查询能力
# =====================================================================

@tool
def get_weather(city: str) -> str:
    # 使用 @tool 后，Agent 可以在需要时自动调用这个函数。
    """获取指定城市的天气信息。

    Args:
        city: 城市名称。
    """
    return f"{city}的天气为晴朗，25°C。"


# =====================================================================
# 2. 创建 Agent —— 设置默认系统提示词
# =====================================================================

agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    # create_agent 里的 system_prompt 是 Agent 的默认系统提示词。
    # 如果 invoke 里又传入 system 消息，模型会同时看到两条系统级指令。
    system_prompt="你是能查询任何问题的助手"
)


# =====================================================================
# 3. 发起 invoke —— 同时传入 system 和 user 消息
# =====================================================================

response = agent.invoke({  # type: ignore
    # invoke 的核心输入是 messages，格式类似 OpenAI / LangChain 的聊天消息列表。
    "messages": [
        # 这里额外传入 system 消息，用来观察它和 create_agent 的 system_prompt 如何共同影响回答。
        # 这条 system 消息比上面的默认提示词更具体，会把 Agent 限定成“天气查询助手”。
        # 两条系统提示词冲突时，模型通常更容易遵守更具体、更靠近用户问题的那条。
        {'role': 'system', "content": "你是一个天气查询助手，只回答天气相关的问题，其他问题请直接回答：我不清楚这问题答案。"},
        # 切换下面两行 user 消息，可以分别观察“工具调用”和“拒答非天气问题”的效果。
        # {"role": "user", "content": "查询北京的天气"},
        # 当前问题不是天气问题，所以会触发上面 system 消息里的拒答规则。
        {"role": "user", "content": "100加上50等于多少？"}
    ]
})


# =====================================================================
# 4. 查看消息轨迹 —— 观察 Agent 的完整执行链路
# =====================================================================

# 如果只关心最终回答，可以打印最后一条消息的 content。
# print(response)
# print(response["messages"][-1].content)

# 格式化输出所有消息，方便观察 HumanMessage、AIMessage、ToolMessage 的完整执行链路。
for msg in response["messages"]:
    msg.pretty_print()
