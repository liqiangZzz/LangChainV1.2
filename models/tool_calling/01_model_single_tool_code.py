from langchain.tools import tool
from langchain_core.messages import HumanMessage

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义工具 —— 给模型暴露天气查询能力
# =====================================================================

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气

    Args:
        location: 地点名称。
    """
    return f' {location}的天气是晴朗的。'


# =====================================================================
# 2. 绑定工具 —— bind_tools 只生成工具调用意图，不自动执行工具
# =====================================================================

model_with_tools = deepseek_llm.bind_tools([get_weather])


# =====================================================================
# 3. 准备消息 —— 构造用户问题并保存对话历史
# =====================================================================

messages = []
human_message = HumanMessage(content="上海天气怎么样？")
messages.append(human_message)

# message 结果
# [HumanMessage(content='上海天气怎么样？', additional_kwargs={}, id='0', type='human')]


# =====================================================================
# 4. 获取工具调用意图 —— 模型返回 tool_calls
# =====================================================================

response = model_with_tools.invoke(messages)
messages.append(response)

# message 结果
# [
#     HumanMessage(content="上海天气怎么样？"),
#     AIMessage(
#         content="好的，我来查询一下上海的天气情况。",
#         tool_calls=[{"name": "get_weather", "args": {"location": "上海"}}]
#     )
# ]


# =====================================================================
# 5. 手动执行工具 —— 把工具结果追加为 ToolMessage
# =====================================================================

if response.tool_calls:
    for tool_call in response.tool_calls:
        if tool_call['name'] == 'get_weather':
            # 调用工具并获取结果
            tool_result = get_weather.invoke(tool_call)
            messages.append(tool_result)

# message 结果
# [
#     HumanMessage(content="上海天气怎么样？"),
#     AIMessage(
#         content="好的，我来帮您查询上海的天气情况。",
#         tool_calls=[{"name": "get_weather", "args": {"location": "上海"}, "id": "call_00_Q2EzMrpyKnVr6J5aRoAc2446", "type": "tool_call"}]
#     ),
#     ToolMessage(content="上海的天气是晴朗的。", name="get_weather", tool_call_id="call_00_Q2EzMrpyKnVr6J5aRoAc2446")
# ]


# =====================================================================
# 6. 二次调用模型 —— 根据工具结果生成最终回答
# =====================================================================

final_response = model_with_tools.invoke(messages)
print("final_response", final_response)
print(final_response.content)
