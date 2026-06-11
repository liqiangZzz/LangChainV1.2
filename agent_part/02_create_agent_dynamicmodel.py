from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool

from my_llm import deepseek_llm


# 1. 定义动态模型选择中间件
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂度选择调用策略。

    当前项目只有 deepseek-chat，因此这里不切换到 deepseek-reasoner 或其他厂商模型。
    如果以后增加便宜的备用模型，只需要在这里替换 selected_model 即可。
    """
    message_count = len(request.state["messages"])
    is_complex_conversation = message_count >= 3

    if is_complex_conversation:
        print("[模型选择] 当前对话较长：继续使用 deepseek-chat，避免切换到推理模型。")
    else:
        print("[模型选择] 当前对话较短：使用 deepseek-chat。")

    selected_model = deepseek_llm

    return handler(request.override(model=selected_model))


# 2.定义工具
@tool
def get_current_location() -> str:
    """获取当前位置。"""
    return "当前位置为北京市。"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气为晴朗，25°C。"

# 2. 创建Agent，并传入动态模型选择中间件
agent = create_agent(
    model=deepseek_llm,  # 当前只使用 deepseek-chat 作为默认模型
    tools=[get_current_location, get_weather],  # 集成工具
    system_prompt="你是一个助手，可以帮助用户回答各种问题。",
    middleware=[dynamic_model_selection]  # 挂载中间件
)

# 模拟一个对话的调用，包含获取当前位置和天气信息
if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "获取当前位置，并告诉我天气情况"}]})
    print(result["messages"][-1].content)
