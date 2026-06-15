"""演示只有一个大模型时，如何创建“动态模型”智能体。

动态模型 middleware 的核心作用，是在每次调用模型前根据当前对话状态选择模型。
当前项目只复用一个公共 DeepSeek 模型，因此本示例不会切换模型厂商或使用更贵的
推理模型，而是为同一个模型准备“快速回答”和“严谨分析”两组配置，再动态选择。

Agent 调用工具前后都会请求模型，所以该 middleware 在一次 invoke 中可能执行多次。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import tool

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# 两个对象仍然使用公共模型的模型名称，只是生成参数不同。
# model_copy 不会修改 init_chat_model_llm.py 中共享的 deepseek_llm，
# 因此不会影响其他示例。
FAST_MODEL = deepseek_llm.model_copy(
    update={
        "temperature": 0.7,
        "max_tokens": 512,
    }
)

PRECISE_MODEL = deepseek_llm.model_copy(
    update={
        "temperature": 0.1,
        "max_tokens": 1024,
    }
)

# 出现这些词时，即使消息数量不多，也优先使用更稳定的模型配置。
COMPLEX_QUERY_KEYWORDS = ("分析", "比较", "规划", "步骤", "原因", "总结")


def get_latest_user_text(messages: list) -> str:
    """从消息列表中提取最近一条用户消息的文本。"""
    for message in reversed(messages):
        # Agent state 中通常是 LangChain 消息对象。
        if getattr(message, "type", None) == "human":
            return str(message.content)

        # 同时兼容示例 invoke 时传入的 {"role": "user", ...} 字典。
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))

    return ""


def is_complex_request(messages: list) -> bool:
    """根据对话轮次、问题长度和关键词判断是否需要严谨分析策略。"""
    # 提取最近一条用户消息，只判断用户当前问题，不把模型回复和工具结果
    # 误当成用户输入参与长度及关键词检查。
    latest_user_text = get_latest_user_text(messages)

    # 条件一：判断 Agent 是否已经进入多步执行流程。
    # 初次请求通常只有一条 HumanMessage；模型决定调用工具后，state 中还会加入
    # AIMessage 和 ToolMessage。因此消息数量达到 3 条时，一般表示 Agent 正在根据
    # 工具结果继续推理，此时使用更稳定的严谨分析配置。
    has_multiple_steps = len(messages) >= 3

    # 条件二：判断用户问题是否较长。
    # 较长的问题通常包含更多背景、限制条件或多个任务，需要模型更仔细地理解上下文。
    # 这里的 60 是教学示例使用的经验阈值，实际项目可以根据业务数据进行调整。
    has_long_question = len(latest_user_text) >= 60

    # 条件三：判断问题中是否包含复杂任务关键词。
    # any() 会依次检查 COMPLEX_QUERY_KEYWORDS 中的关键词；只要有一个关键词出现在
    # 用户问题中就返回 True，例如“分析原因”或“比较并总结”。
    has_complex_keyword = any(
        keyword in latest_user_text
        for keyword in COMPLEX_QUERY_KEYWORDS
    )

    # 三个条件满足任意一个，就将当前请求视为复杂请求。
    # middleware 随后会为该请求选择低 temperature、较大 max_tokens 的严谨分析配置；
    # 如果全部不满足，则选择响应更灵活、输出更短的快速回答配置。
    return has_multiple_steps or has_long_question or has_complex_keyword


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """在每次模型调用前，动态选择公共 DeepSeek 模型的运行配置。"""
    messages = request.state["messages"]

    if is_complex_request(messages):
        selected_model = PRECISE_MODEL
        strategy_name = "严谨分析"
    else:
        selected_model = FAST_MODEL
        strategy_name = "快速回答"

    print(
        f"[动态模型] 使用 {selected_model.model_name} / {strategy_name}策略，"
        f"当前消息数：{len(messages)}"
    )

    # override 只修改当前这一次模型请求，不会改变 Agent 的默认模型。
    return handler(request.override(model=selected_model))


@tool
def get_current_location() -> str:
    """获取用户当前所在城市。"""
    return "当前位置为北京市。"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气为晴朗，25°C。"


def build_agent():
    """创建带动态模型选择 middleware 的天气查询 Agent。"""
    return create_agent(
        # deepseek_llm 是默认模型；middleware 会在每次调用前覆盖本次使用的配置。
        model=deepseek_llm,
        tools=[get_current_location, get_weather],
        system_prompt=(
            "你是一个天气助手。"
            "当用户询问当前位置的天气时，先调用工具获取位置，"
            "再根据位置调用天气工具，最后用简洁中文回答。"
        ),
        middleware=[dynamic_model_selection],
    )


def ask_agent(agent, user_query: str) -> str:
    """调用 Agent，并返回最后一条模型消息的文本内容。"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]}
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    weather_agent = build_agent()
    query = "请获取我的当前位置，并查询当地天气。"

    print(f"用户问题：{query}")
    print(f"Agent 回答：{ask_agent(weather_agent, query)}")
