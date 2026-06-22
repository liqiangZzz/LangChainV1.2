"""演示使用 ``stream_mode="values"`` 查看每一步的完整 Agent 状态。

与只返回节点增量的 ``updates`` 不同，``values`` 每次都会返回当前完整状态，
因此消息列表会随着 Agent 执行不断增长。
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@tool
def get_weather(city: str) -> str:
    """获取指定城市的模拟天气。

    Args:
        city: 城市名称。
    """
    return f"{city}天气晴朗，温度 25°C。"


def main() -> None:
    """流式执行 Agent，并打印每一步的完整消息状态。"""
    agent = create_agent(
        model=deepseek_llm,
        tools=[get_weather],
        system_prompt="你是一个天气助手，需要查询天气时必须调用工具。",
    )

    # values 模式每次返回的是“当前完整 state”，不是某个节点刚写入的增量。
    # 因此 messages 会从用户消息开始，随着模型回复、工具结果、最终回答逐步变长。
    for step, state in enumerate(
        agent.stream(
            {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
            stream_mode="values",
        ),
        start=1,
    ):
        messages = state.get("messages", [])
        print(f"\n第 {step} 次完整状态，消息数量：{len(messages)}")

        if messages:
            # 打印当前完整状态中的最后一条消息，方便观察本步骤新增了什么。
            messages[-1].pretty_print()


if __name__ == "__main__":
    main()
