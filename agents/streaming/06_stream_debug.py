"""演示使用 ``stream_mode="debug"`` 查看完整调试事件。

``debug`` 会同时输出检查点、任务开始和任务结束事件，信息最完整，
但输出也最多，通常只在排查 Agent 执行流程时使用。
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from my_llm import deepseek_llm


@tool
def get_weather(city: str) -> str:
    """获取指定城市的模拟天气。"""
    return f"{city}天气晴朗，温度 25°C。"


def print_debug_event(event: dict[str, Any]) -> None:
    """打印 debug 事件的类型、步骤和关键内容。"""
    event_type = event.get("type")
    payload = event.get("payload", {})

    print(f"步骤：{event.get('step')}，事件类型：{event_type}")

    if event_type == "checkpoint":
        metadata = payload.get("metadata", {})
        print(f"检查点来源：{metadata.get('source')}")
        print(f"下一节点：{payload.get('next', [])}")
    else:
        print(f"任务节点：{payload.get('name')}")
        if event_type == "task_result":
            print(f"执行错误：{payload.get('error')}")

    print("-" * 50)


def main() -> None:
    """流式执行 Agent，并打印检查点和任务调试事件。"""
    agent = create_agent(
        model=deepseek_llm,
        tools=[get_weather],
        system_prompt="你是一个天气助手，需要查询天气时必须调用工具。",
    )

    for event in agent.stream(
        {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
        stream_mode="debug",
    ):
        print_debug_event(event)


if __name__ == "__main__":
    main()
