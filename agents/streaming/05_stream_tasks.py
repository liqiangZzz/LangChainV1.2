"""演示使用 ``stream_mode="tasks"`` 查看节点任务的开始和结束事件。

任务开始事件包含节点名称、输入和触发条件；任务结束事件包含执行结果或错误。
该模式适合观察 Agent 调度了哪些节点，以及每个节点是否执行成功。
"""

from typing import Any

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


def print_task_event(event: dict[str, Any]) -> None:
    """区分任务开始和结束事件，并打印关键字段。

    Args:
        event: 流式事件数据。
    """
    task_name = event.get("name")

    # tasks 模式中，任务开始事件通常包含 input；任务结束事件通常包含 result/error。
    # 这里用是否存在 input 来区分开始和结束，便于观察节点调度过程。
    if "input" in event:
        print(f"任务开始：{task_name}")
        print(f"触发条件：{event.get('triggers', [])}")
    else:
        print(f"任务结束：{task_name}")
        print(f"执行错误：{event.get('error')}")
        print(f"结果字段：{list(event.get('result', {}))}")

    print("-" * 50)


def main() -> None:
    """流式执行 Agent，并打印所有任务事件。"""
    agent = create_agent(
        model=deepseek_llm,
        tools=[get_weather],
        system_prompt="你是一个天气助手，需要查询天气时必须调用工具。",
    )

    # tasks 模式只关注任务级别事件，适合排查 Agent 调用了哪些节点、是否报错。
    # 它不会像 messages 模式那样逐 token 输出模型文本。
    for event in agent.stream(
        {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
        stream_mode="tasks",
    ):
        print_task_event(event)


if __name__ == "__main__":
    main()
