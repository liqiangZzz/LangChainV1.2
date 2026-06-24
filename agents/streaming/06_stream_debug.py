"""演示使用 ``stream_mode="debug"`` 查看完整调试事件。

``debug`` 会同时输出检查点、任务开始和任务结束事件，信息最完整，
但输出也最多，通常只在排查 Agent 执行流程时使用。
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义工具 —— 用一次工具调用生成调试事件
# =====================================================================

@tool
def get_weather(city: str) -> str:
    """获取指定城市的模拟天气。

    Args:
        city: 城市名称。
    """
    return f"{city}天气晴朗，温度 25°C。"


# =====================================================================
# 2. 打印 debug 事件 —— 同时处理 checkpoint 和 task
# =====================================================================

def print_debug_event(event: dict[str, Any]) -> None:
    """打印 debug 事件的类型、步骤和关键内容。

    Args:
        event: 流式事件数据。
    """
    event_type = event.get("type")
    payload = event.get("payload", {})

    print(f"步骤：{event.get('step')}，事件类型：{event_type}")

    if event_type == "checkpoint":
        # checkpoint 事件表示图在某一步保存或推进了状态。
        # metadata.source 能帮助判断检查点来自输入、循环还是节点执行。
        metadata = payload.get("metadata", {})
        print(f"检查点来源：{metadata.get('source')}")
        print(f"下一节点：{payload.get('next', [])}")
    else:
        # task / task_result 事件用于观察具体节点的开始和结束。
        print(f"任务节点：{payload.get('name')}")
        if event_type == "task_result":
            # error 为 None 表示该节点正常结束；非 None 时可据此定位失败节点。
            print(f"执行错误：{payload.get('error')}")

    print("-" * 50)


# =====================================================================
# 3. 执行 debug 流 —— 查看最完整的 Agent 运行细节
# =====================================================================

def main() -> None:
    """流式执行 Agent，并打印检查点和任务调试事件。"""
    agent = create_agent(
        model=deepseek_llm,
        tools=[get_weather],
        system_prompt="你是一个天气助手，需要查询天气时必须调用工具。",
    )

    # debug 模式信息最全，通常用于排查执行流程；日常展示不建议直接使用。
    for event in agent.stream(
        {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
        stream_mode="debug",
    ):
        print_debug_event(event)


# =====================================================================
# 4. 运行示例 —— 用于排查节点执行流程
# =====================================================================

if __name__ == "__main__":
    main()
