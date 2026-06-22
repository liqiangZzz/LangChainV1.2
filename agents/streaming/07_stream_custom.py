"""演示使用 ``custom`` 流模式发送工具内部的自定义进度事件。

``stream_mode="custom"`` 不会自动生成业务事件。工具需要通过
``get_stream_writer()`` 获取当前执行上下文中的写入器，再主动发送进度数据。
"""

import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@tool
def generate_sales_report() -> str:
    """模拟生成销售报告，并持续发送生成进度。"""
    # writer 需要在 Agent/Graph 的执行上下文中获取。
    # 传给 writer 的数据会原样出现在 custom 流中。
    # writer 只能在工具运行时调用；如果脱离 Agent 执行上下文，通常拿不到当前流写入器。
    writer = get_stream_writer()

    writer(
        {
            "report_type": "sales",
            "progress": 0,
            "message": "开始生成销售报告",
        }
    )

    for i in range(1, 5):
        # 这里用 sleep 模拟耗时任务，让 custom 流能连续看到进度事件。
        time.sleep(0.5)
        progress = i * 25
        writer(
            {
                "report_type": "sales",
                "progress": progress,
                "message": f"销售报告生成进度：{progress}%",
            }
        )

    return "销售报告：总收入 150 万元，同比增长 12%"


@tool
def generate_inventory_report() -> str:
    """模拟生成库存报告，并持续发送分析进度。"""
    # custom 事件的结构完全由业务决定，只要调用 writer(...) 即可发送给外层 stream。
    writer = get_stream_writer()

    progress_events = (
        (0, "开始库存分析"),
        (50, "检查当前库存量"),
        (100, "库存报告生成完成"),
    )
    for progress, message in progress_events:
        writer(
            {
                "report_type": "inventory",
                "progress": progress,
                "message": message,
            }
        )
        if progress < 100:
            time.sleep(0.5)

    return "当前库存量为 10000 件，库存充足，无异常"


def print_custom_event(event: dict[str, Any]) -> None:
    """以便于阅读的格式打印工具发送的自定义事件。

    Args:
        event: 流式事件数据。
    """
    # event 就是工具内部 writer(...) 传出的字典，custom 模式不会额外包装成 state。
    print(
        f"报告类型：{event['report_type']}，"
        f"进度：{event['progress']}%，"
        f"状态：{event['message']}"
    )
    print("-" * 40)


def main() -> None:
    """创建报告 Agent，并监听两个工具发送的自定义进度。"""
    reporting_agent = create_agent(
        model=deepseek_llm,
        tools=[generate_sales_report, generate_inventory_report],
        system_prompt="根据用户要求调用对应的报告生成工具；需要多份报告时依次调用。",
    )

    # custom 模式只返回 writer(...) 主动写入的数据，不返回完整 Agent 状态。
    # 如果工具没有调用 writer(...)，外层就收不到业务进度事件。
    for event in reporting_agent.stream(
        {"messages": [{"role": "user", "content": "生成销售报告和库存报告"}]},
        stream_mode="custom",
    ):
        print_custom_event(event)


if __name__ == "__main__":
    main()
