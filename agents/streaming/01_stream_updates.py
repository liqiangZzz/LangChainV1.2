"""演示使用 ``stream_mode="updates"`` 查看 Agent 的节点状态更新。

``updates`` 是 ``agent.stream()`` 的默认流模式，会在模型节点或工具节点完成后
立即产出该节点写入的状态增量。本示例仍显式传入该参数，便于理解流模式的作用。
它适合观察 Agent 的执行步骤；如果希望查看完整状态，应使用 ``values``，
如果希望查看持久化检查点事件，则应使用 ``checkpoints``。
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@tool
def query_customer_data(customer_id: str) -> dict[str, Any]:
    """查询客户基本信息。

    Args:
        customer_id: 客户唯一标识。

    Returns:
        包含客户姓名、等级和加入日期的模拟记录。
    """
    return {
        "customer_id": customer_id,
        "name": "张三",
        "level": "VIP",
        "join_date": "2023-01-15",
    }


@tool
def check_order_history(customer_id: str) -> dict[str, Any]:
    """查询客户订单历史。

    Args:
        customer_id: 客户唯一标识。

    Returns:
        包含订单数量和累计消费的模拟数据。
    """
    return {
        "customer_id": customer_id,
        "total_orders": 15,
        "total_spent": 25800.00,
    }


@tool
def get_current_promotions() -> dict[str, Any]:
    """获取当前可用的模拟促销活动。"""
    return {
        "promotions": ["老用户优惠", "会员专属折扣"],
        "valid_until": "2027-01-31",
    }


def create_customer_service_agent() -> CompiledStateGraph:
    """创建带内存检查点的客户服务 Agent。"""
    return create_agent(
        model=deepseek_llm,
        system_prompt=(
            "你是一个客户服务助手，负责查询客户资料、订单历史和促销活动。"
            "需要业务数据时必须调用对应工具，并根据工具结果回答。"
        ),
        tools=[
            query_customer_data,
            check_order_history,
            get_current_promotions,
        ],
        # checkpointer 会按 thread_id 保存执行状态，便于同一会话后续继续使用。
        # 它和 stream_mode 的职责不同：前者负责持久化，后者决定流式输出内容。
        checkpointer=InMemorySaver(),
    )


def print_stream_updates() -> None:
    """流式执行 Agent，并逐步打印模型节点和工具节点产生的消息。"""
    agent = create_customer_service_agent()
    config = {"configurable": {"thread_id": "customer-service-demo"}}
    user_input = {
        "messages": [
            {
                "role": "user",
                "content": "查询客户ID为12345的完整信息和可用优惠活动",
            }
        ]
    }

    # updates 是 agent.stream() 的默认流模式，每次返回：
    # {节点名称: 该节点写入的状态增量}。
    # 这里仍显式写出 stream_mode="updates"，便于学习和切换其他流模式。
    # 处理逻辑必须放在 for 循环内部，否则只能读取流结束后的最后一个 chunk。
    for chunk in agent.stream(
        user_input,
        config=config,
        stream_mode="updates",
    ):
        for node_name, state_update in chunk.items():
            messages = state_update.get("messages", [])

            print(f"\n节点：{node_name}")
            if messages:
                # 模型节点通常产生 AIMessage，工具节点通常产生 ToolMessage。
                messages[-1].pretty_print()
            else:
                # 某些节点更新可能不包含 messages，直接打印更新内容便于调试。
                print(state_update)

            print("-" * 50)


if __name__ == "__main__":
    print_stream_updates()
