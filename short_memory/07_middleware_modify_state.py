"""演示 after_model middleware 如何根据结构化输出修改 Agent state。

流程重点：
1. 查询订单后，模型生成 OrderQueryResult 结构化结果。
2. after_model 读取 structured_response，把商品名写入 product_name state。
3. 下一轮查询“这个订单的库存”时，库存工具从 state 中读取 product_name。
"""
from typing import Any, Dict, Union

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from models.init_chat_model.init_chat_model_llm import deepseek_llm


class OrderQueryResult(BaseModel):
    """Order query result"""
    order_id: str
    product_name: str
    price: float
    status: str


class InventoryQueryResult(BaseModel):
    """库存查询响应结构"""
    product_name: str  # 商品名称
    stock_quantity: int  # 库存数量


MOCK_DATABASE = {
    "orders": {
        "order_001": OrderQueryResult(order_id="order_001", product_name="华为手机", price=1999.00, status="已发货"),
        "order_002": OrderQueryResult(order_id="order_002", product_name="苹果电脑", price=2999.00, status="待发货"),
        "order_003": OrderQueryResult(order_id="order_003", product_name="三星显示器", price=3999.00, status="已签收"),

    },
    "inventory": {
        "华为手机": InventoryQueryResult(product_name="华为手机", stock_quantity=50),
        "苹果电脑": InventoryQueryResult(product_name="苹果电脑", stock_quantity=20),
        "三星显示器": InventoryQueryResult(product_name="三星显示器", stock_quantity=30)
    }
}


class OrderState(AgentState, total=False):
    """声明 after_model 会写入的自定义状态字段。"""

    product_name: str


@tool
def get_order_info(order_id: str, runtime: ToolRuntime) -> Command:
    """
    根据订单ID获取订单信息，包括订单ID、商品名称、价格和状态。
    """
    print(f"[工具调用] get_order_info(order_id={order_id})")
    order_info = MOCK_DATABASE["orders"].get(order_id)
    if order_info:
        print(f"[工具结果] 查到订单商品：{order_info.product_name}")
        return Command(
            update={
                # 这里故意不更新 product_name。
                # 本示例要突出 after_model：由 after_model 根据结构化输出写入 state。
                "messages": [ToolMessage(
                    content=f"订单ID: {order_info.order_id}, 商品名称: {order_info.product_name}, 价格: {order_info.price}, 状态: {order_info.status}",
                    tool_call_id=runtime.tool_call_id
                )]
            }

        )
    else:
        print(f"[工具结果] 未找到订单：{order_id}")
        return Command(
            update={
                "messages": [ToolMessage(
                    content="未找到该订单",
                    tool_call_id=runtime.tool_call_id
                )]
            }
        )


@tool
def get_inventory_info(runtime: ToolRuntime) -> Command:
    """
    根据 state 中保存的商品名查询库存。
    """
    product_name = runtime.state.get("product_name")
    print(f"[工具调用] get_inventory_info() 从 state 读取 product_name={product_name}")
    if not product_name:
        return Command(update={
            "messages": [ToolMessage(
                content="还没有保存订单商品名，请先查询订单信息。",
                tool_call_id=runtime.tool_call_id
            )]
        })

    inventory_info = MOCK_DATABASE["inventory"].get(product_name)
    if inventory_info:
        print(f"[工具结果] 查到库存：{inventory_info.product_name} -> {inventory_info.stock_quantity}")
        return Command(update={
            "messages": [ToolMessage(
                content=f"商品名称: {inventory_info.product_name}, 库存数量: {inventory_info.stock_quantity}",
                tool_call_id=runtime.tool_call_id
            )]
        })

    print(f"[工具结果] 未找到库存商品：{product_name}")
    return Command(update={
        "messages": [ToolMessage(
            content=f"未找到商品 {product_name} 的库存信息",
            tool_call_id=runtime.tool_call_id
        )]
    })


@after_model
def manage_order_state(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """模型返回后运行，根据订单结构化结果更新 product_name state。"""
    # after_model 会在每次模型调用后执行。
    # 只有最终生成 OrderQueryResult 时，才把商品名写入 state。
    print("[after_model] 模型调用结束，开始检查 structured_response。")
    if "structured_response" not in state:
        print("[after_model] 当前没有 structured_response，不更新 product_name。")
        return None

    structured_response = state['structured_response']
    print(f"[after_model] structured_response={structured_response}")

    if isinstance(structured_response, OrderQueryResult):
        print(f"[after_model] 识别为订单结果，写入 product_name={structured_response.product_name}")
        return {"product_name": structured_response.product_name}

    print("[after_model] 不是订单结果，保持 product_name 不变。")
    return None


def build_agent():
    """创建使用 after_model 修改 state 的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[get_order_info, get_inventory_info],
        # manage_order_state 会在模型返回后读取 structured_response 并更新 state。
        middleware=[manage_order_state],
        checkpointer=InMemorySaver(),
        state_schema=OrderState,
        response_format=ToolStrategy(Union[OrderQueryResult, InventoryQueryResult])
    )


def main() -> None:
    """运行两组对话，观察 after_model 如何更新 product_name state。"""
    agent = build_agent()
    config = {"configurable": {"thread_id": "session001"}}

    print("\n[第1步] 查询订单 order_001")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "查询订单ID为order_001的信息"}]
    }, config=config)
    print("[第1步结果] 结构化输出:", response["structured_response"])
    print("[第1步状态] after_model 保存的 product_name:", agent.get_state(config=config).values.get("product_name"))
    print("***" * 20)

    print("\n[第2步] 查询“这个订单”的库存，库存工具会从 state 读取 product_name")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "查询这个订单的库存信息"}]
    }, config=config)
    print("[第2步结果] 结构化输出:", response2["structured_response"])
    print("[第2步状态] 当前 product_name:", agent.get_state(config=config).values.get("product_name"))
    print("***" * 20)

    print("\n[第3步] 查询订单 order_002，after_model 会覆盖 product_name")
    response3 = agent.invoke({
        "messages": [{"role": "user", "content": "查询订单ID为order_002的信息"}]
    }, config=config)
    print("[第3步结果] 结构化输出:", response3["structured_response"])
    print("[第3步状态] after_model 更新后的 product_name:", agent.get_state(config=config).values.get("product_name"))
    print("***" * 20)

    print("\n[第4步] 再次查询“这个订单”的库存，应使用 order_002 对应商品")
    response4 = agent.invoke({
        "messages": [{"role": "user", "content": "查询这个订单的库存信息"}]
    }, config=config)
    print("[第4步结果] 结构化输出:", response4["structured_response"])
    print("[第4步状态] 当前 product_name:", agent.get_state(config=config).values.get("product_name"))


if __name__ == "__main__":
    main()
