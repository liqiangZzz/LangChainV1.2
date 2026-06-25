"""
演示 approve / reject / edit 三种人工决策。

示例通过批量修改商品折扣展示 edit 的作用：
人工可以在工具真正执行前修改商品 ID 或折扣率。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm

# =====================================================================
# 1. 准备示例数据和工具 —— 用商品改价模拟敏感批量操作
# =====================================================================

PRODUCTS = {
    "P001": {"name": "AirPods Pro", "price": 1000},
    "P002": {"name": "iPhone 15", "price": 2000},
    "P003": {"name": "MacBook Air", "price": 3000},
}


@tool
def batch_update_discount(product_ids: list, discount_rate: float) -> str:
    """
    批量更新商品折扣。
    Args:
        product_ids: list - 商品ID列表
        discount_rate: float - 折扣率
    Returns:
        str - 操作结果
    """

    results = []
    for pid in product_ids:
        if pid in PRODUCTS:
            product_name = PRODUCTS[pid]["name"]
            price = PRODUCTS[pid]["price"]
            new_price = price * discount_rate
            results.append(f"{product_name}: 原价¥{price} → ¥{new_price}")
    return "折扣更新完成：\n" + "\n".join(results)


@tool
def query_product(product_id: str) -> str:
    """
    查询商品信息
    Args:
        product_id: str - 商品ID
    Returns:
        str - 商品信息¬
    """
    p = PRODUCTS.get(product_id)
    if not p:
        return "未找到"

    product_name = p["name"]
    price = p["price"]
    return f"{product_name} 当前售价 ¥{price}"


# =====================================================================
# 2. 创建 Agent —— 批量改折扣支持 approve / reject / edit
# =====================================================================

agent = create_agent(
    model=deepseek_llm,
    tools=[batch_update_discount, query_product],
    middleware=[
        # 本示例重点演示 edit：人工可以在执行工具前修改工具参数。
        HumanInTheLoopMiddleware(
            interrupt_on={
                "query_product": False,
                "batch_update_discount": {
                    # approve 表示继续执行，reject 表示拒绝，edit 表示修改工具参数后继续。
                    "allowed_decisions": ["approve", "reject", "edit"],
                    # description 会出现在中断信息中，方便人工确认当前风险操作。
                    "description": "请确认是否更新折扣率?",
                },
            },
            description_prefix="需要人工介入，请确认操作。",
        ),
    ],
    # 中断后需要 checkpointer 保存执行现场，后续才能用同一个 thread_id 恢复。
    checkpointer=InMemorySaver(),
    system_prompt="你是一个智能助手，可以回答用户问题。"
)


# =====================================================================
# 3. 发起调用 —— 让模型生成批量折扣更新工具调用
# =====================================================================

# thread_id 用来标识同一条可恢复会话；首次调用和恢复调用必须保持一致。
config = {"configurable": {"thread_id": "session_123"}}

# 首次调用会让模型生成批量改价工具调用；真正执行前会被中间件拦截。
result = agent.invoke({  # type: ignore
    "messages":
        [
            {
                "role": "user",
                "content": "将 P001 和P002 商品给我打5折"
            }
        ]
}, config=config, version="v2")

if result.interrupts:

    print("触发中断 result ---> ", result)

    # =====================================================================
    # 4. 展示中断信息 —— 查看原始商品 ID 和折扣率
    # =====================================================================

    # action_requests 中保存了被拦截的工具调用名称、参数和描述。
    req = result.interrupts[0].value["action_requests"][0]

    print("Agent 暂停！请确认操作。")
    print(f"---待调用工具：{req['name']}")
    print(f"---工具参数：{req['args']}")
    print(f"---中断描述：{req['description']}")

    print("-" * 80)
    # review_configs 中保存当前中断允许的人工决策类型。
    allowed_decisions = result.interrupts[0].value["review_configs"][0]["allowed_decisions"]

    # =====================================================================
    # 5. 人工决策 —— edit 可以改工具名和 args 后再执行
    # =====================================================================

    while True:

        decision = input(f"    请输入确认操作，从{allowed_decisions}中选择一种：").strip().lower()
        if decision == "approve":
            # approve：不修改参数，直接执行模型原本生成的工具调用。
            command_cmd = Command(
                resume={"decisions": [{"type": "approve"}]}
            )
            break

        elif decision == "reject":
            # reject：拒绝执行本次工具调用，并把拒绝理由交回给模型。
            reason = input("请输入拒绝理由：").strip()
            if not reason:
                reason = "用户拒绝了该操作。"

            command_cmd = Command(
                resume={"decisions": [{"type": "reject", "message": f"原因：{reason}"}]}
            )
            break
        elif decision == "edit":
            # edit：人工重新指定商品 ID 和折扣率，再恢复执行修改后的工具调用。
            product_ids_text = input("请输入新的商品ID列表，用逗号分隔：").strip()
            new_product_ids = [
                product_id.strip()
                for product_id in product_ids_text.split(",")
                if product_id.strip()
            ]
            new_discount_rate = float(input("请输入折扣：").strip())

            command_cmd = Command(
                resume={"decisions": [{
                    "type": "edit",
                    "edited_action": {
                        "name": "batch_update_discount",
                        "args": {"product_ids": new_product_ids, "discount_rate": new_discount_rate}
                    }
                }]}
            )
            break
        else:
            print(f"    请输入正确的确认操作！从{allowed_decisions}中选择一种。当前的输入：{decision}")

    # =====================================================================
    # 6. 恢复执行 —— 使用人工确认或编辑后的决策继续流程
    # =====================================================================

    # 使用 Command(resume=...) 恢复同一个 thread_id 的中断流程。
    result2 = agent.invoke(  # type: ignore
        command_cmd,
        config=config,
        version="v2"
    )

    print("Agent 恢复执行，result:", result2)
    print("[Agent 回复]：", result2.value["messages"][-1].content)

else:
    # 没有触发中断时，直接读取最终消息。
    print(" result ", result.value["messages"][-1].content)
