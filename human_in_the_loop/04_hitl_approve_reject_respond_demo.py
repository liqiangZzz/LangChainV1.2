"""
演示 respond 人工决策。

示例把 ask_customer 设计成占位工具，由人工输入模拟客户回复；
respond 会把人工回复作为工具结果交回给 Agent 继续执行。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义示例工具 —— ask_customer 由人工 respond 提供结果
# =====================================================================

@tool
def ask_customer(question: str) -> str:
    """
    向客户询问确认信息（占位工具——由人工回复来实现）
    注意：这个工具本身不做任何事，它的"返回值"就是人工的回复
    """
    # 正常流程下这个函数体不会被执行——respond 决策会跳过工具执行
    # 但如果有人错误地配置为 approve，这个函数会被调用
    raise RuntimeError("ask_customer 必须由人工回复，不允许直接执行！")


@tool
def query_order(order_id: str) -> str:
    """查询订单信息"""
    return f"订单 {order_id}：已付款，待发货，金额 ¥299.00"


@tool
def update_shipping_address(order_id: str, address: str) -> str:
    """更新收货地址"""
    return f"订单 {order_id} 的收货地址已更新为：{address}"


# =====================================================================
# 2. 创建 Agent —— 只拦截 ask_customer，聚焦 respond
# =====================================================================

agent = create_agent(
    model=deepseek_llm,
    tools=[ask_customer, query_order, update_shipping_address],
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            # respond 适合“工具本身不执行，由人工直接提供工具返回值”的场景。
            "ask_customer": {
                "allowed_decisions": ["respond"],
                "description": "人工介入：请回复确认的信息",
            },
            "query_order": False,
            # 这里不拦截真实更新工具，方便聚焦演示 ask_customer 的人工回复。
            "update_shipping_address": False,
        },
        description_prefix="消息中断，需要人工介入。"
    )],
    # checkpointer 保存中断现场；respond 后要用同一个 thread_id 继续执行。
    checkpointer=InMemorySaver(),
    system_prompt=(
        "你是电商客服助手。如果需要向客户确认信息，请使用 ask_customer 工具。"
        "收到客户回复后，再执行后续操作（如修改地址、处理退款等）。"
    )
)


# =====================================================================
# 3. 发起调用 —— 让 Agent 先向客户确认新地址
# =====================================================================

# thread_id 标识一次可恢复的对话流程。
config = {"configurable": {"thread_id": "session_id001"}}

# 首次调用通常会触发 ask_customer 中断，等待人工模拟客户回复。
result = agent.invoke({  # type: ignore
    "messages": [{
        "role": "user",
        "content": "订单 ORD-001 的客户想修改收货地址，帮我跟客户确认一下新地址"
    }]
}, config=config, version="v2")


# =====================================================================
# 4. 处理中断循环 —— 可能连续处理多个 respond
# =====================================================================

# 外层循环处理 Agent 层面的中断：恢复后如果又产生新中断，会继续处理。
while result.interrupts:
    print("触发中断 --> ", result)

    # 1.输出中断信息
    req = result.interrupts[0].value["action_requests"][0]
    print("Agent 暂停！请确认操作。")
    print(f"---待调用工具：{req['name']}")
    print(f"---工具参数：{req['args']}")
    print(f"---中断描述：{req['description']}")

    print("-" * 80)

    allowed_decisions = result.interrupts[0].value["review_configs"][0]["allowed_decisions"]

    print(f"---允许的人工决策：{allowed_decisions}")

    # =====================================================================
    # 5. 人工输入回复 —— respond 直接成为工具返回值
    # =====================================================================

    # 内层循环处理本次中断的人工输入校验，直到拿到合法决策。
    while True:
        decision = input(f"\n请输入决策类型 (选择其中一个：{allowed_decisions}): ").strip().lower()
        if decision not in allowed_decisions:
            print(f"无效输入，请输入 {allowed_decisions} 中的一个，当前输入：{decision}")
            continue

        if decision == "approve":
            command_cmd = Command(
                resume={"decisions": [{"type": "approve"}]}
            )
            break
        elif decision == "reject":
            reason = input("请输入拒绝理由：").strip()
            if not reason:
                reason = "用户拒绝了该操作。"
            command_cmd = Command(
                resume={"decisions": [{"type": "reject", "message": f"原因：{reason}"}]}
            )
            break
        elif decision == "respond":
            # respond：把人工输入作为 ask_customer 的工具返回值交回给模型。
            confirm = input("请输入确认信息: ").strip()
            if not confirm:
                print("确认信息不能为空，请重新输入。")
                continue

            command_cmd = Command(
                resume={"decisions": [{"type": "respond", "message": confirm}]}
            )
            break
        else:
            print(f"暂未处理该决策类型：{decision}")

    # =====================================================================
    # 6. 恢复执行 —— 把人工回复交回 Agent
    # =====================================================================

    # 恢复后把新结果覆盖回 result，外层循环才能判断是否还有后续中断。
    result = agent.invoke(command_cmd, config=config, version="v2")
    print("Agent 继续执行...")
    print("恢复执行后 result:", result)

print(f"[Agent回复]: {result.value['messages'][-1].content}")
