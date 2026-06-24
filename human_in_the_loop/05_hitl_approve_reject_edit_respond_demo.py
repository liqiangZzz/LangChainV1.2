"""
演示 approve / reject / edit / respond 四种人工决策的组合用法。

示例先用 respond 模拟客户回复地址，再对真实更新地址工具进行
approve、reject 或 edit 审批，展示多轮人机协同流程。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义示例工具 —— respond 获取地址，真实工具执行更新
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
# 2. 创建 Agent —— 同时演示 respond 与 approve/reject/edit
# =====================================================================

agent = create_agent(
    model=deepseek_llm,
    tools=[ask_customer, query_order, update_shipping_address],
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            # respond：人工直接模拟客户回复，作为 ask_customer 的工具返回值。
            "ask_customer": {
                "allowed_decisions": ["respond"],
                "description": "人工介入：请回复确认的信息",
            },
            "query_order": False,
            # approve/reject/edit：对真实地址更新操作做人工审批或参数修正。
            "update_shipping_address": {
                "allowed_decisions": ["approve", "reject", "edit"],
                "description": "人工介入：请确认是否更新收货地址",
            },
        },
        description_prefix="消息中断，需要人工介入。"
    )],
    # checkpointer 保存中断前后的图状态，保证多次人工介入可以连续恢复。
    checkpointer=InMemorySaver(),
    system_prompt=(
        "你是电商客服助手。如果需要向客户确认信息，请使用 ask_customer 工具。"
        "收到客户回复的新地址后，必须调用 update_shipping_address 工具更新订单收货地址。"
        "如果人工审核对 update_shipping_address 的参数进行了 edit，说明人工已确认最终地址，"
        "你必须以 edit 后的工具执行结果为准，不要再使用客户原始回复重新更新地址。"
        "update_shipping_address 成功后，直接总结最终更新结果，不要再次调用更新地址工具。"
    )
)


# =====================================================================
# 3. 发起调用 —— 先确认地址，再准备更新订单
# =====================================================================

# thread_id 是恢复中断的会话标识；多轮 resume 必须使用同一个值。
config = {"configurable": {"thread_id": "session_id001"}}

# 首次调用通常先触发 ask_customer，让人工用 respond 提供客户的新地址。
result = agent.invoke({  # type: ignore
    "messages": [{
        "role": "user",
        "content": "订单 ORD-001 的客户想修改收货地址，帮我跟客户确认一下新地址"
    }]
}, config=config, version="v2")


# =====================================================================
# 4. 处理多轮中断 —— 先 respond，再审批真实更新工具
# =====================================================================

# 外层循环处理 Agent 可能连续产生的多个中断。
# 例如：先 respond 客户地址，再 approve/reject/edit 真正的更新地址工具。
while result.interrupts:
    print("触发中断 --> ", result)

    # action_requests 展示当前被拦截的工具名、参数和说明。
    req = result.interrupts[0].value["action_requests"][0]
    print("Agent 暂停！请确认操作。")
    print(f"---待调用工具：{req['name']}")
    print(f"---工具参数：{req['args']}")
    print(f"---中断描述：{req['description']}")

    print("-" * 80)

    allowed_decisions = result.interrupts[0].value["review_configs"][0]["allowed_decisions"]

    print(f"---允许的人工决策：{allowed_decisions}")

    # =====================================================================
    # 5. 人工决策分流 —— 四种策略在同一段循环里统一处理
    # =====================================================================

    # 内层循环只负责本次中断的人工输入，直到构造出合法的 Command。
    while True:
        decision = input(f"\n请输入决策类型 (选择其中一个：{allowed_decisions}): ").strip().lower()
        if decision not in allowed_decisions:
            print(f"无效输入，请输入 {allowed_decisions} 中的一个，当前输入：{decision}")
            continue

        if decision == "approve":
            # approve：按模型原始参数执行当前工具调用。
            command_cmd = Command(
                resume={"decisions": [{"type": "approve"}]}
            )
            break
        elif decision == "reject":
            # reject：拒绝当前工具调用，并把原因作为工具错误结果返回给模型。
            reason = input("请输入拒绝理由：").strip()
            if not reason:
                reason = "用户拒绝了该操作。"
            command_cmd = Command(
                resume={"decisions": [{"type": "reject", "message": f"原因：{reason}"}]}
            )
            break
        elif decision == "respond":
            # respond：不执行 ask_customer 函数体，直接把人工回复作为工具结果。
            confirm = input("请输入确认信息: ").strip()
            if not confirm:
                print("确认信息不能为空，请重新输入。")
                continue

            command_cmd = Command(
                resume={"decisions": [{"type": "respond", "message": confirm}]}
            )
            break
        elif decision == "edit":
            # edit：修改当前 update_shipping_address 的 address 参数后再执行。
            new_address = input("请输入新的收货地址：").strip()
            if not new_address:
                print("新地址不能为空，请重新输入。")
                continue
            command_cmd = Command(
                resume={"decisions": [{
                    "type": "edit",
                    "edited_action": {
                        "name": "update_shipping_address",
                        "args": {"order_id": req["args"]["order_id"], "address": new_address}
                    }
                }]}
            )
            break
        else:
            print(f"暂未处理该决策类型：{decision}")

    # =====================================================================
    # 6. 恢复执行 —— 如果恢复后还有中断，继续回到外层循环
    # =====================================================================

    # 恢复执行后更新 result；如果恢复后又触发中断，外层循环会继续处理。
    result = agent.invoke(command_cmd, config=config, version="v2")
    print("Agent 继续执行...")
    print("恢复执行后 result:", result)

print(f"[Agent回复]: {result.value['messages'][-1].content}")
