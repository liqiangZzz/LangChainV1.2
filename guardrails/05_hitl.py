# =====================================================================
# 1. 导入依赖
# =====================================================================

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm

# =====================================================================
# 2. 模拟订单数据库
# =====================================================================
ORDERS = {
    "ORD001": {"user": "张三", "amount": 200, "status": "已付款"},
    "ORD002": {"user": "李四", "amount": 3000, "status": "已付款"},
    "ORD003": {"user": "王五", "amount": 15000, "status": "已发货"},
}


# =====================================================================
# 3. 定义工具 —— 查订单放行，退款走自定义审批
# =====================================================================

@tool
def query_order(order_id: str) -> str:
    """
    查询订单信息
    Args：
      order_id: 订单编号，如 ORD002
    Returns：
      订单信息
    """
    order = ORDERS.get(order_id)
    if not order:
        return f"订单 {order_id} 不存在"
    return (f"订单 {order_id}：用户 {order['user']}，"
            f"金额 ¥{order['amount']}，状态 {order['status']}")


@tool
def process_refund(order_id: str) -> str:
    """
    执行退款操作
    Args：
      order_id: 订单编号，如 ORD002
    Returns：
      退款成功消息
    """
    order = ORDERS.get(order_id)
    if not order:
        return f"订单 {order_id} 不存在，无法退款"
    return f"订单 {order_id}（{order['user']}）已退款，退款金额{order['amount']}"


# =====================================================================
# 5. 创建智能体 —— 挂载自定义中间件
# =====================================================================
agent = create_agent(
    model=deepseek_llm,
    tools=[query_order, process_refund],
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "process_refund": {
                "allowed_decisions": ["approve", "reject"],
                "description": "订单退款操作：可批准或拒绝退款请求",
            },
            "query_order": False
        }
    )],
    checkpointer=InMemorySaver(),
    system_prompt="你是一个电商助手，可以处理用户订单问题。",
)

config = {"configurable": {"thread_id": "session_01"}}

# =====================================================================
# 6. 流式交互主循环 —— 外层轮次 + 内层中断/恢复
# =====================================================================

# 6.1 外层循环：对话轮次，直到用户退出
while True:
    try:
        user_input = input("\n[你]: ").strip()

        if user_input.lower() in ("quit", "exit", "退出", "q"):
            print("已退出，再见！")
            break

        if not user_input:
            continue

        # 本轮对话的输入：首次是普通消息，中断后会被替换为 Command(resume=...)
        next_input = {"messages": [{"role": "user", "content": user_input}]}

        # 6.2 内层循环：同一轮对话中可能发生多轮中断/恢复
        while True:
            print("[Agent 回复]：", end="", flush=True)

            interrupted = False  # 本轮 stream 是否因中断而跳出

            for chunk in agent.stream(  # type: ignore
                    next_input,
                    config=config,
                    stream_mode=["updates", "messages"],
                    version="v2"
            ):
                # print("chunk:", chunk)
                # 6.3 处理非中断消息流 —— 逐 token 打印
                if chunk['type'] == 'messages':
                    token_data = chunk['data']
                    # token_data 是 (token_chunk, metadata) 元组
                    token_chunk = token_data[0]

                    # 如果 token_chunk 有 context 属性，则认为是中间结果，不打印
                    if token_chunk.content:
                        # end="" 避免打印时自动换行，flush=True 及时刷新输出
                        print(token_chunk.content, end="", flush=True)

                # 6.4 处理更新流 —— 解析中断信号，收集人工决策
                elif chunk['type'] == 'updates':
                    update_data = chunk['data']

                    # stream_mode=["updates"] 下，中断的 chunk 结构为：
                    #   {"__interrupt__": (Interrupt对象, ...)}
                    # key 是 "__interrupt__"（没有 s），值是 Interrupt 对象的 tuple
                    # Interrupt 对象有 .value 属性，其值为 HITLRequest 字典
                    if isinstance(update_data, dict) and "__interrupt__" in update_data:
                        interrupted = True

                        print("=" * 60)
                        print("Agent中断已触发！Agent 等待人工确认…")
                        print("=" * 60)

                        # 提取中断详情：Interrupt 对象的 .value 才是 HITLRequest
                        interrupt_tuple = update_data["__interrupt__"]
                        hitl_request = interrupt_tuple[0].value
                        action_requests = hitl_request['action_requests']
                        review_configs = hitl_request['review_configs']

                        print(f"\n  本次中断包含 {len(action_requests)} 个待审批操作：")

                        for i, req in enumerate(action_requests):
                            review_config = review_configs[i]
                            print(
                                f"[{i + 1}] {req['name']}  |  参数: {req['args']}  |  允许: {review_config['allowed_decisions']}")

                        # 按顺序逐个收集决策（多工具同时中断时，顺序必须一致）
                        decisions = []
                        for i, req in enumerate(action_requests):
                            review_config = review_configs[i]
                            allowed_decisions = review_config['allowed_decisions']

                            print(f"\n  ── 操作 [{i + 1}] ：工具：{req['name']} ──")

                            # 循环：等待用户输入有效决策
                            while True:
                                decision = input(f"\n  ──>>>> 请输入决策 [{review_config['allowed_decisions']}]: ")
                                if decision in allowed_decisions:
                                    break
                                print(f"      只允许: {allowed_decisions}")

                            if decision == 'approve':
                                decisions.append({"type": "approve"})
                                print(f"      已批准：按原参数执行")
                            elif decision == 'reject':
                                reason = input(f"      拒绝原因: ").strip() or "操作被人工拒绝"
                                decisions.append({"type": "reject", "message": reason})
                                print(f"      已拒绝：原因 {reason}")

                        print(f"\n  提交决策: {decisions}")

                        # 构造恢复命令，替换下一轮 stream 的输入
                        next_input = Command(resume={"decisions": decisions})

                        # 跳出 for 循环，回到内层 while 顶部，用 Command(resume=...) 重新 stream
                        # 这个 break 不是结束对话，而是结束当前这一次 agent.stream(...)。把“人工审批结果”带回去，再作为新的输入重新执行 agent.stream()，让 Agent 从中断点继续跑下去。
                        break

            # ── 内层 while 的分岔口 ──
            if not interrupted:
                # for 循环正常结束 → 本轮对话完成，跳出内层 while，回到外层等下一句
                print()  # 末尾换行
                break
            else:
                # for 循环被 break（有中断）→ 内层 while 回到顶部，用 next_input 恢复执行
                # 打印一个空行作为中断→恢复执行分割标志
                print("\nAgent正在恢复执行…\n")

    except Exception as e:
        print(f"\n调用过程中出现错误：{e}")
