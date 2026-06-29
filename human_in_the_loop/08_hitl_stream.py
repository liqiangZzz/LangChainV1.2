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
# 2. 定义工具 —— 广播/邮件需审批，查天气放行
# =====================================================================

@tool
def send_broadcast(message: str) -> str:
    """发送全员广播通知"""
    return f"已发送：{message} 广播"


@tool
def send_email(content: str) -> str:
    """发送邮件"""
    return f"邮件已发送，内容：{content}"


@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city} 今天晴，22~28°C"


# =====================================================================
# 3. 创建智能体 —— 配置 HumanInTheLoopMiddleware 中断规则
# =====================================================================
agent = create_agent(
    model=deepseek_llm,
    tools=[send_broadcast, send_email, get_weather],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_broadcast": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "是否发送广播"
                },
                "send_email": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "是否发送邮件"
                },
                "get_weather": False,
            },
            description_prefix="人工介入，请确认以下操作：",
        )
    ],
    checkpointer=InMemorySaver(),
    system_prompt="你是一个助手，可以查询天气、发送全员广播通知和发送邮件。",
)

config = {"configurable": {"thread_id": "session_01"}}

# =====================================================================
# 4. 流式交互主循环 —— 外层轮次 + 内层中断/恢复
# =====================================================================

# 4.1 外层循环：对话轮次，直到用户退出
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

        # 4.2 内层循环：同一轮对话中可能发生多轮中断/恢复
        while True:
            print("[Agent 回复]：", end="", flush=True)

            interrupted = False   # 本轮 stream 是否因中断而跳出

            for chunk in agent.stream(  # type: ignore
                    next_input,
                    config=config,
                    stream_mode=["updates", "messages"],
                    version="v2"
            ):
                # print("chunk:", chunk)
                # 4.3 处理非中断消息流 —— 逐 token 打印
                if chunk['type'] == 'messages':
                    token_data = chunk['data']
                    # token_data 是 (token_chunk, metadata) 元组
                    token_chunk = token_data[0]

                    # 如果 token_chunk 有 context 属性，则认为是中间结果，不打印
                    if token_chunk.content:
                        # end="" 避免打印时自动换行，flush=True 及时刷新输出
                        print(token_chunk.content, end="", flush=True)

                # 4.4 处理更新流 —— 解析中断信号，收集人工决策
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
