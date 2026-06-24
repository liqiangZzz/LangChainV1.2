"""
演示一次中断中包含多个待审批工具调用的处理方式。

示例让 Agent 同时发起重启服务、发送通知、修改配置三个敏感操作；
人工需要按 action_requests 顺序依次给出决策，再一次性恢复执行。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义示例工具 —— 用运维场景模拟多个敏感操作
# =====================================================================

@tool
def restart_service(service_name: str, environment: str) -> str:
    """重启指定的微服务"""
    # 示例工具只返回模拟结果，不真正操作线上服务。
    return f"服务[{service_name}]（{environment}环境）已成功重启，耗时 3.2 秒"


@tool
def send_notification(channel: str, title: str, content: str) -> str:
    """向指定渠道发送通知消息"""
    # 示例中不真正发送外部通知，只演示敏感操作审批。
    return (f"已通过[{channel}]发送通知：\n   标题：{title}\n   内容：{content}")


@tool
def update_config(config_key: str, config_value: str) -> str:
    """更新系统配置项"""
    # 示例中不真正修改配置中心，只返回模拟更新结果。
    return f"配置项[{config_key}]已更新为[{config_value}]"


@tool
def query_service_status(service_name: str) -> str:
    """查询服务当前运行状态"""
    return f"服务[{service_name}]当前状态：CPU 92%, 内存 78%, 错误率 15%"


# =====================================================================
# 2. 创建 Agent —— 给多个敏感工具配置同一种人工审批策略
# =====================================================================

agent = create_agent(
    model=deepseek_llm,
    tools=[restart_service, send_notification, update_config, query_service_status],
    # 多个工具调用在同一次中断中恢复时，也依赖 checkpointer 保存图状态。
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 下面三个都是敏感写操作，需要人工逐一批准或拒绝。
                "restart_service": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "敏感操作：重启生产服务，请确认是否继续执行？",
                },
                "send_notification": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "敏感操作：向外发送通知，请确认是否继续执行？",
                },
                "update_config": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "敏感操作：修改系统配置，请确认是否继续执行？",
                },
                # 查询操作是只读操作，无需审批，直接放行。
                "query_service_status": False,
            },
            description_prefix="需要人工介入，请确认操作。"
        ),
    ],
    system_prompt="你是运维工程师的 AI 助手。处理故障时，你可以同时执行多个操作来提高效率。"
)


# =====================================================================
# 3. 发起请求 —— 让模型在同一轮里生成多个工具调用
# =====================================================================

# thread_id 标识本次可恢复流程；提交多个决策时仍要使用同一个 thread_id。
config = {"configurable": {"thread_id": "session_id001"}}

# 用户要求“三个操作现在一起执行”，模型可能在同一步生成多个工具调用。
result = agent.invoke({  # type: ignore
    "messages": [
        {
            "role": "user",
            "content": (
                "订单服务（order-service）在 production 环境出现大量超时错误，"
                "请立即执行以下三个操作：\n"
                "1）重启[production 环境]环境的[订单服务]；\n"
                "2）给[运维告警群]发送通知，标题[订单服务紧急重启]，内容[因超时率过高，正在重启订单服务]；\n"
                "3）把配置项[order.max_retry]改成 5。\n"
                "这三个操作现在一起执行，不要逐个处理。"
            )
        },
    ],
}, config=config, version='v2')

if result.interrupts:
    print("触发中断", result)

    interrupt_value = result.interrupts[0].value

    # =====================================================================
    # 4. 解析中断信息 —— action_requests 与 review_configs 必须按顺序匹配
    # =====================================================================

    # action_requests 和 review_configs 的顺序一一对应；
    # 后面构造 decisions 时必须保持相同顺序。
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]

    print(f"\nAgent 已暂停！本次中断包含 {len(action_requests)} 个待审批操作：\n")

    for idx, action_request in enumerate(action_requests):
        print(f" ==== 操作 [{idx+1}] ====")
        print(f" 工具名称: {action_request['name']}")
        print(f" 参数:     {action_request['args']}")
        print(f" 允许决策: {review_configs[idx]['allowed_decisions']}")
        print(f" 描述:     {action_request['description']}")
        print(f"====================="*3)

    action_order = " → ".join(f"[{idx + 1}]" for idx in range(len(action_requests)))
    print(f"\n注意：请按操作顺序 {action_order} 依次提供 {len(action_requests)} 个决策。")

    # =====================================================================
    # 5. 人工逐个确认 —— 每个操作都要收集一个对应决策
    # =====================================================================

    print("\n" + "=" * 70)
    print("【人工确认】对每个操作逐一做出决策")
    print("=" * 70)

    decisions = []
    for i, req in enumerate(action_requests):
        print(f"\n **** 正在确认操作 [{i}]：{req['name']} ****")
        print(f" 参数: {req['args']}")
        allowed = review_configs[i]["allowed_decisions"]

        while True:
            decision = input(f"   请输入决策（{'/'.join(allowed)}）: ").strip().lower()
            if decision not in allowed:
                print(f"无效输入，请输入 {allowed} 中的一个，当前输入：{decision}")
                continue

            if decision == "approve":
                # approve：批准当前位置的工具调用，保持原参数执行。
                decisions.append({"type": "approve"})
                break
            elif decision == "reject":
                # reject：拒绝当前位置的工具调用，模型会收到一条工具错误结果。
                reason = input(f"请输入拒绝原因: ").strip()
                if not reason:
                    reason = "人工拒绝了该操作"
                decisions.append({"type": "reject", "message": f"原因： {reason}"})
                break

    # 确认决策列表
    print(f"\n即将提交的决策列表：{decisions}")

    # =====================================================================
    # 6. 恢复执行 —— 一次性提交 decisions 列表让 Agent 继续跑
    # =====================================================================

    print("\n" + "=" * 70)
    print("【恢复 Agent 执行】")
    print("=" * 70)

    # 多操作中断需要一次性提交 decisions 列表，列表顺序对应 action_requests。
    result2 = agent.invoke(Command(resume={"decisions": decisions}), config=config, version='v2')
    print(f"\n最终结果：\n{result2.value['messages'][-1].content}")
else:
    print(f"\n[Agent 回复]: {result.value['messages'][-1].content}")
