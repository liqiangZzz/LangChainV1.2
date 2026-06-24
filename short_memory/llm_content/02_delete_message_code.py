"""
演示 Delete Message：通过删除旧消息解决 LLM 上下文过长问题。

Agent 使用 InMemorySaver 按 thread_id 保存短期记忆。随着多轮对话增加，
state["messages"] 会越来越长。这个示例在 before_model middleware 中检查消息数量，
当消息超过上限时，使用 RemoveMessage 删除较早的消息，只保留最近上下文再调用模型。

和 01_trim_message_code.py 的区别：
- 01_trim_message_code.py 是先用 trim_messages 算出保留结果，再通过
  RemoveMessage(id=REMOVE_ALL_MESSAGES) 清空全部旧 messages，最后重写保留结果。
- 本文件不清空全部 messages，也不重新写入保留消息。
- 本文件只构造 RemoveMessage(id=某条旧消息.id)，让 LangGraph 精准删除这些旧消息。
- 也就是说，01 是“清空全部 + 重写保留结果”，02 是“按 message.id 精准删除旧消息”。

本示例会调用真实 DeepSeek 模型，运行前需要确认 .env 中配置了 DeepSeek 信息。
"""
from typing import Any, Dict

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from models.init_chat_model.init_chat_model_llm import deepseek_llm

# 模型调用前最多保留最近 3 条对话消息。
# 这里为了用三轮对话就能看出删除效果，所以阈值设置得比较小。
# 真实项目里可以根据模型上下文窗口、业务保留策略、消息类型来调整。
#
# 在本示例的三轮对话里，第 3 轮调用模型前会出现 5 条消息：
# Human(第1轮) -> AI(第1轮) -> Human(第2轮) -> AI(第2轮) -> Human(第3轮)
# 删除前 2 条后，模型只看到最近一轮完整问答和当前用户问题。
MAX_MESSAGES_BEFORE_MODEL = 3


# =====================================================================
# 1. 生成删除指令 —— RemoveMessage 只带 id，不搬运消息内容
# =====================================================================
def build_delete_messages(messages: list[BaseMessage]) -> list[RemoveMessage]:
    """构造要删除的 RemoveMessage 列表。

    RemoveMessage 不直接保存消息内容，而是通过消息 id 告诉 LangGraph：
    从 state["messages"] 中删除这些 id 对应的旧消息。
    """
    # 当前消息数量没有超过上限时，不需要删除任何消息。
    # 返回空列表表示本轮 before_model 不更新 messages。
    if len(messages) <= MAX_MESSAGES_BEFORE_MODEL:
        return []

    # 只保留最后 MAX_MESSAGES_BEFORE_MODEL 条消息。
    # messages[:-MAX_MESSAGES_BEFORE_MODEL] 就是需要删除的较早历史消息。
    old_messages = messages[:-MAX_MESSAGES_BEFORE_MODEL]
    delete_messages = []

    for message in old_messages:
        # RemoveMessage 依赖消息 id 定位要删除的消息。
        # 正常由 LangGraph 管理的 HumanMessage / AIMessage 都会有 id。
        # 如果某条消息没有 id，就跳过，避免构造无效删除指令。
        if message.id is None:
            continue

        # 这里不是删除 Python list 里的对象，而是生成一条“删除指令”。
        # LangGraph 收到 RemoveMessage 后，会在合并 state 时删除对应 id 的消息。
        delete_messages.append(RemoveMessage(id=message.id))

    return delete_messages


# =====================================================================
# 2. 打印消息摘要 —— 聚焦 type/content，别被 metadata 淹没
# =====================================================================
def print_messages_summary(title: str, messages: list[BaseMessage]) -> None:
    """只打印消息类型和内容，避免输出 response_metadata 影响阅读。

    Args:
        title: 打印输出标题或图书标题筛选条件。
        messages: 消息列表或当前状态中的 messages。
    """
    print(title)
    for index, message in enumerate(messages, start=1):
        # message.type 通常是 human / ai / system / tool。
        # 这里只看 type 和 content，方便观察哪些旧消息被删除、哪些消息被保留。
        print(f"{index} -> {message.type}: {message.content}")


# =====================================================================
# 3. before_model 删除旧消息 —— 精准移除超出上限的历史
# =====================================================================
@before_model
def delete_old_messages_before_model(state: AgentState, runtime: Runtime) -> Dict[str, Any]:
    """模型调用前删除较早的 messages，避免上下文持续增长。

    before_model 的执行时机：
    1. 用户调用 agent.invoke(...)。
    2. 新的用户消息先进入 state["messages"]。
    3. Agent 真正请求 DeepSeek 前，先运行这个 middleware。
    4. 如果这里返回 RemoveMessage，旧消息会从 state 中删除。
    5. DeepSeek 看到的是删除后的短期记忆。

    Args:
        runtime: 工具或 middleware 的运行时对象，可读取 context、state、store 等信息。
    """
    messages = state["messages"]
    delete_messages = build_delete_messages(messages)

    # 打印删除前的完整 messages，方便和删除后的保留列表对比。
    print_messages_summary("\n[before_model] 删除前 messages:", messages)

    if not delete_messages:
        # 返回 {} 表示不修改 state。
        print("[before_model] 当前消息未超过上限，不删除。")
        return {}

    # delete_messages 里只有 RemoveMessage，没有原始消息内容。
    # 为了打印“删除后会保留什么”，这里用要删除的 id 反推出保留列表。
    delete_ids = {message.id for message in delete_messages}
    kept_messages = [message for message in messages if message.id not in delete_ids]

    print(f"[before_model] 删除旧消息数量：{len(delete_messages)}")
    print_messages_summary("[before_model] 删除后将保留 messages:", kept_messages)

    # 这里只返回 RemoveMessage，LangGraph 会按 id 从 state["messages"] 中删除旧消息。
    # 与 trim_messages 示例不同，这里不是“清空后重写”，而是精准删除指定消息。
    # 对比 01_trim_message_code.py：
    # 01 返回 [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed_messages]，
    # 表示清空全部 messages 后再写回保留结果。
    # 这里返回的 delete_messages 只包含若干 RemoveMessage(id=旧消息.id)，
    # 表示只删除这些旧消息，其他消息原样留在 state 中。
    return {"messages": delete_messages}


# =====================================================================
# 4. 创建 Agent —— 把删除策略挂到模型调用前
# =====================================================================
def build_agent():
    """按项目常用方式创建带 Delete Message 能力的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[],
        # checkpointer 负责保存每个 thread_id 对应的消息状态。
        checkpointer=InMemorySaver(),
        system_prompt=SystemMessage(content="你是一个助手，需要帮助时请随时告诉我。"),
        # 每次真正调用模型前，先检查并删除过旧的短期记忆消息。
        middleware=[delete_old_messages_before_model],
    )


# =====================================================================
# 5. 查看当前记忆 —— 删除后真实 state 以这里为准
# =====================================================================
def print_state_messages(agent, config: dict) -> None:
    """打印当前 thread_id 下实际保存的短期记忆。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
    """
    # get_state(config=...) 可以查看 checkpointer 中当前 thread_id 对应的状态。
    # 这里重点看 messages，因为 Delete Message 操作的就是这个字段。
    state = agent.get_state(config=config).values
    messages = state.get("messages", [])

    print_messages_summary("[当前短期记忆 messages]", messages)

    print("[当前短期记忆 messages_count]", {"messages_count": len(messages)})


# =====================================================================
# 6. 封装单轮调用 —— 每轮都输出回复和短期记忆
# =====================================================================
def invoke_and_print(agent, config: dict, user_content: str) -> None:
    """发送一轮用户消息，并打印模型回复和当前短期记忆。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
        user_content: 本轮用户输入内容。
    """
    # invoke 输入里的 messages 只包含“本轮新增用户消息”。
    # 由于使用了相同 thread_id，历史消息会由 InMemorySaver 自动接上。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
    )

    # result["messages"] 是本轮完成后的完整消息状态。
    # 最后一条通常就是模型最终回复。
    print("[用户]", user_content)
    print("[模型回复]", result["messages"][-1].content)
    print_state_messages(agent, config)
    print("-" * 60)


# =====================================================================
# 7. 运行三轮演示 —— 第三轮进入模型前删除最早消息
# =====================================================================
def main() -> None:
    """连续三轮调用 Agent，观察旧消息何时被删除。"""
    agent = build_agent()
    # 三轮调用使用同一个 thread_id，InMemorySaver 才会持续保存短期记忆。
    config = {"configurable": {"thread_id": "delete-message-session001"}}

    # 第 1 轮：写入用户身份和订单信息。
    invoke_and_print(
        agent,
        config,
        "你好，我叫张三。我的订单号是 A1001，买的是无线耳机。",
    )
    # 第 2 轮：继续追加售后问题，让 messages 增长到 4 条。
    invoke_and_print(
        agent,
        config,
        "耳机左耳没声音，充电盒也接触不良。我希望优先换新。",
    )
    # 第 3 轮：进入模型前会有 5 条消息，before_model 会删除最早的 2 条。
    invoke_and_print(agent, config, "请根据最近沟通内容，帮我整理一段售后说明。")


if __name__ == "__main__":
    main()
