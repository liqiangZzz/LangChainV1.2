"""
演示 Summarize Message：通过消息摘要解决 LLM 上下文过长问题。

Agent 使用 InMemorySaver 按 thread_id 保存短期记忆。随着多轮对话增加，
state["messages"] 会越来越长。这个示例在 before_model middleware 中检查消息数量：
当消息超过上限时，先把较早的消息总结成一条摘要消息，再保留最近几条原始消息。

和前两个示例的区别：
- 01_trim_message_code.py：直接丢弃较早消息，只保留最近消息。
- 02_delete_message_code.py：按 message.id 精准删除较早消息。
- 本文件：不只是删除旧消息，而是先把旧消息压缩成摘要，再和最近消息一起发给模型。

本示例会调用真实 DeepSeek 模型。触发摘要时会额外调用一次 DeepSeek 生成摘要，
运行前需要确认 .env 中配置了 DeepSeek 信息，并注意 API 额度消耗。
"""
from typing import Any, Dict

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# 模型调用前最多保留最近 3 条原始消息。
# 如果消息超过这个数量，就把更早的消息总结成一条摘要消息。
# 这里设置得较小，是为了三轮对话就能看到摘要效果。
# 真实项目里可以根据模型上下文窗口和业务需要调大。
MAX_RECENT_MESSAGES = 3


def format_messages_for_summary(messages: list[BaseMessage]) -> str:
    """把旧消息整理成摘要模型容易阅读的文本。

    Args:
        messages: 消息列表或当前状态中的 messages。
    """
    lines = []
    for index, message in enumerate(messages, start=1):
        # message.type 用来标明消息角色，例如 human / ai / system / tool。
        # 摘要模型看到角色后，更容易判断哪些是用户诉求、哪些是模型已经回复过的内容。
        lines.append(f"{index}. {message.type}: {message.content}")
    return "\n".join(lines)


def summarize_old_messages(old_messages: list[BaseMessage]) -> SystemMessage:
    """调用 DeepSeek，把较早的对话消息压缩成一条 SystemMessage 摘要。

    Args:
        old_messages: 需要被摘要压缩的较早消息列表。
    """
    # old_messages 是即将被压缩的较早历史消息。
    # recent_messages 不会进入这里，因为最近消息会原样保留给最终回答模型。
    old_messages_text = format_messages_for_summary(old_messages)

    # 注意：这是一次额外的模型调用，只用于生成摘要。
    # 后面 agent 还会再调用一次模型，用“摘要 + 最近消息”生成最终回复。
    summary_response = deepseek_llm.invoke(
        [
            # 这里的 SystemMessage 是给“摘要模型”看的，不是最终 Agent 的 system_prompt。
            SystemMessage(content="你负责压缩对话历史，请只保留后续回答需要的关键信息。"),
            HumanMessage(
                content=(
                    "请把下面的较早对话总结成简洁摘要，保留用户身份、订单信息、问题、"
                    "诉求和已经确认过的上下文。不要补充原文没有的信息。\n\n"
                    f"{old_messages_text}"
                )
            ),
        ]
    )

    # 把摘要包装成 SystemMessage 放回对话历史。
    # 这样最终回答模型会把它当作背景上下文，而不是当作用户的新问题。
    return SystemMessage(content=f"较早对话摘要：{summary_response.content}")


def build_summarized_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """生成摘要后的 messages。

    返回结果结构：
    1. 一条 SystemMessage 摘要，代表较早历史。
    2. 最近 MAX_RECENT_MESSAGES 条原始消息，保证模型还能看到最新上下文细节。
    """
    if len(messages) <= MAX_RECENT_MESSAGES:
        # 消息还不多时，不需要摘要，直接保留原 messages。
        return messages

    # 切分历史：
    # - old_messages：较早消息，会被摘要压缩。
    # - recent_messages：最近消息，保持原文，避免丢失最新细节。
    old_messages = messages[:-MAX_RECENT_MESSAGES]
    print_messages_summary(
        "\n[before_model] 较早消息 old_messages:",
        old_messages,
        source="会被摘要",
    )
    recent_messages = messages[-MAX_RECENT_MESSAGES:]
    print_messages_summary(
        "\n[before_model] 最近消息 recent_messages:",
        recent_messages,
        source="原样保留",
    )

    # 摘要化较早消息。
    summary_message = summarize_old_messages(old_messages)

    # 最终传给模型的上下文 = 一条摘要 + 最近几条原始消息（*代表解包，把消息列表展开）。
    return [summary_message, *recent_messages]


def print_messages_summary(title: str, messages: list[BaseMessage], source: str = "") -> None:
    """只打印消息类型和内容，避免 response_metadata 干扰观察。

    Args:
        title: 打印输出标题或图书标题筛选条件。
        messages: 消息列表或当前状态中的 messages。
        source: 打印消息时附加的来源标签。
    """
    print(title)
    for index, message in enumerate(messages, start=1):
        # AIMessage 通常包含较长的 response_metadata。
        # 示例只打印 type/content，方便聚焦“摘要前后 messages 的变化”。
        # source 用来标记这批消息的来源：
        # - 会被摘要：会进入 summarize_old_messages，被压缩成一条摘要。
        # - 原样保留：不会进入摘要，会直接保留到最终上下文。
        # - 最终送入模型：摘要处理完成后，真正给主模型回答用的上下文。
        source_text = f"[{source}] " if source else ""
        print(f"{index} ---> {source_text}{message.type}: {message.content}")


@before_model
def summarize_messages_before_model(state: AgentState, runtime: Runtime) -> Dict[str, Any]:
    """模型调用前把较早消息摘要化，避免上下文持续增长。

    before_model 的执行顺序：
    1. 用户调用 agent.invoke(...)。
    2. 本轮用户消息先加入 state["messages"]。
    3. 请求 DeepSeek 正式回答前，先执行这个 middleware。
    4. 如果消息过多，就额外调用一次 DeepSeek 生成摘要。
    5. 清空旧 messages，写入“摘要消息 + 最近消息”。
    6. DeepSeek 基于压缩后的上下文生成本轮最终回复。

    Args:
        runtime: 工具或 middleware 的运行时对象，可读取 context、state、store 等信息。
    """
    messages = state["messages"]
    print_messages_summary("\n[before_model] 摘要前 messages:", messages, source="原始state")

    if len(messages) <= MAX_RECENT_MESSAGES:
        # 返回空 dict 表示本轮不修改 state，Agent 会直接拿当前 messages 调模型。
        print("[before_model] 当前消息未超过上限，不生成摘要。")
        return {}

    # 消息数量超过上限时，生成摘要后的 messages。
    summarized_messages = build_summarized_messages(messages)
    # 打印摘要后的 messages。
    print_messages_summary(
        "[before_model] 摘要后 messages:",
        summarized_messages,
        source="最终送入模型",
    )

    # messages 默认是追加式 reducer。
    # 如果直接 return {"messages": summarized_messages}，新 messages 会追加到旧 messages 后面。
    # 这里先用 REMOVE_ALL_MESSAGES 清空旧历史，再写入摘要后的 messages。
    # 这个写法和 01_trim_message_code.py 类似：都是“清空全部 + 写回压缩后的结果”。
    # 不同点是：01 写回的是 trim 后的原始消息；这里写回的是“摘要消息 + 最近原始消息”。
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *summarized_messages]}


def build_agent():
    """按项目常用方式创建带 Summarize Message 能力的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[],
        # checkpointer 负责保存每个 thread_id 对应的消息状态。
        checkpointer=InMemorySaver(),
        system_prompt=SystemMessage(content="你是一个助手，需要帮助时请随时告诉我。"),
        # 每次真正调用模型前，先检查是否需要把较早消息摘要化。
        middleware=[summarize_messages_before_model],
    )


def print_state_messages(agent, config: dict) -> None:
    """打印当前 thread_id 下实际保存的短期记忆。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
    """
    # get_state(config=...) 读取 checkpointer 中指定 thread_id 的状态。
    # 摘要后，state["messages"] 里应该能看到一条“较早对话摘要”的 SystemMessage。
    state = agent.get_state(config=config).values
    messages = state.get("messages", [])
    print_messages_summary("[当前短期记忆]", messages, source="已保存state")
    print(f"[当前短期记忆] messages_count={len(messages)}")


def invoke_and_print(agent, config: dict, user_content: str) -> None:
    """发送一轮用户消息，并打印模型回复和当前短期记忆。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
        user_content: 本轮用户输入内容。
    """
    # 这里只传入本轮新增的用户消息。
    # 历史消息由 InMemorySaver 根据 thread_id 自动从 state 里取出并拼接。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
    )

    # result["messages"] 是本轮 invoke 完成后的消息列表。
    # 最后一条通常是本轮最终 AI 回复。
    print("[用户]", user_content)
    print("[模型回复]", result["messages"][-1].content)
    print_state_messages(agent, config)
    print("-" * 60)


def main() -> None:
    """连续三轮调用 Agent，观察旧消息如何被摘要压缩。"""
    agent = build_agent()
    # 三轮调用使用同一个 thread_id，InMemorySaver 才会持续保存短期记忆。
    config = {"configurable": {"thread_id": "summarize-message-session001"}}

    # 第 1 轮：写入用户身份和订单信息。
    invoke_and_print(
        agent,
        config,
        "你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。",
    )
    # 第 2 轮：追加售后问题和处理偏好。
    invoke_and_print(
        agent,
        config,
        "耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。",
    )
    # 第 3 轮：进入模型前消息超过上限，before_model 会先摘要较早消息。
    invoke_and_print(agent, config, "请根据我们聊过的信息，帮我整理一段售后说明。")


if __name__ == "__main__":
    main()
