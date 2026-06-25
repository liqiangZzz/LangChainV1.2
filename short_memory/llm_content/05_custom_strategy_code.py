"""
演示 Custom Strategies：自定义消息保留策略，解决 LLM 上下文过长问题。

前几个示例分别演示了：
- 01_trim_message_code.py：按固定规则截断消息。
- 02_delete_message_code.py：按 message.id 删除旧消息。
- 03_summarize_message_code.py：手写摘要，把旧消息压缩成摘要。
- 04_summarization_middleware_code.py：使用内置 SummarizationMiddleware 自动摘要。

本文件演示“自定义策略”：
不是简单保留最近 N 条，也不是把旧消息全部摘要，而是根据业务规则决定哪些消息重要。
例如客服场景中，用户姓名、订单号、手机号尾号、商品、故障、换新诉求都很重要，
这些消息即使比较早，也应该尽量保留；普通寒暄或已经不重要的旧回复可以丢弃。

本示例会调用真实 DeepSeek 模型，运行前需要确认 .env 中配置了 DeepSeek 信息。
"""
from typing import Any, Dict

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from models.init_chat_model.init_chat_model_llm import deepseek_llm

# 最近消息保留数量。
# 最近消息通常包含当前问题和它前面的最新上下文，所以无论是否命中关键词，都优先保留。
KEEP_RECENT_MESSAGES = 3

# 关键消息最大保留数量。
# 如果历史里命中关键词的消息很多，只保留最后几条关键消息，避免上下文继续无限增长。
# 注意：这是“较早历史中的关键消息”保留数量，不包含 recent_messages。
KEEP_IMPORTANT_MESSAGES = 2

# 自定义业务关键词。
# 这里不是 token 策略，而是业务策略：只要消息内容命中这些关键词，就认为它对后续回答重要。
# 这个列表就是自定义策略的核心。换成别的业务时，通常只需要替换关键词或判断函数。
IMPORTANT_KEYWORDS = (
    "我叫",
    "姓名",
    "张三",
    "订单号",
    "A1001",
    "无线耳机",
    "手机号",
    "尾号",
    "左耳",
    "充电盒",
    "换新",
    "不维修",
)


# =====================================================================
# 1. 判断关键消息 —— 命中业务关键词就尽量保留
# =====================================================================
def is_important_message(message: BaseMessage) -> bool:
    """判断一条消息是否应该被自定义策略保留。

    Args:
        message: 待处理的消息对象。
    """
    content = str(message.content)

    # 只用内容判断，避免依赖具体模型 response_metadata。
    # 真实项目里也可以结合 message.type、工具结果、结构化 state 等信息综合判断。
    # 例如：
    # - 只保留 human 消息中的订单信息。
    # - 保留 tool 消息中的查询结果。
    # - 保留结构化 state 里标记为 high_priority 的消息。
    return any(keyword in content for keyword in IMPORTANT_KEYWORDS)


# =====================================================================
# 2. 消息去重 —— 合并关键消息和最近消息时保持原顺序
# =====================================================================
def deduplicate_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """按 message.id 去重，并保持原有顺序。

    Args:
        messages: 消息列表或当前状态中的 messages。
    """
    deduplicated_messages = []
    seen_ids = set()

    for message in messages:
        # LangGraph 管理的消息通常都有 id。
        # 如果没有 id，就用对象本身的 id(message) 做兜底，只用于本地去重。
        message_id = message.id or id(message)
        if message_id in seen_ids:
            continue

        seen_ids.add(message_id)
        deduplicated_messages.append(message)

    return deduplicated_messages


# =====================================================================
# 3. 应用自定义策略 —— 业务关键消息 + 最近上下文
# =====================================================================
def apply_custom_strategy(messages: list[BaseMessage]) -> list[BaseMessage]:
    """应用自定义消息保留策略。

    策略分成两部分：
    1. 从较早历史中挑出最后几条“业务关键消息”。
    2. 始终保留最近几条原始消息。

    最终结果 = 关键消息 + 最近消息，并按原始顺序去重。

    Args:
        messages: 消息列表或当前状态中的 messages。
    """
    if len(messages) <= KEEP_RECENT_MESSAGES:
        # 消息还不多时，不需要做自定义筛选。
        # 返回原 messages，before_model 中会判断长度不变并跳过重写 state。
        return messages

    # 先把消息切成两段：
    # - old_messages：较早历史，可能已经不全部重要，需要通过业务规则筛选。
    # - recent_messages：最近上下文，通常包含当前用户问题，必须原样保留。
    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]

    # 从较早历史里挑出命中业务关键词的消息。
    # 这些消息可能包含订单号、商品、故障、诉求等关键信息。
    important_messages = [message for message in old_messages if is_important_message(message)]
    # 只保留最后 KEEP_IMPORTANT_MESSAGES 条关键消息。
    # 这样可以防止“所有历史都命中关键词”时，上下文还是无限增长。
    important_messages = important_messages[-KEEP_IMPORTANT_MESSAGES:]

    print_messages_summary("\n[before_model] 较早消息 old_messages:", old_messages, source="候选历史")
    print_messages_summary("\n[before_model] 命中关键策略 important_messages:", important_messages, source="业务保留")
    print_messages_summary("\n[before_model] 最近消息 recent_messages:", recent_messages, source="最近保留")

    # important_messages 和 recent_messages 可能有重叠。
    # 例如某条消息既命中关键词，又属于最近消息。
    # 合并后去重，避免同一条消息重复发送给模型。
    selected_messages = deduplicate_messages([*important_messages, *recent_messages])
    print_messages_summary("[before_model] 合并去重 selected_messages:", selected_messages, source="策略结果")

    return selected_messages


# =====================================================================
# 4. 打印消息摘要 —— 用来源标签看清策略筛选过程
# =====================================================================
def print_messages_summary(title: str, messages: list[BaseMessage], source: str = "") -> None:
    """只打印消息类型和内容，方便观察自定义策略保留了什么。

    Args:
        title: 打印输出标题或图书标题筛选条件。
        messages: 消息列表或当前状态中的 messages。
        source: 打印消息时附加的来源标签。
    """
    print(title)
    for index, message in enumerate(messages, start=1):
        # source 标识这批消息在策略中的角色：
        # - 原始state：before_model 看到的完整消息。
        # - 候选历史：较早历史，可能被丢弃，也可能被保留。
        # - 业务保留：命中关键词、决定从旧历史中保留下来的消息。
        # - 最近保留：最近上下文，始终原样保留。
        # - 策略结果/最终送入模型：最终写回 state 并发给模型的消息。
        source_text = f"[{source}] " if source else ""
        print(f"{index} ---> {source_text}{message.type}: {message.content}")


# =====================================================================
# 5. before_model 执行策略 —— 清空旧历史后写回精选消息
# =====================================================================
@before_model
def custom_strategy_before_model(state: AgentState, runtime: Runtime) -> Dict[str, Any]:
    """模型调用前应用自定义消息保留策略。

    before_model 的执行顺序：
    1. 用户调用 agent.invoke(...)。
    2. 本轮用户消息先进入 state["messages"]。
    3. Agent 请求 DeepSeek 前，先运行这个 middleware。
    4. middleware 根据业务关键词和最近上下文挑选消息。
    5. 清空旧 messages，写入自定义策略保留后的 messages。
    6. DeepSeek 基于这些保留消息生成最终回复。

    Args:
        runtime: 工具或 middleware 的运行时对象，可读取 context、state、store 等信息。
    """
    messages = state["messages"]
    print_messages_summary("\n[before_model] 策略执行前 messages:", messages, source="原始state")

    selected_messages = apply_custom_strategy(messages)

    if len(selected_messages) == len(messages):
        # 长度没有变化，说明当前消息数量还少，或者策略没有删掉任何消息。
        # 返回 {} 表示不修改 state。
        print("[before_model] 当前消息未超过策略处理范围，不重写 messages。")
        return {}

    print_messages_summary("[before_model] 策略执行后 messages:", selected_messages, source="最终送入模型")

    # messages 默认是追加式 reducer。
    # 因此这里要先用 REMOVE_ALL_MESSAGES 清空旧 messages，
    # 再写回自定义策略筛选后的 selected_messages。
    # 这个写法和 01/03 类似，都是“先清空全部，再写回处理后的结果”。
    # 不同的是，这里处理后的结果不是 trim 或 summary，而是业务规则筛选结果。
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *selected_messages]}


# =====================================================================
# 6. 创建 Agent —— 把业务消息保留策略挂到模型调用前
# =====================================================================
def build_agent():
    """按项目常用方式创建带自定义策略的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[],
        # checkpointer 负责保存每个 thread_id 对应的消息状态。
        checkpointer=InMemorySaver(),
        system_prompt=SystemMessage(content="你是一个助手，需要帮助时请随时告诉我。"),
        # 每次真正调用模型前，先执行自定义消息保留策略。
        middleware=[custom_strategy_before_model],
    )


# =====================================================================
# 7. 查看当前记忆 —— 确认策略最终保留了哪些消息
# =====================================================================
def print_state_messages(agent, config: dict) -> None:
    """打印当前 thread_id 下实际保存的短期记忆。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
    """
    # get_state(config=...) 读取 checkpointer 中指定 thread_id 的状态。
    # 如果 before_model 执行了自定义策略，这里看到的就是策略处理后的 messages，
    # 再加上本轮模型刚生成的 AI 回复。
    state = agent.get_state(config=config).values
    messages = state.get("messages", [])
    print_messages_summary("[当前短期记忆]", messages, source="已保存state")
    print(f"[当前短期记忆] messages_count={len(messages)}")


# =====================================================================
# 8. 封装单轮调用 —— 每轮都打印回复和策略后的状态
# =====================================================================
def invoke_and_print(agent, config: dict, user_content: str) -> None:
    """发送一轮用户消息，并打印模型回复和当前短期记忆。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
        user_content: 本轮用户输入内容。
    """
    # invoke 只传入本轮新增用户消息。
    # 历史消息由 InMemorySaver 根据同一个 thread_id 自动接上。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
    )

    # result["messages"] 是本轮完成后的消息列表。
    # 最后一条通常就是模型最终回复。
    print("[用户]", user_content)
    print("[模型回复]", result["messages"][-1].content)
    print_state_messages(agent, config)
    print("-" * 60)


# =====================================================================
# 9. 运行多轮演示 —— 让普通消息和关键消息拉开差异
# =====================================================================
def main() -> None:
    """连续多轮调用 Agent，观察自定义策略如何保留业务关键消息。"""
    agent = build_agent()
    # 多轮调用使用同一个 thread_id，InMemorySaver 才会持续保存短期记忆。
    config = {"configurable": {"thread_id": "custom-strategy-session001"}}

    # 第 1 轮：包含身份、订单号、商品、手机号尾号，是业务关键消息。
    invoke_and_print(
        agent,
        config,
        "你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。",
    )
    # 第 2 轮：普通过渡问题，不一定需要长期保留。
    invoke_and_print(agent, config, "你先简单回复我一下，说明你已经准备好了。")
    # 第 3 轮：包含故障和诉求，是业务关键消息。
    invoke_and_print(
        agent,
        config,
        "耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。",
    )
    # 第 4 轮：触发自定义策略，保留关键消息和最近上下文。
    invoke_and_print(agent, config, "请根据我们聊过的信息，帮我整理一段售后说明。")


if __name__ == "__main__":
    main()
