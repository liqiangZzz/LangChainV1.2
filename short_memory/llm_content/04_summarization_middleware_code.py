"""
演示内置 SummarizationMiddleware：自动摘要消息，解决 LLM 上下文过长问题。

03_summarize_message_code.py 是手写摘要逻辑：
1. 自己在 before_model 中判断 messages 数量。
2. 自己调用 deepseek_llm 生成摘要。
3. 自己用 RemoveMessage(id=REMOVE_ALL_MESSAGES) 清空旧消息并写回摘要结果。

本文件使用 LangChain 内置的 SummarizationMiddleware：
1. middleware 自动判断是否达到摘要触发条件。
2. middleware 自动调用指定 model 生成摘要。
3. middleware 自动保留最近消息，并用摘要替代较早对话历史。

本示例会调用真实 DeepSeek 模型。触发摘要时会额外调用一次 DeepSeek 生成摘要，
运行前需要确认 .env 中配置了 DeepSeek 信息，并注意 API 额度消耗。

官方文档：
https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization
"""
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# 摘要触发条件：当消息数量 >= 4 时触发摘要。
# 这里不用 ("tokens", 4000)，是因为 deepseek-v4-flash 当前不支持
# get_num_tokens_from_messages()，按 messages 数量触发更稳定、也更容易观察。
SUMMARY_TRIGGER = ("messages", 4)

# 摘要后保留最近 2 条原始消息。
# 较早消息会被 middleware 压缩进摘要，最近消息保留原文，避免丢失最新问题细节。
SUMMARY_KEEP = ("messages", 2)


def print_messages_summary(title: str, messages: list[BaseMessage]) -> None:
    """只打印消息类型和内容，方便观察摘要前后的短期记忆。"""
    print(title)
    for index, message in enumerate(messages, start=1):
        print(f"{index} ---> {message.type}: {message.content}")


def build_agent():
    """按项目常用方式创建带内置 SummarizationMiddleware 的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[],
        # checkpointer 负责保存每个 thread_id 对应的消息状态。
        checkpointer=InMemorySaver(),
        system_prompt=SystemMessage(content="你是一个助手，需要帮助时请随时告诉我。"),
        middleware=[
            SummarizationMiddleware(
                # 生成摘要用的模型。这里为了和项目保持一致，继续复用 deepseek_llm。
                # 如果生产环境想降低成本，可以换成更便宜的摘要模型。
                model=deepseek_llm,
                # trigger 表示什么时候触发摘要。
                # ("messages", 4) 的含义是：消息数量达到 4 条时触发。
                trigger=SUMMARY_TRIGGER,
                # keep 表示摘要后保留多少最近上下文。
                # ("messages", 2) 的含义是：保留最近 2 条原始消息。
                keep=SUMMARY_KEEP,
                # 自定义摘要提示词。必须包含 {messages} 占位符，
                # middleware 会把要摘要的旧消息填充到这个位置。
                summary_prompt=(
                    "请将下面较早的对话历史压缩成简洁中文摘要。\n"
                    "要求：保留用户身份、订单信息、商品、故障问题、用户诉求和已确认上下文；"
                    "不要补充原文没有的信息。\n\n"
                    "{messages}"
                ),
            )
        ],
    )


def print_state_messages(agent, config: dict) -> None:
    """打印当前 thread_id 下由 checkpointer 保存的短期记忆。"""
    state = agent.get_state(config=config).values
    messages = state.get("messages", [])
    print_messages_summary("[当前短期记忆]", messages)
    print(f"[当前短期记忆] messages_count={len(messages)}")


def invoke_and_print(agent, config: dict, user_content: str) -> None:
    """发送一轮用户消息，并打印模型回复和当前短期记忆。"""
    # invoke 只传入本轮新增用户消息。
    # 历史消息由 InMemorySaver 根据同一个 thread_id 自动接上。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
    )

    print("[用户]", user_content)
    print("[模型回复]", result["messages"][-1].content)
    print_state_messages(agent, config)
    print("-" * 60)


def main() -> None:
    """连续三轮调用 Agent，观察内置 middleware 何时自动摘要。"""
    agent = build_agent()
    # 三轮调用使用同一个 thread_id，InMemorySaver 才会持续保存短期记忆。
    config = {"configurable": {"thread_id": "summarization-middleware-session001"}}

    # 第 1 轮结束后，state 中通常有 2 条消息：Human + AI。
    invoke_and_print(
        agent,
        config,
        "你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。",
    )
    # 第 2 轮调用模型前，消息数量达到 3 条；第 2 轮结束后通常有 4 条消息。
    invoke_and_print(
        agent,
        config,
        "耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。",
    )
    # 第 3 轮调用模型前，新用户消息进入 state 后消息数量会达到 5 条，
    # 满足 trigger=("messages", 4)，SummarizationMiddleware 会自动摘要旧消息。
    invoke_and_print(agent, config, "请根据我们聊过的信息，帮我整理一段售后说明。")


if __name__ == "__main__":
    main()
