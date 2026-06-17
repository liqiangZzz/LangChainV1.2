"""
演示 Agent 的短期记忆。

InMemorySaver 会把同一个 thread_id 下的消息状态保存在内存中。
因此第二次调用时，Agent 可以读取前一次对话里的信息。

注意：
- 这是内存版 checkpointer，程序退出后记忆会丢失。
- 这个示例会调用真实模型，运行前需要确认 .env 中配置了 DeepSeek 信息。
"""
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from models.init_chat_model.init_chat_model_llm import deepseek_llm


def main() -> None:
    """使用相同 thread_id 执行两轮对话，观察短期记忆效果。"""
    agent = create_agent(
        model=deepseek_llm,
        tools=[],
        # checkpointer 负责保存每个 thread_id 对应的消息状态。
        checkpointer=InMemorySaver(),
        system_prompt=SystemMessage(content="你是一个助手，需要帮助时请随时告诉我。"),
    )

    # thread_id 表示会话 ID。两次调用使用同一个 thread_id，才会共享上下文记忆。
    config = {"configurable": {"thread_id": "session001"}}

    first_response = agent.invoke(  # type: ignore
        {"messages": [{"role": "user", "content": "你好，我叫张三"}]},
        config=config,
    )

    print("第一轮回复：")
    print(first_response["messages"][-1].content)
    print("-" * 50)

    # 第二轮没有再次提供名字，但 Agent 可以从同一 thread_id 的历史消息中读取。
    second_response = agent.invoke(  # type: ignore
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config=config,
    )

    print("第二轮回复：")
    print(second_response["messages"][-1].content)


if __name__ == "__main__":
    main()
