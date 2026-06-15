"""使用检查点保存同一会话的消息状态。

InMemorySaver 是内存版 checkpointer，适合演示短期会话记忆。
程序退出后数据会丢失，不属于持久化的长期记忆。
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from models.init_chat_model.init_chat_model_llm import deepseek_llm


def main() -> None:
    """使用相同 thread_id 执行两轮对话并打印检查点。"""
    agent = create_agent(
        model=deepseek_llm,
        tools=[],
        system_prompt="你是一个友好的助手，需要记住用户在当前会话中提供的信息。",
        checkpointer=InMemorySaver(),
    )

    # 相同的 thread_id 表示两次调用属于同一个会话。
    config = {"configurable": {"thread_id": "session-001"}}
    questions = ["我叫张三，你是谁？", "我叫什么名字？"]

    for question in questions:
        print(f"\n用户：{question}")

        # checkpoints 不是默认模式，需要显式指定。
        # 每次创建检查点时，都会返回当前完整状态。
        for checkpoint in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            stream_mode="checkpoints",
        ):
            messages = checkpoint["values"].get("messages", [])
            step = checkpoint["metadata"].get("step")
            print(f"检查点步骤：{step}，消息数量：{len(messages)}")

        # 相同 thread_id 的最终状态可以通过 get_state() 读取。
        state = agent.get_state(config)
        print("助手：", state.values["messages"][-1].content)


if __name__ == "__main__":
    main()
