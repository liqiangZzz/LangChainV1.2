"""演示使用 ``stream_mode="messages"`` 实时输出模型生成的文本。

``messages`` 模式会返回 ``(message_chunk, metadata)``：

- ``message_chunk`` 是模型本次生成的消息片段。
- ``metadata`` 包含当前节点、执行步骤等运行信息。

该模式不是默认值，使用时需要显式指定。
"""

from langchain.agents import create_agent

from my_llm import deepseek_llm


def main() -> None:
    """流式调用 Agent，并逐段打印模型回复。"""
    agent = create_agent(
        model=deepseek_llm,
        tools=[],
        system_prompt="你是一个回答简洁的中文助手。",
    )

    print("模型回复：", end="", flush=True)
    current_node = None

    for message_chunk, metadata in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请用三句话介绍 LangChain Agent。",
                }
            ]
        },
        stream_mode="messages",
    ):
        # metadata 可以用来判断当前消息片段来自哪个 LangGraph 节点。
        node_name = metadata.get("langgraph_node")
        if node_name != current_node:
            current_node = node_name
            print(f"\n[{node_name}] ", end="", flush=True)

        # 模型生成文本时，message_chunk.content 通常是字符串。
        # 工具调用参数等非文本片段不在这个基础示例中展开。
        if isinstance(message_chunk.content, str):
            print(message_chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
