"""
演示 Agent 的系统提示词写法。

重点观察：
1. create_agent 的 system_prompt 可以传字符串，也可以传 SystemMessage。
2. system_prompt 用来限定 Agent 的身份、能力边界和回答方式。
3. 工具函数的名称、参数类型和 docstring 会帮助模型判断什么时候调用工具。
"""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@tool
def add_numbers(a: int, b: int):
    # @tool 会把普通 Python 函数包装成 Agent 可调用的工具。
    # 类型标注 a: int、b: int 能帮助模型生成正确的工具参数。
    """计算并返回两个数的和。"""
    return f"和为：{a + b}"


if __name__ == '__main__':
    agent = create_agent(
        model=deepseek_llm,
        tools=[add_numbers],
        # 这里使用 SystemMessage 作为系统提示词，作用类似直接传入字符串。
        # 它告诉 Agent：你有一个“计算两个数之和”的工具可以使用。
        system_prompt=SystemMessage(content="你是一个助手，你可以计算两个数的和。")
    )

    response = agent.invoke({ # type: ignore
        # invoke 传入用户消息后，Agent 会自行判断是否需要调用 add_numbers 工具。
        "messages": [
            # 这个问题包含三段加法，而工具一次只接收两个数。
            # 可以观察 Agent 是否会分步调用工具，或直接用模型自己完成计算。
            {"role": "user", "content": "10加上20再加上30是多少？"}
        ]
    })

    # 打印完整 response，可以看到消息列表、工具调用、token 用量等详细信息。
    print(response)
    # 只打印最终回答内容，适合日常查看结果。
    print(response["messages"][-1].content)
    # pretty_print 会把最后一条消息用更清晰的格式输出。
    response["messages"][-1].pretty_print()
