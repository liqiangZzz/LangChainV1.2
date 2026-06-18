"""演示 Agent 自定义状态。

自定义状态步骤：
1. 定义一个继承自 AgentState 的类，将自定义状态字段添加到类中。
2. 创建 Agent 时，把自定义状态类传给 state_schema。
3. invoke 时可以传入自定义状态；同一个 thread_id 下，checkpointer 会保存这些状态。

注意：自定义状态只是保存在 Agent state 里，不会自动进入模型上下文。
如果希望模型稳定使用这些字段，需要通过 dynamic_prompt 或 middleware 显式注入。
"""
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@tool
def get_user_info(name: str) -> str:
    """
    根据姓名查询用户信息
    Args:
        name (str): 要查询的用户姓名
    Returns:
        str: 包含用户信息的字符串
    """
    user_db = {
        "张三": {"age": 28, "hobby": "旅游、滑雪、喝茶"},
        "李四": {"age": 32, "hobby": "编程、阅读、电影"},
    }
    info = user_db.get(name, {"age": "未知", "hobby": "未知"})
    return f"姓名: {name}, 年龄: {info['age']}岁, 爱好: {info['hobby']}"


# 自定义状态类
class CustomState(AgentState):
    # 这里声明的是“Agent state 允许保存哪些自定义字段”。
    # 不是要求每次 invoke 都必须传这些字段。
    # 这些字段可以在 invoke 输入中写入，也可以由 tool/middleware 在运行过程中写入。
    # 但只要希望 invoke 传入的字段能稳定进入 state，就应该先在这里声明。
    user_id: str  # 用户唯一标识
    hobby: list[str]  # 用户爱好
    other_info: dict[str, Any]  # 用户其他信息


@dynamic_prompt
def custom_state_prompt(request: ModelRequest) -> str:
    """把自定义 state 注入到系统提示词中，确保模型能看到这些字段。

    Args:
        request: 当前模型、工具或 middleware 调用请求。
    """
    # state_schema 只是让 Agent 能保存这些字段。
    # dynamic_prompt 会在每次调用模型前运行，把 state 转成模型能看到的系统提示词。
    # 也就是说：state 负责“存数据”，dynamic_prompt 负责“把数据告诉模型”。
    user_id = request.state.get("user_id", "未知")
    hobby = request.state.get("hobby", [])
    other_info = request.state.get("other_info", {})

    print("[dynamic_prompt] 从 state 读取用户信息:", {"user_id": user_id, "hobby": hobby, "other_info": other_info})
    return f"""你是一个助手，需要帮助时请随时告诉我。
                当前会话已保存的用户状态：
                - user_id: {user_id}
                - hobby: {hobby}
                - other_info: {other_info}
                当用户询问自己的信息时，请优先根据以上状态回答。"""


def build_agent():
    """创建带自定义状态和动态提示词的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[get_user_info],
        # state_schema 负责声明并保存自定义状态字段。
        state_schema=CustomState,
        # middleware 中的 dynamic_prompt 负责把 state 显式放进模型上下文。
        middleware=[custom_state_prompt],
        checkpointer=InMemorySaver(),
    )


def print_saved_state(agent, config: dict) -> None:
    """只打印自定义状态，避免完整 messages 输出过长。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
    """
    state = agent.get_state(config=config)
    print(
        "保存的自定义状态:",
        {
            "user_id": state.values.get("user_id"),
            "hobby": state.values.get("hobby"),
            "other_info": state.values.get("other_info"),
        },
    )


def main() -> None:
    """运行两轮对话，演示第二轮可以读取第一轮保存的自定义状态。"""
    agent = build_agent()
    config = {"configurable": {"thread_id": "session001"}}

    # 第一次 invoke 时传入 user_id/hobby/other_info。
    # 这些字段会按照 CustomState 的声明写入 Agent state，并由 checkpointer 绑定到 thread_id。
    # 但“写入 state”不等于“模型一定能看到”，模型看到它们依赖上面的 dynamic_prompt。
    # 如果字段名写错，例如写成 userId，后续 request.state.get("user_id") 就读不到。
    first_response = agent.invoke(  # type: ignore
        {
            "messages": [{"role": "user", "content": "你好，我是张三"}],
            "user_id": "user_001",
            "hobby": ["编程", "阅读", "电影"],
            "other_info": {"age": 25, "gender": "男"},
        },
        config=config,
    )

    print("第一轮回复:")
    print(first_response["messages"][-1].content)
    print_saved_state(agent, config)
    print("-" * 60)

    # 第二轮不再传 user_id/hobby/other_info，验证同一个 thread_id 是否能读取上轮状态。
    # dynamic_prompt 会再次运行，从已保存的 state 中读取这些字段并注入系统提示词。
    # 如果第一轮没有传入这些字段，也没有 tool/middleware 写入，dynamic_prompt 只能读到默认值。
    second_response = agent.invoke(  # type: ignore
        {"messages": [{"role": "user", "content": "你知道我的用户信息吗？请直接列出来。"}]},
        config=config,
    )

    print("第二轮回复:")
    print(second_response["messages"][-1].content)
    print_saved_state(agent, config)


if __name__ == "__main__":
    main()
