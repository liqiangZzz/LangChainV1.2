"""演示工具如何修改 Agent state。

普通工具返回字符串时，只会把结果追加到 messages 中。
如果工具返回 Command(update=...)，就可以同时更新自定义 state 字段。
"""
from langchain.agents import AgentState, create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@tool
def get_info(runtime: ToolRuntime) -> str:
    """读取当前 Agent state 中保存的用户信息。"""
    name = runtime.state.get("user_name", "未知")
    hobby = runtime.state.get("hobby", [])
    return f"用户：{name}，用户爱好：{','.join(hobby)}"


@tool
def update_info(user_name: str, hobby: list[str], runtime: ToolRuntime) -> Command:
    """
    更新用户信息。

    注意：
    普通 tool 返回 str 时，LangGraph 只会自动追加 ToolMessage 到 messages。

    如果希望 tool 修改 AgentState，
    需要返回 Command(update={...})。

    Command.update 中的字段会被 LangGraph 合并到当前 state。

    Args:
        name: 用户姓名。
        hobby: 用户爱好列表。
    Returns:
        Command: 更新后的 Agent state 和工具消息。
    """
    if not user_name or not hobby:
        # 工具返回 Command 时，也要写入一条 ToolMessage，
        # 用来回应本次模型发起的 tool_call。
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="错误：姓名或爱好不能为空",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    # Command 是这个示例的重点：
    # update 中不仅可以追加 ToolMessage，还可以更新自定义 state 字段。
    # ====================================================
    # Command(update=...) 是工具修改 state 的核心
    #
    # 注意：
    # 这里不是直接修改 runtime.state
    #
    # 不要：
    # runtime.state["user_name"] = user_name
    #
    # 正确：
    # 返回 Command，让 LangGraph runtime
    # 在工具结束后统一更新 state。
    # ====================================================
    return Command(
        update={
            # ----------------------------
            # 自定义 state 字段更新
            # ----------------------------
            #
            # 这里的 key 必须存在于：
            #
            # class CustomState(AgentState):
            #     user_name: str
            #     hobby: list[str]
            #
            # LangGraph 收到后：
            #
            # state["user_name"] = user_name
            # state["hobby"] = hobby
            #
            "user_name": user_name,
            "hobby": hobby,

            # messages 更新
            "messages": [
                ToolMessage(
                    # ToolMessage 会进入 messages，告诉模型工具执行结果，它不是保存用户资料用的。
                    content=f"已更新用户档案：姓名={user_name}, 爱好={','.join(hobby)}",
                    # 必须对应当前 tool_call_id，否则工具消息不会进入 messages
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class CustomState(AgentState, total=False):
    """
   自定义 Agent 状态。

    AgentState 默认只有：
        messages

    这里扩展两个字段：
            user_name
            hobby

    之后工具返回：

    Command(
        update={
            "user_name": "...",
            "hobby": [...]
        }
    )

    LangGraph 才知道这些字段属于 state。
    """

    user_name: str
    hobby: list[str]


def build_agent():
    """创建允许工具修改 state 的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[get_info, update_info],
        # state_schema 声明 Command(update=...) 中允许更新哪些自定义字段。
        state_schema=CustomState,
        system_prompt=(
            "你是一个助手。"
            "当用户提供姓名和爱好时，必须调用 update_info 保存到 state。"
            "当用户询问已保存的信息时，必须调用 get_info 查询 state。"
        ),
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    """运行两轮对话，演示工具写入并更新 state。"""
    agent = build_agent()
    config = {"configurable": {"thread_id": "session001"}}

    result = agent.invoke(  # type: ignore
        {"messages": [{"role": "user", "content": "我叫王五，我的爱好是钓鱼和唱歌，请保存。"}]},
        config=config,
    )

    print("模型回复:", result["messages"][-1].content)
    print("-" * 60)

    result2 = agent.invoke(  # type: ignore
        {"messages": [{"role": "user", "content": "请把我的爱好更新为钓鱼、唱歌和旅游。"}]},
        config=config,
    )
    print("模型回复:", result2["messages"][-1].content)
    print("=" * 20)
    print("当前状态:", agent.get_state(config=config))


if __name__ == "__main__":
    main()
