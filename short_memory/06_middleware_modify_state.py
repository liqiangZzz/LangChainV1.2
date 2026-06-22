"""演示 middleware 如何修改 Agent state。

和 05_tool_modify_state.py 不同：
- 工具修改 state：只有模型调用工具时才会发生。
- middleware 修改 state：可以挂在 Agent/模型调用前后，自动统计或补充状态。
"""
from typing import Any, Dict

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@tool
def get_weather(city: str) -> str:
    """根据城市名称获取天气信息

    Args:
        city: 城市名称。
    """
    # 工具只需要返回普通字符串。
    # LangGraph 会把这个工具结果自动包装成 ToolMessage，并追加到 state["messages"]。
    return f"{city}天气晴朗"


@before_model
def before_model(state: AgentState, runtime: Runtime) -> Dict[str, Any]:
    """模型调用前运行，用当前 messages 重新统计工具调用次数。

    Args:
        state: 当前 Agent state。
        runtime: 工具或 middleware 的运行时对象，可读取 context、state、store 等信息。
    """
    # before_model 会在每次请求模型前执行。
    # 执行时机：用户消息、历史消息、已完成的工具消息进入 state 之后，真正调用 LLM 之前。
    # 这里统计的是“已经执行完成的工具结果”数量，也就是 messages 里的 ToolMessage 数量。
    # tool_call_count 不是手动 +1 得来的，而是每次重新扫描 messages 统计出来的。
    # 注意：模型刚决定调用工具时，工具还没执行，所以这里统计不到那次工具调用。
    messages_count = len(state["messages"])
    tool_call_count = len([msg for msg in state["messages"] if isinstance(msg, ToolMessage)])
    model_call_count = state.get("model_call_count", 0)

    print(
        "[before_model] 即将调用模型 | "
        f"messages_count={messages_count}, "
        f"tool_call_count={tool_call_count}, "
        f"model_call_count={model_call_count}"
    )

    # middleware 通过 return dict 更新 state。
    # 这里没有直接写 state["tool_call_count"] = tool_call_count。
    # 真正修改 state 的时机是：before_model 函数返回后，LangGraph 会把这个 dict
    # 合并进 Agent state，相当于更新 state["tool_call_count"]。
    # 更新完成后，模型调用时拿到的 state 中就包含最新的 tool_call_count。
    return {"tool_call_count": tool_call_count}


@after_model
def after_model(state: AgentState, runtime: Runtime) -> Dict[str, Any]:
    """模型调用后运行，用来累计模型调用次数。

    Args:
        state: 当前 Agent state。
        runtime: 工具或 middleware 的运行时对象，可读取 context、state、store 等信息。
    """
    # after_model 会在模型返回后执行。
    # 执行时机：LLM 已经生成 AIMessage，并追加到 state["messages"] 之后。
    # 这里读取的是本次模型调用结束后的 state，再基于旧值计算新的模型调用次数。
    old_model_call_count = state.get("model_call_count", 0)
    new_model_call_count = old_model_call_count + 1
    messages_count = len(state["messages"])
    tool_call_count = state.get("tool_call_count", 0)

    print(
        "[after_model] 模型调用结束 | "
        f"model_call_count: {old_model_call_count} -> {new_model_call_count}, "
        f"messages_count={messages_count}, "
        f"tool_call_count={tool_call_count}"
    )

    # 真正修改 state 的时机是：after_model 函数返回后。
    # LangGraph 会把返回值合并进 Agent state，相当于执行：
    # state["model_call_count"] = new_model_call_count
    # 因为本例配置了 checkpointer，同一个 thread_id 下一轮还能继续读到这个累计值。
    return {"model_call_count": new_model_call_count}


class CustomState(AgentState, total=False):
    """声明 middleware 会写入的自定义状态字段。"""

    tool_call_count: int
    model_call_count: int


def build_agent():
    """创建带 before_model / after_model 统计逻辑的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[get_weather],
        state_schema=CustomState,
        # state_schema 声明 tool_call_count / model_call_count 是允许保存到 state 的字段。
        # middleware 返回这些字段时，LangGraph 才能把它们作为自定义状态持久化。
        # middleware 按列表注册，让 Agent 在指定生命周期节点自动执行这些函数。
        middleware=[before_model, after_model],
        checkpointer=InMemorySaver(),
    )


def print_final_state(agent, config: dict) -> None:
    """只打印本例关心的统计字段。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
    """
    state = agent.get_state(config=config).values
    # get_state 读取的是 checkpointer 中指定 thread_id 的最新 state。
    # 这里看到的统计值，来自 before_model / after_model 返回 dict 后被合并保存的结果。
    print(
        "[当前统计]",
        {
            "model_call_count": state.get("model_call_count", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "messages_count": len(state.get("messages", [])),
        },
    )


def main() -> None:
    """运行三轮对话，观察 before_model 和 after_model 的统计变化。"""
    agent = build_agent()
    config = {"configurable": {"thread_id": "session001"}}

    print("\n[第1轮] 普通对话，通常不需要调用工具")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "你好，我叫张三"}]
    }, config=config)
    print("[模型回复]", result["messages"][-1].content)
    print_final_state(agent, config)
    print("*" * 60)

    print("\n[第2轮] 查询天气，通常会触发工具调用")
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "北京天气如何"}]
    }, config=config)
    print("[模型回复]", result2["messages"][-1].content)
    print_final_state(agent, config)
    print("*" * 60)

    print("\n[第3轮] 询问历史信息，观察统计是否继续累加")
    result3 = agent.invoke({
        "messages": [{"role": "user", "content": "你知道我的信息吗？ 直接列出来"}]
    }, config=config)
    print("[模型回复]", result3["messages"][-1].content)
    print_final_state(agent, config)
    print("*" * 60)

    print("\n[第4轮] 查询天气，通常会触发工具调用")
    result4 = agent.invoke({
        "messages": [{"role": "user", "content": "上海天气这么样？"}]
    }, config=config)
    print("[模型回复]", result4["messages"][-1].content)
    print_final_state(agent, config)
    print("*" * 60)

if __name__ == "__main__":
    main()
