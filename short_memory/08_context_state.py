"""演示 runtime context 和 Agent state 的区别。

context：本次 invoke 传入的运行时上下文，只在当前调用中有效。
state：Agent 会话状态，会被 checkpointer 按 thread_id 保存下来。
"""
import warnings
from typing import Any, Dict, TypedDict

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

from models.init_chat_model.init_chat_model_llm import deepseek_llm


class ConversationContext(TypedDict, total=False):
    """运行时上下文：只对本次 invoke 生效。"""

    user_name: str
    channel: str


class ConversationState(AgentState, total=False):
    """状态：会话中的动态记忆"""

    user_name: str  # 用户名称
    channel: str  # 渠道名称
    call_llm_count: int  # 大模型调用次数


def format_state_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """只保留学习本例需要关注的 state 字段。"""
    return {
        "user_name": state.get("user_name"),
        "channel": state.get("channel"),
        "call_llm_count": state.get("call_llm_count", 0),
        "messages_count": len(state.get("messages", [])),
    }


@tool
def get_weather(location: str, runtime: ToolRuntime[ConversationContext, ConversationState]) -> str:
    """获取指定位置的天气"""
    # 工具可以通过 ToolRuntime 读取当前 state 和 context。
    # runtime 不会暴露给模型，模型只需要填写 location。
    print("[tool] runtime.state 摘要:", format_state_summary(runtime.state))
    print("[tool] runtime.context:", runtime.context)

    return f"{location}的天气是晴天"


@after_model
def save_context_to_state(state: ConversationState, runtime: Runtime[ConversationContext]) -> Dict[str, Any]:
    """模型调用后运行，把本次 context 合并进可持久化 state。"""
    # 一轮用户请求如果触发了工具调用，通常会有两次模型调用：
    # 第一次模型返回 AIMessage(tool_calls=...)，只是“决定要调用工具”。
    # after_model 会在这次模型返回后立即执行，所以它会出现在真正工具调用之前。
    # 然后 Agent 才根据 tool_calls 执行工具。
    # 第二次模型基于工具结果 ToolMessage 生成最终回答。
    # 所以 after_model 在一轮用户请求中可能执行多次。
    last_message = state["messages"][-1]
    has_tool_calls = isinstance(last_message, AIMessage) and bool(last_message.tool_calls)
    if has_tool_calls:
        print("[after_model] 第1次模型返回 tool_calls，Agent 接下来会调用工具。")
        print("[after_model] state 摘要:", format_state_summary(state))
        print("[after_model] context:", runtime.context)

    context = runtime.context or {}

    # context 只在本次 invoke 有效；没有传 context 时，继续沿用 state 中已有值。
    user_name = context.get("user_name") or state.get("user_name", "")
    channel = context.get("channel") or state.get("channel", "")

    # state 会被 checkpointer 保存，同一个 thread_id 下一轮还能继续读取。
    call_llm_count = state.get("call_llm_count", 0) + 1

    state_update = {"user_name": user_name, "channel": channel, "call_llm_count": call_llm_count}
    if has_tool_calls:
        print("[after_model] 即将写入 state:", state_update)

    # middleware 不需要直接 state["user_name"] = user_name。
    # 返回的 dict 会由 LangGraph 合并进 Agent state。
    # 所以后续工具执行时，runtime.state 就能读到这些新值。
    return state_update


def build_agent():
    """创建同时声明 context_schema 和 state_schema 的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[get_weather],
        # state_schema 声明会被 checkpointer 保存的会话状态。
        state_schema=ConversationState,
        # context_schema 声明本次 invoke 可传入的运行时上下文。
        context_schema=ConversationContext,
        checkpointer=InMemorySaver(),
        middleware=[save_context_to_state],
    )


def main() -> None:
    """运行两轮对话，观察 context 和 state 的差异。"""
    # 当前依赖版本在 ToolRuntime.context 参与内部序列化时，可能打印 Pydantic warning。
    # 这个 warning 不影响 context/state 的正常使用；这里仅过滤这类已知提示，保持示例输出清爽。
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings:*", category=UserWarning)

    agent = build_agent()
    config = {"configurable": {"thread_id": "1"}}

    print("\n[第1轮] 传入 context：user_name=张三, channel=微信")
    print("[说明] 一轮对话里只要模型调用了工具，after_model 通常会执行两次。")
    print("[说明] 为了看清流程，这里只打印模型确认要调用工具后的 after_model。")
    resp = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，北京天气怎么样"}]},
        config=config,
        context={"user_name": "张三", "channel": "微信"},
    )
    print("[第1轮回复]", resp["messages"][-1].content)
    print("[第1轮状态]", format_state_summary(agent.get_state(config=config).values))

    print("*" * 60)

    print("\n[第2轮] 不传 context，观察 state 是否保留上一轮信息")
    resp2 = agent.invoke(
        {"messages": [{"role": "user", "content": "上海天气怎么样"}]},
        config=config,
    )
    print("[第2轮回复]", resp2["messages"][-1].content)
    print("[第2轮状态]", format_state_summary(agent.get_state(config=config).values))


if __name__ == "__main__":
    main()
