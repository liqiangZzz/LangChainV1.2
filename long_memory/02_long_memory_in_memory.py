"""
演示 Agent 如何读取进程内长期记忆。

本文件用 InMemoryStore 保存用户资料，用 InMemorySaver 保存同一 thread_id 下的短期记忆。
重点观察：runtime.context.user_id 会决定工具查哪个用户，但同一个 thread_id 的历史消息
仍然可能影响模型后续回答。
"""
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore

from models.init_chat_model.init_chat_model_llm import deepseek_llm

checkpointer = InMemorySaver()
store = InMemoryStore()


# =====================================================================
# 1. 准备长期记忆数据 —— 先把两位用户资料放进 store
# =====================================================================
# InMemoryStore 是长期记忆，这里用它保存“跨会话也可以查询”的业务数据。
# 注意：这里的长期记忆只在当前进程内有效，程序退出后会丢失。
#
# 数据结构：
# - namespace=("users",)：把所有用户资料放在 users 命名空间下。
# - key="user_123" / "user_456"：每个用户自己的唯一键。
# - value={...}：真正保存的用户资料。
store.put(
    ("users",),  # 命名空间：用户数据
    "user_123",  # 键：用户ID
    {"name": "张三", "age": 28, "city": "北京", "hobby": "编程、阅读"}  # 值：用户信息
)
store.put(
    ("users",),
    "user_456",
    {"name": "李四", "age": 32, "city": "上海", "hobby": "旅游、摄影"}
)


# =====================================================================
# 2. 声明运行时上下文 —— user_id 决定工具读取哪份长期记忆
# =====================================================================
@dataclass
class UserContent():
    """本次调用传入的运行时上下文。

    context 只表示“这一次 invoke 要查哪个用户”，不会自动清空短期记忆。
    如果两次 invoke 使用同一个 thread_id，checkpointer 仍会把上一轮对话历史带进来。
    """

    user_id: str


# =====================================================================
# 3. 定义查询工具 —— 从 runtime.store 中捞出当前用户资料
# =====================================================================
@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """从长期记忆中查询用户信息。

    Args:
        runtime: 工具运行时对象，用于读取 context.user_id 和长期记忆 store。

    Returns:
        str: 查询到的用户信息；如果用户不存在，返回未找到提示。
    """
    # runtime.context 来自 agent.invoke(..., context=UserContent(...))。
    # 这里读取 user_id 后，再到 runtime.store 里查询对应用户资料。
    user_id = runtime.context.user_id

    # runtime.store 就是 create_agent(..., store=store) 传入的 InMemoryStore。
    # get(namespace, key) 会返回一个 item；真正的业务数据在 item.value 中。
    user_info = runtime.store.get(("users",), user_id)

    if user_info:
        value = user_info.value
        return (
            f"用户id:{user_id},"
            f"用户姓名：{value['name']},"
            f"用户age:{value['age']},"
            f"用户city:{value['city']},"
            f"用户hobby:{value['hobby']}"
        )

    return f"未找到用户ID为 {user_id} 的用户信息"


# =====================================================================
# 4. 创建 Agent —— 短期记忆和长期记忆在这里一起挂上车
# =====================================================================
agent = create_agent(
    model=deepseek_llm,
    # checkpointer 管的是短期记忆：同一个 thread_id 下的 messages 会被持续保存。
    checkpointer=checkpointer,
    tools=[get_user_info],
    # store 管的是长期记忆：工具可以通过 runtime.store 查询业务数据。
    store=store,
    # context_schema 声明本次 invoke 可以传入 UserContent(user_id=...)。
    context_schema=UserContent,
    # 这个系统提示词要求模型查询用户信息时必须调用 get_user_info 工具。
    system_prompt="每次查询用户信息时，都要调用工具get_user_info",
)


# =====================================================================
# 5. 连续调用两轮 —— 观察同一 thread_id 下短期记忆的影响
# =====================================================================
# 两次调用使用了同一个 thread_id。
# 这意味着第二次调用时，Agent 不只会看到第二次用户问题，
# 还会看到第一次对话留下的短期记忆 messages。
config = {"configurable": {"thread_id": "session001"}}

response = agent.invoke({  # type: ignore
    "messages": [{"role": "user", "content": "你知道我的信息吗？"}]},
    config=config,
    # 第一次调用的运行时上下文是 user_123，所以工具会从长期记忆中查到张三。
    context=UserContent(user_id="user_123")
)
print(response['messages'][-1].content)
print("*" * 70)

response2 = agent.invoke({  # type: ignore
    "messages": [{"role": "user", "content": "你知道我的信息吗？"}]},
    config=config,
    # 第二次调用的运行时上下文变成 user_456，工具查询目标应该是李四。
    #
    # 但如果模型最终仍回答“张三”，原因通常不是长期记忆 store 查错了，
    # 而是因为两次调用使用了同一个 thread_id=session001：
    # checkpointer 会把第一次对话历史保存在短期记忆 messages 中，
    # 第二次调用模型时，这些历史消息会一起进入上下文。
    # 模型可能受上一轮“张三”的短期记忆影响，优先沿用了旧上下文。
    #
    # 如果想让第二次完全不受第一次影响，可以：
    # 1. 使用新的 thread_id，例如 session002；
    # 2. 或者在调用前清理/截断/删除短期记忆 messages；
    # 3. 或者在提示词中明确要求以 runtime.context.user_id 对应的工具结果为准。
    context=UserContent(user_id="user_456")
)
print(response2['messages'][-1].content)
print("*" * 70)
