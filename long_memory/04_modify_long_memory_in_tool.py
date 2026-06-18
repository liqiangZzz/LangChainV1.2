"""
演示在工具中修改长期记忆。

本示例使用 InMemoryStore 保存用户偏好，工具可以通过 ToolRuntime 读取 context 和 store：
- save_user_preference：把用户偏好写入长期记忆。
- get_user_preference：从长期记忆读取用户偏好。

重点观察：
1. thread_id 控制短期记忆，同一个 thread_id 会保留对话历史。
2. user_id 控制长期记忆命名空间，不同 thread_id 也能读取同一个用户的长期记忆。
3. 工具内部可以直接 runtime.store.put/search，从而读写长期记忆。

本示例会调用真实 DeepSeek 模型，运行前需要确认 .env 中配置了 DeepSeek 信息。
"""
from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@dataclass
class UserContent:
    """本次 invoke 的运行时上下文。

    user_id 用来区分当前读写的是哪个用户的长期记忆。
    thread_id 控制短期记忆，user_id 控制长期记忆，两者不是一回事。
    """

    user_id: str


class UserPreference(BaseModel):
    """保存偏好工具的参数结构。

    模型调用 save_user_preference 时，必须按这个结构填写 category 和 preference。
    """

    category: Literal["color", "food", "music"] = Field(
        description="用户偏好类别，必须是 'color', 'food', 'music' 中的一个"
    )
    preference: str = Field(description="具体偏好内容，如 '蓝色'、'意大利面' 等")


@tool(args_schema=UserPreference)
def save_user_preference(category: str, preference: str, runtime: ToolRuntime) -> str:
    """
    将用户偏好保存到长期记忆中。

    Args:
       category: 用户偏好类别，必须是 "color", "food", "music" 中的一个
       preference: 具体偏好内容，如'红色'、'中国美食'等
       runtime: ToolRuntime  # 包含长期记忆存储和上下文
    Returns:
       str: 操作结果描述
    """

    user_id = runtime.context.user_id

    # 长期记忆命名空间。
    # 同一个 user_id 的偏好都写到 (user_id, "preference") 下。
    namespace = (user_id, "preference")

    # 使用 category 作为 key，表示同一类偏好只保留最新值。
    # 例如用户先说喜欢蓝色，后面又说喜欢绿色，那么 color 会被更新为绿色。
    runtime.store.put(namespace, category, {"category": category, "preference": preference})

    return f"已成功保存{user_id}的{category}偏好：{preference}"


@tool
def get_user_preference(runtime: ToolRuntime) -> str:
    """
    从长期记忆中获取用户偏好。

    Args:
       runtime: ToolRuntime  # 包含长期记忆存储和上下文
    Returns:
       str: 用户偏好描述
    """
    user_id = runtime.context.user_id

    # 查询时必须使用和保存时相同的 namespace，才能读到同一用户的偏好。
    namespace = (user_id, "preference")

    # search(namespace) 会返回这个命名空间下的所有记忆记录。
    memories = runtime.store.search(namespace)

    if not memories:
        return "没有找到用户偏好"

    # 格式化所有偏好为字符串列表
    preferences_list = []
    for memory in memories:
        pref = memory.value
        preferences_list.append(f"{pref['category']}: {pref['preference']}")

    preferences_list.sort()
    return "用户偏好：" + ", ".join(preferences_list)


def build_agent():
    """创建可以在工具中读写长期记忆的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        # 写入和读取都要注册为工具。
        # 第一、二轮主要调用 save_user_preference；第三轮需要调用 get_user_preference。
        tools=[save_user_preference, get_user_preference],
        # checkpointer 保存短期记忆，按 thread_id 隔离。
        checkpointer=InMemorySaver(),
        # store 保存长期记忆，按 namespace/key 读写；这里的 InMemoryStore 只在当前进程有效。
        store=InMemoryStore(),
        # context_schema 声明 invoke 时可以传入 UserContent(user_id=...)。
        context_schema=UserContent,
        system_prompt=(
            "当用户要求记住偏好时，必须调用 save_user_preference。"
            "当用户询问已经保存的偏好时，必须调用 get_user_preference。"
            "回答用户时要基于工具返回结果，不要凭空编造偏好。"
        ),
    )


def invoke_and_print(agent, thread_id: str, user_id: str, user_content: str) -> None:
    """发送一轮消息，并打印模型回复。

    Args:
        agent: 已创建好的 Agent。
        thread_id: 短期记忆会话 ID，用于隔离 checkpointer 中的 messages。
        user_id: 长期记忆用户 ID，用于定位 store 中的用户偏好命名空间。
        user_content: 本轮用户输入内容。
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config={"configurable": {"thread_id": thread_id}},
        context=UserContent(user_id=user_id),
    )

    print("[用户]", user_content)
    print("[模型回复]", result["messages"][-1].content)
    print("-" * 60)


def main() -> None:
    """演示同一用户的长期记忆可以跨 thread_id 读取。"""
    memory_agent = build_agent()
    user_id = "current_user"

    print("=== 完整演示：长期记忆的写入与跨线程读取 ===")

    print("第一轮（thread1）：保存颜色偏好")
    invoke_and_print(
        memory_agent,
        thread_id="thread1",
        user_id=user_id,
        user_content="请记住我喜欢的颜色是蓝色",
    )

    print("第二轮（thread1）：保存食物偏好")
    invoke_and_print(
        memory_agent,
        thread_id="thread1",
        user_id=user_id,
        user_content="我还喜欢的食物是意大利面",
    )

    print("第三轮（thread2）：新线程查询所有偏好")
    invoke_and_print(
        memory_agent,
        thread_id="thread2",
        user_id=user_id,
        user_content="告诉我我都喜欢什么颜色和食物",
    )


if __name__ == "__main__":
    main()
