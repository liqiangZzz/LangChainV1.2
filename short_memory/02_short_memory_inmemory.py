"""
演示带工具调用的内存版短期记忆。

本文件在 InMemorySaver 基础上加入 get_user_info 工具，并在最后通过 get_state()
查看同一 thread_id 下保存的 messages 和工具调用结果。
"""
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义查询工具 —— 工具结果也会进入短期记忆 messages
# =====================================================================
@tool
def get_user_info(name: str) -> str:
    """
    根据姓名查询用户信息
    Args：
        name (str): 用户姓名
    Returns:
        str: 用户信息
    """

    user_db = {
        "张三": "张三，男，1990年1月1日出生，汉族，中国公民，本科文化，工程师。",
        "李四": "李四，女，1991年2月2日出生，汉族，中国公民，本科文化，教师。",
    }
    return user_db.get(name, "未找到该用户")


# =====================================================================
# 2. 创建 Agent —— InMemorySaver 让同一 thread_id 保留上下文
# =====================================================================
agent = create_agent(
    model=deepseek_llm,
    tools=[get_user_info],
    system_prompt="你是一个助手，需要帮助时请随时告诉我。",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "session001"}}


# =====================================================================
# 3. 第一轮写入上下文 —— 用户姓名会被保存进会话消息
# =====================================================================
first_response = agent.invoke(  # type: ignore
    {"messages": [{"role": "user", "content": "你好，我叫张三"}]},
    config=config,
)

print("第一轮回复：")
print(first_response["messages"][-1].content)
print("-" * 50)


# =====================================================================
# 4. 第二轮读取上下文 —— 不再提供姓名，依赖短期记忆回答
# =====================================================================
# 第二轮没有再次提供名字，但 Agent 可以从同一 thread_id 的历史消息中读取。
second_response = agent.invoke(  # type: ignore
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config,
)

print("第二轮回复：")
print(second_response["messages"][-1].content)
print("-" * 50)


# =====================================================================
# 5. 查看最终 state —— 看看 messages 和工具结果都被保存成了什么
# =====================================================================
state = agent.get_state(config)
print(type(state))
print(state)
