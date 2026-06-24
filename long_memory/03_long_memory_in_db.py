"""
演示把长期记忆和短期记忆都保存到 MySQL。

- PyMySQLSaver：保存 checkpointer，也就是同一 thread_id 下的短期记忆 messages。
- PyMySQLStore：保存长期记忆 store，例如用户资料、偏好、画像等业务数据。

运行前需要：
1. 配置 MYSQL_DATABASE_URL。
2. 安装 MySQL checkpoint/store 相关依赖。
"""
from dataclasses import dataclass
import warnings

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.mysql import PyMySQLStore

from env_utils import MYSQL_DATABASE_URL
from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义查询工具 —— 工具直接读取 MySQL Store 中的长期记忆
# =====================================================================
@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """根据当前 runtime.context.user_id 从 MySQL 长期记忆中查询用户信息。"""
    user_id = runtime.context.user_id

    user_info = runtime.store.get(("users",), user_id)

    if user_info is None:
        return "没有找到用户信息"

    value = user_info.value
    return f"用户id:{user_id},用户姓名：{value['name']},用户age:{value['age']}，用户city:{value['city']},用户hobby:{value['hobby']}"


# =====================================================================
# 2. 声明运行时上下文 —— 本轮调用要查谁，由 user_id 说了算
# =====================================================================
@dataclass
class UserContext:
    """本次 invoke 传入的运行时上下文。

    user_id 只决定本次工具 get_user_info 去长期记忆 store 里查哪个用户。
    它不会自动清空 checkpointer 中同一个 thread_id 的短期记忆 messages。
    """

    user_id: str


# ToolRuntime.context 参与 LangGraph 内部序列化时，当前依赖版本可能打印
# PydanticSerializationUnexpectedValue warning。这个 warning 不影响工具读取 context，
# 这里只过滤它，避免学习示例输出被 warning 打断。
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:*", category=UserWarning)


# =====================================================================
# 3. 打开 MySQL 记忆组件 —— checkpointer 管短期，store 管长期
# =====================================================================
with (
    PyMySQLSaver.from_conn_string(MYSQL_DATABASE_URL) as checkpointer,
    PyMySQLStore.from_conn_string(MYSQL_DATABASE_URL) as store):
    # 初始化数据库
    checkpointer.setup()
    store.setup()

    store.put(
        ("users",),
        "user_123",
        {"name": "张三", "age": 28, "city": "北京", "hobby": "编程、阅读"}
    )
    store.put(
        ("users",),
        "user_456",
        {"name": "李四", "age": 32, "city": "上海", "hobby": "旅游、摄影"}
    )


    # =================================================================
    # 4. 创建 Agent —— 把 MySQL 版短期记忆和长期记忆都接入
    # =================================================================
    agent = create_agent(
        model=deepseek_llm,
        tools=[get_user_info],
        checkpointer=checkpointer,
        store=store,
        system_prompt="每次查询用户信息时，都要调用工具get_user_info",
        context_schema=UserContext
    )

    # =================================================================
    # 5. 连续调用两轮 —— 看看长期记忆切换和短期记忆延续如何并存
    # =================================================================
    # 两次调用故意使用同一个 thread_id。
    # 这样可以观察：
    # - 长期记忆 store 会根据 runtime.context.user_id 查询不同用户。
    # - 短期记忆 checkpointer 会把第一次对话 messages 带入第二次调用。
    # 所以第二次模型可能先受到上一轮“张三”的短期记忆影响，
    # 再通过工具结果纠正为当前 context=user_456 对应的“李四”。
    config = {"configurable": {"thread_id": "session001"}}

    resp1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你知道我的信息吗？"}]},
        config=config,
        context=UserContext(user_id="user_123")
    )
    print(resp1["messages"][-1].content)
    print("*" * 70)

    resp2 = agent.invoke(
        {"messages": [{"role": "user", "content": "你知道我的信息吗？"}]},
        config=config,
        context=UserContext(user_id="user_456")
    )
    print(resp2["messages"][-1].content)
