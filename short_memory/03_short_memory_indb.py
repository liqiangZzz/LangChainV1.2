"""
演示把 Agent 短期记忆保存到 MySQL。

PyMySQLSaver 会把同一个 thread_id 下的 checkpoint 写入 MySQL。
和 InMemorySaver 不同，程序退出后 MySQL 中的会话状态仍然存在。

运行前准备：
1. 安装依赖：langgraph-checkpoint-mysql[pymysql]、PyMySQL[rsa]。
2. 创建数据库，例如 langchain_db。
3. 在 .env 中配置 MYSQL_DATABASE_URL，例如：
   MYSQL_DATABASE_URL=mysql://langchain_user:你的密码@localhost:3306/langchain_db?charset=utf8mb4

注意：
- 不要把数据库用户名和密码硬编码到代码里。
- 重复使用同一个 thread_id 会继续读取历史会话；想重新测试可以换一个 thread_id。
- 如果要完整重置 checkpoint，请删除 checkpoint 相关表，不要只清空 checkpoint_migrations。
- 这个示例会调用真实模型，运行前需要确认 .env 中配置了 DeepSeek 信息。
"""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from pymysql.err import OperationalError

from env_utils import MYSQL_DATABASE_URL


MYSQL_DATABASE_URL_EXAMPLE = (
    "mysql://langchain_user:你的密码@localhost:3306/langchain_db?charset=utf8mb4"
)


# =====================================================================
# 1. 定义示例工具 —— 第二轮会用它查询短期记忆里的姓名
# =====================================================================
@tool
def get_user_info(name: str) -> str:
    """根据姓名查询用户信息。

    Args:
        name: name 参数。
    """
    user_db = {
        "张三": {"age": 28, "hobby": "旅游、滑雪、喝茶"},
        "李四": {"age": 32, "hobby": "编程、阅读、电影"},
    }
    info = user_db.get(name, {"age": "未知", "hobby": "未知"})
    return f'姓名：{name}, 年龄：{info["age"]}, 爱好：{info["hobby"]}'


# =====================================================================
# 2. 创建 Agent —— MySQL checkpointer 负责持久化会话状态
# =====================================================================
def build_agent(checkpointer):
    """创建使用 MySQL checkpointer 的 Agent。"""
    from models.init_chat_model.init_chat_model_llm import deepseek_llm

    return create_agent(
        model=deepseek_llm,
        tools=[get_user_info],
        system_prompt="你是一个助手，需要帮助时请随时告诉我。",
        checkpointer=checkpointer,
    )


# =====================================================================
# 3. 读取数据库配置 —— 缺配置时直接给出可执行提示
# =====================================================================
def get_mysql_database_url() -> str:
    """读取 MySQL checkpoint 连接地址，并在缺失时给出明确提示。"""
    if not MYSQL_DATABASE_URL:
        raise RuntimeError(f"请先在 .env 中配置 MYSQL_DATABASE_URL，例如：{MYSQL_DATABASE_URL_EXAMPLE}")

    return MYSQL_DATABASE_URL


# =====================================================================
# 4. 初始化 checkpoint 表 —— 顺手兜住常见迁移表误删问题
# =====================================================================
def setup_checkpointer(checkpointer: PyMySQLSaver) -> None:
    """初始化 MySQL checkpoint 表，并把常见的手动删库问题转换成可读提示。

    Args:
        checkpointer: 短期记忆保存器，用于保存 messages 和 state。
    """
    try:
        # 第一次运行前需要 setup() 创建 checkpoint 相关表；重复执行是安全的。
        checkpointer.setup()
    except OperationalError as exc:
        error_code = exc.args[0] if exc.args else None
        error_message = str(exc)
        if error_code == 1061 and "Duplicate key name" in error_message:
            raise RuntimeError(
                "MySQL checkpoint 表结构和迁移记录不一致：索引已经存在，"
                "但 checkpoint_migrations 里的迁移记录被清空或丢失了。\n"
                "如果只是想清空会话数据，保留 checkpoint_migrations，只清空 "
                "checkpoints、checkpoint_blobs、checkpoint_writes。\n"
                "如果想完全重置，请先删除 checkpoint 相关表后再运行本脚本，"
                "让 checkpointer.setup() 重新建表。"
            ) from exc
        raise


# =====================================================================
# 5. 运行两轮对话 —— 用同一 thread_id 验证 MySQL 短期记忆
# =====================================================================
def main() -> None:
    """使用 MySQL 持久化保存同一 thread_id 的会话状态。"""
    database_url = get_mysql_database_url()

    with PyMySQLSaver.from_conn_string(database_url) as checkpointer:
        setup_checkpointer(checkpointer)

        agent = build_agent(checkpointer)

        # thread_id 是持久化会话 ID。相同 thread_id 会读取 MySQL 中已有历史。
        config = {"configurable": {"thread_id": "session001"}}

        first_response = agent.invoke(  # type: ignore
            {"messages": [{"role": "user", "content": "你好，我叫张三"}]},
            config=config,
        )
        print("第一轮回复：")
        print(first_response["messages"][-1].content)
        print("-" * 50)

        # 这里不再次提供姓名，测试 Agent 能否从 MySQL checkpoint 中读取上一轮信息。
        second_response = agent.invoke(  # type: ignore
            {"messages": [{"role": "user", "content": "我叫什么名字？请查询我的用户信息。"}]},
            config=config,
        )
        print("第二轮回复：")
        print(second_response["messages"][-1].content)


if __name__ == "__main__":
    main()
