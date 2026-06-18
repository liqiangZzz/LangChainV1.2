"""
智能电商客服助手综合示例。

本示例把短期记忆、长期记忆、工具更新 state/store、消息摘要和流式输出放在一起：

- 短期记忆：PyMySQLSaver 保存同一个 thread_id 下的 messages 和 state。
- 长期记忆：PyMySQLStore 保存用户偏好，跨 thread_id 也能读取。
- runtime context：UserContext 保存本次调用的用户 ID 和咨询渠道。
- 自定义 state：CustomerSessionState 保存当前会话正在查询的订单号。
- 工具：查询订单、写入偏好、读取偏好推荐商品。
- middleware：SummarizationMiddleware 压缩过长消息，wrap_tool_call 统一处理工具异常。

运行前需要：
1. 配置 MYSQL_DATABASE_URL。
2. 安装 MySQL 相关依赖。
3. 确认 .env 中配置了 DeepSeek 信息。
"""
from typing import Any
import warnings

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import SummarizationMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.mysql.pymysql import PyMySQLStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from env_utils import MYSQL_DATABASE_URL
from models.init_chat_model.init_chat_model_llm import deepseek_llm


# 当前依赖版本在 runtime.context 参与内部序列化时，可能打印 Pydantic warning。
# 这个 warning 不影响工具读取 context；这里过滤掉，保持控制台示例输出清爽。
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:*", category=UserWarning)


class UserContext(BaseModel):
    """每次调用 Agent 时传入的运行时上下文。

    context 不由模型生成，也不进入长期记忆。
    它通常来自登录态、请求头或业务系统。
    """

    user_id: str = Field(description="用户唯一标识")
    channel: str = Field(description="咨询渠道，如 APP、Web、小程序")


class CustomerSessionState(AgentState, total=False):
    """短期记忆 state。

    current_order_id 会被 checkpointer 按 thread_id 保存。
    用户换一个 thread_id 后，这个会话状态不会自动延续。
    """

    current_order_id: str


class OrderQuery(BaseModel):
    """查询订单工具的入参结构。"""

    order_id: str = Field(
        description="用户要查询的订单号，例如 order001 或 order002"
    )


class UserPreferenceUpdate(BaseModel):
    """更新用户长期偏好的入参结构。"""

    category: str = Field(
        description="偏好所属类别，例如 手机、手表、配件、耳机"
    )
    liked_item: str = Field(
        description="用户喜欢或本次订单体现出的具体商品、品牌或型号，例如 华为 P70"
    )


MOCK_ORDERS = {
    "order001": {
        "order_id": "order001",
        "status": "已发货",
        "product": "智能手机",
        "preference_category": "手机",
        "preference_item": "华为 P70",
    },
    "order002": {
        "order_id": "order002",
        "status": "待支付",
        "product": "智能手表",
        "preference_category": "手表",
        "preference_item": "Apple Watch Series 8",
    },
}


def preference_namespace(user_id: str) -> tuple[str, str]:
    """长期记忆命名空间：按用户隔离偏好数据。

    Args:
        user_id: 用户唯一标识，来自 UserContext.user_id。

    Returns:
        tuple[str, str]: LangGraph Store 使用的命名空间，例如
        ("user_customer_001", "preferences")。
    """
    return (f"user_{user_id}", "preferences")


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """获取当前用户信息和会话中的当前订单号。

    Args:
        runtime: 工具运行时对象，可读取 context、state、store 和 tool_call_id。

    Returns:
        str: 当前用户 ID、咨询渠道和当前订单号。
    """
    current_order_id = runtime.state.get("current_order_id", "无")
    return (
        f"用户ID: {runtime.context.user_id}, "
        f"咨询渠道: {runtime.context.channel}, "
        f"当前查询订单号: {current_order_id}"
    )


# args_schema 会暴露给模型，帮助模型知道调用工具时应该填写哪些字段。
@tool(args_schema=OrderQuery)
def query_order_status(order_id: str, runtime: ToolRuntime) -> Command:
    """查询订单状态，并把当前订单号写入短期记忆 state。

    Args:
        order_id: 用户要查询的订单号，例如 order001。
        runtime: 工具运行时对象，用于读取 tool_call_id 并返回 ToolMessage。

    Returns:
        Command: 用于更新短期记忆 state 和 messages 的命令对象。
    """
    order_info = MOCK_ORDERS.get(order_id)
    if order_info is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"错误：订单 [{order_id}] 不存在。",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    tool_message = (
        f"订单 [{order_id}] 状态: {order_info['status']}，商品: {order_info['product']}。"
        f"可用于长期偏好的信息：{order_info['preference_category']} / "
        f"{order_info['preference_item']}。"
    )

    # Command(update=...) 可以同时更新自定义 state 和 messages。
    # 这里 current_order_id 会进入短期记忆，后续推荐工具可以读取。
    return Command(
        update={
            "current_order_id": order_id,
            "messages": [
                ToolMessage(content=tool_message, tool_call_id=runtime.tool_call_id)
            ],
        }
    )


# 使用 args_schema 后，category 和 liked_item 的含义会更清楚，
# 模型更容易从订单工具结果中提取正确偏好并调用本工具。
@tool(args_schema=UserPreferenceUpdate)
def update_user_preference(category: str, liked_item: str, runtime: ToolRuntime) -> str:
    """把用户偏好写入长期记忆 store。

    Args:
        category: 偏好类别，例如 手机、手表、配件。
        liked_item: 具体偏好对象，例如 华为 P70。
        runtime: 工具运行时对象，用于读取 context.user_id 和写入 store。

    Returns:
        str: 写入长期记忆后的确认信息。
    """
    namespace = preference_namespace(runtime.context.user_id)

    # 用 category 作为 key，表示同类偏好保留最新值。
    # 如果想保留多条历史偏好，可以改成 uuid 作为 key。
    runtime.store.put(
        namespace,
        category,
        {"category": category, "liked_item": liked_item},
    )
    return f"已记录长期偏好：用户喜欢 {category} 类的 {liked_item}。"


@tool
def get_recommendation(runtime: ToolRuntime) -> str:
    """读取长期偏好，并结合当前订单给出推荐依据。

    Args:
        runtime: 工具运行时对象，用于读取当前 state 和长期记忆 store。

    Returns:
        str: 基于当前订单和长期偏好的推荐依据。
    """
    namespace = preference_namespace(runtime.context.user_id)
    current_order_id = runtime.state.get("current_order_id", "未知订单")
    preferences = runtime.store.search(namespace)

    if not preferences:
        return f"当前订单 [{current_order_id}] 暂无长期偏好记录，可先根据订单商品做通用推荐。"

    preference_text = "、".join(
        f"{item.value.get('category')}({item.value.get('liked_item')})"
        for item in preferences
    )
    return f"当前订单 [{current_order_id}]；用户长期偏好：{preference_text}。"



@wrap_tool_call
def handle_tool_errors(request: Any, handler: Any) -> ToolMessage:
    """统一处理工具异常，避免原始异常直接打断 Agent。

    Args:
        request: 当前工具调用请求，包含 tool_call 等信息。
        handler: LangChain 提供的工具调用处理函数。

    Returns:
        ToolMessage: 工具正常结果，或包装后的错误消息。
    """
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(
            content=f"调用工具错误，请稍后重试。错误信息：{exc}",
            tool_call_id=request.tool_call["id"],
        )


SYSTEM_PROMPT = """
你是一个智能电商客服助手。

工具使用规则：
1. 用户询问个人信息时，调用 get_user_info。
2. 用户查询订单状态时，调用 query_order_status。
3. 查询到订单后，如果工具结果里包含可用于长期偏好的商品信息，再调用 update_user_preference。
4. 用户要求推荐商品时，调用 get_recommendation。
5. 回答必须基于工具结果，不要编造订单状态或用户偏好。
""".strip()


def build_agent(checkpointer: PyMySQLSaver, store: PyMySQLStore):
    """创建同时使用短期记忆、长期记忆和 middleware 的 Agent。

    Args:
        checkpointer: MySQL 短期记忆保存器，用于保存 messages 和 state。
        store: MySQL 长期记忆存储，用于保存用户偏好等业务数据。

    Returns:
        已配置工具、state_schema、context_schema 和 middleware 的 Agent。
    """
    return create_agent(
        model=deepseek_llm,
        tools=[get_user_info, query_order_status, update_user_preference, get_recommendation],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        state_schema=CustomerSessionState,
        context_schema=UserContext,
        middleware=[
            SummarizationMiddleware(
                model=deepseek_llm,
                summary_prompt="请总结以下对话内容，保留用户问题、订单号、订单状态和偏好信息：{messages}",
                # 消息超过 10 条后触发摘要，只保留最近 5 条原始消息。
                trigger=("messages", 10),
                keep=("messages", 5),
            ),
            handle_tool_errors,
        ],
    )


def print_stream_chunk(chunk: dict[str, Any]) -> None:
    """只打印模型节点和工具节点产生的最新消息。

    Args:
        chunk: agent.stream(...) 每次产出的流式事件片段。
    """
    for step, data in chunk.items():
        if step not in {"model", "tools"}:
            continue

        messages = data.get("messages", [])
        if not messages:
            continue

        messages[-1].pretty_print()


def run_chat_loop(agent, config: dict[str, Any], user_context: UserContext) -> None:
    """控制台交互循环。

    Args:
        agent: 已创建好的客服 Agent。
        config: LangGraph 运行配置，主要包含 configurable.thread_id。
        user_context: 本次会话的运行时上下文，包含 user_id 和 channel。
    """
    print("=" * 50)
    print("智能电商客服助手")
    print("示例问题：查询 order001；给我推荐配件；你知道我的信息吗？")
    print("输入 quit / exit / 退出 / q 结束对话。")
    print("=" * 50)

    while True:
        user_input = input("[你]: ").strip()
        if user_input.lower() in {"quit", "exit", "退出", "q"}:
            print("客服助手: 感谢你的咨询，再见！")
            break

        if not user_input:
            continue

        # messages 必须是列表。每轮只传入本轮新增用户消息；
        # 历史消息由 checkpointer 根据 thread_id 自动接上。
        input_data = {"messages": [{"role": "user", "content": user_input}]}

        print("[客服助手]:")
        try:
            for chunk in agent.stream(input_data, config=config, context=user_context):
                print_stream_chunk(chunk)
        except Exception as exc:
            print(f"调用过程中出现错误: {exc}")


def main() -> None:
    """初始化 MySQL 记忆后启动交互式客服助手。"""
    if not MYSQL_DATABASE_URL:
        raise ValueError("请先在 .env 中配置 MYSQL_DATABASE_URL。")

    with (
        PyMySQLSaver.from_conn_string(MYSQL_DATABASE_URL) as checkpointer,
        PyMySQLStore.from_conn_string(MYSQL_DATABASE_URL) as store,
    ):
        checkpointer.setup()
        store.setup()

        agent = build_agent(checkpointer, store)
        user_context = UserContext(user_id="customer_001", channel="Web")

        # thread_id 决定短期记忆会话。换 thread_id 会开启新的短期会话；
        # 但长期偏好仍然按 user_id 保存在 store 中。
        config = {"configurable": {"thread_id": "customer_001_session_01"}}
        run_chat_loop(agent, config, user_context)


if __name__ == "__main__":
    main()
