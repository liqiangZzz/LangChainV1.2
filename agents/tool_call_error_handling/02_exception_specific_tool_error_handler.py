"""演示使用 wrap_tool_call 按异常类型处理 Agent 工具调用错误。

示例包含查询工单和更新工单两个工具，并主动模拟以下异常：

- ConnectionError：数据库连接失败等临时系统异常。
- PermissionError：当前用户没有执行操作的权限。
- ToolException：工具参数不符合业务规则。
- Exception：未被预料到的其他异常。

middleware 会把异常转换为与原工具调用关联的 ToolMessage。模型收到工具错误消息后，
仍然可以继续生成友好的最终回答，而不是让整个 Agent 执行流程直接中断。
"""

import json
import random

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# 使用字典模拟工单数据库。示例中的查询和更新都只操作内存数据，
# 不会连接真实数据库；脚本结束后，修改过的状态也不会被持久化。
TICKET_DATABASE = {
    "T001": {"title": "登录问题", "status": "处理中", "assignee": "张三"},
    "T002": {"title": "支付失败", "status": "已解决", "assignee": "李四"},
    "T003": {"title": "页面加载慢", "status": "待处理", "assignee": "王五"},
}


@tool
def query_tickets(ticket_id: str) -> str:
    """根据工单 ID 查询工单详情。

    Args:
        ticket_id: 工单 ID，例如 T001。

    Returns:
        包含工单详情的 JSON 字符串。

    Raises:
        ConnectionError: 模拟数据库临时连接失败。
        ToolException: 指定的工单 ID 不存在。
    """
    # random.random() 会返回 [0, 1) 之间的随机数。
    # 小于 0.5 时主动抛出连接异常，因此同一测试用例每次运行的结果可能不同。
    if random.random() < 0.5:
        raise ConnectionError("数据库连接超时，请稍后重试")

    # ToolException 表示工具能够预期的业务错误，例如参数合法但查询不到数据。
    # 它和数据库断开等系统错误不同，适合向模型返回“请检查输入”的提示。
    if ticket_id not in TICKET_DATABASE:
        raise ToolException(f"工单ID {ticket_id} 不存在")

    # ensure_ascii=False 保留负责人等中文内容，方便观察 ToolMessage。
    return json.dumps(TICKET_DATABASE[ticket_id], ensure_ascii=False)


@tool
def update_ticket(
    ticket_id: str,
    new_status: str,
) -> str:
    """更新指定工单的状态。

    Args:
        ticket_id: 工单 ID，例如 T001。
        new_status: 要更新到的新状态。

    Returns:
        状态更新成功后的文本说明。

    Raises:
        ToolException: 工单不存在或状态不在允许范围内。
        PermissionError: 模拟当前用户没有更新权限。
    """
    # 业务允许的状态集合。模型如果生成“完结”等其他值，会触发 ToolException。
    valid_statuses = [
        "待处理",
        "处理中",
        "已解决",
        "已关闭",
    ]

    if ticket_id not in TICKET_DATABASE:
        raise ToolException(f"工单ID {ticket_id} 不存在")

    if new_status not in valid_statuses:
        raise ToolException(f"状态必须是: {', '.join(valid_statuses)}")

    # 以 20% 概率模拟权限校验失败，用于演示 PermissionError 分支。
    # 这里对所有状态都可能触发权限异常，仅用于教学，不代表真实权限规则。
    if random.random() < 0.2:
        raise PermissionError("权限不足：只有管理员可以关闭工单")

    # 校验通过后才修改内存中的工单状态。
    TICKET_DATABASE[ticket_id]["status"] = new_status

    return f"工单 {ticket_id} 状态已更新为: {new_status}"


# wrap_tool_call 会包裹 Agent 发起的每一次工具调用。
# handler(request) 才是真正执行模型所选工具的步骤；在它外层使用 try/except，
# 就可以集中处理所有已注册工具抛出的异常，而不用在每个工具中重复写错误响应。
@wrap_tool_call
def handle_tool_call_error(request: ToolCallRequest, handler):
    """执行工具，并按照异常类型返回不同的 ToolMessage。"""
    try:
        # 工具正常执行时，直接返回原始工具结果。
        return handler(request)
    except ConnectionError as e:
        # 临时基础设施故障通常适合提示用户稍后重试。
        return ToolMessage(
            content=(
                f"系统暂时繁忙：{str(e)}。"
                "建议您稍后重试此操作。"
            ),
            # 每条 ToolMessage 必须关联模型原始工具调用的 ID，
            # 否则模型无法判断这条结果属于哪一次工具调用。
            tool_call_id=request.tool_call["id"],
        )
    except PermissionError as e:
        # 权限错误不应该建议模型修改参数重试，而应提示用户联系管理员。
        return ToolMessage(
            content=(
                f"权限限制：{str(e)}。"
                "如需执行此操作，请联系管理员。"
            ),
            tool_call_id=request.tool_call["id"],
        )
    except ToolException as e:
        # ToolException 表示可预期的业务校验错误，例如工单不存在或状态非法。
        # 模型可以根据此消息向用户解释问题，或尝试生成更合理的参数。
        return ToolMessage(
            content=(
                f"输入验证失败：{str(e)}。"
                "请检查输入参数是否正确。"
            ),
            tool_call_id=request.tool_call["id"],
        )
    except Exception as e:
        # 最后的 Exception 分支负责兜底，避免未知错误直接中断 Agent。
        # 生产环境中还应在这里记录日志和异常堆栈，而不是只返回文本。
        return ToolMessage(
            content=(
                f"意外错误：{str(e)}。"
                "技术团队已收到通知，请稍后重试。"
            ),
            tool_call_id=request.tool_call["id"],
        )


# middleware 列表中的 handle_tool_call_error 会作用于下面注册的所有工具。
agent = create_agent(
    model=deepseek_llm,
    tools=[query_tickets, update_ticket],
    system_prompt=(
        "你是一个助手，"
        "你可以查询工单系统，并更新工单状态。"
    ),
    middleware=[handle_tool_call_error],
)


def test_error() -> None:
    """依次调用 Agent，观察不同工具异常的处理结果。"""
    test_cases = [
        # 工单不存在，预期触发 ToolException。
        "查询工单T999的详情",
        # “完结状态”不在 valid_statuses 中，预期触发 ToolException。
        "把工单T001状态更新为完结状态",
        # 工单存在，但查询工具仍有 50% 概率触发 ConnectionError。
        "查询工单T001的详情",
        # 状态参数合法，但更新工具有 20% 概率触发 PermissionError。
        "关闭工单T002",
    ]

    for query in test_cases:
        try:
            print("=" * 50)
            print(f"用户查询: {query}")

            # 一次 invoke 通常包含以下步骤：
            # 1. 模型理解问题并生成工具调用。
            # 2. middleware 包裹并执行工具。
            # 3. 工具结果或错误 ToolMessage 返回给模型。
            # 4. 模型根据工具消息生成最终回复。
            # 因此每个测试用例可能触发多次真实模型请求。
            response = agent.invoke({  # type: ignore
                "messages": [{"role": "user", "content": query}]
            })

            # 返回消息列表的最后一条通常是模型面向用户生成的最终回答。
            result = response["messages"][-1]
            print(f"助手回复: {result.content}")
        except Exception as e:
            # middleware 只处理工具执行阶段的异常。
            # 模型请求失败、配置错误等 Agent 层异常仍可能传播到这里。
            print(f"系统异常: {e}")


if __name__ == "__main__":
    test_error()
