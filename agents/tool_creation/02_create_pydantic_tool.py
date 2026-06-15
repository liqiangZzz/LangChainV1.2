"""使用 Pydantic 模型定义具有强类型参数的 LangChain 工具。

Pydantic 模型会被转换为工具的参数 Schema，帮助大模型理解每个参数的名称、
类型、可选值和用途。工具真正执行前，Pydantic 还会校验模型生成的参数，
不符合要求的数据不会直接进入工具函数。
"""
import json
from typing import Literal, Optional

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from models.init_chat_model.init_chat_model_llm import deepseek_llm


class TicketQueryInput(BaseModel):
    """定义 query_tickets 工具接收的参数结构和校验规则。"""

    # Optional 表示参数可以不传；Field.description 会写入工具的 JSON Schema，
    # 大模型会参考这些描述决定应该给工具传入哪些参数。
    ticket_id: Optional[str] = Field(default=None, description="工单ID")
    assignee: Optional[str] = Field(default=None, description="负责人姓名")

    # Literal 将 status 限制为三个固定值，避免模型生成任意状态名称。
    status: Optional[Literal["open", "resolved", "closed"]] = Field(
        default=None,
        description="工单状态：open(待处理)，resolved(已解决)，closed(已关闭)",
    )

    # priority 同样使用 Literal 定义枚举值。Pydantic 会在工具执行前验证取值。
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = Field(
        default=None,
        description="优先级：low(低)，medium(中)，high(高)，urgent(紧急)",
    )

    # field_validator 只校验 ticket_id 字段。
    # 当大模型生成了不合法的工单 ID 时，Pydantic 会抛出校验错误，
    # LangChain 可以把错误反馈给模型，让模型重新生成符合要求的工具参数。
    @field_validator("ticket_id")
    @classmethod
    def validate_ticket_id(cls, value: Optional[str]) -> Optional[str]:
        # ticket_id 是可选字段，没有传值时不需要继续校验。
        if value is None:
            return value

        # 工单 ID 统一使用大写，避免大小写不同导致数据库查询不到记录。
        if not value.isupper():
            raise ValueError("工单ID必须使用大写字母")

        # 当前示例中的所有工单编号都以 TK 开头。
        if not value.startswith("TK"):
            raise ValueError("工单ID必须以TK开头")

        return value


# args_schema 明确指定工具参数由 TicketQueryInput 描述和验证。
# 如果不指定，LangChain 只会根据 query_tickets 的函数签名推断参数结构。
@tool(args_schema=TicketQueryInput)
def query_tickets(
    ticket_id: Optional[str] = None,
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
) -> str:
    """查询工单系统，支持按工单 ID、负责人、状态和优先级组合筛选。

    所有参数均为可选参数；没有提供筛选条件时返回全部模拟工单。
    该 docstring 也会成为工具说明的一部分，帮助大模型判断何时调用此工具。
    """
    try:
        # 使用列表模拟工单数据库。真实项目中通常会替换为数据库或 API 查询。
        mock_tickets_db = [
            {"ticket_id": "TK2025012001", "assigner": "张三", "title": "登录页面加载缓慢", "status": "open",
             "priority": "low"},
            {"ticket_id": "TK2025012002", "assigner": "李四", "title": "用户头像上传失败", "status": "open",
             "priority": "medium"},
            {"ticket_id": "TK2025011901", "assigner": "张三", "title": "支付成功通知未发送", "status": "resolved",
             "priority": "high"},
            {"ticket_id": "TK2025011902", "assigner": "马六", "title": "订单查询接口返回空值", "status": "closed",
             "priority": "high"},
        ]

        # 默认保留全部工单，随后根据非空参数逐步缩小查询结果。
        # 多个筛选条件同时存在时，它们之间是“并且”关系。
        filtered_tickets = mock_tickets_db

        if ticket_id:
            filtered_tickets = [ticket for ticket in filtered_tickets if ticket["ticket_id"] == ticket_id]
        if assignee:
            filtered_tickets = [ticket for ticket in filtered_tickets if ticket["assigner"] == assignee]
        if status:
            filtered_tickets = [ticket for ticket in filtered_tickets if ticket["status"] == status]
        if priority:
            filtered_tickets = [ticket for ticket in filtered_tickets if ticket["priority"] == priority]

        # 工具始终返回字符串，便于模型读取工具执行结果并组织最终回答。
        if not filtered_tickets:
            return "未找到符合条件的工单。"

        # ensure_ascii=False 用于保留 JSON 中的中文，避免转成 Unicode 转义字符。
        return json.dumps(filtered_tickets, ensure_ascii=False)
    except Exception as e:
        # 示例中将异常转成文本交给模型；生产环境还应记录异常日志。
        return f"查询工单时发生错误：{str(e)}"


# 将带 Pydantic 参数 Schema 的工具注册到 Agent。
# 当用户描述查询条件时，模型会生成 query_tickets 对应的结构化参数。
agent = create_agent(
    model=deepseek_llm,
    tools=[query_tickets],
    system_prompt=SystemMessage(content="你是一个助手，你可以查询工单系统的工单信息。"),
)

# response1 = agent.invoke({  # type: ignore
#     "messages": [{"role": "user", "content": "请帮我查一下TK2025012001工单的详细信息"}]
# })
#
# print("response1 content", response2["messages"][-1].content)


# response2 = agent.invoke({  # type: ignore
#     "messages": [{"role": "user", "content": "请帮我查一下张三负责的工单"}]
# })
# print("response2 content", response2["messages"][-1].content)


# response3 = agent.invoke({  # type: ignore
#     "messages": [{"role": "user", "content": "请帮我查一下所有高优先级的工单"}]
# })
# print("response3 content", response3["messages"][-1].content)


# response4 = agent.invoke({  # type: ignore
#     "messages": [{"role": "user", "content": "请帮我查一下所有关闭工单"}]
# })
# print("response4 content", response4["messages"][-1].content)


# 测试 Pydantic 参数验证：
# 用户输入的是小写 tk，模型如果原样生成该参数，会触发 ticket_id 的大写校验。
# Agent 可能根据校验错误修正参数后再次调用工具，因此一次 invoke 可能请求模型多次。
response5 = agent.invoke({  # type: ignore
    "messages": [{"role": "user", "content": "请帮我查一下tk2025012001工单的详细信息"}]
})
print(response5)
print("response5 content", response5["messages"][-1].content)
