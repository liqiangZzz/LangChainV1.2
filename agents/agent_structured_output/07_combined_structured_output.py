"""使用组合模型同时返回联系人和活动信息。

与 ``Union[ContactInfo, EventDetails]`` 不同，Union 表示最终结果只能二选一。
本示例使用唯一的 CombinedInfo 作为顶层 Schema，再把 ContactInfo 和
EventDetails 作为两个可选的嵌套字段，因此一次结构化输出可以同时包含两类信息。
"""
import datetime

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from models.init_chat_model.init_chat_model_llm import deepseek_llm


class ContactInfo(BaseModel):
    """从文本中提取的个人联系信息。"""

    name: str = Field(description="联系人姓名")
    email: str = Field(description="电子邮箱地址")


class EventDetails(BaseModel):
    """从文本中提取的活动信息。"""

    event_name: str = Field(description="活动名称")
    # 使用完整限定名，避免字段名 date 与导入的类型名称发生冲突。
    date: datetime.date = Field(description="活动日期")


class CombinedInfo(BaseModel):
    """文本中提取出的联系人信息和活动信息。"""

    # 两个字段均为可选：
    # - 文本同时包含两类信息时，contact 和 event 都有值。
    # - 文本只包含一类信息时，另一字段保持为 None。
    contact: ContactInfo | None = Field(
        default=None,
        description="联系人信息；文本中没有联系人时为 null",
    )
    event: EventDetails | None = Field(
        default=None,
        description="活动信息；文本中没有活动时为 null",
    )


def build_agent():
    """创建使用组合模型输出的结构化信息提取 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[],
        system_prompt=(
            "从用户文本中提取联系人信息和活动信息。"
            "文本包含哪类信息就填写对应字段；没有出现的信息保持为 null。"
            "如果两类信息都存在，必须同时填写 contact 和 event。"
        ),
        response_format=ToolStrategy(
            # 顶层只有一个 CombinedInfo 工具，因此不会像 Union 示例那样
            # 因同时调用 ContactInfo 和 EventDetails 而触发多结构输出错误。
            CombinedInfo,
            tool_message_content="联系人和活动信息提取完成！",
            handle_errors=True,
        ),
    )


def extract_combined_information(user_text: str) -> tuple[CombinedInfo, list]:
    """提取组合信息，并返回结构化对象和完整消息历史。"""
    agent = build_agent()
    result = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_text}]
    })

    return result["structured_response"], result["messages"]


if __name__ == "__main__":
    source_text = (
        "姓名：张三，电子邮箱：zhangsan@example.com，"
        "活动名称：公司年会，活动日期：2024-08-15。"
    )

    combined_info, messages = extract_combined_information(source_text)

    print("完整消息历史：")
    for message in messages:
        message.pretty_print()

    print("\n组合结构化结果：")
    print(combined_info)
    print("字典结果：", combined_info.model_dump())

    if combined_info.contact:
        print("联系人：", combined_info.contact.name)

    if combined_info.event:
        print("活动：", combined_info.event.event_name)
