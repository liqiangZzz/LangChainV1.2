"""演示 ToolStrategy 对结构化输出错误的处理和自动重试。

ContactInfo 和 EventDetails 通过 Union 组成多个候选输出类型。输入文本同时包含
联系人和活动信息时，模型可能一次生成两个结构化结果，但 Agent 最终只允许返回
一个 structured_response，因此可能产生“多个结构化输出”错误。

Union[ContactInfo, EventDetails] 表示最终结果必须在两个 Schema 中选择一个，
并不表示可以同时返回 ContactInfo 和 EventDetails。如果业务需要一次返回两类信息，
应该定义一个包含 contact 和 event 字段的组合模型，而不是使用 Union。

handle_errors 决定 LangChain 收到结构化输出错误后的行为：

- True：使用默认错误消息反馈给模型，并要求模型重新生成。
- 字符串：使用指定的固定错误消息反馈给模型并重试。
- False：关闭错误处理和重试，直接把异常抛给调用方。
"""
import datetime
from typing import Union

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from my_llm import deepseek_llm


class ContactInfo(BaseModel):
    """从文本中提取的个人联系信息。"""

    name: str = Field(description="联系人姓名")
    email: str = Field(description="电子邮箱地址")


class EventDetails(BaseModel):
    """从文本中提取的活动信息。"""

    event_name: str = Field(description="活动名称")
    # 字段名本身叫 date，因此类型使用 datetime.date 的完整限定名，
    # 避免 Pydantic 将字段名称 date 与导入的 date 类型混淆。
    date: datetime.date = Field(description="活动日期")


def build_agent(handle_errors: bool | str = True):
    """创建支持多个候选结构和错误重试的信息提取 Agent。

    Args:
        handle_errors: 结构化输出错误处理策略。True 使用默认错误提示；
            字符串表示使用自定义错误提示；False 表示直接抛出异常。
    """
    return create_agent(
        model=deepseek_llm,
        tools=[],
        system_prompt=(
            "从用户文本中提取最相关的一类信息。"
            "如果文本同时包含联系人和活动信息，只选择用户主要关注的一种类型返回。"
        ),
        response_format=ToolStrategy(
            # Union 表示最终结果可以是 ContactInfo 或 EventDetails。
            # 模型应该根据用户意图选择其中一种结构。
            Union[ContactInfo, EventDetails],
            # 该参数只修改结构化输出成功后写入消息历史的 ToolMessage。
            tool_message_content="信息提取完成！",
            # True：使用 LangChain 默认错误消息并让模型重试。
            # 固定字符串：使用自定义错误消息并让模型重试，例如：
            # "一次只能返回一种信息，请选择最相关的类型后重试。"
            # False：不处理结构化输出错误，agent.invoke() 会直接抛出异常。
            handle_errors=handle_errors,
        ),
    )


def extract_information(
    user_text: str,
    handle_errors: bool | str = True,
) -> tuple[ContactInfo | EventDetails, list]:
    """提取一种结构化信息，并返回结果对象和完整消息历史。"""
    agent = build_agent(handle_errors=handle_errors)
    result = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_text}]
    })

    # 自动重试成功后，最终结构化对象保存在 structured_response 中。
    return result["structured_response"], result["messages"]


if __name__ == "__main__":
    # 这段文本故意同时包含两种 Schema 的字段，用于观察模型是否会一次生成
    # ContactInfo 和 EventDetails，以及 handle_errors 如何处理该错误。
    source_text = (
        "请提取以下文本中的主要内容："
        "姓名：张三，电子邮箱：zhangsan@example.com，"
        "活动名称：公司年会，活动日期：2024-08-15。"
    )

    structured_response, messages = extract_information(
        source_text,
        handle_errors=True,
        # 可以替换为下面任一种配置观察差异：
        # handle_errors="一次只能返回一种信息，请选择最相关的类型后重试。",
        # handle_errors=False,
    )

    print("完整消息历史：")
    for message in messages:
        # 如果首次输出不符合要求，可以在消息历史中观察错误提示和模型重试过程。
        message.pretty_print()

    print("\n最终结构化结果：")
    print(structured_response)
    print("结果类型：", type(structured_response).__name__)
