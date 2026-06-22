"""自定义 Agent 生成结构化输出后的 ToolMessage 内容。

ToolStrategy 会让模型通过一次内部工具调用生成 ContactInfo。默认情况下，
该工具执行成功后会向消息历史写入一条系统生成的 ToolMessage。
tool_message_content 可以替换这条消息的文本，便于记录更符合业务语义的提示。

注意：tool_message_content 只影响消息历史中的 ToolMessage，不会改变
structured_response 中经过 Pydantic 校验的 ContactInfo 对象。
"""
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from models.init_chat_model.init_chat_model_llm import deepseek_llm


class ContactInfo(BaseModel):
    """从用户文本中提取的个人联系信息。"""

    # Field.description 会进入结构化输出 Schema，
    # 帮助模型理解每个字段应该提取什么内容。
    name: str = Field(description="联系人姓名")
    email: str = Field(description="电子邮箱地址")
    phone: str = Field(description="电话号码")


def build_agent():
    """创建带自定义结构化输出 ToolMessage 的信息提取 Agent。"""
    return create_agent(
        model=deepseek_llm,
        system_prompt=(
            "你是一个专业的联系信息提取器，"
            "负责从用户文本中提取姓名、电子邮箱和电话号码。"
        ),
        response_format=ToolStrategy(
            # ContactInfo 决定 structured_response 的字段和类型。
            ContactInfo,
            # 结构化输出工具成功后，消息历史中的 ToolMessage 会使用这段文本，
            # 而不是默认的结构化输出成功提示。
            tool_message_content="联系信息提取完成",
            # 关闭结构化输出错误的自动处理与重试。
            # 如果 【模型缺少必填字段、字段类型错误或生成多个结构化结果】，
            # Agent 会直接抛出异常，方便开发阶段定位 Schema 和输出问题。
            # 生产环境若希望模型根据错误信息自行修正，通常应使用默认值 True。
            handle_errors=False,
        ),
    )


def extract_contact_info(user_text: str) -> tuple[ContactInfo, list]:
    """调用 Agent，并返回结构化联系人对象和完整消息历史。

    Args:
        user_text: 待分析或抽取信息的用户文本。
    """
    agent = build_agent()
    result = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_text}]
    })

    # structured_response 是经过 ContactInfo 校验后的 Pydantic 对象。
    contact_info = result["structured_response"]
    # messages 中可以观察 HumanMessage、AIMessage 和自定义内容的 ToolMessage。
    return contact_info, result["messages"]


if __name__ == "__main__":
    contact_info, messages = extract_contact_info(
        "张三的电子邮箱是 zhangsan@example.com，电话号码是 123-456-7890。"
    )

    print("完整消息历史：")
    for message in messages:
        # pretty_print() 方便观察自定义的“联系信息提取完成”消息出现在哪一步。
        message.pretty_print()

    print("\n结构化结果：")
    print(contact_info)
    print("字典结果：", contact_info.model_dump())
