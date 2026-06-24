"""使用自定义函数处理 ToolStrategy 的结构化输出校验错误。

本示例故意要求模型先生成不符合 ProductEvaluation Schema 的数据：

- rating=10 超出 Field(ge=1, le=5) 规定的 1 至 5 范围。
- sentiment="复杂" 不在 Literal["正面", "负面", "中性"] 中。

Pydantic 校验失败后，LangChain 会调用 custom_error_handler，把返回的错误提示写入
ToolMessage 并反馈给模型重试。模型修正数据后，结果才会写入 structured_response。
"""
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from pydantic import BaseModel, Field

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义错误处理函数 —— 按异常类型生成重试提示
# =====================================================================

def custom_error_handler(error: Exception) -> str:
    """根据结构化输出错误类型，返回给模型不同的修正提示。

    Args:
        error: 结构化输出或工具调用过程中捕获的异常对象。
    """
    error_str = str(error)

    # 这些日志只用于学习和调试，可以观察 LangChain 传入的真实异常类型。
    print(f"捕获到错误类型: {type(error).__name__}")
    print(f"错误详情: {error_str}")

    if isinstance(error, StructuredOutputValidationError):
        # Pydantic 字段缺失、类型错误、数值越界或 Literal 不匹配时进入这里。
        return (
            "评价数据不符合 Schema：rating 必须是 1 到 5 的整数，"
            "sentiment 只能是正面、负面或中性。请修正后重新生成。"
        )

    if isinstance(error, MultipleStructuredOutputsError):
        # 模型一次生成多个 ProductEvaluation 结果时进入这里。
        return "检测到多个结构化结果，请只返回最相关的一条产品评价。"

    # 兜底分支必须插入真实错误文本，原代码返回的是固定字面量 error_str。
    return f"结构化输出发生错误：{error_str}。请修正后重试。"


# =====================================================================
# 2. 定义 Schema —— 故意设置严格字段约束
# =====================================================================

class ProductEvaluation(BaseModel):
    """产品评价的结构化分析结果。"""

    # 不设置默认值，使三个字段都成为必填项，避免缺少信息时静默使用空字符串。
    product_name: str = Field(description="产品名称")
    rating: int = Field(description="评分，必须是1到5之间的整数", ge=1, le=5)
    sentiment: Literal["正面", "负面", "中性"] = Field(
        description="评价的情感倾向",
    )


# =====================================================================
# 3. 创建 Agent —— 将自定义函数传给 handle_errors
# =====================================================================

def build_agent():
    """创建使用自定义结构化输出错误处理器的产品评价 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[],
        system_prompt=(
            "你是一个产品评价分析助手，请结构化输出评价结果。"
            "为了演示 Schema 校验错误：如果评价中提到10分或满分，首次尝试将 rating 设为10；"
            "如果评价情感复杂，首次尝试将 sentiment 设为复杂。"
            "收到工具校验错误后，必须按照错误提示修正为 Schema 允许的值。"
        ),
        response_format=ToolStrategy(
            ProductEvaluation,
            tool_message_content="产品评价分析完成",
            # ToolStrategy 没有 error_handler 参数。
            # 自定义错误处理函数应通过 handle_errors 传入。
            handle_errors=custom_error_handler,
        ),
    )


# =====================================================================
# 4. 调用 Agent —— 返回修正后的结构化评价
# =====================================================================

def analyze_product_review(user_text: str) -> tuple[ProductEvaluation, list]:
    """分析产品评价，并返回结构化结果和完整消息历史。

    Args:
        user_text: 待分析或抽取信息的用户文本。
    """
    agent = build_agent()
    response = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_text}]
    })

    return response["structured_response"], response["messages"]


# =====================================================================
# 5. 运行示例 —— 观察校验失败、错误提示和模型重试
# =====================================================================

if __name__ == "__main__":
    evaluation, messages = analyze_product_review(
        "DeepSeek手机非常棒，我给10分，超级喜欢！"
    )

    print("完整消息历史：")
    for message in messages:
        # 消息历史中可以观察首次校验失败、自定义 ToolMessage 和模型重试过程。
        message.pretty_print()

    print("\n最终结构化结果：")
    print(evaluation)
    print("字典结果：", evaluation.model_dump())
