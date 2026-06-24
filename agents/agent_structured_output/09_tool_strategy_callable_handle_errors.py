"""演示 ToolStrategy 的 handle_errors 接收 Callable。

handle_errors 不只能接收 True、False 或固定字符串，也可以接收一个
Callable[[Exception], str]。当结构化输出校验失败时，LangChain 会把异常对象
传给这个函数，并把函数返回的字符串作为 ToolMessage 反馈给模型重试。
"""
from typing import Callable, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from pydantic import BaseModel, Field

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义 Callable 类型和退款审核 Schema
# =====================================================================

HandleErrorsCallable = Callable[[Exception], str]


class RefundDecision(BaseModel):
    """退款审核的结构化结果。"""

    order_id: str = Field(description="订单编号，例如 ORD-1001")
    approved: bool = Field(description="是否同意退款")
    reason: str = Field(description="退款审核原因")
    risk_level: Literal["低", "中", "高"] = Field(description="退款风险等级")


# =====================================================================
# 2. 定义 Callable 错误处理器 —— 将异常转换为模型重试提示
# =====================================================================

def refund_error_handler(error: Exception) -> str:
    """把结构化输出异常转换成模型可理解的重试提示。

    Args:
        error: ToolStrategy 捕获到的结构化输出异常。
    """
    print(f"handle_errors Callable 捕获到错误: {type(error).__name__}")

    if isinstance(error, StructuredOutputValidationError):
        return (
            "退款审核结果不符合 Schema。请只返回一个 RefundDecision："
            "order_id 必须是订单编号，approved 必须是布尔值，"
            "risk_level 只能是低、中或高。"
        )

    if isinstance(error, MultipleStructuredOutputsError):
        return "一次只能返回一个退款审核结果，请选择最主要的订单重新生成。"

    return f"结构化输出失败：{error}。请按 RefundDecision Schema 重新生成。"


# =====================================================================
# 3. 创建 Agent —— 显式传入 Callable[[Exception], str]
# =====================================================================

def build_agent():
    """创建使用 Callable 自定义 handle_errors 的退款审核 Agent。"""
    handle_errors: HandleErrorsCallable = refund_error_handler

    return create_agent(
        model=deepseek_llm,
        tools=[],
        system_prompt=(
            "你是一个退款审核助手，需要根据用户描述输出结构化退款审核结果。"
            "如果用户说风险很大，首次尝试可以把 risk_level 写成严重，"
            "用于演示 Callable 错误处理器如何提示你修正为低、中或高。"
        ),
        response_format=ToolStrategy(
            RefundDecision,
            tool_message_content="退款审核结构化结果已生成",
            # 这里显式传入 Callable[[Exception], str]。
            # 校验失败时，LangChain 会调用 refund_error_handler(error)，
            # 再把返回的字符串作为 ToolMessage 反馈给模型重试。
            handle_errors=handle_errors,
        ),
    )


# =====================================================================
# 4. 调用 Agent —— 返回退款审核结构化结果
# =====================================================================

def review_refund(user_text: str) -> tuple[RefundDecision, list]:
    """执行退款审核，并返回结构化结果和完整消息历史。

    Args:
        user_text: 用户提供的退款申请描述。
    """
    agent = build_agent()
    result = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_text}]
    })

    return result["structured_response"], result["messages"]


# =====================================================================
# 5. 运行示例 —— 观察 Callable 处理非法 risk_level
# =====================================================================

if __name__ == "__main__":
    decision, messages = review_refund(
        "订单 ORD-1001 申请退款，用户称商品严重损坏，风险很大。"
    )

    print("完整消息历史：")
    for message in messages:
        # 如果首次结构化输出不符合 Schema，这里可以看到 Callable 返回的重试提示。
        message.pretty_print()

    print("\n最终结构化结果：")
    print(decision)
    print("字典结果：", decision.model_dump())
