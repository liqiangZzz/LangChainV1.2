"""使用 TypedDict 定义 Agent 的结构化输出 Schema。

TypedDict 用字典键和类型标注描述数据结构，适合希望最终结果仍是普通 dict 的场景。
LangChain 会读取字段类型以及 Annotated 中的说明，将 CustomerAnalysis 转换为
结构化输出 Schema；最终结果保存在 Agent 状态的 structured_response 字段中。
"""
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool

from models.init_chat_model.init_chat_model_llm import deepseek_llm


@tool
def search_customer_database(query: str) -> str:
    """在模拟客户数据库中搜索客户信息。

    Args:
        query: 客户查询字符串，例如“张三”或“李四”。

    Returns:
        包含客户等级、最近购买日期和累计消费的客户记录。
    """
    # 统一处理查询字符串。中文没有大小写差异，lower() 主要用于兼容英文内容。
    normalized_query = query.lower()

    if "张三" in normalized_query:
        return "客户记录：张三，VIP客户，最近购买日期：2024-01-15，累计消费：$15,000"
    if "李四" in normalized_query:
        return "客户记录：李四，普通客户，最近购买日期：2023-12-20，累计消费：$3,200"

    return f"关于客户 {query}，无记录"


@tool
def send_email(customer: str) -> str:
    """向指定客户发送模拟感谢邮件。

    Args:
        customer: 客户姓名，例如“张三”或“李四”。

    Returns:
        模拟邮件发送结果。
    """
    return f"已向 {customer} 发送感谢邮件"


# TypedDict 默认 total=True，表示每个键都是必填项。
# 这里设置 total=False，使所有键都可以省略，从而支持“找不到客户时返回空字典”。
# 如果只有部分字段可选，也可以使用 Required 和 NotRequired 分别标记。
class CustomerAnalysis(TypedDict, total=False):
    """客户分析结果。"""

    # Annotated 的第一个参数是真实字段类型，后面的字符串是字段说明。
    # LangChain 会把说明写入工具调用使用的 JSON Schema，帮助模型正确填值。
    customer_name: Annotated[str, "客户姓名"]
    customer_tier: Annotated[
        Literal["潜在客户", "普通客户", "VIP客户", "流失风险"],
        "客户等级，只能是潜在客户、普通客户、VIP客户或流失风险",
    ]
    recent_activity: Annotated[str, "客户最近一次购买日期或活动说明"]
    spending_level: Annotated[
        Literal["低消费", "中消费", "高消费"],
        "消费水平，只能是低消费、中消费或高消费",
    ]
    email_sent: Annotated[bool, "是否已经成功调用工具发送感谢邮件"]


def build_agent():
    """创建使用 TypedDict 结构化输出的客户分析 Agent。"""
    return create_agent(
        model=deepseek_llm,
        system_prompt=(
            "请分析指定客户，并严格遵守以下规则：\n"
            "1. 先调用 search_customer_database 搜索客户信息。\n"
            "2. 只有查询结果明确显示为 VIP 客户时，才调用 send_email 发送感谢邮件。\n"
            "3. 根据累计消费判断消费水平：低于 5000 为低消费，"
            "5000 至 10000 为中消费，高于 10000 为高消费。\n"
            "4. email_sent 只有在 send_email 工具成功执行后才设为 true；"
            "普通客户应设为 false。\n"
            "5. 如果找不到客户，或用户问题与客户分析无关，不发送邮件，"
            "并让所有结构化字段保持为空。"
        ),
        tools=[search_customer_database, send_email],
        # ToolStrategy 根据 TypedDict 的键、类型和 Annotated 说明生成输出 Schema。
        response_format=ToolStrategy(CustomerAnalysis),
    )


def analyze_customer(user_query: str) -> CustomerAnalysis:
    """调用 Agent，并返回符合 CustomerAnalysis 结构的普通字典。

    Args:
        user_query: 用户输入的问题文本。
    """
    agent = build_agent()
    result = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_query}]
    })

    # 配置 response_format 后，结构化结果位于 structured_response。
    return result["structured_response"]


if __name__ == "__main__":
    analysis = analyze_customer("请分析客户李四")

    # TypedDict 只用于类型检查和 Schema 描述，运行时结果仍然是普通 dict，
    # 不需要调用 Pydantic 的 model_dump() 或 dataclass 的 asdict()。
    print("TypedDict 结果：", analysis)
    print("客户姓名：", analysis.get("customer_name"))
    print("邮件是否发送：", analysis.get("email_sent"))
