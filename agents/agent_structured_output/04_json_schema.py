"""使用原生 JSON Schema 定义 Agent 的结构化输出格式。

JSON Schema 不依赖 Pydantic、dataclass 或 TypedDict，可以直接通过字典描述字段类型、
枚举值、必填规则和额外字段限制。ToolStrategy 会让模型按照该 Schema 生成结果，
成功的结构化数据保存在 Agent 状态的 structured_response 字段中。
"""
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
    # 中文没有大小写差异，lower() 主要用于兼容查询中的英文内容。
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
        customer: 客户姓名。

    Returns:
        模拟邮件发送结果。
    """
    return f"已向 {customer} 发送感谢邮件"


# JSON Schema 的最外层类型是 object，表示最终结果是一个字典对象。
customer_analysis_schema = {
    # title 和 description 用于说明整个结构化结果的名称与用途。
    "title": "CustomerAnalysis",
    "type": "object",
    "description": "客户分析结果",
    # properties 声明对象允许出现的字段，以及每个字段的类型和说明。
    "properties": {
        "customer_name": {
            "type": "string",
            "description": "客户姓名",
        },
        "customer_tier": {
            "type": "string",
            "description": "客户等级，只能是潜在客户、普通客户、VIP客户或流失风险",
            # enum 将模型输出限制在指定的四个字符串中。
            "enum": ["潜在客户", "普通客户", "VIP客户", "流失风险"],
        },
        "recent_activity": {
            "type": "string",
            "description": "客户最近一次购买日期或活动说明",
        },
        "spending_level": {
            "type": "string",
            "description": "消费水平，只能是低消费、中消费或高消费",
            "enum": ["低消费", "中消费", "高消费"],
        },
        "email_sent": {
            "type": "boolean",
            "description": "是否已经成功调用工具发送感谢邮件",
        },
    },
    # required 为空表示所有字段都可以省略。
    # 这样找不到客户或问题无关时，模型可以返回空对象 {}。
    # 如果业务要求始终返回完整结果，可改成所有字段名组成的列表。
    "required": [],
    # 禁止模型生成 properties 中没有定义的其他字段，保证输出结构稳定。
    "additionalProperties": False,
}


def build_agent():
    """创建使用 JSON Schema 结构化输出的客户分析 Agent。"""
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
            "并返回空对象。"
        ),
        tools=[search_customer_database, send_email],
        # 直接把 JSON Schema 字典交给 ToolStrategy，无需额外定义 Python 数据类。
        response_format=ToolStrategy(customer_analysis_schema),
    )


def analyze_customer(user_query: str) -> dict:
    """调用 Agent，并返回符合 JSON Schema 的普通字典。"""
    agent = build_agent()
    result = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_query}]
    })

    # 配置 response_format 后，结构化结果保存在 structured_response。
    return result["structured_response"]


if __name__ == "__main__":
    analysis = analyze_customer("请分析客户张三")

    # JSON Schema 没有对应的 Python 模型类，运行时结果直接是普通 dict。
    print("JSON Schema 结果：", analysis)
    # 字段不是必填项，使用 get() 可以安全读取空对象或缺失字段。
    print("客户姓名：", analysis.get("customer_name"))
    print("邮件是否发送：", analysis.get("email_sent"))
