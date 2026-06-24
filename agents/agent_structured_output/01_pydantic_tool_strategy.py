"""使用 Pydantic Schema 约束 Agent 的结构化输出。

Agent 会先调用客户查询工具；如果客户是 VIP，还会调用邮件工具。
最终分析结果由 ToolStrategy 按 CustomerAnalysis 模型进行校验，并保存在
Agent 返回状态的 structured_response 字段中。
"""
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义工具和 Pydantic Schema —— 查询客户并约束最终输出
# =====================================================================

@tool
def search_customer_database(query: str) -> str:
    """在模拟客户数据库中搜索客户信息。

    Args:
        query: 客户查询字符串，例如“张三”或“李四”。

    Returns:
        包含客户等级、最近购买日期和累计消费的客户记录。
    """
    # 中文没有大小写差异，但 lower() 可以兼容查询中可能出现的英文内容。
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


# ToolStrategy 的 Pydantic Schema 必须继承 BaseModel。
# 所有字段都允许为 None，这样在没有客户记录或问题与客户无关时，
# 模型可以返回空参数，最终通过 model_dump(exclude_none=True) 得到空字典。
class CustomerAnalysis(BaseModel):
    """客户分析结果。"""

    customer_name: str | None = Field(default=None, description="客户姓名")
    customer_tier: Literal["潜在客户", "普通客户", "VIP客户", "流失风险"] | None = Field(
        default=None,
        description="客户等级，只能是潜在客户、普通客户、VIP客户或流失风险",
    )
    recent_activity: str | None = Field(
        default=None,
        description="客户最近一次购买日期或活动说明",
    )
    spending_level: Literal["低消费", "中消费", "高消费"] | None = Field(
        default=None,
        description="消费水平，只能是低消费、中消费或高消费",
    )
    email_sent: bool | None = Field(
        default=None,
        description="是否已经成功调用工具发送感谢邮件",
    )


# =====================================================================
# 2. 创建 Agent —— 使用 ToolStrategy 生成结构化结果
# =====================================================================

def build_agent():
    """创建具有工具调用和 Pydantic 结构化输出能力的客户分析 Agent。"""
    return create_agent(
        model=deepseek_llm,
        system_prompt=(
            "请分析指定客户的情况，并严格遵守以下规则：\n"
            "1. 先调用 search_customer_database 查询客户的最新记录。\n"
            "2. 只有查询结果明确显示为 VIP 客户时，才调用 send_email 发送感谢邮件。\n"
            "3. 根据累计消费判断消费水平：低于 5000 为低消费，"
            "5000 至 10000 为中消费，高于 10000 为高消费。\n"
            "4. email_sent 只有在 send_email 工具成功执行后才设为 true；"
            "普通客户应设为 false。\n"
            "5. 如果找不到客户，或用户问题与客户分析无关，不发送邮件，"
            "并让所有结构化字段保持为空。"
        ),
        tools=[search_customer_database, send_email],
        # 本示例显式使用工具调用策略生成并校验 CustomerAnalysis。
        response_format=ToolStrategy(CustomerAnalysis),
    )


# =====================================================================
# 3. 调用 Agent —— 从 structured_response 读取 Pydantic 对象
# =====================================================================

def analyze_customer(user_query: str) -> CustomerAnalysis:
    """调用 Agent 并返回经过 Pydantic 校验的客户分析结果。

    Args:
        user_query: 用户输入的问题文本。
    """
    agent = build_agent()
    result = agent.invoke({  # type: ignore
        "messages": [{"role": "user", "content": user_query}]
    })

    print("完整结果：", result)
    # 配置 response_format 后，成功的结构化结果保存在 structured_response 中。
    return result["structured_response"]


# =====================================================================
# 4. 运行示例 —— 打印对象和过滤空值后的字典
# =====================================================================

if __name__ == "__main__":
    analysis = analyze_customer(
        "请分析客户张三"
        # "请分析客户李四"
        # "请分析客户王五"
        # "今天天气如何""
    )

    print("Pydantic 对象：", analysis)
    # exclude_none=True 会移除值为 None 的字段。
    # 对于无关问题或不存在的客户，输出结果可以显示为 {}。
    print("结构化字典：", analysis.model_dump(exclude_none=True))
