"""
演示 Agent 的动态系统提示词。

dynamic_prompt 会在 Agent 每次调用模型前运行，根据 runtime context
动态生成 system_prompt。这个示例用 query_type 区分普通客服和 VIP 客服。

注意：@dynamic_prompt 包装后的函数要传给 create_agent 的 middleware，
不能传给 system_prompt。system_prompt 只接收固定字符串或消息对象。

注意：Agent 调用工具时可能会多次调用模型。为了省时间和额度，main 默认只跑一个场景。
"""
import json
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import tool

from models.init_chat_model.init_chat_model_llm import deepseek_llm

# =====================================================================
# 1. 准备模拟数据 —— 订单库和 FAQ 知识库
# =====================================================================

# 模拟订单数据库。真实项目里通常会换成数据库/API 查询。
ORDER_DATABASE = {
    "ORD123456": {"status": "已发货", "items": ["手机X1"], "create_time": "2025-01-15"},
    "ORD654321": {"status": "待付款", "items": ["耳机Y1"], "create_time": "2025-01-18"},
}

# 模拟 FAQ 知识库。这里用简单字典演示工具调用流程。
FAQ_KNOWLEDGE_BASE = {
    "退货": "支持7天无理由退货，商品需完好且包装齐全。",
    "保修": "电子产品享受1年免费保修服务。",
    "发货": "下单后48小时内发货，偏远地区可能延迟。",
}


# =====================================================================
# 2. 定义运行时上下文 —— 给 dynamic_prompt 传业务参数
# =====================================================================

class AgentContext(TypedDict):
    """Agent 运行时上下文，用来给动态提示词传参。"""

    # query_type 不是用户消息的一部分，而是运行时额外传给 Agent 的业务参数。
    # dynamic_support_prompt 会读取它，决定使用普通客服还是 VIP 客服提示词。
    query_type: str


# =====================================================================
# 3. 准备提示词模板 —— 普通客服与 VIP 客服两种模式
# =====================================================================

BASE_SUPPORT_PROMPT = "你是一名专业的电商客服助手。请根据工具查询结果，准确、清晰地回答用户问题。"

NORMAL_SUPPORT_PROMPT = f"""{BASE_SUPPORT_PROMPT}
当前角色：一线客服助手
工作要求：
1. 快速响应：优先使用工具获取订单/政策信息。
2. 直接解答：对于明确问题，直接给出基于知识的答案。
3. 简洁友好：回复要简单明了，避免复杂术语。"""

VIP_SUPPORT_PROMPT = f"""{BASE_SUPPORT_PROMPT}
当前角色：高级支持专员
工作要求：
1. 深度分析：仔细分析用户描述，识别潜在的根本问题。
2. 精准分类：将问题明确归类，如物流问题、产品质量、售后申请。
3. 方案规划：若工具能解决，提供具体步骤；若需人工，明确告知后续流程。
请使用更专业、严谨的语言。"""


# =====================================================================
# 4. 定义工具 —— 查询订单和检索 FAQ
# =====================================================================

@tool
def query_order_info(order_id: str) -> str:
    """根据订单ID查询订单的详细信息，包括状态、商品列表和创建时间。

    Args:
        order_id: 订单号。
    """
    order_data = ORDER_DATABASE.get(order_id)
    if not order_data:
        return f"错误：未找到订单 {order_id}。"

    return json.dumps(order_data, ensure_ascii=False)


@tool
def search_faq(keyword: str) -> str:
    """根据关键词从知识库中检索相关的政策条款或解决方案。

    Args:
        keyword: FAQ 搜索关键词。
    """
    for topic, answer in FAQ_KNOWLEDGE_BASE.items():
        if topic in keyword:
            return answer

    return f"未找到与'{keyword}'直接相关的政策，请尝试其他关键词或联系人工客服。"


# =====================================================================
# 5. 编写 dynamic_prompt —— 每次模型调用前生成系统提示词
# =====================================================================

@dynamic_prompt
def dynamic_support_prompt(request: ModelRequest) -> str:
    """根据 context["query_type"] 动态选择系统提示词。

    Args:
        request: 当前模型、工具或 middleware 调用请求。
    """
    # dynamic_prompt 不是只在 invoke 开始时运行一次。
    # 当 Agent 调用工具后再次请求模型生成下一步动作/最终回答时，它也可能再次运行。
    query_type = request.runtime.context.get("query_type", "normal")
    print(f"[动态提示词] query_type={query_type}")

    if query_type == "vip":
        return VIP_SUPPORT_PROMPT

    return NORMAL_SUPPORT_PROMPT


# =====================================================================
# 6. 创建与调用 Agent —— 通过 context 切换提示词
# =====================================================================

def build_agent():
    """创建客服 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[query_order_info, search_faq],
        # dynamic_prompt 属于 Agent middleware，要放在 middleware 列表中。
        # LangChain 会在运行时调用它，生成本轮模型请求使用的系统提示词。
        middleware=[dynamic_support_prompt],
        # context_schema 声明 context 里允许传入哪些字段，便于类型检查和示例理解。
        context_schema=AgentContext,
    )


def ask_support_agent(agent, user_query: str, query_type: str = "normal") -> str:
    """调用 Agent 并返回最终回答。

    Args:
        agent: 已创建好的 Agent 实例。
        user_query: 用户输入的问题文本。
        query_type: 本次客服问题类型。
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        # 这里的 context 会进入 request.runtime.context，
        # dynamic_support_prompt 正是通过它判断当前客服模式。
        context={"query_type": query_type},
    )
    return result["messages"][-1].content


# =====================================================================
# 7. 运行示例 —— 默认只跑 VIP 场景以节省额度
# =====================================================================

if __name__ == "__main__":
    agent = build_agent()
    user_query = "我的订单ORD654321已签收，但是物品坏了怎么办？"

    print(f"用户问题：{user_query}\n")
    # print("普通客服模式：")
    # print(ask_support_agent(agent, user_query, query_type="normal"))

    # 如果想在脚本里直接对比 VIP 提示词效果，再取消下面两行注释。
    # 默认注释掉是为了避免一次运行触发两轮真实模型调用，节省时间和 API 额度。
    print("\nVIP客服模式：")
    print(ask_support_agent(agent, user_query, query_type="vip"))
