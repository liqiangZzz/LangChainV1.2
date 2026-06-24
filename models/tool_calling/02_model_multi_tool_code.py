from langchain.tools import tool
from langchain_core.messages import HumanMessage

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义工具 —— 股票价格和财经新闻两个能力
# =====================================================================

# @tool 会把普通 Python 函数包装成 LangChain 工具，函数名、参数和 docstring 会提供给模型参考。
@tool
def get_stock_price(company: str, timeframe: str = 'today') -> str:
    """
   获取指定公司的股票价格信息
   Args:
       company: 公司名称（如：苹果公司, 微软公司, 谷歌公司）
       timeframe: 时间范围（today-今日, week-本周, month-本月）
   """

    # 模拟股票数据：这里不请求真实行情接口，只用于演示工具调用流程。
    mock_data = {
        "苹果公司": {"today": 185.20, "week": 183.50, "month": 180.75},
        "微软公司": {"today": 415.86, "week": 412.30, "month": 405.42},
        "谷歌公司": {"today": 15.42, "week": 15.20, "month": 14.85}
    }

    if company in mock_data:
        price = mock_data[company].get(timeframe, "未知时间范围")
        return f"{company} {timeframe}股票价格: {price}美元"
    else:
        return "未找到该公司"


@tool
def search_news(company: str) -> str:
    """搜索指定公司的财经新闻

    Args:
        company: 公司名称
    Return:
        公司的财经新闻，每个新闻占一行
    """
    # 模拟新闻数据：模型只能通过工具拿到这些本地准备好的新闻内容。
    mock_news = {
        "苹果公司": [
            "苹果发布新款iPhone，股价上涨3%",
            "苹果与欧盟达成反垄断和解协议",
            "苹果将在印度扩大生产规模"
        ],
        "微软公司": [
            "微软Azure云业务季度增长超预期",
            "微软完成对Nuance的收购",
            "微软推出新一代AI助手Copilot"
        ],
        "谷歌公司": [
            "谷歌发布新AI模型，性能提升20%",
            "谷歌与OpenAI合作，开发新的AI助手",
            "谷歌在欧洲展开AI研究项目"
        ]
    }

    news_list = mock_news.get(company, [f'未找到{company}公司'])
    return "\n".join(news_list)


# =====================================================================
# 2. 绑定工具 —— 让模型可以选择一个或多个工具
# =====================================================================

# bind_tools 只是让模型知道“有哪些工具可以调用”，并不会帮我们真正执行工具函数。
model_with_tools = deepseek_llm.bind_tools([get_stock_price, search_news])


# =====================================================================
# 3. 建立工具映射 —— 根据 tool_call.name 找到真实函数
# =====================================================================

# 后续模型返回 tool_calls 时，只会给出工具名和参数；代码需要通过这个映射找到真实函数。
tools_map = {
    "get_stock_price": get_stock_price,
    "search_news": search_news  # 注意这里名字要和函数名一致
}


# =====================================================================
# 4. 准备消息 —— 保存用户问题、工具调用和工具结果
# =====================================================================

# messages 保存完整对话历史：用户问题、模型工具调用请求、工具返回结果、最终回答。
messages = []
# human_message = HumanMessage(content="苹果公司今天的股价是多少？最近有什么新闻？")
human_message = HumanMessage(content="比较一下微软和苹果的股价")
# human_message = HumanMessage(content="腾讯最近有什么重大新闻？")
messages.append(human_message)


# =====================================================================
# 5. 循环处理工具调用 —— 直到模型不再返回 tool_calls
# =====================================================================

# 使用循环处理多轮工具调用：模型可能先请求工具，拿到工具结果后再继续请求其他工具。
while True:
    # 每次循环模型都会看到更新后的 message 列表
    response = model_with_tools.invoke(messages)  # model_with_tools.invoke(messages) 时，模型（LLM）会返回一个 AIMessage 对象。
    messages.append(response)

    if response.tool_calls:
        # tool_calls 是模型想调用的工具列表，可能一次返回一个，也可能一次返回多个。
        print(f"检测到工具调用: {[tc['name'] for tc in response.tool_calls]}")
        for tool_call in response.tool_calls:
            # 每个 tool_call 里包含 name、args、id 等信息。
            tool_name = tool_call["name"]
            # 动态获取工具并执行
            selected_tool = tools_map.get(tool_name)

            if selected_tool:
                # 关键：selected_tool.invoke(tool_call) 会自动返回 ToolMessage
                # 里面包含了 tool_call_id，模型必须靠这个 ID 来匹配结果
                tool_result = selected_tool.invoke(tool_call)
                messages.append(tool_result)
            else:
                # 容错：如果模型乱生成了不存在的工具名
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    content=f"错误：工具 {tool_name} 不存在。",
                    tool_call_id=tool_call["id"]
                ))
    else:
        # 模型输出普通文字，没有 tool_calls 了
        print("\n--- 最终答案 ---")
        print(response.content)
        break
