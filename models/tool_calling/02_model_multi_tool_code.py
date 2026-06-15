from langchain.tools import tool
from langchain_core.messages import HumanMessage

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# 1.定义工具
# 定义股票查询工具
@tool
def get_stock_price(company: str, timeframe: str = 'today') -> str:
    """
   获取指定公司的股票价格信息
   Args:
       company: 公司名称（如：苹果公司, 微软公司, 谷歌公司）
       timeframe: 时间范围（today-今日, week-本周, month-本月）
   """

    # 模拟股票数据
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
    # 模拟新闻数据
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


# 2. 初始化模型并绑定工具
model_with_tools = deepseek_llm.bind_tools([get_stock_price, search_news])

# 3. 建立工具查找字典（这在生产环境是标准做法）
tools_map = {
    "get_stock_price": get_stock_price,
    "search_news": search_news  # 注意这里名字要和函数名一致
}

# 4.准备messages
messages = []
# human_message = HumanMessage(content="苹果公司今天的股价是多少？最近有什么新闻？")
human_message = HumanMessage(content="比较一下微软和苹果的股价")
# human_message = HumanMessage(content="腾讯最近有什么重大新闻？")
messages.append(human_message)

# 5.工具调用
while True:
    # 每次循环模型都会看到更新后的 message 列表
    response = model_with_tools.invoke(messages)  # model_with_tools.invoke(messages) 时，模型（LLM）会返回一个 AIMessage 对象。
    messages.append(response)

    if response.tool_calls:
        print(f"检测到工具调用: {[tc['name'] for tc in response.tool_calls]}")
        for tool_call in response.tool_calls:
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

