"""使用 wrap_tool_call middleware 统一处理 Agent 工具调用异常。

正常流程是：模型生成工具调用 -> Agent 执行工具 -> 工具结果返回给模型。
如果工具直接抛出异常，Agent 执行流程可能中断。wrap_tool_call 可以包裹每一次
工具执行，把异常转换为 ToolMessage，再交给模型生成友好的最终回答。
"""
import requests
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义可能失败的工具 —— 模拟外部股票服务异常
# =====================================================================

@tool
def get_stock_price(symbol: str) -> str:
    """获取指定股票代码的当前价格。

    Args:
        symbol: 股票代码，例如 TCEHY。

    Returns:
        股票当前价格的文本描述。

    Raises:
        Exception: 股票服务请求失败时抛出，由工具调用 middleware 统一处理。
    """
    print(f"=====调用股票查询工具: {symbol}")

    try:
        # 这里使用不存在的示例接口，目的是模拟超时、连接失败等外部服务异常。
        # timeout 可以防止请求长时间阻塞 Agent 的执行流程。
        response = requests.get(f"https://api.xxx.com/stocks/{symbol}", timeout=1)
        # 4xx、5xx 响应默认不会让 requests.get() 抛出异常，
        # 调用 raise_for_status() 后才能统一进入下面的异常处理逻辑。
        response.raise_for_status()

        # requests 返回的是 Response 对象，需要先解析 JSON 再读取价格字段。
        response_data = response.json()
        return f"股票 {symbol} 当前价格为 {response_data['price']}"
    except requests.exceptions.RequestException as e:
        # 工具本身只记录并抛出异常，不在这里决定如何向用户解释错误。
        # 最终面向模型的错误消息由 handle_tool_call_error 统一生成。
        print(f"=====查询股票数据失败: {str(e)}")
        raise Exception(f"查询股票数据失败: {str(e)}")


# =====================================================================
# 2. 编写工具调用 middleware —— 把异常转换为 ToolMessage
# =====================================================================

# wrap_tool_call 会包裹 Agent 发起的每一次工具调用。
# handler 代表原本的工具执行逻辑，调用 handler(request) 才会真正执行工具。
@wrap_tool_call
def handle_tool_call_error(request: ToolCallRequest, handler):
    """执行工具并将未处理异常转换为 ToolMessage。

    Args:
        request: 当前模型、工具或 middleware 调用请求。
        handler: LangChain 提供的默认处理函数。
    """
    # request 中包含工具名称、模型生成的参数、工具调用 ID 和当前运行上下文。
    # 学习时可以打印 request，观察模型传给工具的完整调用信息。
    print("request", request)

    try:
        # 没有发生异常时，直接返回工具原本的执行结果。
        return handler(request)
    except Exception as e:
        # 工具执行失败时不继续向外抛出异常，而是构造一条工具结果消息。
        # 模型会读取这条消息，并向用户解释当前服务不可用。
        return ToolMessage(
            content=f"当前股票查询服务不可用，错误信息: {str(e)}",
            # tool_call_id 必须与模型发起的工具调用 ID 一致。
            # 这样 LangChain 才能知道该 ToolMessage 对应的是哪一次工具调用。
            tool_call_id=request.tool_call["id"],
        )


# =====================================================================
# 3. 创建 Agent —— 注册工具和统一错误处理 middleware
# =====================================================================

# 将错误处理 middleware 挂载到 Agent。
# 当模型选择 get_stock_price 后，工具执行会先经过 handle_tool_call_error。
agent = create_agent(
    model=deepseek_llm,
    tools=[get_stock_price],
    system_prompt="你是一个助手，你可以查询股票价格。",
    middleware=[handle_tool_call_error],
)


# =====================================================================
# 4. 发起调用 —— 观察工具异常如何变成模型可读的错误消息
# =====================================================================

# 完整执行过程：
# 1. DeepSeek 根据用户问题生成 get_stock_price(symbol="TCEHY") 工具调用。
# 2. handle_tool_call_error 调用 handler(request)，开始执行股票查询工具。
# 3. 模拟接口请求失败，异常被 middleware 捕获并转换为 ToolMessage。
# 4. DeepSeek 读取错误 ToolMessage，生成面向用户的最终回答。
# 该过程通常会调用模型两次，因此会消耗真实 API 额度。
result = agent.invoke({  # type: ignore
    "messages": [{"role": "user", "content": "查询TCEHY的股票价格"}]
})

print("result", result)
print(result["messages"][-1].content)
