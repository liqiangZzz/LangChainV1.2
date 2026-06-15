"""演示使用 ``agent.ainvoke()`` 异步调用旅行规划 Agent。

异步调用不会改变 Agent 的工具选择和执行逻辑，只是调用方可以使用 ``await``
等待结果，从而避免阻塞事件循环。示例中的天气、交通和景点数据均为本地模拟数据。
"""

import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph

from my_llm import deepseek_llm


@tool
def get_weather(city: str) -> str:
    """获取指定城市的模拟天气信息。

    Args:
        city: 城市名称。

    Returns:
        城市天气；没有对应数据时返回“信息暂缺”。
    """
    weather_data = {
        "北京": "晴朗，15°C",
        "上海": "多云，18°C",
        "广州": "小雨，22°C",
        "深圳": "晴间多云，25°C",
    }
    return f"{city}的天气：{weather_data.get(city, '信息暂缺')}"


@tool
def get_transport_info(from_city: str, to_city: str) -> str:
    """查询两个城市之间的模拟交通信息。

    Args:
        from_city: 出发城市名称。
        to_city: 到达城市名称。

    Returns:
        两个城市之间的交通时间说明。
    """
    transport_data = {
        "北京-上海": "高铁约4.5小时，航班约2小时",
        "上海-广州": "高铁约7小时，航班约2.5小时",
        "广州-深圳": "高铁约0.5小时，驾车约1.5小时",
    }
    route = f"{from_city}-{to_city}"
    return f"{from_city}到{to_city}：{transport_data.get(route, '请查询具体班次')}"


@tool
def get_scenic_spots(city: str, interest_type: str = "通用") -> str:
    """根据兴趣类型推荐城市中的模拟景点。

    Args:
        city: 要查询景点的城市名称。
        interest_type: 兴趣类型，例如“历史”“现代”或“通用”。

    Returns:
        与城市和兴趣类型匹配的景点推荐。
    """
    scenic_spots_data = {
        "北京": {
            "历史": "推荐：故宫、天坛、长城",
            "美食": "推荐：全聚德烤鸭、王府井小吃街",
            "通用": "推荐：故宫、长城、颐和园、天坛",
        },
        "上海": {
            "现代": "推荐：外滩、东方明珠、陆家嘴",
            "文化": "推荐：博物馆、艺术馆、田子坊",
            "通用": "推荐：外滩、迪士尼、南京路",
        },
    }
    city_spots = scenic_spots_data.get(city, {})
    recommendation = city_spots.get(
        interest_type,
        city_spots.get("通用", "信息暂缺"),
    )
    return f"{city}{interest_type}景点：{recommendation}"


def create_travel_agent() -> CompiledStateGraph:
    """创建能够查询天气、交通和景点信息的旅行规划 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[get_weather, get_transport_info, get_scenic_spots],
        system_prompt=(
            "你是一个专业的旅行规划助手，能够帮助用户查询天气、交通和景点信息。"
            "拒绝回答与旅行规划无关的问题。"
        ),
    )


async def invoke_travel_agent() -> dict[str, Any]:
    """异步调用旅行规划 Agent，并返回包含完整消息历史的状态字典。"""
    agent = create_travel_agent()

    # ainvoke 的输入格式与同步 invoke 相同，区别是调用时需要使用 await。
    # Agent 可能先调用多个工具，再请求模型根据工具结果生成最终回答。
    return await agent.ainvoke(  # type: ignore[return-value]
        {
            "messages": [
                # {
                #     "role": "user",
                #     "content": "请帮我查询北京的天气信息，并推荐一些历史类型的景点",
                # },
               # 可以替换为下面的问题，观察交通工具调用或非旅行问题的拒答：
                {
                    "role": "user",
                    "content": (
                        "我要从北京出发到上海，请查询上海天气，"
                        "并推荐一些现代类型的景点"
                    ),
                },
                # {"role": "user", "content": "天空为什么是蓝色的"},
            ]
        }
    )


if __name__ == "__main__":
    # asyncio.run() 负责创建事件循环并执行异步入口函数。
    response = asyncio.run(invoke_travel_agent())

    print("完整 Agent 状态：", response)
    # messages 最后一项通常是 Agent 面向用户生成的最终 AIMessage。
    print("最终回答：", response["messages"][-1].content)
