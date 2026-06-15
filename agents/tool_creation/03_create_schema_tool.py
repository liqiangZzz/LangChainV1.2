"""使用 JSON Schema 定义 LangChain 工具的参数结构。

与 Pydantic 模型不同，本示例直接使用字典编写 JSON Schema。
LangChain 会把该 Schema 提供给大模型，让模型了解工具支持哪些参数、
每个参数的数据类型、用途以及允许的取值范围。
"""
import json
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# JSON Schema 的最外层类型是 object，表示工具接收一组键值参数。
book_query_schema = {
    "type": "object",
    # properties 用于声明对象中可以出现的字段。
    # 每个字段的 description 都会提供给大模型，帮助它从用户问题中提取参数。
    "properties": {
        "title": {
            "type": "string",
            "description": "图书标题或标题关键词，支持模糊匹配",
        },
        "author": {"type": "string", "description": "图书作者"},
        "category": {
            "type": "string",
            # enum 限制 category 只能从以下值中选择，
            # 防止模型向工具传入数据库不支持的分类名称。
            "enum": ["技术", "历史", "文学", "经济"],
            "description": "图书类别",
        },
    },
    # required 为空表示三个参数都不是必填项。
    # JSON Schema 的 required 只能指定“哪些字段必须存在”，不能直接表达
    # “title、author、category 至少传一个”。本示例不传条件时会返回全部图书。
    # 如果业务上必须至少传一个条件，可以使用 anyOf 组合规则进行约束。
    "required": [],

    # anyOf 表示下面的规则至少满足一条。
    # 每条规则分别要求 title、author 或 category 字段必须存在，
    # 因此取消注释后，就可以实现“三个查询条件至少传入一个”。
    # "anyOf": [
    #     {"required": ["title"]},
    #     {"required": ["author"]},
    #     {"required": ["category"]},
    # ],
}


# args_schema 指定 query_books 的参数结构来自上面的 JSON Schema，
# 而不是只根据 Python 函数签名自动推断。
@tool(args_schema=book_query_schema)
def query_books(
    title: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """根据标题、作者或分类查询企业图书库。

    三个参数可以单独使用，也可以组合使用；多个条件之间是“并且”关系。
    没有提供任何筛选条件时，返回全部模拟图书。
    """
    try:
        # 使用列表模拟图书数据库。真实项目中通常会替换为数据库或 API 查询。
        mock_books_db = [
            {"book_id": "BK1001", "title": "人工智能导论", "author": "张明", "category": "技术"},
            {"book_id": "BK1002", "title": "机器学习实战", "author": "李华", "category": "技术"},
            {"book_id": "BK1003", "title": "中国近代史", "author": "王伟", "category": "历史"},
            {"book_id": "BK1004", "title": "红楼梦", "author": "曹雪芹", "category": "文学"},
            {"book_id": "BK1005", "title": "经济学原理", "author": "刘强", "category": "经济"},
            {"book_id": "BK1006", "title": "文学导论", "author": "张明", "category": "文学"},
            {"book_id": "BK1007", "title": "Python编程基础", "author": "王丽", "category": "技术"},
        ]

        # 默认保留全部数据，然后按照实际传入的非空参数逐步过滤。
        filtered_books = mock_books_db

        if title:
            # 标题使用包含判断，因此传入“Python”可以匹配“Python编程基础”。
            # lower() 让英文标题关键词匹配时忽略大小写。
            filtered_books = [book for book in filtered_books if title.lower() in book["title"].lower()]

        if author:
            # 作者和分类使用精确匹配。
            filtered_books = [book for book in filtered_books if book["author"] == author]

        if category:
            filtered_books = [book for book in filtered_books if book["category"] == category]

        if not filtered_books:
            return "未找到符合条件的图书。"

        # 除了图书列表，同时返回结果数量，方便模型概括查询结果。
        result = {
            "total_count": len(filtered_books),
            "books": filtered_books,
        }

        # 工具返回 JSON 字符串，模型会读取该结果并生成面向用户的最终回答。
        # ensure_ascii=False 保留中文，indent=2 便于直接打印观察。
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        # 学习示例直接返回错误文本；生产环境还应记录异常日志。
        return f"查询图书时发生错误：{str(e)}"


# 将 query_books 注册给 Agent。
# 用户用自然语言提出查询条件后，模型会按照 book_query_schema 生成工具参数。
agent = create_agent(
    model=deepseek_llm,
    tools=[query_books],
    system_prompt=SystemMessage(content="你是一个图书管理员，你可以查询图书信息。"),
)

# 测试1：按图书种类精确查询
# response1 = agent.invoke({  # type: ignore
#     "messages": [{"role": "user", "content": "请帮我查一下历史类图书"}]
# })
# print("=== 测试1：按图书种类精确查询 ===")
# print(response1["messages"][-1].content)


# 测试2：多条件组合查询
# response2 = agent.invoke({  # type: ignore
#     "messages": [{"role": "user", "content": "我想找张明写的技术类图书"}]
# })
# print(response2)
# print("\n=== 测试2：多条件组合查询 ===")
# print(response2["messages"][-1].content)


# 测试3：关键词模糊查询。
# 该调用会访问真实 DeepSeek API，并可能经历“模型生成工具参数 -> 执行工具 ->
# 模型读取工具结果并回答”两次模型请求。
response3 = agent.invoke({  # type: ignore
    "messages": [{"role": "user", "content": "搜索包含'Python'关键词的图书"}]
})
print("\n=== 测试3：关键词模糊查询 ===")
print(response3["messages"][-1].content)
