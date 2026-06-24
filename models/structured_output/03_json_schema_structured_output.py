"""
使用 JSON Schema 模型定义结构化输出
"""
from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义 JSON Schema —— 描述嵌套电影信息结构
# =====================================================================

# 返回嵌套结构
# 使用 JSON Schema（最灵活，跨语言友好）
json_schema = {
    "title": "MovieInfo",
    "description": "包含上映年份、导演、完整的演员名单及角色、以及电影评分",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "电影标题"},
        "year": {"type": "integer", "description": "电影上映年份"},
        "director": {"type": "string", "description": "电影导演"},
        "rating": {"type": "number", "description": "电影评分"},
        "actors": {
            "type": "array",
            "description": "电影演员列表",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "演员名称"},
                    "role": {"type": "string", "description": "饰演的角色"}
                },
                "required": ["name", "role"]
            }
        }
    },
    "required": ["title", "year", "director", "rating", "actors"]
}


# =====================================================================
# 2. 绑定结构化输出 —— 使用 JSON Schema 约束模型返回
# =====================================================================

model_with_structure = deepseek_llm.with_structured_output(json_schema)


# =====================================================================
# 3. 发起调用 —— 按 JSON Schema 获取结构化字典
# =====================================================================

response = model_with_structure.invoke(
    "请详细提取电影《泰坦尼克号》的信息。"
    "注意：必须包含上映年份、导演、完整的演员名单及角色、以及电影评分。"
)

if response:
    print(type(response))
    print(f"电影名: {response['title']}")
    print(f"上映年份: {response['year']}")
    print(f"导演: {response['director']}")
    print(f"演员列表:{response['actors']}")
    print(f"评分: {response['rating']}")
else:
    print("模型未能生成结构化数据")


# =====================================================================
# 4. 简单结构示例 —— 可按需切换为非嵌套 JSON Schema
# =====================================================================

# 返回简单结构
# 使用 JSON Schema（最灵活，跨语言友好）
# json_schema = {
#     "title": "MovieInfo",
#     "description": "包含上映年份、导演以及电影评分",
#     "type": "object",
#     "properties": {
#         "title": {"type": "string", "description": "电影标题"},
#         "year": {"type": "integer", "description": "电影上映年份"},
#         "director": {"type": "string", "description": "电影导演"},
#         "rating": {"type": "number", "description": "电影评分"},
#     },
#     "required": ["title", "year", "director", "rating"]
# }

# 绑定 JSON Schema 到模型
# model_with_structure = deepseek_llm.with_structured_output(json_schema)

# 调用模型
# response = model_with_structure.invoke(
#     '请详细提取电影《泰坦尼克号》的信息。注意：必须包含上映年份、导演以及电影评分。')
#
# print(type(response))
# print(f"电影名: {response['title']}")
# print(f"上映年份: {response['year']}")
# print(f"导演: {response['director']}")
# print(f"评分: {response['rating']}")
