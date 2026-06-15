"""
使用TypedDict定义嵌套结构化输出模型
"""
from typing import TypedDict, Annotated

from my_llm import deepseek_llm


class ActorTypedDict(TypedDict):
    name: Annotated[str, "演员名称"]
    role: Annotated[str, "饰演的角色"]


class MovieTypedDict(TypedDict):
    title: Annotated[str, "电影标题"]
    year: Annotated[int, "电影上映年份"]
    director: Annotated[str, "电影导演"]
    rating: Annotated[float, "电影评分"]
    actors: Annotated[list[ActorTypedDict], "电影演员列表"]


# 设置模型结构化输出
model_with_structured_output = deepseek_llm.with_structured_output(MovieTypedDict)

# 调用模型并获取结构化输出
response = model_with_structured_output.invoke(
    '请详细提取电影《泰坦尼克号》的信息。注意：必须包含上映年份、导演、完整的演员名单及角色、以及电影评分。')

print(type(response))
print(f"电影名: {response['title']}")
print(f"上映年份: {response['year']}")
print(f"导演: {response['director']}")
print(f"演员列表:{response['actors']}")
print(f"评分: {response['rating']}")

# class MovieTypedDict(TypedDict):
#     title: Annotated[str, "电影标题"]
#     year: Annotated[int, "电影上映年份"]
#     director: Annotated[str, "电影导演"]
#     rating: Annotated[float, "电影评分"]
#
# # 设置模型结构化输出
# model_with_structured_output = deepseek_llm.with_structured_output(MovieTypedDict)
#
#
# # 调用模型并获取结构化输出
# response = model_with_structured_output.invoke(
#     '请详细提取电影《泰坦尼克号》的信息。注意：必须包含上映年份、导演以及电影评分。')
#
# print(type(response))
# print(response)
