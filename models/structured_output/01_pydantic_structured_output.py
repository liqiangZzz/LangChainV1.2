"""
pydantic 模型返回结构化数据
"""
from pydantic import BaseModel, Field

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# 1. 定义嵌套的 Pydantic 模型
class Actor(BaseModel):
    name: str = Field(description="演员名称")
    role: str = Field(description="饰演的角色")


class Movie(BaseModel):
    title: str = Field(description="电影名称")
    year: int = Field(description="电影上映年份")
    director: str = Field(description="电影导演")
    actors: list[Actor] = Field(description="电影演员列表")  # 定义列表字段，嵌套结构
    rating: float = Field(description="电影评分")


# 2. 初始化模型并绑定输出结构
model_with_structured_output = deepseek_llm.with_structured_output(Movie, include_raw=True)

# 3. 调用模型，直接获取 Movie 实例
response = model_with_structured_output.invoke('请详细提取电影《泰坦尼克号》的信息。注意：必须包含上映年份、导演、完整的演员名单及角色、以及电影评分。')

print(type(response))
print(response)


# 简单结构
# class Movie(BaseModel):
#     title: str = Field(description="电影名称")
#     year: int = Field(description="电影上映年份")
#     director: str = Field(description="电影导演")
#     rating: float = Field(description="电影评分")
#
#  设置模型结构化输出
# model_with_structured_output = deepseek_llm.with_structured_output(Movie)

# 调用模型并获取结构化输出
# resp = model_with_structured_output.invoke('请详细提取电影《泰坦尼克号》的信息。注意：必须包含上映年份、导演以及电影评分。')
#
# print(type(resp))
# print(resp)
