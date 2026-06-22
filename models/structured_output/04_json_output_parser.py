"""
输出解析器
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from models.init_chat_model.init_chat_model_llm import deepseek_llm


# 1. 定义 Pydantic 模型
class Movie(BaseModel):
    title: str = Field(description="电影名称")
    year: int = Field(description="电影上映年份")


# 2. 创建解析器 (Parser)
parser = JsonOutputParser(pydantic_object=Movie)

# 3. 创建提示模板，并注入格式指令
# parser.get_format_instructions() 会自动生成一份告诉模型如何输出 JSON 的“说明书”
prompt = ChatPromptTemplate.from_template(
    template="""回答用户问题。
                {format_instructions}
                问题：{question}
             """,
    # 提前绑定格式说明，简化调用时的传参
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 构建 LCEL 链
# 数据流向：Prompt -> LLM -> Parser
chain = prompt | deepseek_llm | parser

# 5. 调用链
# 此时返回的是一个已经解析好的 Python 字典（或 Movie 实例）
response = chain.invoke({"question": "请提取电影《泰坦尼克号》的信息。"})

print("解析后的结果：", response)
print("类型：", type(response))
