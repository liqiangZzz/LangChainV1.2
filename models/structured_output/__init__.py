"""
聊天模型结构化输出示例包。

本包演示如何使用 with_structured_output() 或 JsonOutputParser，
让 DeepSeek 返回符合指定结构的数据。当前示例统一围绕电影信息提取，
便于比较不同 Schema 定义方式和返回类型。

主要文件：

1. 01_pydantic_structured_output.py
   使用嵌套的 Pydantic BaseModel 定义电影和演员信息，并通过
   include_raw=True 同时保留原始消息、解析结果和解析错误。

2. 02_typed_dict_structured_output.py
   使用 TypedDict 和 Annotated 定义嵌套结构，最终返回普通字典。

3. 03_json_schema_structured_output.py
   直接使用 JSON Schema 字典声明字段、数组结构和必填规则，
   最终返回符合 Schema 的普通字典。

4. 04_json_output_parser.py
   使用 ChatPromptTemplate、JsonOutputParser 和 LCEL 构建解析链，
   通过格式指令要求模型生成 JSON，并将结果解析为字典。

运行注意事项：

- 四个示例都会调用真实 DeepSeek 模型并消耗 API 额度。
- 模型生成的电影事实可能不准确；结构化输出只能约束格式，不能保证事实真实性。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 包时触发模型调用。
"""
