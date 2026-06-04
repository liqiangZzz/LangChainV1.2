"""
结构化输出模块 - 让 AI 返回格式化的数据

本模块演示了如何使用 LangChain 的 with_structured_output() 方法，
让大语言模型返回符合预定义结构的格式化数据，而非自由文本。

核心功能：
- 将 AI 响应自动解析为结构化的 Python 对象（字典或 Pydantic 实例）
- 确保输出格式的一致性和可验证性
- 支持复杂嵌套结构（如列表、嵌套对象）

包含四种结构化输出方式：

1. Pydantic 模型 (01_pydantic_structured_optput.py)
   - 使用 BaseModel 和 Field 定义数据结构
   - 支持类型验证和字段描述
   - 返回 Pydantic 实例，可直接访问属性
   - 适合需要强类型检查和数据验证的场景

2. TypedDict (02_typeddict_structured_output.py)
   - 使用 TypedDict 和 Annotated 定义结构
   - 轻量级，无需额外依赖
   - 返回字典类型，通过键访问
   - 适合简单的结构化需求

3. JSON Schema (03_jsonschema_structured_output.py)
   - 直接使用 JSON Schema 定义结构
   - 最灵活，跨语言友好
   - 支持复杂的验证规则
   - 适合需要与其他系统交互的场景

4. Output Parser (04_parser_output.py)
   - 使用 JsonOutputParser 和 LCEL 链式调用
   - 结合 Prompt 模板，更灵活的控制
   - 通过 format_instructions 自动生成格式说明
   - 适合需要自定义提示词的场景

应用场景：
- 数据提取：从文本中提取结构化信息
- API 数据生成：生成符合 API 要求的 JSON 数据
- 分类任务：将内容分类到预定义的类别
- 知识图谱构建：提取实体和关系
- 表单填充：自动填充结构化表单

技术要点：
- 所有示例均使用 DeepSeek 模型
- 支持简单结构和复杂嵌套结构（如列表中包含对象）
- 可通过 include_raw=True 获取原始响应和解析后的数据
"""
