"""
Agent 工具创建示例包。

本包演示如何为 LangChain Agent 创建工具，以及如何通过函数签名、Pydantic 模型
和 JSON Schema 描述工具参数。学习顺序建议按照文件编号从小到大阅读。

主要文件：

1. 01_create_tool.py
   演示使用 @tool 装饰器把普通 Python 函数注册为 Agent 工具：
   - 通过工具名称和 description 告诉模型工具的用途。
   - 根据函数参数生成基础工具参数结构。
   - 模拟根据员工 ID 查询员工信息。

2. 02_create_pydantic_tool.py
   演示使用 Pydantic 模型定义强类型工具参数：
   - 使用 BaseModel 和 Field 描述参数名称、类型及用途。
   - 使用 Literal 限制状态和优先级的可选值。
   - 使用 field_validator 校验工单 ID。
   - 支持按工单 ID、负责人、状态和优先级组合筛选。

3. 03_create_schema_tool.py
   演示直接使用 JSON Schema 定义工具参数：
   - 使用 type、properties 和 description 描述参数。
   - 使用 enum 限制图书分类的可选值。
   - 使用 required 或 anyOf 控制必填参数规则。
   - 支持按标题关键词、作者和分类查询图书。

运行方式：

- 请从项目根目录使用模块方式运行，例如：
  python -m agents.tool_creation.01_create_tool
  python -m agents.tool_creation.02_create_pydantic_tool
  python -m agents.tool_creation.03_create_schema_tool

运行注意事项：

- 三个示例都会调用真实 DeepSeek 模型并消耗 API 额度。
- Agent 通常需要先让模型生成工具参数，执行工具后再调用模型组织最终回答，
  因此一次 agent.invoke(...) 可能触发多次模型请求。
- 示例中的员工、工单和图书数据均为本地模拟数据，不会访问真实业务系统。
- 本包的 __init__.py 只提供说明，不导入示例模块，避免 import 包时触发模型调用。

"""
