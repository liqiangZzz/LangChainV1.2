"""
tool 装饰器创建工具
"""
import json

from langchain.agents import create_agent
from langchain_core.tools import tool

from my_llm import deepseek_llm


@tool('get_employee_info', description='根据员工Id查询员工的详细信息，包括姓名、部门和职位')
def get_employee_info(employee_id: str):
    """
    根据员工Id查询员工的详细信息，包括姓名、部门和职位

    Args:
        employee_id (str): 员工的唯一标识，例如'E001'
    Returns:
        str: 返回包含员工详细信息的JSON 字符串
    """

    # 模拟一个简单的数据库
    mock_employee_database = {
        "E001": {"name": "张三", "department": "技术部", "position": "高级软件工程师", "email": "zhangsan@company.com"},
        "E002": {"name": "李四", "department": "市场部", "position": "市场经理", "email": "lisi@company.com"},
        "E003": {"name": "王五", "department": "人力资源部", "position": "招聘专员", "email": "wangwu@company.com"}
    }

    print(f"正在查询数据库，员工ID: {employee_id}")

    # 从数据库中查询员工信息
    employee_info = mock_employee_database.get(employee_id)
    if employee_info:
        return json.dumps(employee_info, ensure_ascii=False)
    else:
        return f"未找到员工ID为{employee_id}的员工信息"


agent = create_agent(
    model=deepseek_llm,
    tools=[get_employee_info],
    system_prompt="你是一个HR部门员工，请根据员工ID查询员工的详细信息，并返回JSON格式的详细信息"
)

result = agent.invoke({  # type: ignore
    "messages": [
        {"role": "user", "content": "请查询员工ID为E002的员工信息"}
    ]
})

print(type(result))
print("result:", result)

print(result["messages"][-1].content)