"""
演示 HumanInTheLoopMiddleware 的基础用法。

示例中 read_file 直接放行，delete_file 在执行前触发人工审批，
用于观察中断信息、人工确认和 Command(resume=...) 恢复流程。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义示例工具 —— 一个放行工具，一个敏感工具
# =====================================================================

@tool
def read_file(file_path: str) -> str:
    """读取指定文件。"""
    # 示例工具不真正访问本地文件，避免学习代码误读或泄露真实文件内容。
    return f"文件 {file_path} 已成功读取！"


@tool
def delete_file(file_path: str) -> str:
    """删除指定文件。"""
    # 这里同样只返回模拟结果，重点演示 delete_file 被人工审批拦截。
    return f"文件 {file_path} 已成功删除！"


# =====================================================================
# 2. 创建 Agent —— 给 delete_file 配置人工中断
# =====================================================================

agent = create_agent(
    model=deepseek_llm,
    tools=[read_file, delete_file],
    middleware=[
        # HumanInTheLoopMiddleware 会在指定工具执行前暂停 Agent，
        # 把工具名、参数和允许的审批动作交给人工确认。
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_file": False,  # 读取文件工具不会中断，人工不介入
                "delete_file": True  # 删除文件工具会中断 Agent，等待人工审批
            },
            description_prefix="需要人工介入，请确认操作。"
        ),
    ],
    # checkpointer 用来保存图执行状态；中断后恢复必须能找到同一个会话状态。
    checkpointer=InMemorySaver(),
    system_prompt="你是一个智能助手，可以回答用户的问题。"
)


# =====================================================================
# 3. 发起调用 —— 观察 delete_file 触发的中断
# =====================================================================

# thread_id 标识一次可恢复的对话线程；恢复中断时必须继续使用同一个 thread_id。
config = {"configurable": {"thread_id": "session_id001"}}

# 首次调用会先执行 read_file；当 Agent 准备调用 delete_file 时触发中断。
result = agent.invoke({  # type: ignore
    "messages": [
        {
            "role": "user",
            "content": "帮我读取a.txt文件，最后删除这个文件"
        }
    ]
}, config=config, version="v2")

print("result", result)


# =====================================================================
# 4. 读取中断信息 —— 展示待审批工具和允许决策
# =====================================================================

# 获取中断信息。
# 这里传入 version="v2"，是为了观察 GraphOutput 返回对象。
# GraphOutput 中：
# - interrupts 保存人工中断信息
# - value 保存当前图执行返回的 state
if result.interrupts:
    print(f"Agent 已暂停！等待人工确认中...")
    interrupt_value = result.interrupts[0].value

    # action_requests 记录本次被拦截的工具调用信息，便于展示给人工审核。
    req = interrupt_value["action_requests"][0]
    print(f"待确认执行的工具：{req['name']}")
    print(f"工具参数：{req['args']}")
    print(f"描述：{req['description']}")

    # review_configs 记录该中断支持的处理方式，例如 approve、edit、reject。
    allowed_decisions = interrupt_value["review_configs"][0]["allowed_decisions"]
    print(f"用户可以确认的操作：{allowed_decisions}")

    print('***' * 20)

    # =====================================================================
    # 5. 恢复执行 —— 用 approve 决策继续被暂停的工具调用
    # =====================================================================

    # 人工确认后，用 Command(resume=...) 恢复同一个 thread_id 的中断流程。
    result2 = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        version="v2",
    )
    print("result2", result2)
    print(result2.value["messages"][-1].content)
else:
    # 如果没有触发中断，直接读取最终消息。
    print(result.value["messages"][-1].content)
