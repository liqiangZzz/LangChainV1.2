"""
演示 approve / reject 两种人工决策。

示例中读取文件直接执行，删除文件会在工具调用前暂停；
人工可以批准继续执行，也可以拒绝并把原因交回给模型。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm


# =====================================================================
# 1. 定义示例工具 —— read_file 放行，delete_file 需要审批
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
# 2. 创建 Agent —— delete_file 支持 approve / reject
# =====================================================================

agent = create_agent(
    model=deepseek_llm,
    tools=[read_file, delete_file],
    middleware=[
        # 这里演示“允许人工批准或拒绝”的中断配置：
        # read_file 自动执行，delete_file 在真正调用工具前暂停等待人工决策。
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_file": False,
                "delete_file": {
                    # approve 表示继续执行 delete_file；reject 表示拒绝本次工具调用。
                    "allowed_decisions": ["approve", "reject"],
                    # description 会出现在中断信息中，方便人工确认当前风险操作。
                    "description": "请确认是否删除文件。",
                },
            },
            description_prefix="需要人工介入，请确认操作。",
        ),
    ],
    # 中断后需要 checkpointer 保存执行现场，后续才能用同一个 thread_id 恢复。
    checkpointer=InMemorySaver(),
    system_prompt="你是一个助手，需要根据用户输入执行相应的操作。",
)


# =====================================================================
# 3. 发起调用 —— 读取后删除，删除前触发人工中断
# =====================================================================

# thread_id 用来标识同一条可恢复会话；首次调用和恢复调用必须保持一致。
config = {"configurable": {"thread_id": "session_123"}}

# 首次调用会先执行 read_file；当模型准备调用 delete_file 时触发人工中断。
result = agent.invoke({  # type: ignore
    "messages":
        [
            {
                "role": "user",
                "content": "请读取文件 /home/user/file.txt,然后再帮我删除这个文件"
            }
        ]
}, config=config, version="v2")

if result.interrupts:

    print("触发中断 result ---> ", result)

    # =====================================================================
    # 4. 展示中断信息 —— 让人工知道即将执行什么
    # =====================================================================

    # action_requests 中保存了被拦截的工具调用名称、参数和描述。
    req = result.interrupts[0].value["action_requests"][0]

    print("Agent 暂停！请确认操作。")
    print(f"---待调用工具：{req['name']}")
    print(f"---工具参数：{req['args']}")
    print(f"---中断描述：{req['description']}")

    print("-" * 80)
    # review_configs 中保存当前中断允许的人工决策类型。
    allowed_decisions = result.interrupts[0].value["review_configs"][0]["allowed_decisions"]

    # =====================================================================
    # 5. 人工决策 —— approve 继续执行，reject 拒绝执行
    # =====================================================================

    while True:
        decision = input(f"请输入决策({allowed_decisions})：").strip().lower()
        if decision == "approve":
            # approve 会让 Agent 继续执行原本被中断的 delete_file 工具调用。
            command_cmd = Command(
                resume={"decisions": [{"type": "approve"}]}
            )
            break

        elif decision == "reject":
            # reject 会拒绝本次工具调用，并把拒绝原因作为工具错误消息交回给模型。
            reason = input("请输入拒绝理由：").strip()
            if not reason:
                reason = "用户拒绝了该操作。"

            command_cmd = Command(
                resume={"decisions": [{"type": "reject", "message": f"原因：{reason}"}]}
            )
            break
        else:
            print(f"无效输入，请输入 {allowed_decisions} 中的一个，当前输入：{decision}")

    # =====================================================================
    # 6. 恢复执行 —— 把人工决策交回 Agent
    # =====================================================================

    # 使用 Command(resume=...) 恢复同一个 thread_id 的中断流程。
    result2 = agent.invoke(  # type: ignore
        command_cmd,
        config=config,
        version="v2"
    )

    print(result2)
    print(" result ", result2.value["messages"][-1].content)

else:
    # 没有触发中断时，直接读取最终消息。
    print(" result ", result.value["messages"][-1].content)
