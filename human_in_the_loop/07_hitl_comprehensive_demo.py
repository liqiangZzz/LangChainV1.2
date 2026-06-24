"""
HITL 综合案例：文件管理助手。

本文件把 HumanInTheLoopMiddleware 的四种人工决策策略放在一个文件管理场景中：
- approve：批准敏感工具按原参数执行。
- reject：拒绝执行，并把原因反馈给 Agent。
- edit：人工修改工具参数后再执行。
- respond：跳过占位工具执行，直接把人工回复交还给 Agent。

示例使用 VIRTUAL_FS 字典模拟文件系统，不会读写或删除真实磁盘文件。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from models.init_chat_model.init_chat_model_llm import deepseek_llm

# =====================================================================
# 1. 模拟文件系统 —— 用字典代替真实磁盘，演示更安心
# =====================================================================

VIRTUAL_FS = {
    "readme.txt": "欢迎使用文件管理系统 v1.0\n作者：张三",
    "config.json": '{"debug": true, "max_connections": 100}',
    "data.csv": "name,age,city\n张三,18,北京\n李四,19,上海",
}


def _show_file_list() -> str:
    """返回当前文件列表"""
    if not VIRTUAL_FS:
        return "    文件（空目录）"
    lines = []
    for name, content in VIRTUAL_FS.items():
        lines.append(f"    文件：{name}（{len(content)} 字符）")
    return "\n".join(lines)


# =====================================================================
# 2. 定义文件工具 —— 读放行，写可编辑，删需确认
# =====================================================================
@tool
def list_files() -> str:
    """列出当前目录中的所有文件"""
    return _show_file_list()


@tool
def read_file(file_path: str) -> str:
    """
    读取指定文件的内容
    Args:
        file_path: 文件路径，如 "readme.txt"
    Returns:
        文件内容
    """
    print(f"读取文件: {file_path}")
    content = VIRTUAL_FS.get(file_path)
    if content is None:
        return f"[错误] 文件{file_path}不存在。当前目录中的文件：{_show_file_list()}"
    return f"文件{file_path}的内容：\n{content}"


@tool
def write_file(file_path: str, content: str) -> str:
    """
    创建新文件或覆盖写入已有文件
    Args:
        file_path: 文件路径，如 "notes.txt"
        content: 要写入的文件内容
    Returns:
        操作结果
    """
    is_new = file_path not in VIRTUAL_FS
    VIRTUAL_FS[file_path] = content
    action = "创建" if is_new else "更新"
    return f"文件{file_path}已{action}（写入 {len(content)} 字符）"


@tool
def delete_file(file_path: str) -> str:
    """
    永久删除指定文件
    Args:
        file_path: 文件路径，如 "data.csv"
    Returns:
        操作结果
    """
    if file_path not in VIRTUAL_FS:
        return f"[错误] 文件{file_path}不存在，无法删除。\n当前目录中的文件：{_show_file_list()}"
    # pop 作用：删除指定键及对应值，返回值为该键对应的值。
    removed_content = VIRTUAL_FS.pop(file_path)
    return f"文件{file_path}已永久删除（原文件 {len(removed_content)} 字符）"


# =====================================================================
# 3. 定义人工回复占位工具 —— respond 决策会绕过函数体
# =====================================================================
@tool
def ask_user(question: str) -> str:
    """
    向用户确认信息或征求意见。当你需要确认某个操作、或需要用户做出选择时调用。
    这是一个占位工具：它的"返回值"就是人工在介入环节通过 respond 决策给出的回复，
    工具本身的函数体不会被执行。
    Args:
        question: 需要用户确认的问题
    Returns:
        人工在介入环节通过 respond 决策给出的回复
    """
    # 注意：这个函数体不应该被执行。如果被执行了，说明 HITL 配置有误
    # ask_user 只应该允许 respond 决策，不应被 approve。
    raise RuntimeError(
        "ask_user 是占位工具，不应该被直接执行！"
        "请检查 interrupt_on 配置：ask_user 应仅允许 ['respond']。"
    )


# =====================================================================
# 4. 创建 Agent —— 按工具风险配置不同 HITL 策略
# =====================================================================

agent = create_agent(
    model=deepseek_llm,
    tools=[list_files, read_file, write_file, delete_file, ask_user],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 读取文件：只读操作，直接放行
                "read_file": False,

                # 写入文件：允许批准/修改参数/拒绝
                # 人工介入时如果觉得内容不合适，可以 edit 修改；也可以直接 reject
                "write_file": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "文件写入操作：可修改文件名或内容后再执行",
                },

                # 删除文件：只允许批准/拒绝（不允许修改参数）
                # 防止人工介入时误操作，把"删 A"改成"删 B"
                "delete_file": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "文件删除操作：请确认是否永久删除此文件",
                },

                # 询问用户：只允许人工直接回复
                # 这是一个占位工具，人工通过 respond 直接给答案
                "ask_user": {
                    "allowed_decisions": ["respond"],
                    "description": "向用户确认信息：请直接输入你的回复内容",
                },
            },
            description_prefix="需要人工介入",
        ),
    ],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "你是智能文件管理助手，帮助用户管理文件系统。你具备如下功能：\n"
        "1. 调用工具进行文件读写、删除操作。\n"
        "2. 当有问题需要用户确认时，请调用ask_user工具向用户确认。\n"
    ),
)


# =====================================================================
# 5. 统一处理中断 —— 多个 action_request 要按顺序提交决策
# =====================================================================
def handle_interrupts(result, agent, config):
    """
    处理一轮或多轮中断，直到 Agent 不再触发中断为止。

    返回值：最终的 GraphOutput（此时 result.interrupts 为空）
    """
    while result.interrupts:
        interrupt_data = result.interrupts[0].value

        action_requests = interrupt_data["action_requests"]
        review_configs = interrupt_data["review_configs"]

        # =================================================================
        # 5.1 展示待审操作 —— 先让人工看清楚 Agent 想做什么
        # =================================================================
        print(f"\n{'─' * 60}")
        print(f" Agent中断 —— {len(action_requests)} 个操作需要人工介入")
        print(f"{'─' * 60}")

        for i, req in enumerate(action_requests):
            cfg = review_configs[i]
            print(f"\n  [{i+1}] 工具名称 : {req['name']}")
            print(f"      参数    : {req['args']}")
            print(f"      允许决策 : {cfg['allowed_decisions']}")

        # =================================================================
        # 5.2 逐个收集决策 —— 决策顺序必须与 action_requests 一致
        # =================================================================
        decisions = []

        print(f"\n{'·' * 40}")
        print("请按顺序对以上操作做出决策：")
        print(f"{'·' * 40}")

        for i, req in enumerate(action_requests):
            allowed = review_configs[i]["allowed_decisions"]

            print(f"\n 操作 [{i+1}] {req['name']}")
            # 展示当前参数供人工介入参考
            if req.get("args"):
                for k, v in req["args"].items():
                    print(f"     参数: {k} = {v}")

            # 可用的决策类型及说明
            hint_map = {
                "approve": "批准，按原参数执行工具",
                "edit":    "修改参数后执行工具",
                "reject":  "拒绝执行，附带反馈说明",
                "respond": "跳过工具执行，直接返回人工回复",
            }
            print("     可选操作：")
            for a in allowed:
                print(f"       > {a} — {hint_map.get(a)}")

            # 等待有效输入：只接受当前工具配置允许的决策类型。
            while True:
                decision = input(f"      >>> 输入操作 ({'/'.join(allowed)}): ").strip().lower()
                if decision in allowed:
                    break
                print(f"      无效输入，该操作只允许: {allowed}")

            # =================================================================
            # 5.3 构建决策对象 —— approve/edit/reject/respond 四种形态
            # =================================================================
            if decision == "approve":
                decisions.append({"type": "approve"})
                print(f"      已批准 —— 工具将按原参数执行")

            elif decision == "edit":
                print(f"      请输入修改后的参数（直接回车保留原值）：")
                new_args = {}
                for k, v in req["args"].items():
                    new_val = input(f"         {k} [原值: {str(v)}]: ").strip()
                    if new_val == "":
                        new_args[k] = v  # 保留原值
                    else:
                        # 直接使用用户输入的字符串
                        new_args[k] = new_val
                decisions.append({
                    "type": "edit",
                    "edited_action": {"name": req["name"], "args": new_args},
                })
                print(f"      已修改参数: {new_args}")

            elif decision == "reject":
                reason = input(f"      请输入拒绝原因: ").strip()
                if not reason:
                    reason = "操作被人工拒绝"
                decisions.append({"type": "reject", "message": reason})
                print(f"      已拒绝:{reason}")

            elif decision == "respond":
                reply = input(f"      请输入回复内容: ").strip()
                if not reply:
                    reply = "已确认，没有补充信息。"
                decisions.append({"type": "respond", "message": reply})
                print(f"      已回复:{reply}")

        # =================================================================
        # 5.4 提交决策并恢复执行 —— 一次 resume 带回整批 decisions
        # =================================================================
        print(f"\n{'─' * 60}")
        print(f"提交决策列表:{decisions}")
        print(f"{'─' * 60}")

        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            version="v2",
        )

    return result


# =====================================================================
# 6. 主交互循环 —— 自然语言驱动文件助手，必要时人工介入
# =====================================================================
def main():
    print("=" * 60)
    print("智能文件管理助手 —— 综合使用 HITL 四种决策类型")
    print("=" * 60)
    print("命令说明：")
    print("  1.可以自然语言对话模拟读写文件、删除文件")
    print("  2.输入 ls/list/dir 查看当前文件列表")
    print("  3.输入 exit/quit/q 退出")
    print("-----------------")

    config = {"configurable": {"thread_id": "session_01"}}

    while True:
        # 获取用户输入。
        user_input = input("\n你: ").strip()

        if not user_input:
            continue

        lower_input = user_input.lower()

        # 退出命令直接结束控制台循环。
        if lower_input in ("exit", "quit", "q"):
            print("  再见！")
            break

        # 文件列表命令不走 Agent，直接查看虚拟文件系统。
        if lower_input in ("ls", "list", "dir"):
            print(f"  当前文件：\n{_show_file_list()}")
            continue

        # 其他自然语言输入交给 Agent，由模型决定是否调用工具。
        print("Agent 思考中…")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            version="v2",
        )

        # 处理中断：如果工具触发 HITL，这里会一轮或多轮收集人工决策。
        result = handle_interrupts(result, agent, config)

        # 输出最终回复。
        final_msg = result.value["messages"][-1]
        print(f"\nAgent回复: {final_msg.content}")


if __name__ == "__main__":
    main()
