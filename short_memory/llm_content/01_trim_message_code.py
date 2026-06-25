"""
演示 Agent 短期记忆中的 trim_messages 消息截断。

Agent 使用 InMemorySaver 保存同一个 thread_id 下的完整消息历史。对话变长后，
before_model middleware 会在每次调用模型前截断 messages，只把系统提示和最近上下文
发送给 DeepSeek，避免短期记忆无限增长导致上下文过长。

和 02_delete_message_code.py 的区别：
- 本文件先用 trim_messages 算出“应该保留哪些消息”。
- 然后使用 RemoveMessage(id=REMOVE_ALL_MESSAGES) 清空全部旧 messages。
- 最后把 trim_messages 得到的消息列表重新写回 state。
- 也就是说，本文件是“清空全部 + 重写保留结果”，不是按某几条旧消息 id 精准删除。

本示例会调用真实 DeepSeek 模型，运行前需要确认 .env 中配置了 DeepSeek 信息。
"""
from typing import Any, Dict

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from models.init_chat_model.init_chat_model_llm import deepseek_llm

# 这里演示“按消息条数截断”：最多保留最近 4 条消息。
# 由于 DeepSeek-V4-Flash 当前没有实现 get_num_tokens_from_messages()，
# 示例不做不准确的 token 估算，而是用 len 作为 token_counter。
# 在 token_counter=len 的语义下，max_tokens 表示“最多保留多少条消息”。
MAX_SHORT_MEMORY_MESSAGES = 4


# =====================================================================
# 1. 计算截断结果 —— 用 trim_messages 选出最近上下文
# =====================================================================
def trim_short_memory(messages: list[BaseMessage]) -> list[BaseMessage]:
    """保留最近对话，丢弃较早的短期记忆。

    Args:
        messages: 消息列表或当前状态中的 messages。
    """
    return trim_messages(
        messages,
        max_tokens=MAX_SHORT_MEMORY_MESSAGES,
        token_counter=len,
        # strategy="last" 表示从最后一条消息往前保留，也就是保留最近上下文。
        strategy="last",
        # 如果历史里有 SystemMessage，截断后仍保留系统提示，避免模型丢失角色设定。
        include_system=True,
        # 截断后的对话从 human 消息开始，避免出现 AI 单方面回复开头的残缺上下文。
        start_on="human",
        # 不切开单条消息；如果某条消息太长，会整条丢弃，而不是保留半截文本。
        allow_partial=False,
    )


# =====================================================================
# 2. before_model 重写 messages —— 清空旧历史，再写回截断结果
# =====================================================================
@before_model
def trim_messages_before_model(state: AgentState, runtime: Runtime) -> Dict[str, Any]:
    """模型调用前截断短期记忆中的 messages。

    before_model 会在 Agent 每次真正请求大模型前执行。
    这里截断的是即将发送给模型的 messages，同时也会把截断后的 messages 写回 state。

    Args:
        runtime: 工具或 middleware 的运行时对象，可读取 context、state、store 等信息。
    """
    messages = state["messages"]
    print("[before_model] trim messages | 原始消息：", messages)
    trimmed_messages = trim_short_memory(messages)
    print("[before_model] trim messages | 截断后消息：", trimmed_messages)

    print(
        "[before_model] trim messages | "
        f"原始消息数={len(messages),type(messages)}, "
        f"截断后消息数={len(trimmed_messages),type(trimmed_messages)}"
    )

    # Agent state 里的 messages 默认是追加式更新：
    # 如果这里只 return {"messages": trimmed_messages}，新消息会追加到旧消息后面，
    # 旧历史不会被真正替换。
    # RemoveMessage(id=REMOVE_ALL_MESSAGES) 表示先清空旧 messages，
    # 再把截断后的 messages 写回去。
    # 这和 02_delete_message_code.py 不同：
    # 02 是 return {"messages": [RemoveMessage(id=某条旧消息.id), ...]}，
    # 只删除指定 id 的旧消息；这里是先清空全部，再重写截断后的结果。
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed_messages]}


# =====================================================================
# 3. 创建 Agent —— 每次调用模型前自动执行截断 middleware
# =====================================================================
def build_agent():
    """按项目常用方式创建带短期记忆截断能力的 Agent。"""
    return create_agent(
        model=deepseek_llm,
        tools=[],
        # checkpointer 负责保存每个 thread_id 对应的消息状态。
        checkpointer=InMemorySaver(),
        system_prompt=SystemMessage(content="你是一个助手，需要帮助时请随时告诉我。"),
        # 注册 middleware 后，每轮 invoke 进入模型前都会先执行 trim_messages_before_model。
        middleware=[trim_messages_before_model],
    )


# =====================================================================
# 4. 打印短期记忆 —— 看 checkpointer 里实际保存了哪些消息
# =====================================================================
def print_state_messages(agent, config: dict) -> None:
    """打印当前 checkpointer 中保存的消息数量。

    get_state() 读取的是同一个 thread_id 下的短期记忆状态。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
    """
    state = agent.get_state(config=config).values
    messages = state.get("messages", [])
    print(
        "[当前短期记忆]",
        {   "messages": messages,
            "messages_count": len(messages),
        },
    )


# =====================================================================
# 5. 封装单轮调用 —— 一边聊天，一边观察记忆变化
# =====================================================================
def invoke_and_print(agent, config: dict, user_content: str) -> None:
    """发送一轮用户消息，并打印模型回复和当前记忆状态。

    Args:
        agent: 已创建好的 Agent 实例。
        config: LangGraph 运行配置，通常包含 configurable.thread_id。
        user_content: 本轮用户输入内容。
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
    )

    print("[用户]", user_content)
    print("[模型回复]", result["messages"][-1].content)
    print_state_messages(agent, config)
    print("-" * 60)


# =====================================================================
# 6. 运行三轮演示 —— 第三轮触发最近消息截断
# =====================================================================
def main() -> None:
    """连续多轮调用 Agent，观察消息历史如何被截断。"""
    agent = build_agent()
    # thread_id 表示同一个会话。三轮 invoke 使用同一个 thread_id，
    # InMemorySaver 才会把上一轮 messages 带到下一轮。
    config = {"configurable": {"thread_id": "trim-message-session001"}}

    # 第 1 轮：写入用户基础信息和订单信息。
    invoke_and_print(
        agent,
        config,
        "你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。",
    )
    # 第 2 轮：继续追加售后问题和处理偏好，让短期记忆变长。
    invoke_and_print(
        agent,
        config,
        "耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。",
    )
    # 第 3 轮：要求模型基于最近上下文回答，此时 before_model 会先截断 messages。
    invoke_and_print(agent, config, "请基于我们最近聊过的信息，帮我整理一段售后说明。")


if __name__ == "__main__":
    main()


"""

[before_model] trim messages | 原始消息： [HumanMessage(content='你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。', additional_kwargs={}, response_metadata={}, id='137c5f4d-092e-478b-898d-79450fdf332b')]

[before_model] trim messages | 截断后消息： [HumanMessage(content='你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。', additional_kwargs={}, response_metadata={}, id='137c5f4d-092e-478b-898d-79450fdf332b')]

[before_model] trim messages | 原始消息数=(1, <class 'list'>), 截断后消息数=(1, <class 'list'>)

[用户] 你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。

[模型回复] 您好，张三！感谢您提供的信息。我已记录您的订单号 A1001，购买的是无线耳机，手机尾号 8899。请问您有什么需要我帮忙的吗？比如查询订单状态、处理售后问题，或其他服务？

[当前短期记忆] {'messages': [

  HumanMessage(content='你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。', additional_kwargs={}, response_metadata={}, id='137c5f4d-092e-478b-898d-79450fdf332b'),

  AIMessage(content='您好，张三！感谢您提供的信息。我已记录您的订单号 A1001，购买的是无线耳机，手机尾号 8899。请问您有什么需要我帮忙的吗？比如查询订单状态、处理售后问题，或其他服务？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 42, 'total_tokens': 92, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 42}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '56b3d2a8-95a2-48e3-8eed-ed4e732c52a3', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-2e80-7832-aaf1-cecd144f29bf-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 42, 'output_tokens': 50, 'total_tokens': 92, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})],    'messages_count': 2}

---------------------------------------------------------------------------------------------------------------------------------


[before_model] trim messages | 原始消息： [
  HumanMessage(content='你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。', additional_kwargs={}, response_metadata={}, id='137c5f4d-092e-478b-898d-79450fdf332b'),

  AIMessage(content='您好，张三！感谢您提供的信息。我已记录您的订单号 A1001，购买的是无线耳机，手机尾号 8899。请问您有什么需要我帮忙的吗？比如查询订单状态、处理售后问题，或其他服务？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 42, 'total_tokens': 92, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 42}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '56b3d2a8-95a2-48e3-8eed-ed4e732c52a3', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-2e80-7832-aaf1-cecd144f29bf-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 42, 'output_tokens': 50, 'total_tokens': 92, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}),

  HumanMessage(content='耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。', additional_kwargs={}, response_metadata={}, id='d64b7be1-1b09-44de-9a99-66df3946667f')]


[before_model] trim messages | 截断后消息： [
  HumanMessage(content='你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。', additional_kwargs={}, response_metadata={}, id='137c5f4d-092e-478b-898d-79450fdf332b'),

  AIMessage(content='您好，张三！感谢您提供的信息。我已记录您的订单号 A1001，购买的是无线耳机，手机尾号 8899。请问您有什么需要我帮忙的吗？比如查询订单状态、处理售后问题，或其他服务？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 42, 'total_tokens': 92, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 42}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '56b3d2a8-95a2-48e3-8eed-ed4e732c52a3', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-2e80-7832-aaf1-cecd144f29bf-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 42, 'output_tokens': 50, 'total_tokens': 92, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}),

  HumanMessage(content='耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。', additional_kwargs={}, response_metadata={}, id='d64b7be1-1b09-44de-9a99-66df3946667f')]

[before_model] trim messages | 原始消息数=(3, <class 'list'>), 截断后消息数=(3, <class 'list'>)
[用户] 耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。
[模型回复] 您好，张三。非常理解您对耳机问题的困扰，我们很愿意帮您处理。

根据“三包”政策中关于网络交易商品的规定，对于性能故障，通常7天内可以退货，15天内可以换货。您的订单信息显示购买已经超过15天了吗？如果还在15天内，我这边可以优先为您申请“换新”服务。

另外，为了核实您的订单详情并为您开启特殊换新通道，我需要先验证一下您的身份。请问您订单上留的【收件人姓名】和【完整手机号码】方便提供一下吗？这步操作是为了确保快速为您处理，避免流程延误。


[当前短期记忆] {'messages': [
  HumanMessage(content='你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。', additional_kwargs={}, response_metadata={}, id='137c5f4d-092e-478b-898d-79450fdf332b'),

  AIMessage(content='您好，张三！感谢您提供的信息。我已记录您的订单号 A1001，购买的是无线耳机，手机尾号 8899。请问您有什么需要我帮忙的吗？比如查询订单状态、处理售后问题，或其他服务？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 42, 'total_tokens': 92, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 42}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '56b3d2a8-95a2-48e3-8eed-ed4e732c52a3', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-2e80-7832-aaf1-cecd144f29bf-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 42, 'output_tokens': 50, 'total_tokens': 92, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}),

  HumanMessage(content='耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。', additional_kwargs={}, response_metadata={}, id='d64b7be1-1b09-44de-9a99-66df3946667f'),

  AIMessage(content='您好，张三。非常理解您对耳机问题的困扰，我们很愿意帮您处理。\n\n根据“三包”政策中关于网络交易商品的规定，对于性能故障，通常7天内可以退货，15天内可以换货。您的订单信息显示购买已经超过15天了吗？如果还在15天内，我这边可以优先为您申请“换新”服务。\n\n另外，为了核实您的订单详情并为您开启特殊换新通道，我需要先验证一下您的身份。请问您订单上留的【收件人姓名】和【完整手机号码】方便提供一下吗？这步操作是为了确保快速为您处理，避免流程延误。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 134, 'prompt_tokens': 116, 'total_tokens': 250, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 116}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '07feff6e-6732-4137-8051-2a6178911a0b', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-33b8-7d72-ab1e-acd6bc3dca18-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 116, 'output_tokens': 134, 'total_tokens': 250, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})], 'messages_count': 4}


---------------------------------------------------------------------------------------------------------------------------------



[before_model] trim messages | 原始消息： [
  HumanMessage(content='你好，我叫张三。我的订单号是 A1001，买的是无线耳机，手机号尾号是 8899。', additional_kwargs={}, response_metadata={}, id='137c5f4d-092e-478b-898d-79450fdf332b'),

  AIMessage(content='您好，张三！感谢您提供的信息。我已记录您的订单号 A1001，购买的是无线耳机，手机尾号 8899。请问您有什么需要我帮忙的吗？比如查询订单状态、处理售后问题，或其他服务？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 42, 'total_tokens': 92, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 42}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '56b3d2a8-95a2-48e3-8eed-ed4e732c52a3', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-2e80-7832-aaf1-cecd144f29bf-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 42, 'output_tokens': 50, 'total_tokens': 92, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}),

  HumanMessage(content='耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。', additional_kwargs={}, response_metadata={}, id='d64b7be1-1b09-44de-9a99-66df3946667f'),

  AIMessage(content='您好，张三。非常理解您对耳机问题的困扰，我们很愿意帮您处理。\n\n根据“三包”政策中关于网络交易商品的规定，对于性能故障，通常7天内可以退货，15天内可以换货。您的订单信息显示购买已经超过15天了吗？如果还在15天内，我这边可以优先为您申请“换新”服务。\n\n另外，为了核实您的订单详情并为您开启特殊换新通道，我需要先验证一下您的身份。请问您订单上留的【收件人姓名】和【完整手机号码】方便提供一下吗？这步操作是为了确保快速为您处理，避免流程延误。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 134, 'prompt_tokens': 116, 'total_tokens': 250, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 116}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '07feff6e-6732-4137-8051-2a6178911a0b', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-33b8-7d72-ab1e-acd6bc3dca18-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 116, 'output_tokens': 134, 'total_tokens': 250, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}),

  HumanMessage(content='请基于我们最近聊过的信息，帮我整理一段售后说明。', additional_kwargs={}, response_metadata={}, id='760a2d30-28e9-422e-8b0e-8ae7f04689ad')]


[before_model] trim messages | 截断后消息： [

  HumanMessage(content='耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。', additional_kwargs={}, response_metadata={}, id='d64b7be1-1b09-44de-9a99-66df3946667f'),

  AIMessage(content='您好，张三。非常理解您对耳机问题的困扰，我们很愿意帮您处理。\n\n根据“三包”政策中关于网络交易商品的规定，对于性能故障，通常7天内可以退货，15天内可以换货。您的订单信息显示购买已经超过15天了吗？如果还在15天内，我这边可以优先为您申请“换新”服务。\n\n另外，为了核实您的订单详情并为您开启特殊换新通道，我需要先验证一下您的身份。请问您订单上留的【收件人姓名】和【完整手机号码】方便提供一下吗？这步操作是为了确保快速为您处理，避免流程延误。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 134, 'prompt_tokens': 116, 'total_tokens': 250, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 116}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '07feff6e-6732-4137-8051-2a6178911a0b', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-33b8-7d72-ab1e-acd6bc3dca18-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 116, 'output_tokens': 134, 'total_tokens': 250, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}),

  HumanMessage(content='请基于我们最近聊过的信息，帮我整理一段售后说明。', additional_kwargs={}, response_metadata={}, id='760a2d30-28e9-422e-8b0e-8ae7f04689ad')]
[before_model] trim messages | 原始消息数=(5, <class 'list'>), 截断后消息数=(3, <class 'list'>)
[用户] 请基于我们最近聊过的信息，帮我整理一段售后说明。
[模型回复] 好的，根据我们之前的沟通，我为您整理了一段初步的售后说明。请您核对，我会根据您最终确认的信息进行完善。

---

**售后处理说明**

**【产品信息】**
产品名称：无线耳机
问题描述：
1.  左耳无声，无法正常播放音频。
2.  充电盒接触不良，无法正常为耳机充电。

**【客户诉求】**
要求优先进行换新处理，不接受维修。

**【处理方案建议】**
*   **前提条件：** 根据您订单的购买时间，是否仍在“三包”规定的15天换货期内？
    *   **如在15天内：** 可直接执行换新流程。
    *   **如已超过15天：** 需按照保修政策进行检测后处理，无法直接换新。
*   **身份验证：** 为了核实订单信息并启动换新流程，需要您提供订单上的【收件人姓名】和【完整手机号码】。
*   **建议流程：** 请您确认上述信息，我将在身份验证通过后，为您启动换新处理。

---

**请您确认：**
1.  上述描述的问题和需求是否准确？
2.  您的耳机是否仍在15天换货期内？如果不是，您希望如何处理？
3.  您方便提供订单上的【收件人姓名】和【完整手机号码】用于验证吗？

等您确认后，我会将这条说明更新为您正式提交售后的依据。
[当前短期记忆] {'messages': [
  HumanMessage(content='耳机左耳没声音，充电盒也接触不良。我希望优先换新，不想维修。', additional_kwargs={}, response_metadata={}, id='d64b7be1-1b09-44de-9a99-66df3946667f'),

  AIMessage(content='您好，张三。非常理解您对耳机问题的困扰，我们很愿意帮您处理。\n\n根据“三包”政策中关于网络交易商品的规定，对于性能故障，通常7天内可以退货，15天内可以换货。您的订单信息显示购买已经超过15天了吗？如果还在15天内，我这边可以优先为您申请“换新”服务。\n\n另外，为了核实您的订单详情并为您开启特殊换新通道，我需要先验证一下您的身份。请问您订单上留的【收件人姓名】和【完整手机号码】方便提供一下吗？这步操作是为了确保快速为您处理，避免流程延误。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 134, 'prompt_tokens': 116, 'total_tokens': 250, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 116}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '07feff6e-6732-4137-8051-2a6178911a0b', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-33b8-7d72-ab1e-acd6bc3dca18-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 116, 'output_tokens': 134, 'total_tokens': 250, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}),

  HumanMessage(content='请基于我们最近聊过的信息，帮我整理一段售后说明。', additional_kwargs={}, response_metadata={}, id='760a2d30-28e9-422e-8b0e-8ae7f04689ad'),


  AIMessage(content='好的，根据我们之前的沟通，我为您整理了一段初步的售后说明。请您核对，我会根据您最终确认的信息进行完善。\n\n---\n\n**售后处理说明**\n\n**【产品信息】**\n产品名称：无线耳机\n问题描述：\n1.  左耳无声，无法正常播放音频。\n2.  充电盒接触不良，无法正常为耳机充电。\n\n**【客户诉求】**\n要求优先进行换新处理，不接受维修。\n\n**【处理方案建议】**\n*   **前提条件：** 根据您订单的购买时间，是否仍在“三包”规定的15天换货期内？\n    *   **如在15天内：** 可直接执行换新流程。\n    *   **如已超过15天：** 需按照保修政策进行检测后处理，无法直接换新。\n*   **身份验证：** 为了核实订单信息并启动换新流程，需要您提供订单上的【收件人姓名】和【完整手机号码】。\n*   **建议流程：** 请您确认上述信息，我将在身份验证通过后，为您启动换新处理。\n\n---\n\n**请您确认：**\n1.  上述描述的问题和需求是否准确？\n2.  您的耳机是否仍在15天换货期内？如果不是，您希望如何处理？\n3.  您方便提供订单上的【收件人姓名】和【完整手机号码】用于验证吗？\n\n等您确认后，我会将这条说明更新为您正式提交售后的依据。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 304, 'prompt_tokens': 187, 'total_tokens': 491, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 187}, 'model_provider': 'deepseek', 'model_name': 'DeepSeek-V4-Flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '18c3c818-9fd5-41f0-9411-ebbfba2e0e41', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019ed8b3-4034-7321-ac32-5dbbe90bd443-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 187, 'output_tokens': 304, 'total_tokens': 491, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})], 'messages_count': 4}
---------------------------------------------------------------------------------------------------------------------------------

"""
