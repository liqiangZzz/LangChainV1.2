from typing import Dict, Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_agent
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from models.init_chat_model.init_chat_model_llm import deepseek_llm, glm_llm

BANNED_KEYWORDS = ["暴力", "枪支", "毒品", "色情", "赌博"]


@tool
def publish_post(topic: str) -> str:
    """根据主题发布帖子"""
    return f"已发布【{topic}】的帖子。"


@before_agent(can_jump_to=["end"])
def content_input_filter(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    if not state["messages"]:
        return None

    print("state",state)

    first_message = state["messages"][-1]

    for keyword in BANNED_KEYWORDS:
        if keyword in first_message.content:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"您输入包含违禁词{keyword}，无法生成帖子。"
                }],
                "jump_to": "end"
            }

    return None


agent = create_agent(
    model= deepseek_llm,
    tools=[publish_post],
    middleware=[content_input_filter],
    system_prompt="你是一个社区论坛内容助手，可以生成对应主题的帖子并发布"
)

result1 = agent.invoke({  # type: ignore
    "messages": [{
        "role": "user",
        "content": "帮我写一篇50字关于暴力画面的帖子并发布"
    }]
})
print(result1["messages"][-1].content)

print("-----------------" * 4)

result2 = agent.invoke({ # type: ignore
    "messages": [{
        "role": "user",
        "content": "帮我写一篇50字的AI技术文章并发布"
    }]
})
print(result2["messages"][-1].content)
