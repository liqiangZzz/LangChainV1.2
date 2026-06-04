"""


"""
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import StdOutCallbackHandler, BaseCallbackHandler
from langchain_core.runnables import ConfigurableField

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# 1. 定义自定义回调处理器：用于监控模型运行全过程
class MyCustomCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized, prompts, **kwargs):
        """当模型开始运行时触发"""

        # 从 kwargs 字典中安全地获取 config 传入的运行信息
        print("kwargs:", kwargs)

        # 从 kwargs 中提取配置信息
        run_name = kwargs.get("name")  # 对应 config 中的 'run_name'
        tags = kwargs.get("tags", [])  # 获取标签列表
        metadata = kwargs.get("metadata", {})  # 获取元数据
        run_id = kwargs.get("run_id")  # 本次运行的唯一标识 ID

        print(f"[LLM开始] 运行名称: {run_name}")
        print(f"          标签: {tags}")
        print(f"          元数据: {metadata}")
        print(f"          运行ID: {run_id}")

    # 可以在这里执行更复杂的操作，比如：
    # 1. 将信息记录到日志系统或数据库
    # 2. 根据 metadata 中的 user_id 进行用户行为分析
    # 3. 根据 tags 对运行进行分类和监控
    # 4. 触发自定义事件，比如发送通知到监控系统
    def on_llm_end(self, response, **kwargs):
        """当模型运行结束并返回结果时触发"""

        # **kwargs 中的参数有如下
        print("kwargs:", kwargs)

        # 提取运行结束时的信息
        run_id = kwargs.get("run_id")
        print(f"[LLM结束] 运行ID: {run_id}")
        # 可以在这里记录运行结束的信息，比如消耗的令牌数等
        # 从response中提取模型消耗令牌数
        print("response:", response)

        # 修复点：LangChain 1.x 推荐使用 usage_metadata 来获取 Token 统计
        # 如果模型不支持或没返回，则设为空字典防止报错
        usage = getattr(response, "usage_metadata", {})

        if usage:
            total_tokens = usage.get("total_tokens", 0)
            print(f"统计: 消耗总令牌数 (Total Tokens): {total_tokens}")
        else:
            # 兼容性备选方案：从响应元数据中提取
            meta_usage = response.response_metadata.get("token_usage", {})
            print(f"统计: 消耗总令牌数 (从元数据提取): {meta_usage.get('total_tokens', '未知')}")


# 实例化回调处理器
custom_handler = MyCustomCallbackHandler()

deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    # 指定可调整参数
    # configurable_fields=("model",  "temperature", "max_tokens"),
).configurable_fields(  # <--- 这里用点(.)接上，效果和你写在参数里是一样的
    # 格式：模型属性名 = ConfigurableField(id="你在config里用的名字")
    model_name=ConfigurableField(id="model"),  # 将内部的 model_name 映射为外部的 model
    temperature=ConfigurableField(id="temperature"),
    max_tokens=ConfigurableField(id="max_tokens")

)
# 1. 初始化模型并直接通过链式调用指定可调整字段

# 2. 准备 config 字典
config = {
    "run_name": "joke_generation",  # 在LangSmith中这次运行会显示为 "joke_generation"
    "tags": ["tag1", "tag2"],  # 打上标签便于分类查找
    "metadata": {"user_id": "123"},  # 记录用户ID
    "callbacks": [custom_handler],  # 启用自定义回调
    "configurable": {
        "model": "deepseek-reasoner",  # 配置模型参数
        "temperature": 0.7,  # 配置温度参数
        "max_tokens": 1000  # 配置最大令牌数
    }
}

# 3. 调用模型并传入 config
response = deepseek_llm.invoke(
    "给我讲个AI相关的笑话",
    config=config
)

print(response.content)
