"""速率限制"""
import time

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 初始化速率限制器
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 每10s 允许1个请求
    check_every_n_seconds=0.1,  # 每100毫秒检查一次，控制更精细
    max_bucket_size=5,  # 允许短时间突发5个请求，应对流量小高峰
)

deepseek_llm = init_chat_model(
    model='deepseek-chat',
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    rate_limiter=rate_limiter  # 关键步骤
)

for i in range(3):
    response = deepseek_llm.invoke('你好，你是谁？')
    print(response.content)
    print(time.time())  # 打印当前时间，用于查看请求间隔，返回时间单位为秒
