from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_llm: BaseChatModel = init_chat_model(
    model='deepseek-chat',
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)
