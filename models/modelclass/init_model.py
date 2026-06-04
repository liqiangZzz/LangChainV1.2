"""
   测试连接各个大模型
"""
from models.modelclass.class_llm import deepseek_llm2

# print(deepseek_llm.invoke('请介绍一下你自己'))

print(deepseek_llm2.invoke('请介绍一下你自己'))
