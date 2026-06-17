"""
环境变量工具模块

该模块用于加载和管理项目所需的环境变量配置。
使用 python-dotenv 从 .env 文件中读取环境变量，
并提供 API 相关的配置常量。
"""
import os

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量，override=True 表示覆盖已存在的环境变量
load_dotenv(override=True)

# DeepSeek API 密钥
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# DeepSeek API 基础 URL
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')

# MySQL checkpoint 数据库连接地址
MYSQL_DATABASE_URL = os.getenv('MYSQL_DATABASE_URL')
