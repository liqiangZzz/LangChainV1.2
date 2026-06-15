# AGENTS.md

本文件定义自动化编程助手在本仓库中的协作约定。项目使用说明和学习内容放在
`README.md` 与各包的 `__init__.py` 中，不在这里重复维护。

## 项目定位

这是一个 LangChain 学习与示例项目，主要使用 DeepSeek 演示模型调用、Agent、
工具调用、结构化输出、middleware、流式处理、批处理和异步调用。

主要目录与入口：

- `agents/`：Agent 基础能力及相关专题示例。
- `models/`：聊天模型调用及相关专题示例。
- `quick_start.py`：Agent 快速开始入口。
- `my_llm.py`：项目共享模型实例。
- `env_utils.py`：加载并导出模型服务所需环境变量。
- `docs/skills/`：项目文档维护 Skill。
- `scripts/`：只读审计脚本。

## 协作方式

- 默认使用中文回答。
- 需求明确时直接执行；缺少关键信息且会影响结果时再简短提问。
- 修改前先了解相关目录、现有代码风格和 `git status --short`。
- 保持改动范围小，不重构、删除或回滚与当前任务无关的内容。
- 完成后说明实际修改、验证结果以及仍存在的风险或未完成项。

## 代码与命名

- 使用 Python 编写示例，保持代码清晰、可直接阅读和运行。
- 包、模块、函数和变量使用 `lower_snake_case`，类名使用 `PascalCase`。
- 按学习顺序组织的示例使用 `NN_topic_name.py`，例如
  `03_model_invocation_config.py`。
- 复合词和缩写使用下划线分隔，例如 `json_schema`、`tool_strategy`。
- 示例说明、注释和文档字符串优先使用中文。
- 只为复杂或容易误解的逻辑添加必要注释，不解释显而易见的代码。
- 重命名模块或包时，同步更新代码引用、`__init__.py`、README 和运行命令。

## LangChain 示例约定

- Agent 示例放在 `agents/` 的对应主题包中，模型示例放在 `models/` 中。
- 优先复用 `my_llm.py` 的共享模型；只有演示初始化方式或特殊模型配置时，
  才在示例中单独创建模型。
- 环境变量统一从 `env_utils.py` 读取，不硬编码 API Key、Token、密码或服务地址。
- 示例应保留必要的输出，方便观察消息、工具调用和结构化结果。
- 很多脚本在模块顶层调用真实模型。不要为了检查导入而直接导入具体示例模块；
  可以安全导入只包含说明的包级 `__init__.py`。
- 执行真实模型、推理模型、联网或多轮 Agent 示例前，注意 API 额度、网络和环境配置，
  且不得打印敏感信息。

## 文档维护

- `__init__.py` 只维护包级导航和运行风险，不导入示例模块。
- 维护包说明时使用 `docs/skills/package-init-doc-maintainer/SKILL.md`，
  并先运行 `python scripts/audit_init_docs.py`。
- 维护根目录 README 时使用 `docs/skills/project-readme-maintainer/SKILL.md`，
  并先运行 `python scripts/audit_project_readme.py`。
- 文档只描述仓库中已经存在的代码；README 已存在时进行增量更新。
- 新建包后先维护该包及上级包的 `__init__.py`，再判断 README 是否需要更新。
- `AGENTS.md` 只保存长期有效的项目级规则，普通模块清单和学习细节不写入本文件。

## 验证

根据改动范围选择最小有效验证：

- 单文件语法检查：`python -m py_compile <file>`
- 包或目录语法检查：`python -m compileall -q <path>`
- 包说明审计：`python scripts/audit_init_docs.py`
- README 审计：`python scripts/audit_project_readme.py`
- 格式问题检查：`git diff --check`

不要仅为验证而运行会产生真实 API 请求的示例。若确需运行，应说明额度消耗；
如果受依赖、网络或环境变量限制无法验证，也要明确说明。

## 安全与 Git

- 不泄露或提交 `.env`、API Key、Token、密码等敏感信息。
- 未经明确授权，不执行删除目录、重置 Git、覆盖配置等破坏性操作。
- 不覆盖用户已有改动，只处理当前任务相关文件。
- 依赖安装、外部网络访问、线上部署或外部服务登录按需说明并请求授权。
- 不主动提交、推送或创建分支，除非用户明确要求。
- 不处理与当前任务无关的未提交文件。
