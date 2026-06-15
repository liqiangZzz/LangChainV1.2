# AGENTS.md

本文件是本项目的协作说明。所有自动化编程助手在本仓库中工作时，都应优先遵循这里的约定。

## 项目概览

这是一个 LangChain 学习与示例项目，主要演示模型调用、流式输出、异步调用、批处理、工具调用、结构化输出等能力。

主要文件和目录：

- `quick_start.py`：Agent 快速示例入口。
- `my_llm.py`：统一创建 DeepSeek Chat 模型实例。
- `env_utils.py`：加载 `.env` 并导出 API 相关环境变量。
- `models/`：按主题组织的 LangChain 示例脚本。

## 默认沟通方式

- 默认使用中文回答。
- 需求明确时直接修改代码，不只给方案。
- 如果信息不足且会影响实现结果，先简短提问。
- 完成后说明改了什么、如何验证、是否还有风险或未完成项。

## 代码风格

- 使用 Python 编写示例，遵循当前项目的简洁脚本风格。
- 文件、函数、变量命名尽量贴合现有目录和示例命名习惯。
- Python 包、模块、函数和变量使用 `lower_snake_case`，类名使用 `PascalCase`。
- 按学习顺序组织的示例文件统一使用 `NN_topic_name.py`，例如
  `03_model_invocation_config.py`。
- 缩写也作为独立单词处理，例如使用 `json_schema`、`tool_strategy`，
  避免 `jsonschema`、`toolstrategy` 等粘连写法。
- 示例说明、注释和文档字符串优先使用中文。
- 复杂逻辑可以添加简短注释；不要添加解释显而易见代码的空洞注释。
- 保持改动范围小，不做与当前任务无关的重构。
- 不随意移动、重命名或删除已有示例文件，除非用户明确要求。
- 重命名已有模块时，必须同步更新 README、`__init__.py`、运行命令和代码引用。

## LangChain 约定

- 复用 `my_llm.py` 中的 `deepseek_llm`，避免在示例文件里重复创建模型实例。
- 环境变量统一从 `env_utils.py` 读取，不在代码中硬编码 API Key、Base URL、Token、密码等敏感信息。
- 新增示例时优先放在 `models/` 下对应主题目录；如果是新主题，可以新建清晰命名的子目录。
- 示例脚本应尽量可直接运行，并保留必要的 `print` 输出，方便学习和调试。
- 涉及真实模型调用时，注意这会消耗外部 API 额度；执行前确认 `.env` 配置是否存在即可，不要打印密钥内容。

## 包说明维护 Skill

- 维护 Python 包的 `__init__.py` 说明时，优先参考项目内 skill：`docs/skills/package-init-doc-maintainer/SKILL.md`。
- 该 skill 会进一步引用 `docs/skills/package-init-doc-maintainer/references/package-init-doc-guide.md` 作为当前项目的包说明维护细则。
- 该文档只描述当前已有代码承载的知识点；尚未写成代码的技术点不要提前写成完整闭环。
- 维护 `__init__.py` 包说明时，先运行 `python scripts/audit_init_docs.py`，再按扫描结果决定是否调整。
- `__init__.py` 只写包级说明，不导入示例模块，避免 import 包时触发真实 LLM 调用。

## README 维护 Skill

- 创建或维护根目录 `README.md` 时，优先参考项目内 skill：`docs/skills/project-readme-maintainer/SKILL.md`。
- README 只描述当前已有代码、目录和文档，不提前写尚未实现的技术点。
- README 已存在时优先增量维护，只更新与当前代码变化相关的小节，不默认重写整份文档。
- 维护 README 前先运行 `python scripts/audit_project_readme.py`，再按扫描结果决定是否调整。
- README 面向项目使用者和学习者；包级细节应放在对应 `__init__.py` 中。

## 新增模块后的文档流程

- 新建 Python 包后，先运行 `python scripts/audit_init_docs.py` 查看包说明状态。
- 先维护新包的 `__init__.py`，再检查上级包的 `__init__.py` 是否需要增加子包导航。
- 再运行 `python scripts/audit_project_readme.py`，判断根目录 `README.md`
  是否需要补充目录结构、学习模块或运行命令。
- `AGENTS.md` 只维护项目级协作规则；普通新增模块不需要写入本文件。

## 安全边界

- 不泄露 `.env`、API Key、Token、密码等敏感信息。
- 不执行破坏性命令，例如删除文件、重置 Git、清空目录、覆盖重要配置，除非用户明确授权。
- 不回滚或覆盖用户已有改动。若发现工作区有未提交修改，只处理当前任务相关文件。
- 涉及依赖安装、外部网络访问、线上部署或登录外部服务时，先说明当前限制并按需请求授权。

## 验证建议

根据改动类型选择最小有效验证：

- 语法检查：`python -m py_compile <file>`
- 运行单个示例：`python <script.py>`
- 批量检查 Python 文件：`python -m compileall .`

如果运行示例会触发真实 LLM 调用，应在结果中说明这一点；如果因为缺少依赖、网络或环境变量无法验证，也要明确说明。

## Git 协作

- 开始修改前先查看 `git status --short`。
- 不主动提交、推送或创建分支，除非用户明确要求。
- 不处理与当前任务无关的未提交文件。
- 最终回复中列出本次实际改动的文件。
