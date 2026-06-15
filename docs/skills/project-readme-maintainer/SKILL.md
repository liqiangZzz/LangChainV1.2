---
name: project-readme-maintainer
description: Use this project-local skill when creating or maintaining the repository README.md for this LangChain learning project. It audits existing project files and updates README content only from code and documentation already present in the repository.
---

# Project README Maintainer

这是本仓库内使用的项目级 skill，不是 Codex 全局 skill。
它只负责创建或维护项目根目录的 `README.md`。

## 核心边界

- README 面向项目使用者和学习者，说明“项目是什么、怎么配置、怎么运行、有哪些目录”。
- README 只能描述当前仓库已经存在的代码、目录和文档。
- README 必须增量维护：已有 README 时优先保留原有结构和内容，只更新与当前代码变化相关的小节。
- 只有 README 不存在，或用户明确要求重写时，才创建完整 README 或做大范围改写。
- 不把尚未写成代码的技术点提前写成完整学习路线。
- 不在 README 中泄露 `.env`、API Key、Token、密码等敏感信息。
- 涉及真实 LLM 调用、推理模型、联网、额度消耗时，必须提醒。
- 不负责维护包级 `__init__.py`；这类任务使用 `package-init-doc-maintainer`。

## 使用流程

快捷入口：

```bash
python scripts/audit_project_readme.py
```

手动流程：

1. 先读取项目约定：

   - `AGENTS.md`
   - `docs/skills/project-readme-maintainer/references/readme-guide.md`

2. 执行 README 扫描：

   ```bash
   python scripts/audit_project_readme.py
   ```

3. 根据扫描结果读取必要文件：

   - `README.md`，如果存在
   - `AGENTS.md`
   - 顶层入口脚本，例如 `quick_start.py`、`my_llm.py`、`env_utils.py`
   - 主要目录说明，例如 `models/__init__.py`、`agents/__init__.py`
   - 必要时读取更深层 `__init__.py`

4. 新增模块时，先确认对应 `__init__.py` 已经维护，再根据包说明增量更新 README：

   - 新增一级目录或重要子包时，补充目录结构或学习模块
   - 新增可直接运行的入口时，补充运行命令
   - 普通内部文件变化不要求逐个写入 README

5. 只根据实际代码和已有说明增量维护 README；已有内容合理时不要重写。

6. Markdown 文件不需要 `py_compile`。如果同时修改了脚本，再执行语法检查。

## 当前配套资源

- `docs/skills/project-readme-maintainer/references/readme-guide.md`
  README 维护细则。

- `scripts/audit_project_readme.py`
  扫描 README 是否存在，并列出可用于生成 README 的项目入口和包说明。

## 适合处理的请求

- “帮我创建 README”
- “根据已有代码更新 README”
- “检查 README 是否和当前目录不一致”
- “新增模块后，把 README 的目录说明补上”

## 不适合处理的请求

- 维护各包的 `__init__.py` 说明。
- 为未实现技术点提前生成完整学习路线。
- 修改 `.env`、密钥、线上配置或依赖安装流程。
- 自动安装依赖或运行会消耗 API 额度的示例。
