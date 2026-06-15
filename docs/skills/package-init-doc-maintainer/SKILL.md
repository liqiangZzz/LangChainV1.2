---
name: package-init-doc-maintainer
description: Use this project-local skill when maintaining Python package __init__.py documentation for this LangChain learning project. It scans package docstring status, reads existing code, and updates only package-level descriptions based on code already present in the repository.
---

# Package Init Doc Maintainer

这是本仓库内使用的项目级 skill，不是 Codex 全局 skill。
它用于维护当前 LangChain 学习项目中 Python 包的 `__init__.py` 包级说明。

## 核心边界

- 只描述当前仓库里已经存在的代码、目录和示例。
- 不把尚未写成代码的技术点提前写成完整文档。
- 如果发现某个知识点还没有示例代码，只能标记为“待补充”或暂不写入。
- 不在 `__init__.py` 中导入示例模块，避免 `import models` 或 `import agents` 时触发真实 LLM 调用。
- 涉及真实模型调用、推理模型、联网、额度消耗时，必须提醒。

## 使用流程

扫描入口：

```bash
python scripts/audit_init_docs.py
```

维护流程：

1. 先读取项目约定：

   - `AGENTS.md`
   - `docs/skills/package-init-doc-maintainer/references/package-init-doc-guide.md`

2. 如果任务涉及 `__init__.py` 包说明，先运行：

   ```bash
   python scripts/audit_init_docs.py
   ```

3. 根据扫描结果读取目标目录：

   - 目标目录的 `__init__.py`
   - 同目录下已有示例文件
   - 如果目标是新增子包，必须读取上一级包的 `__init__.py`

4. 只根据实际代码增量更新文档：

   - 更新目标包的 `__init__.py`
   - 新增子包时，在上一级 `__init__.py` 中补充简短子包导航
   - 不在包说明中导入示例模块

5. 修改后执行最小验证：

   ```bash
   python -m py_compile <changed python file>
   ```

   Markdown 文件不需要 py_compile；如果同时改了脚本，则检查脚本。

## 当前配套资源

- `docs/skills/package-init-doc-maintainer/references/package-init-doc-guide.md`
  包级 `__init__.py` 文档增量维护细则，记录当前仓库的维护边界和专项流程。

- `scripts/audit_init_docs.py`
  扫描项目内 Python 包和 `__init__.py` 文档状态。

## 适合处理的请求

- “检查新增模块的 `__init__.py` 是否需要补说明”
- “根据已有代码更新某个包说明”
- “扫描包级说明是否和代码不一致”
- “帮我把新写的示例加到包级说明里”
- “新增子包后，同时更新子包和上级包说明”
- “补充已有示例的学习说明或注释”

## 不适合处理的请求

- 为尚未实现的技术点提前生成完整学习路线。
- 维护 README、学习路线、专题文档等非 `__init__.py` 文档。
- 把当前项目规则推广成全局规则。
- 修改 `.env`、密钥、线上配置或依赖安装流程。
- 为了文档美观而重命名、移动或删除已有示例文件。
