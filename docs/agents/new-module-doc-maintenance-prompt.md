# 新增模块文档维护提示词

使用前将 `<模块路径>` 替换为实际路径，例如 `agents/tool_creation`，然后复制下面整段内容：

```text
新增模块 <模块路径> 已完成。

1. 使用 package-init-doc-maintainer 完善该模块及上级包的 __init__.py。
2. 使用 project-readme-maintainer 增量更新根目录 README.md。
3. 检查是否影响项目协作规则；只有确实需要时才更新 AGENTS.md。
4. 完成后运行相关审计、语法检查和格式检查。
5. 最终说明实际修改的文件、验证结果，以及是否存在未完成项或风险。
```
