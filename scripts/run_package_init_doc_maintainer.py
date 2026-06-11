"""
包级 __init__.py 文档维护入口脚本。

该脚本用于串起项目内文档维护流程：
1. 读取项目内 skill: docs/skills/package-init-doc-maintainer/SKILL.md
2. 执行扫描脚本: scripts/audit_init_docs.py
3. 提示继续参考 skill reference: docs/skills/package-init-doc-maintainer/references/package-init-doc-guide.md

脚本只读项目文件，不会修改代码或文档。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_FILE = PROJECT_ROOT / "docs" / "skills" / "package-init-doc-maintainer" / "SKILL.md"
REFERENCE_FILE = (
    PROJECT_ROOT
    / "docs"
    / "skills"
    / "package-init-doc-maintainer"
    / "references"
    / "package-init-doc-guide.md"
)
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_init_docs.py"


def read_skill_summary(path: Path) -> str:
    """读取 skill 的摘要部分，避免输出过长或截断代码块。"""
    if not path.exists():
        return f"未找到文件：{path.relative_to(PROJECT_ROOT)}"

    lines = path.read_text(encoding="utf-8").splitlines()
    end_index = len(lines)
    for index, line in enumerate(lines):
        if line.strip() == "手动流程：":
            end_index = index
            break

    return "\n".join(lines[:end_index]).strip()


def run_audit() -> str:
    """执行 __init__.py 文档扫描脚本。"""
    result = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return f"扫描失败：\n{result.stderr.strip()}"

    return result.stdout.strip()


def main() -> None:
    print("=" * 80)
    print("项目内 Skill：package-init-doc-maintainer")
    print("=" * 80)
    print(read_skill_summary(SKILL_FILE))

    print("\n" + "=" * 80)
    print("执行扫描：scripts/audit_init_docs.py")
    print("=" * 80)
    print(run_audit())

    print("\n" + "=" * 80)
    print("下一步")
    print("=" * 80)
    print(f"1. 继续参考：{REFERENCE_FILE.relative_to(PROJECT_ROOT)}")
    print("2. 根据扫描结果选择需要维护的包。")
    print("3. 只读取目标包已有代码，再增量更新对应 __init__.py。")
    print("4. 修改后执行：python -m py_compile <changed __init__.py>")


if __name__ == "__main__":
    main()
