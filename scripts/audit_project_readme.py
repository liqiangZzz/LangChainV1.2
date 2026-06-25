"""
扫描 README 维护所需的项目信息。

该脚本只读文件，不会修改项目内容。它用于辅助创建或维护根目录 README.md：
- 检查 README.md 是否存在。
- 列出顶层 Python 入口文件。
- 列出项目内 Python 包说明的首行。
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".idea", ".venv", "__pycache__", "venv", "env", "ENV"}


def should_skip(path: Path) -> bool:
    """should_skip 函数。

    Args:
        path: 待检查的文件或目录路径。
    """
    return any(part in SKIP_DIRS for part in path.parts)


def get_doc_first_line(path: Path) -> str:
    """get_doc_first_line 函数。

    Args:
        path: 待检查的文件或目录路径。
    """
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return ""

    try:
        module = ast.parse(content)
    except SyntaxError:
        return ""

    doc = ast.get_docstring(module) or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def iter_package_docs() -> list[tuple[Path, str]]:
    package_docs: list[tuple[Path, str]] = []
    for init_file in PROJECT_ROOT.rglob("__init__.py"):
        relative = init_file.relative_to(PROJECT_ROOT)
        if should_skip(relative):
            continue
        package_docs.append((init_file.parent, get_doc_first_line(init_file)))
    return sorted(package_docs)


def main() -> None:
    readme = PROJECT_ROOT / "README.md"
    print(f"README.md: {'exists' if readme.exists() else 'missing'}")

    top_level_py = sorted(
        file.name
        for file in PROJECT_ROOT.glob("*.py")
        if file.is_file()
    )
    print("\n顶层 Python 文件:")
    for file_name in top_level_py:
        print(f"  - {file_name}")

    print("\n项目包说明:")
    for package_dir, first_line in iter_package_docs():
        rel_dir = package_dir.relative_to(PROJECT_ROOT)
        doc = first_line if first_line else "无 docstring"
        print(f"  - {rel_dir}: {doc}")

    print("\n建议读取:")
    print("  - AGENTS.md")
    if readme.exists():
        print("  - README.md")
    print("  - agents/__init__.py")
    print("  - models/__init__.py")


if __name__ == "__main__":
    main()
