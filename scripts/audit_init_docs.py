"""
扫描项目内 Python 包的 __init__.py 文档状态。

该脚本只读文件，不会修改项目内容。它用于辅助维护包级说明：
- 找出 Python 包目录。
- 展示包内脚本列表。
- 判断 __init__.py 是否为空，是否包含文档字符串。
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


def get_doc_summary(init_file: Path) -> tuple[str, str]:
    """get_doc_summary 函数。

    Args:
        init_file: __init__.py 文件路径。
    """
    content = init_file.read_text(encoding="utf-8").strip()
    if not content:
        return "empty", ""

    try:
        module = ast.parse(content)
    except SyntaxError:
        return "syntax-error", ""

    doc = ast.get_docstring(module) or ""
    if not doc:
        return "no-docstring", ""

    first_line = doc.strip().splitlines()[0] if doc.strip() else ""
    return "has-docstring", first_line


def iter_project_packages() -> list[Path]:
    packages: list[Path] = []
    for init_file in PROJECT_ROOT.rglob("__init__.py"):
        if should_skip(init_file.relative_to(PROJECT_ROOT)):
            continue
        packages.append(init_file.parent)
    return sorted(packages)


def main() -> None:
    packages = iter_project_packages()

    if not packages:
        print("未发现项目级 Python 包。")
        return

    for package_dir in packages:
        init_file = package_dir / "__init__.py"
        status, first_line = get_doc_summary(init_file)
        py_files = sorted(
            file.name
            for file in package_dir.glob("*.py")
            if file.name != "__init__.py"
        )

        rel_dir = package_dir.relative_to(PROJECT_ROOT)
        print(f"\n[{rel_dir}]")
        print(f"__init__.py: {status}")
        if first_line:
            print(f"doc: {first_line}")

        if py_files:
            print("files:")
            for file_name in py_files:
                print(f"  - {file_name}")
        else:
            print("files: 无直接 .py 示例文件")


if __name__ == "__main__":
    main()
