"""Shared script import setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_project_root(script_file: str, remove_shadowing_utils: bool) -> Path:
    project_root: Path = find_project_root(script_file)
    project_root_text: str = str(project_root)
    next_paths: list[str] = [
        path
        for path in sys.path
        if path != project_root_text
        and not _contains_shadowing_utils(path, remove_shadowing_utils)
    ]
    sys.path = [project_root_text, *next_paths]
    return project_root


def find_project_root(script_file: str) -> Path:
    for parent in Path(script_file).resolve().parents:
        if (
            (parent / "requirements.txt").is_file()
            and (parent / "utils").is_dir()
            and (parent / "feature_extraction").is_dir()
            and (parent / "models").is_dir()
        ):
            return parent
    raise FileNotFoundError(f"Could not resolve project root for script: {script_file}")


def _contains_shadowing_utils(path: str, remove_shadowing_utils: bool) -> bool:
    if not remove_shadowing_utils:
        return False
    return os.path.isfile(os.path.join(path, "utils.py"))
