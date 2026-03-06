"""Shared logging helpers (Tee stdout mirror)."""

from __future__ import annotations

import sys
from pathlib import Path


class Tee:
    """Mirror sys.stdout to both the console and a log file."""

    def __init__(self, fname: Path) -> None:
        fname.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(fname, "w", encoding="utf-8")  # noqa: SIM115
        self.stdout = sys.stdout

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.file.write(data)

    def flush(self) -> None:
        self.stdout.flush()
        if not self.file.closed:
            self.file.flush()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        sys.stdout = self.stdout
        self.file.close()
