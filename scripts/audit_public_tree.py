#!/usr/bin/env python3
"""Fail when tracked source files contain machine-specific or secret values."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).relative_to(REPOSITORY_ROOT)
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# Join sensitive tokens so this audit file does not flag its own definitions.
IPV4 = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
LOCAL_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")
LOCALHOST = re.compile(r"\blocal" + r"host\b", re.IGNORECASE)
SECRET_ASSIGNMENT = re.compile(
    r"\b(?:api[_-]?key|pass(?:word|wd)?|secret|token)\b\s*[:=]\s*"
    r"['\"](?!<|replace|example)([^'\"]+)['\"]",
    re.IGNORECASE,
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [Path(item) for item in result.stdout.decode().split("\0") if item]


def main() -> int:
    violations = []
    checks = (
        ("IPv4 address", IPV4),
        ("local home path", LOCAL_PATH),
        ("localhost default", LOCALHOST),
        ("embedded secret", SECRET_ASSIGNMENT),
    )

    for relative_path in tracked_files():
        if relative_path == SELF or relative_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        path = REPOSITORY_ROOT / relative_path
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in checks:
                if pattern.search(line):
                    violations.append(f"{relative_path}:{line_number}: {label}")

    if violations:
        print("Public-tree audit failed:")
        print("\n".join(f"  {violation}" for violation in violations))
        return 1

    print("Public-tree audit passed: no machine-specific endpoints or secrets found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
