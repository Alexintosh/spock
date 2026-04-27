#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def main() -> int:
    diary = Path(__file__).resolve().parents[1] / "diary"
    entries = sorted(path for path in diary.glob("*.md") if path.name != "README.md")
    if len(entries) < 2:
        raise SystemExit("expected at least two diary entries")
    for entry in entries:
        text = entry.read_text(encoding="utf-8")
        for heading in ("## Goal", "## Verification"):
            if heading not in text:
                raise SystemExit(f"{entry} missing {heading}")
        if len(text.split()) < 500:
            raise SystemExit(f"{entry} is too short to be an explanatory diary entry")
    print(f"diary entries valid: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
