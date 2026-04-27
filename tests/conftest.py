from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, List

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--spock-bin-dir",
        action="store",
        default=None,
        help="Directory containing built spock CLI executables.",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def spock_bin_dirs(pytestconfig: pytest.Config) -> List[Path]:
    configured = pytestconfig.getoption("--spock-bin-dir") or os.environ.get("SPOCK_BIN_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_dir():
            return [path, path / "apps", path / "tools", path / "bin"]

    build_dir = os.environ.get("SPOCK_BUILD_DIR")
    candidates = []
    if build_dir:
        root = Path(build_dir).expanduser().resolve()
        candidates.extend([root, root / "apps", root / "tools", root / "bin"])

    candidates.extend(
        [
            REPO_ROOT / "build",
            REPO_ROOT / "build" / "apps",
            REPO_ROOT / "build" / "tools",
            REPO_ROOT / "build" / "bin",
        ]
    )

    return [path for path in candidates if path.is_dir()]


@pytest.fixture(scope="session")
def cli_path(spock_bin_dirs: List[Path]) -> Callable[[str], Path]:
    def resolve(name: str) -> Path:
        for bin_dir in spock_bin_dirs:
            candidate = bin_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        found = shutil.which(name)
        if found:
            return Path(found)

        pytest.skip(f"{name} is not built; set --spock-bin-dir, SPOCK_BIN_DIR, or SPOCK_BUILD_DIR")

    return resolve
