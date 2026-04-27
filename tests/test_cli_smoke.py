from __future__ import annotations

import os
import subprocess
from typing import Sequence, Union

import pytest


CommandPart = Union[os.PathLike[str], str]


def run_cli(command: Sequence[CommandPart], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


@pytest.mark.parametrize("name", ["spock-bench", "spock-check"])
def test_planned_cli_help_smoke(cli_path, name: str) -> None:
    result = run_cli([cli_path(name), "--help"])

    assert result.returncode == 0, result.stderr
    assert result.stdout or result.stderr


@pytest.mark.parametrize("mode", ["pp520", "tg128"])
def test_spock_bench_accepts_planned_modes(cli_path, mode: str) -> None:
    result = run_cli([cli_path("spock-bench"), "--mode", mode], timeout=120)

    assert result.returncode == 0, result.stderr


def test_barrier_probe_help_smoke(cli_path) -> None:
    result = run_cli([cli_path("vk_barrier_probe"), "--help"])

    assert result.returncode == 0, result.stderr
    assert result.stdout or result.stderr
