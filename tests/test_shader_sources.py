from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


SHADERS = [
    "trivial_compute.comp",
    "persistent_barrier_probe.comp",
]


@pytest.mark.parametrize("shader_name", SHADERS)
def test_shader_source_exists(repo_root: Path, shader_name: str) -> None:
    shader = repo_root / "shaders" / shader_name

    assert shader.is_file()
    assert shader.read_text(encoding="utf-8").startswith("#version 450")


@pytest.mark.parametrize("shader_name", SHADERS)
def test_shader_source_compiles_with_glslang_if_available(repo_root: Path, shader_name: str) -> None:
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    shader = repo_root / "shaders" / shader_name
    result = subprocess.run(
        [glslang, "-V", str(shader), "-o", "/dev/null"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
