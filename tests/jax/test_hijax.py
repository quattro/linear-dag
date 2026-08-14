# pattern: Mixed (unavoidable)
# Reason: Metadata contract tests read repository configuration and assert its
# pure supported-runtime policy before the private HiJAX adapter is introduced.

from __future__ import annotations

import tomllib

from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_supported_runtime_metadata_and_lock_are_consistent() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    build_requires = pyproject["build-system"]["requires"]
    runtime_requires = project["dependencies"]
    supported_python = SpecifierSet(project["requires-python"])

    assert Version("3.11") not in supported_python
    assert all(Version(version) in supported_python for version in ("3.12", "3.13", "3.14"))
    assert Version("3.15") not in supported_python
    assert "Programming Language :: Python :: 3.11" not in project["classifiers"]
    assert pyproject["tool"]["hatch"]["envs"]["all"]["matrix"][0]["python"] == ["3.12", "3.13", "3.14"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py312"

    for requirements in (build_requires, runtime_requires):
        assert "jax==0.11.0" in requirements
        assert "jaxlib==0.11.0" in requirements
        assert "numpy>=2.1" in requirements
        assert "scipy>=1.15" in requirements

    ignored_paths = (_REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "uv.lock" not in ignored_paths

    lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())
    assert lock["requires-python"] == ">=3.12, <3.15"
    resolved_versions = {package["name"]: package["version"] for package in lock["package"] if "version" in package}
    assert resolved_versions["jax"] == "0.11.0"
    assert resolved_versions["jaxlib"] == "0.11.0"
