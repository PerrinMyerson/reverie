"""Run every ```python block in docs/python.md, so the guide stays true."""

from __future__ import annotations

import re

import pytest

from conftest import REPO_ROOT

GUIDE = REPO_ROOT / "docs" / "python.md"
BLOCKS = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.S)


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_guide_example(index, monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)
    exec(compile(BLOCKS[index], f"{GUIDE}:block{index}", "exec"), {"__name__": "__docs__"})


def test_guide_has_examples():
    assert len(BLOCKS) >= 8
