"""Pytest fixtures shared across the package's test suite.

Conventions:
- Pure-Python fixtures live here.
- File-backed fixtures (synthetic HDF5 banks, mini CSVs, etc.) go in
  `tests/fixtures/` and are constructed via `tmp_path` factories below.
- No real data files in the repo.

Delete this docstring and the placeholder fixture below once the package
has real tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def example_fixture() -> dict[str, int]:
    """Placeholder fixture — delete once real tests land."""
    return {"answer": 42}
