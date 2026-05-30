"""Packaging sanity checks.

These tests guard against the kinds of drift that quietly break installs
or PyPI metadata:

* ``jno.__version__`` matches the version declared in ``pyproject.toml``
  (single source of truth via importlib.metadata).
* ``CITATION.cff`` advertises the same version, so the GitHub "Cite this
  repository" button doesn't reference an outdated release.
* Repository URLs in ``pyproject.toml`` are non-empty and point at the
  expected GitHub / GitHub-Pages locations.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CITATION = REPO_ROOT / "CITATION.cff"


def _read_pyproject() -> dict:
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover — guard for older interpreters
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(PYPROJECT.read_text())


def _pyproject_version() -> str:
    return _read_pyproject()["project"]["version"]


def _citation_version() -> str:
    text = CITATION.read_text()
    m = re.search(r'^version:\s*"?([^"\s]+)"?', text, flags=re.MULTILINE)
    assert m is not None, "CITATION.cff must declare a version: field"
    return m.group(1)


def test_jno_version_matches_pyproject():
    """jno.__version__ is derived from installed-package metadata."""
    import jno

    assert jno.__version__ == _pyproject_version(), (
        f"jno.__version__ ({jno.__version__!r}) does not match pyproject.toml "
        f"version ({_pyproject_version()!r}). If you changed the version, make "
        "sure the package is re-installed (pixi install / pip install -e .)."
    )


def test_citation_version_matches_pyproject():
    """CITATION.cff version is kept in sync with the release number."""
    assert _citation_version() == _pyproject_version(), (
        f"CITATION.cff version ({_citation_version()!r}) does not match pyproject.toml version ({_pyproject_version()!r})."
    )


def test_project_urls_are_sensible():
    """PyPI metadata URLs should point at the actual GitHub project."""
    urls = _read_pyproject()["project"]["urls"]

    homepage = urls.get("Homepage", "")
    docs = urls.get("Documentation", "")
    source = urls.get("Source", "")

    assert "github.com/FhG-IISB" in homepage, homepage
    assert "github.com/FhG-IISB" in source, source
    # Docs are deployed to https://fhg-iisb.github.io/jNO/ by docs-pages.yml.
    assert "fhg-iisb.github.io" in docs, docs
    assert "jNO_docs" not in docs, f"Documentation URL still points at the legacy `jNO_docs` site: {docs}"


@pytest.mark.parametrize(
    "classifier",
    [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
)
def test_pypi_classifiers_present(classifier):
    classifiers = _read_pyproject()["project"]["classifiers"]
    assert classifier in classifiers, f"Expected PyPI classifier {classifier!r} not found in pyproject.toml"
