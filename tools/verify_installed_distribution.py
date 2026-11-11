"""Exercise an installed distribution without importing the source checkout."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import numpy_vector_store
from numpy_vector_store import VectorStore


class DistributionVerificationError(RuntimeError):
    """Raised when an installed distribution fails its smoke test."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DistributionVerificationError(message)


def verify_installed_distribution() -> None:
    """Verify package metadata, search behavior, and archive persistence."""
    package_file = numpy_vector_store.__file__
    _require(package_file is not None, "numpy_vector_store has no module file")

    installed_path = Path(package_file).resolve()
    source_path = (Path.cwd() / "src").resolve()
    _require(
        source_path not in installed_path.parents,
        f"Imported the source checkout instead of the distribution: {installed_path}",
    )

    installed_version = version("numpy-vector-store")
    _require(
        numpy_vector_store.__version__ == installed_version,
        "Package metadata and numpy_vector_store.__version__ do not agree",
    )

    store = VectorStore[dict[str, str]](dimensions=2)
    store.add([[1.0, 0.0], [0.0, 1.0]], [{"id": "x"}, {"id": "y"}])
    hits = store.cosine_search([1.0, 0.0], top_k=1)
    _require(len(hits) == 1, "Expected exactly one search result")
    _require(hits[0].index == 0, "Search returned the wrong row")
    _require(hits[0].metadata == {"id": "x"}, "Search returned wrong metadata")
    _require(bool(np.isclose(hits[0].value, 1.0)), "Search returned the wrong score")

    with TemporaryDirectory() as temporary_directory:
        archive = Path(temporary_directory) / "store.npz"
        store.save(archive)
        reopened = VectorStore[dict[str, str]].open(archive)

    _require(reopened.dimensions == 2, "Reopened store has the wrong dimensions")
    _require(len(reopened) == 2, "Reopened store has the wrong row count")
    reopened_row = reopened.get(1)
    _require(
        reopened_row is not None and reopened_row[1] == {"id": "y"},
        "Reopened store did not preserve metadata",
    )

    print(f"Installed distribution verified: {installed_version} from {installed_path}")


if __name__ == "__main__":
    verify_installed_distribution()
