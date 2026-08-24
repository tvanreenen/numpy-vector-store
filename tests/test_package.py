"""Tests for the package-level public API."""

from dataclasses import FrozenInstanceError

import pytest

import numpy_vector_store


def test_package_public_api_is_explicit():
    """Test the supported top-level names are declared in __all__."""
    assert set(numpy_vector_store.__all__) == {
        "VectorHit",
        "VectorStore",
        "__version__",
    }
    assert all(hasattr(numpy_vector_store, name) for name in numpy_vector_store.__all__)
    assert isinstance(numpy_vector_store.__version__, str)


def test_vector_hit_fields_are_read_only():
    """Test search-result fields cannot be reassigned."""
    hit = numpy_vector_store.VectorHit(index=1, value=0.5, metadata={"id": 1})

    with pytest.raises(FrozenInstanceError):
        hit.index = 2  # type: ignore[misc]


def test_vector_hit_hashability_follows_metadata():
    """Test opaque unhashable metadata keeps its normal behavior."""
    hit = numpy_vector_store.VectorHit(index=1, value=0.5, metadata={"id": 1})

    with pytest.raises(TypeError, match="unhashable"):
        hash(hit)
