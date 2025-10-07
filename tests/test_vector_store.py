"""Tests for the VectorStore class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from numpy_vector_store import VectorStore


def add_single_vector(store, vector, metadata=None):
    """Helper function to add a single vector using the batch method."""
    vector_2d = np.atleast_2d(vector)
    metadata_array = np.array([metadata or {}])
    store.add_vectors(vector_2d, metadata_array)
    return len(store.vectors) - 1  # Return the index of the added vector


class TestVectorStore:
    """Test cases for VectorStore."""

    def test_init(self):
        """Test VectorStore initialization."""
        store = VectorStore(dimensions=3)
        assert store.dimensions == 3
        assert len(store.vectors) == 0

    def test_add_single_vector(self):
        """Test adding a single vector using the batch method."""
        store = VectorStore(dimensions=3)
        vector_2d = np.atleast_2d(np.array([1.0, 2.0, 3.0]))
        metadata_array = np.array([{"id": "test"}])

        store.add_vectors(vector_2d, metadata_array)
        assert len(store.vectors) == 1

    def test_add_vectors_wrong_dimension(self):
        """Test adding vectors with wrong dimension raises error."""
        store = VectorStore(dimensions=3)
        vectors_2d = np.array([[1.0, 2.0]])  # Wrong dimension (2 instead of 3)
        metadata_array = np.array([{"id": "test"}])

        with pytest.raises(ValueError):
            store.add_vectors(vectors_2d, metadata_array)

    def test_search(self):
        """Test vector search functionality."""
        store = VectorStore(dimensions=3)

        # Add some test vectors
        vectors_2d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        metadata_array = np.array(
            [{"name": "x-axis"}, {"name": "y-axis"}, {"name": "z-axis"}]
        )
        store.add_vectors(vectors_2d, metadata_array)

        # Search for vector similar to x-axis
        query = np.array([0.9, 0.1, 0.0])
        results = store.search(query, top_k=2)

        assert len(results) == 2
        assert results[0][0] == 0  # x-axis should be first
        assert results[0][1] > 0.9  # High similarity

    def test_search_with_score_cutoff(self):
        """Test search with similarity cutoff."""
        store = VectorStore(dimensions=3)

        # Add some test vectors
        add_single_vector(store, np.array([1.0, 0.0, 0.0]), {"name": "x-axis"})
        add_single_vector(store, np.array([0.0, 1.0, 0.0]), {"name": "y-axis"})
        add_single_vector(store, np.array([0.0, 0.0, 1.0]), {"name": "z-axis"})

        # Search with high cutoff (should return fewer results)
        query = np.array([0.9, 0.1, 0.0])
        results = store.search(query, top_k=3, score_cutoff=0.8)

        # Should only return x-axis (high similarity)
        assert len(results) == 1
        assert results[0][0] == 0  # x-axis
        assert results[0][1] > 0.8  # High similarity

    def test_get(self):
        """Test retrieving vector and metadata by index."""
        store = VectorStore(dimensions=2)
        vector = np.array([1.0, 2.0])
        metadata = {"test": "data"}
        add_single_vector(store, vector, metadata)

        # Test successful retrieval
        entry = store.get(0)
        assert entry is not None
        retrieved_vector, retrieved_metadata = entry
        np.testing.assert_array_almost_equal(
            retrieved_vector, vector / np.linalg.norm(vector)
        )
        assert retrieved_metadata == metadata

        # Test out of bounds
        assert store.get(1) is None

    def test_clear(self):
        """Test clearing the store."""
        store = VectorStore(dimensions=2)
        add_single_vector(store, np.array([1.0, 2.0]))
        assert len(store.vectors) == 1

        store.clear()
        assert len(store.vectors) == 0

    def test_add_vectors(self):
        """Test adding multiple vectors in batch."""
        store = VectorStore(dimensions=2)

        # Prepare vectors as 2D NumPy array (most efficient)
        vectors_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Prepare metadata as NumPy array
        metadata_array = np.array([{"id": 1}, {"id": 2}, {"id": 3}])

        store.add_vectors(vectors_2d, metadata_array)
        assert len(store.vectors) == 3
        assert store.get(0)[1] == {"id": 1}

    def test_add_multiple_vectors_wrong_dimension(self):
        """Test adding multiple vectors with wrong dimension raises error."""
        store = VectorStore(dimensions=3)

        # Wrong dimension vectors (all have 2 dimensions instead of 3)
        vectors_2d = np.array(
            [
                [1.0, 2.0],  # Wrong dimension (2 instead of 3)
                [3.0, 4.0],  # Wrong dimension (2 instead of 3)
            ]
        )

        metadata_array = np.array([{"id": 1}, {"id": 2}])

        with pytest.raises(ValueError):
            store.add_vectors(vectors_2d, metadata_array)

    def test_save_and_load(self):
        """Test saving and loading vectors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            # Create store with file path
            store1 = VectorStore(dimensions=2, file_path=file_path)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "test"})
            store1.save()

            # Create new store and explicitly load
            store2 = VectorStore(dimensions=2, file_path=file_path)
            store2.load()

            assert len(store2.vectors) == 1
            assert store2.get(0)[1] == {"id": "test"}

        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_explicit_load(self):
        """Test explicit loading when file exists."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            # Create and save vectors
            store1 = VectorStore(dimensions=2, file_path=file_path)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "explicit_test"})
            store1.save()

            # Create new store and explicitly load
            store2 = VectorStore(dimensions=2, file_path=file_path)
            store2.load()

            assert len(store2.vectors) == 1
            assert store2.get(0)[1] == {"id": "explicit_test"}

        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_context_manager(self):
        """Test context manager auto-save functionality."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            # Test context manager with file path
            with VectorStore(dimensions=2, file_path=file_path) as store:
                add_single_vector(store, np.array([1.0, 2.0]), {"id": "context_test"})
                # Should auto-save on exit

            # Verify the file was saved
            assert Path(file_path).exists()

            # Load and verify content (explicit load)
            store2 = VectorStore(dimensions=2, file_path=file_path)
            store2.load()
            assert len(store2.vectors) == 1
            assert store2.get(0)[1] == {"id": "context_test"}

        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_context_manager_no_file(self):
        """Test context manager without file path (no auto-save)."""
        with VectorStore(dimensions=2) as store:
            add_single_vector(store, np.array([1.0, 2.0]), {"id": "no_file_test"})
            assert len(store.vectors) == 1
        # Should not crash, just no auto-save

    def test_structured_metadata_schema(self):
        """Test VectorStore with structured metadata schema."""
        schema = {"id": "U10", "score": "f4", "active": "?"}
        store = VectorStore(dimensions=2, metadata_schema=schema)

        assert store._use_structured is True
        assert store._metadata_schema == schema
        assert store.metadata.dtype.names == ("id", "score", "active")

    def test_structured_metadata_add_vectors(self):
        """Test adding vectors with structured metadata."""
        schema = {"id": "U10", "score": "f4"}
        store = VectorStore(dimensions=2, metadata_schema=schema)

        vectors_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        metadata_array = np.array(
            [("test1", 0.8), ("test2", 0.9)], dtype=[("id", "U10"), ("score", "f4")]
        )

        store.add_vectors(vectors_2d, metadata_array)
        assert len(store.vectors) == 2
        assert store.metadata.dtype.names == ("id", "score")

    def test_structured_metadata_get(self):
        """Test getting vectors with structured metadata."""
        schema = {"id": "U10", "score": "f4"}
        store = VectorStore(dimensions=2, metadata_schema=schema)

        vectors_2d = np.array([[1.0, 2.0]])
        metadata_array = np.array(
            [("test1", 0.8)], dtype=[("id", "U10"), ("score", "f4")]
        )
        store.add_vectors(vectors_2d, metadata_array)

        vector, metadata = store.get(0)
        assert metadata == {"id": "test1", "score": 0.8}

    def test_structured_metadata_clear(self):
        """Test clearing store with structured metadata."""
        schema = {"id": "U10", "score": "f4"}
        store = VectorStore(dimensions=2, metadata_schema=schema)

        vectors_2d = np.array([[1.0, 2.0]])
        metadata_array = np.array(
            [("test1", 0.8)], dtype=[("id", "U10"), ("score", "f4")]
        )
        store.add_vectors(vectors_2d, metadata_array)

        store.clear()
        assert len(store.vectors) == 0
        assert store.metadata.dtype.names == ("id", "score")

    def test_search_wrong_dimensions(self):
        """Test search with wrong query dimensions."""
        store = VectorStore(dimensions=3)
        query = np.array([1.0, 2.0])  # Wrong dimension (2 instead of 3)

        with pytest.raises(ValueError, match="Query vector dimension"):
            store.search(query)

    def test_search_empty_store(self):
        """Test search on empty store."""
        store = VectorStore(dimensions=3)
        query = np.array([1.0, 2.0, 3.0])

        results = store.search(query)
        assert results == []

    def test_load_file_not_exists(self):
        """Test loading when file doesn't exist."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            # Delete the file so it doesn't exist
            Path(file_path).unlink()

            store = VectorStore(dimensions=2, file_path=file_path)
            store.load()  # Should not crash
            assert len(store.vectors) == 0
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_already_loaded(self):
        """Test loading when already loaded."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            # Create and save vectors
            store1 = VectorStore(dimensions=2, file_path=file_path)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "test"})
            store1.save()

            # Load once
            store2 = VectorStore(dimensions=2, file_path=file_path)
            store2.load()
            assert len(store2.vectors) == 1

            # Load again (should not reload)
            store2.load()
            assert len(store2.vectors) == 1

        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_save_no_file_path(self):
        """Test save with no file path."""
        store = VectorStore(dimensions=2)
        add_single_vector(store, np.array([1.0, 2.0]), {"id": "test"})

        # Should not crash
        store.save()

    def test_save_empty_vectors(self):
        """Test save with empty vectors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store = VectorStore(dimensions=2, file_path=file_path)
            store.save()  # Should not crash with empty vectors
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_add_vectors_mismatched_lengths(self):
        """Test adding vectors with mismatched vector and metadata lengths."""
        store = VectorStore(dimensions=2)
        vectors_2d = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 vectors
        metadata_array = np.array([{"id": 1}])  # 1 metadata item

        with pytest.raises(ValueError, match="Number of vectors must match"):
            store.add_vectors(vectors_2d, metadata_array)

    def test_structured_metadata_complex_types(self):
        """Test structured metadata with complex field types."""
        schema = {"id": "U10", "data": "O"}  # Object type for complex data
        store = VectorStore(dimensions=2, metadata_schema=schema)

        vectors_2d = np.array([[1.0, 2.0]])
        metadata_array = np.array(
            [("test1", {"nested": "data"})], dtype=[("id", "U10"), ("data", "O")]
        )

        store.add_vectors(vectors_2d, metadata_array)
        assert len(store.vectors) == 1

    def test_search_no_valid_results(self):
        """Test search when no results meet the score cutoff."""
        store = VectorStore(dimensions=3)

        # Add vectors
        vectors_2d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        metadata_array = np.array([{"name": "x-axis"}, {"name": "y-axis"}])
        store.add_vectors(vectors_2d, metadata_array)

        # Search with very high cutoff (no results should match)
        query = np.array([0.1, 0.1, 0.1])
        results = store.search(query, score_cutoff=0.9)

        assert results == []

    def test_structured_metadata_non_string_type(self):
        """Test structured metadata with non-string field type."""
        schema = {"id": "U10", "data": dict}  # Non-string type
        store = VectorStore(dimensions=2, metadata_schema=schema)

        vectors_2d = np.array([[1.0, 2.0]])
        metadata_array = np.array(
            [("test1", {"nested": "data"})], dtype=[("id", "U10"), ("data", "O")]
        )

        store.add_vectors(vectors_2d, metadata_array)
        assert len(store.vectors) == 1
