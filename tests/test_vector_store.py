"""Tests for the VectorStore class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from numpy_vector_store import VectorHit, VectorStore


def add_single_vector(store, vector, metadata=None):
    """Helper function to add a single vector."""
    store.add(np.atleast_2d(vector), [metadata or {}])
    return len(store) - 1


class TestVectorStore:
    """Test cases for VectorStore."""

    def test_init(self):
        """Test VectorStore initialization."""
        store = VectorStore(dimensions=3)
        assert store.dimensions == 3
        assert store.normalize is True
        assert len(store.vectors) == 0
        assert len(store) == 0

    def test_init_allows_raw_vector_storage(self):
        """Test VectorStore can preserve raw vectors."""
        store = VectorStore(dimensions=3, normalize=False)

        assert store.normalize is False

    def test_init_rejects_non_positive_dimensions(self):
        """Test VectorStore rejects invalid dimensions."""
        with pytest.raises(ValueError, match="dimensions"):
            VectorStore(dimensions=0)

    def test_add_preferred_api(self):
        """Test adding vectors with the preferred add API."""
        store = VectorStore[dict[str, int]](dimensions=2)

        store.add([[1.0, 0.0], [0.0, 1.0]], [{"id": 1}, {"id": 2}])

        assert len(store) == 2
        assert store.get(1)[1] == {"id": 2}

    def test_add_normalizes_vectors(self):
        """Test add normalizes vectors at insert time."""
        store = VectorStore[dict[str, str]](dimensions=2)

        store.add([[3.0, 4.0]], [{"id": "v"}])

        np.testing.assert_array_almost_equal(store.vectors[0], np.array([0.6, 0.8]))

    def test_add_preserves_raw_vectors_when_normalize_false(self):
        """Test add preserves original vectors when normalize=False."""
        store = VectorStore[dict[str, str]](dimensions=2, normalize=False)

        store.add([[3.0, 4.0]], [{"id": "v"}])

        np.testing.assert_array_equal(store.vectors[0], np.array([3.0, 4.0]))
        np.testing.assert_array_equal(store.get(0)[0], np.array([3.0, 4.0]))

    def test_add_rejects_wrong_dimension(self):
        """Test add rejects vectors with wrong dimensions."""
        store = VectorStore(dimensions=3)

        with pytest.raises(ValueError, match="Vector dimensions"):
            store.add([[1.0, 2.0]], [{"id": "test"}])

    def test_add_requires_2d_vectors(self):
        """Test add rejects non-2D vector arrays."""
        store = VectorStore(dimensions=3)

        with pytest.raises(ValueError, match="2D"):
            store.add([1.0, 2.0, 3.0], [{"id": "test"}])

    def test_add_requires_1d_metadata(self):
        """Test add rejects non-1D metadata arrays."""
        store = VectorStore(dimensions=3)

        with pytest.raises(ValueError, match="1D"):
            store.add([[1.0, 2.0, 3.0]], np.array([[{"id": "test"}]]))

    def test_add_rejects_mismatched_lengths(self):
        """Test add rejects mismatched vector and metadata counts."""
        store = VectorStore(dimensions=2)

        with pytest.raises(ValueError, match="Number of vectors"):
            store.add([[1.0, 2.0], [3.0, 4.0]], [{"id": 1}])

    def test_add_rejects_zero_norm_vector(self):
        """Test add rejects zero-norm vectors."""
        store = VectorStore(dimensions=3)

        with pytest.raises(ValueError, match="zero-norm"):
            store.add([[0.0, 0.0, 0.0]], [{"id": "zero"}])

    def test_add_rejects_zero_norm_vector_when_normalize_false(self):
        """Test add rejects zero vectors because cosine search is always available."""
        store = VectorStore(dimensions=3, normalize=False)

        with pytest.raises(ValueError, match="zero-norm"):
            store.add([[0.0, 0.0, 0.0]], [{"id": "zero"}])

    def test_cosine_search_returns_vector_hits(self):
        """Test preferred cosine_search returns typed vector hits."""
        store = VectorStore[dict[str, str]](dimensions=3)
        store.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [{"name": "x-axis"}, {"name": "y-axis"}],
        )

        results = store.cosine_search([0.9, 0.1, 0.0], top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], VectorHit)
        assert results[0].index == 0
        assert results[0].metadata == {"name": "x-axis"}
        assert results[0].value == pytest.approx(0.9938837)

    def test_cosine_search_with_min_value(self):
        """Test preferred cosine_search min_value filtering."""
        store = VectorStore[dict[str, str]](dimensions=3)
        store.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [{"name": "x-axis"}, {"name": "y-axis"}, {"name": "z-axis"}],
        )

        results = store.cosine_search([0.9, 0.1, 0.0], top_k=3, min_value=0.8)

        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].metadata == {"name": "x-axis"}

    def test_cosine_search_with_python_prefiltered_rows(self):
        """Test within_rows accepts indexes built from Python metadata filtering."""
        store = VectorStore[dict[str, object]](dimensions=3)
        store.add(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
            [
                {"product": "A", "title": "intro"},
                {"product": "B", "title": "setup"},
                {"product": "A", "title": "guide"},
            ],
        )
        rows = [
            i for i, metadata in enumerate(store.metadata) if metadata["product"] == "A"
        ]

        results = store.cosine_search([1.0, 0.0, 0.0], within_rows=rows)

        assert [hit.index for hit in results] == [0, 2]

    def test_cosine_search_with_numpy_prefiltered_rows(self):
        """Test within_rows accepts indexes produced from NumPy metadata masks."""
        store = VectorStore[int](dimensions=3)
        metadata_table = np.array(
            [("intro", "A", 2024), ("setup", "B", 2024), ("guide", "A", 2023)],
            dtype=[("title", "U20"), ("product", "U10"), ("year", "i4")],
        )
        store.add(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
            np.arange(len(metadata_table)),
        )
        rows = np.flatnonzero(
            (metadata_table["product"] == "A") & (metadata_table["year"] >= 2024)
        )

        results = store.cosine_search([1.0, 0.0, 0.0], within_rows=rows)

        assert len(results) == 1
        assert results[0].index == 0
        assert metadata_table[results[0].metadata]["title"] == "intro"

    def test_cosine_search_empty_within_rows(self):
        """Test within_rows can restrict search to an empty row set."""
        store = VectorStore[dict[str, str]](dimensions=3)
        store.add([[1.0, 0.0, 0.0]], [{"id": "x"}])

        assert store.cosine_search([1.0, 0.0, 0.0], within_rows=[]) == []

    def test_cosine_search_rejects_invalid_within_rows(self):
        """Test within_rows validates shape, dtype, and bounds."""
        store = VectorStore[dict[str, str]](dimensions=3)
        store.add([[1.0, 0.0, 0.0]], [{"id": "x"}])

        with pytest.raises(ValueError, match="1D"):
            store.cosine_search([1.0, 0.0, 0.0], within_rows=[[0]])

        with pytest.raises(ValueError, match="integer"):
            store.cosine_search([1.0, 0.0, 0.0], within_rows=[0.0])

        with pytest.raises(IndexError, match="outside"):
            store.cosine_search([1.0, 0.0, 0.0], within_rows=[1])

    def test_cosine_search_rejects_wrong_query_dimensions(self):
        """Test cosine_search rejects wrong query dimensions."""
        store = VectorStore(dimensions=3)

        with pytest.raises(ValueError, match="Query vector dimension"):
            store.cosine_search([1.0, 2.0])

    def test_cosine_search_rejects_non_positive_top_k(self):
        """Test cosine_search rejects non-positive top_k values."""
        store = VectorStore(dimensions=3)
        store.add([[1.0, 0.0, 0.0]], [{"id": "x"}])

        with pytest.raises(ValueError, match="top_k"):
            store.cosine_search([1.0, 0.0, 0.0], top_k=0)

        with pytest.raises(ValueError, match="top_k"):
            store.cosine_search([1.0, 0.0, 0.0], top_k=-1)

    def test_cosine_search_empty_store(self):
        """Test cosine_search on an empty store."""
        store = VectorStore(dimensions=3)

        assert store.cosine_search([1.0, 2.0, 3.0]) == []

    def test_cosine_search_rejects_zero_norm_query(self):
        """Test cosine_search rejects zero-norm query vectors."""
        store = VectorStore(dimensions=3)
        store.add([[1.0, 0.0, 0.0]], [{"id": "x"}])

        with pytest.raises(ValueError, match="zero-norm"):
            store.cosine_search([0.0, 0.0, 0.0])

    def test_cosine_search_no_valid_results(self):
        """Test cosine_search when no results meet min_value."""
        store = VectorStore(dimensions=3)
        store.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [{"name": "x-axis"}, {"name": "y-axis"}],
        )

        results = store.cosine_search([0.1, 0.1, 0.1], min_value=0.9)

        assert results == []

    def test_cosine_search_with_raw_vectors(self):
        """Test cosine_search computes true cosine when normalize=False."""
        store = VectorStore[dict[str, str]](dimensions=2, normalize=False)
        store.add(
            [[10.0, 0.0], [0.0, 2.0], [3.0, 4.0]],
            [{"id": "x"}, {"id": "y"}, {"id": "diag"}],
        )

        results = store.cosine_search([1.0, 0.0], top_k=3)

        assert [hit.index for hit in results] == [0, 2, 1]
        assert [hit.value for hit in results] == pytest.approx([1.0, 0.6, 0.0])

    def test_dot_search_with_raw_vectors(self):
        """Test dot_search ranks by true dot product when normalize=False."""
        store = VectorStore[dict[str, str]](dimensions=2, normalize=False)
        store.add(
            [[10.0, 0.0], [0.0, 2.0], [3.0, 4.0]],
            [{"id": "x"}, {"id": "y"}, {"id": "diag"}],
        )

        results = store.dot_search([1.0, 0.0], top_k=3)

        assert all(isinstance(hit, VectorHit) for hit in results)
        assert [hit.index for hit in results] == [0, 2, 1]
        assert [hit.value for hit in results] == pytest.approx([10.0, 3.0, 0.0])

    def test_dot_search_with_normalized_vectors(self):
        """Test dot_search uses stored unit vectors when normalize=True."""
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add(
            [[10.0, 0.0], [3.0, 4.0], [0.0, 2.0]],
            [{"id": "x"}, {"id": "diag"}, {"id": "y"}],
        )

        results = store.dot_search([2.0, 0.0], top_k=3, min_value=0.5)

        assert [hit.index for hit in results] == [0, 1]
        assert [hit.value for hit in results] == pytest.approx([1.0, 0.6])

    def test_euclidean_search_with_raw_vectors(self):
        """Test euclidean_search ranks by nearest raw vector when normalize=False."""
        store = VectorStore[dict[str, str]](dimensions=2, normalize=False)
        store.add(
            [[1.0, 1.0], [4.0, 5.0], [2.0, 1.0]],
            [{"id": "near"}, {"id": "far"}, {"id": "also-near"}],
        )

        results = store.euclidean_search([1.0, 2.0], top_k=3)

        assert all(isinstance(hit, VectorHit) for hit in results)
        assert [hit.index for hit in results] == [0, 2, 1]
        assert [hit.value for hit in results] == pytest.approx(
            [1.0, np.sqrt(2.0), np.sqrt(18.0)]
        )

    def test_euclidean_search_with_max_value(self):
        """Test euclidean_search max_value filters by distance."""
        store = VectorStore[dict[str, str]](dimensions=2, normalize=False)
        store.add(
            [[1.0, 1.0], [4.0, 5.0], [2.0, 1.0]],
            [{"id": "near"}, {"id": "far"}, {"id": "also-near"}],
        )

        results = store.euclidean_search([1.0, 2.0], top_k=3, max_value=1.5)

        assert [hit.index for hit in results] == [0, 2]

    def test_euclidean_search_with_normalized_vectors(self):
        """Test euclidean_search uses normalized stored vectors by default."""
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add(
            [[10.0, 0.0], [3.0, 4.0]],
            [{"id": "x"}, {"id": "diag"}],
        )

        results = store.euclidean_search([2.0, 0.0], top_k=2)

        assert [hit.index for hit in results] == [0, 1]
        assert [hit.value for hit in results] == pytest.approx([0.0, np.sqrt(0.8)])

    def test_metric_searches_support_within_rows(self):
        """Test dot and Euclidean search support prefiltered row indexes."""
        store = VectorStore[dict[str, str]](dimensions=2, normalize=False)
        store.add(
            [[10.0, 0.0], [0.0, 2.0], [3.0, 4.0]],
            [{"id": "x"}, {"id": "y"}, {"id": "diag"}],
        )

        dot_results = store.dot_search([1.0, 0.0], within_rows=[1, 2])
        euclidean_results = store.euclidean_search([1.0, 0.0], within_rows=[1, 2])

        assert [hit.index for hit in dot_results] == [2, 1]
        assert [hit.index for hit in euclidean_results] == [1, 2]

    def test_metric_searches_validate_common_inputs(self):
        """Test added search methods share common validation behavior."""
        store = VectorStore(dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "x"}])

        with pytest.raises(ValueError, match="top_k"):
            store.dot_search([1.0, 0.0], top_k=0)

        with pytest.raises(ValueError, match="Query vector dimension"):
            store.euclidean_search([1.0])

        assert store.dot_search([1.0, 0.0], within_rows=[]) == []
        with pytest.raises(IndexError, match="outside"):
            store.euclidean_search([1.0, 0.0], within_rows=[1])

    def test_get(self):
        """Test retrieving vector and metadata by index."""
        store = VectorStore(dimensions=2)
        vector = np.array([1.0, 2.0])
        metadata = {"test": "data"}
        add_single_vector(store, vector, metadata)

        entry = store.get(0)

        assert entry is not None
        retrieved_vector, retrieved_metadata = entry
        np.testing.assert_array_almost_equal(
            retrieved_vector, vector / np.linalg.norm(vector)
        )
        assert retrieved_metadata == metadata
        assert store.get(1) is None

    def test_clear(self):
        """Test clearing the store."""
        store = VectorStore(dimensions=2)
        add_single_vector(store, np.array([1.0, 2.0]))

        store.clear()

        assert len(store.vectors) == 0
        assert len(store.metadata) == 0

    def test_save_and_load(self):
        """Test saving and loading vectors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store1 = VectorStore(dimensions=2, file_path=file_path)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "test"})
            store1.save()

            store2 = VectorStore(dimensions=2, file_path=file_path)
            store2.load()

            assert len(store2.vectors) == 1
            assert store2.get(0)[1] == {"id": "test"}
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_save_writes_minimal_persistence_contract(self):
        """Test saves contain only vectors and metadata arrays."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store = VectorStore[dict[str, str]](dimensions=2, file_path=file_path)
            store.add([[1.0, 2.0]], [{"id": "test"}])
            store.save()

            with np.load(file_path, allow_pickle=True) as data:
                assert set(data.files) == {"vectors", "metadata"}
                assert data["vectors"].dtype == np.float32
                assert data["metadata"][0] == {"id": "test"}
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_supports_minimal_format(self):
        """Test files with vectors and metadata load."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                vectors=np.array([[1.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "legacy"}], dtype=object),
            )

            store = VectorStore[dict[str, str]](dimensions=2, file_path=file_path)
            store.load()

            assert len(store) == 1
            assert store.get(0)[1] == {"id": "legacy"}
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_explicit_load_when_file_exists(self):
        """Test explicit loading when file exists."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store1 = VectorStore(dimensions=2, file_path=file_path)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "explicit_test"})
            store1.save()

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
            with VectorStore(dimensions=2, file_path=file_path) as store:
                add_single_vector(store, np.array([1.0, 2.0]), {"id": "context_test"})

            store2 = VectorStore(dimensions=2, file_path=file_path)
            store2.load()
            assert len(store2.vectors) == 1
            assert store2.get(0)[1] == {"id": "context_test"}
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_context_manager_no_file(self):
        """Test context manager without file path."""
        with VectorStore(dimensions=2) as store:
            add_single_vector(store, np.array([1.0, 2.0]), {"id": "no_file_test"})
            assert len(store.vectors) == 1

    def test_load_file_not_exists(self):
        """Test loading when file doesn't exist."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            Path(file_path).unlink()

            store = VectorStore(dimensions=2, file_path=file_path)
            store.load()

            assert len(store.vectors) == 0
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_already_loaded(self):
        """Test loading when already loaded."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store1 = VectorStore(dimensions=2, file_path=file_path)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "test"})
            store1.save()

            store2 = VectorStore(dimensions=2, file_path=file_path)
            store2.load()
            store2.load()

            assert len(store2.vectors) == 1
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_raises_on_vector_dimension_mismatch(self):
        """Test load fails fast when persisted vector dimensions don't match store."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                vectors=np.array([[1.0, 2.0]], dtype=np.float32),
                metadata=np.array([{"id": "test"}], dtype=object),
            )

            store = VectorStore(dimensions=3, file_path=file_path)
            with pytest.raises(ValueError, match="Loaded vector dimension"):
                store.load()
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_raises_on_metadata_length_mismatch(self):
        """Test load fails fast when vectors/metadata lengths are inconsistent."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                vectors=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                metadata=np.array([{"id": "only-one"}], dtype=object),
            )

            store = VectorStore(dimensions=2, file_path=file_path)
            with pytest.raises(ValueError, match="length mismatch"):
                store.load()
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_raises_on_missing_required_arrays(self):
        """Test load fails fast when required persisted arrays are absent."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(file_path, vectors=np.empty((0, 2), dtype=np.float32))

            store = VectorStore(dimensions=2, file_path=file_path)
            with pytest.raises(ValueError, match="metadata"):
                store.load()
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_normalizes_non_zero_vectors(self):
        """Test load normalizes valid vectors once before storing them."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                vectors=np.array([[2.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "test"}], dtype=object),
            )

            store = VectorStore(dimensions=2, file_path=file_path)
            store.load()

            np.testing.assert_array_almost_equal(store.vectors[0], np.array([1.0, 0.0]))
            assert store.cosine_search([1.0, 0.0])[0].value == pytest.approx(1.0)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_preserves_raw_vectors_when_normalize_false(self):
        """Test load preserves valid vectors when normalize=False."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                vectors=np.array([[2.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "test"}], dtype=object),
            )

            store = VectorStore(dimensions=2, file_path=file_path, normalize=False)
            store.load()

            np.testing.assert_array_equal(store.vectors[0], np.array([2.0, 0.0]))
            assert store.cosine_search([1.0, 0.0])[0].value == pytest.approx(1.0)
            assert store.dot_search([1.0, 0.0])[0].value == pytest.approx(2.0)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_raises_on_zero_norm_vectors(self):
        """Test load rejects persisted zero-norm vectors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                vectors=np.array([[0.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "zero"}], dtype=object),
            )

            store = VectorStore(dimensions=2, file_path=file_path)
            with pytest.raises(ValueError, match="zero-norm"):
                store.load()
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_load_raises_on_zero_norm_vectors_when_normalize_false(self):
        """Test raw stores reject zero rows because cosine search is always available."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                vectors=np.array([[0.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "zero"}], dtype=object),
            )

            store = VectorStore(dimensions=2, file_path=file_path, normalize=False)
            with pytest.raises(ValueError, match="zero-norm"):
                store.load()
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_save_no_file_path(self):
        """Test save with no file path."""
        store = VectorStore(dimensions=2)
        add_single_vector(store, np.array([1.0, 2.0]), {"id": "test"})

        store.save()

    def test_save_empty_vectors(self):
        """Test save with empty vectors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store = VectorStore(dimensions=2, file_path=file_path)
            store.save()
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_clear_then_save_overwrites_persisted_data(self):
        """Test clear() + save() persists an empty store to disk."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store1 = VectorStore(dimensions=2, file_path=file_path)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "test"})
            store1.save()

            store1.clear()
            store1.save()

            store2 = VectorStore(dimensions=2, file_path=file_path)
            store2.load()
            assert len(store2.vectors) == 0
            assert len(store2.metadata) == 0
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_save_and_load_raw_vectors_round_trip(self):
        """Test normalize=False saves and loads raw vectors without mode metadata."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store1 = VectorStore(dimensions=2, file_path=file_path, normalize=False)
            store1.add([[3.0, 4.0]], [{"id": "raw"}])
            store1.save()

            with np.load(file_path, allow_pickle=True) as data:
                assert set(data.files) == {"vectors", "metadata"}
                np.testing.assert_array_equal(data["vectors"][0], np.array([3.0, 4.0]))

            store2 = VectorStore(dimensions=2, file_path=file_path, normalize=False)
            store2.load()

            np.testing.assert_array_equal(store2.get(0)[0], np.array([3.0, 4.0]))
            assert store2.get(0)[1] == {"id": "raw"}
        finally:
            Path(file_path).unlink(missing_ok=True)
