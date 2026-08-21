"""Tests for the VectorStore class."""

import os
import stat
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import numpy_vector_store.vector_store as vector_store_module
from numpy_vector_store import VectorHit, VectorStore


@dataclass
class MetadataRecord:
    """Structured metadata payload used by regression tests."""

    identifier: int
    label: str


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

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("dimensions", 3),
            ("normalize", False),
            ("file_path", Path("other.npz")),
        ],
    )
    def test_configuration_properties_are_read_only(self, name, value):
        """Test configuration and archive binding cannot be assigned directly."""
        store = VectorStore(dimensions=2)

        with pytest.raises(AttributeError, match=name):
            setattr(store, name, value)

    def test_removed_persistence_entry_points_are_not_supported(self, tmp_path):
        """Test 0.4 constructor binding and instance loading are gone."""
        with pytest.raises(TypeError, match="file_path"):
            VectorStore(dimensions=2, file_path=tmp_path / "vectors.npz")  # type: ignore[call-arg]

        store = VectorStore(dimensions=2)
        with pytest.raises(AttributeError, match="load"):
            store.load()  # type: ignore[attr-defined]

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

    @pytest.mark.parametrize(
        "magnitude",
        [
            np.float32(1e20),
            np.nextafter(np.float32(0), np.float32(1)),
        ],
    )
    def test_add_normalizes_extreme_finite_vectors(self, magnitude):
        """Test normalization avoids float32 overflow and underflow."""
        store = VectorStore[dict[str, str]](dimensions=2)

        store.add([[magnitude, magnitude]], [{"id": "extreme"}])

        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        np.testing.assert_allclose(store.vectors[0], expected, rtol=1e-6)

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

    @pytest.mark.parametrize(
        "payload",
        [
            ("tuple", 1),
            ["list", 2],
            MetadataRecord(identifier=3, label="dataclass"),
            "scalar",
            {"id": 5},
        ],
    )
    def test_add_preserves_opaque_metadata_payload(self, payload):
        """Test structured payloads remain individual metadata rows."""
        store = VectorStore(dimensions=2)

        store.add([[1.0, 0.0]], [payload])

        assert store.metadata.shape == (1,)
        assert store.get(0)[1] == payload

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

    def test_add_allows_zero_norm_vector_when_normalize_false(self):
        """Test raw stores preserve zero vectors for dot and Euclidean search."""
        store = VectorStore(dimensions=3, normalize=False)

        store.add([[0.0, 0.0, 0.0]], [{"id": "zero"}])

        np.testing.assert_array_equal(store.get(0)[0], np.zeros(3))
        assert store.dot_search([1.0, 0.0, 0.0])[0].value == pytest.approx(0.0)
        assert store.euclidean_search([3.0, 4.0, 0.0])[0].value == pytest.approx(5.0)

    @pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
    def test_add_rejects_non_finite_vectors(self, non_finite):
        """Test add rejects non-finite values without changing the store."""
        store = VectorStore(dimensions=2)

        with pytest.raises(ValueError, match="non-finite"):
            store.add([[non_finite, 1.0]], [{"id": "invalid"}])

        assert len(store) == 0
        assert len(store.metadata) == 0

    def test_add_rejects_values_outside_float32_without_warning(self):
        """Test out-of-range values produce only the validation error."""
        store = VectorStore(dimensions=2)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ValueError, match="non-finite"):
                store.add([[1e100, 1.0]], [{"id": "out-of-range"}])

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

    @pytest.mark.parametrize(
        "magnitude",
        [
            np.float32(1e20),
            np.nextafter(np.float32(0), np.float32(1)),
        ],
    )
    def test_cosine_search_normalizes_extreme_finite_queries(self, magnitude):
        """Test query normalization avoids float32 overflow and underflow."""
        store = VectorStore(dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "x"}])

        result = store.cosine_search([magnitude, magnitude])[0]

        assert result.value == pytest.approx(1 / np.sqrt(2), rel=1e-6)

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

    def test_cosine_search_with_large_raw_vectors(self):
        """Test raw cosine norms do not overflow float32."""
        store = VectorStore[str](dimensions=2, normalize=False)
        store.add([[1e20, 1e20]], ["large"])

        result = store.cosine_search([1e20, 1e20])[0]

        assert result.value == pytest.approx(1.0)

    def test_cosine_search_rejects_selected_zero_norm_stored_vectors(self):
        """Test raw cosine search rejects selected zero vectors."""
        store = VectorStore[str](dimensions=2, normalize=False)
        store.add([[0.0, 0.0], [1.0, 0.0]], ["zero", "x"])

        with pytest.raises(ValueError, match="zero-norm stored vectors"):
            store.cosine_search([1.0, 0.0])

        result = store.cosine_search([1.0, 0.0], within_rows=[1])[0]
        assert result.index == 1
        assert result.value == pytest.approx(1.0)

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

    def test_dot_search_with_large_raw_vectors(self):
        """Test raw dot products do not overflow float32."""
        magnitude = np.finfo(np.float32).max
        store = VectorStore[str](dimensions=2, normalize=False)
        store.add([[magnitude, magnitude]], ["large"])

        result = store.dot_search([magnitude, magnitude])[0]

        expected = 2 * float(magnitude) ** 2
        assert np.isfinite(result.value)
        assert result.value == pytest.approx(expected)

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

    def test_euclidean_search_with_large_raw_vectors(self):
        """Test raw Euclidean distances do not overflow float32."""
        magnitude = np.finfo(np.float32).max
        store = VectorStore[str](dimensions=2, normalize=False)
        store.add([[magnitude, magnitude]], ["large"])

        result = store.euclidean_search([-magnitude, -magnitude])[0]

        expected = np.sqrt(8) * float(magnitude)
        assert np.isfinite(result.value)
        assert result.value == pytest.approx(expected)

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

    def test_unfiltered_metric_search_uses_stored_vector_matrix(self):
        """Test an unfiltered search does not copy the complete vector matrix."""
        store = VectorStore[int](dimensions=2, normalize=False)
        store.add([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 1, 2])
        received_vectors = []

        def capture_vectors(query, vectors):
            received_vectors.append(vectors)
            return np.arange(len(vectors), dtype=np.float32)

        store._metric_search(
            [1.0, 0.0],
            top_k=1,
            within_rows=None,
            values_fn=capture_vectors,
            descending=True,
            min_value=None,
            max_value=None,
        )

        assert received_vectors[0] is store.vectors

    @pytest.mark.parametrize(
        "search_name", ["cosine_search", "dot_search", "euclidean_search"]
    )
    @pytest.mark.parametrize("normalize", [True, False])
    def test_randomized_unfiltered_search_matches_full_row_filter(
        self, search_name, normalize
    ):
        """Test optimized searches match a shuffled full-row selection."""
        rng = np.random.default_rng(20260727)
        vectors = rng.normal(size=(64, 8)).astype(np.float32)
        query = rng.normal(size=8).astype(np.float32)
        store = VectorStore[int](dimensions=8, normalize=normalize)
        store.add(vectors, np.arange(len(vectors)))
        shuffled_rows = rng.permutation(len(store))

        search = getattr(store, search_name)
        unfiltered_results = search(query, top_k=16)
        filtered_results = search(query, top_k=16, within_rows=shuffled_rows)

        assert [hit.index for hit in unfiltered_results] == [
            hit.index for hit in filtered_results
        ]
        assert [hit.metadata for hit in filtered_results] == [
            hit.index for hit in filtered_results
        ]
        assert [hit.value for hit in unfiltered_results] == pytest.approx(
            [hit.value for hit in filtered_results]
        )

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

    @pytest.mark.parametrize(
        "search_name", ["cosine_search", "dot_search", "euclidean_search"]
    )
    @pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
    def test_metric_searches_reject_non_finite_queries(self, search_name, non_finite):
        """Test every metric rejects non-finite query values."""
        store = VectorStore(dimensions=2, normalize=False)
        store.add([[1.0, 0.0]], [{"id": "x"}])

        with pytest.raises(ValueError, match="finite"):
            getattr(store, search_name)([non_finite, 0.0])

    def test_search_rejects_query_outside_float32_without_warning(self):
        """Test an out-of-range query produces only the validation error."""
        store = VectorStore(dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "x"}])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ValueError, match="finite"):
                store.cosine_search([1e100, 0.0])

    @pytest.mark.parametrize(
        ("search_name", "threshold_name"),
        [
            ("cosine_search", "min_value"),
            ("dot_search", "min_value"),
            ("euclidean_search", "max_value"),
        ],
    )
    @pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
    def test_metric_searches_reject_non_finite_thresholds(
        self, search_name, threshold_name, non_finite
    ):
        """Test every metric rejects non-finite result thresholds."""
        store = VectorStore(dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "x"}])

        with pytest.raises(ValueError, match=threshold_name):
            getattr(store, search_name)([1.0, 0.0], **{threshold_name: non_finite})

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
            store1 = VectorStore(dimensions=2)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "test"})
            store1.save(file_path)

            store2 = VectorStore.open(file_path)

            assert len(store2.vectors) == 1
            assert store2.get(0)[1] == {"id": "test"}
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_save_and_load_with_extensionless_path(self, tmp_path):
        """Test extensionless paths resolve to the same .npz file."""
        file_path = tmp_path / "vectors"
        expected_path = tmp_path / "vectors.npz"

        store1 = VectorStore(dimensions=2)
        add_single_vector(store1, np.array([1.0, 2.0]), {"id": "test"})
        store1.save(file_path)

        assert store1.file_path == expected_path
        assert expected_path.exists()

        store2 = VectorStore.open(file_path)

        assert len(store2) == 1
        assert store2.get(0)[1] == {"id": "test"}

    def test_save_with_path_binds_an_unbound_store(self, tmp_path):
        """Test the preferred first save writes and binds its destination."""
        extensionless_path = tmp_path / "vectors"
        expected_path = tmp_path / "vectors.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "first"}])

        store.save(path=extensionless_path)

        assert store.file_path == expected_path
        opened = VectorStore[dict[str, str]].open(expected_path)
        assert opened.get(0)[1] == {"id": "first"}

    def test_save_with_path_rebinds_after_successful_save_as(self, tmp_path):
        """Test Save As binds the new path for later pathless saves."""
        original_path = tmp_path / "original.npz"
        copied_path = tmp_path / "copied.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "original"}])
        store.save(original_path)

        store.add([[0.0, 1.0]], [{"id": "copied"}])
        store.save(copied_path)
        store.add([[1.0, 1.0]], [{"id": "later"}])
        store.save()

        assert store.file_path == copied_path
        assert len(VectorStore.open(original_path)) == 1
        assert len(VectorStore.open(copied_path)) == 3

    def test_failed_save_as_preserves_existing_binding(self, tmp_path):
        """Test a failed write cannot redirect later pathless saves."""
        original_path = tmp_path / "original.npz"
        invalid_path = tmp_path / "directory.npz"
        invalid_path.mkdir()
        store = VectorStore(dimensions=2)
        store.save(original_path)

        with pytest.raises(IsADirectoryError):
            store.save(invalid_path)

        assert store.file_path == original_path

    def test_save_replaces_from_closed_same_directory_archive(
        self, tmp_path, monkeypatch
    ):
        """Test save closes a complete local temporary archive before replace."""
        file_path = tmp_path / "vectors.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "first"}])
        store.save(file_path)
        store.add([[0.0, 1.0]], [{"id": "second"}])
        original_savez = np.savez_compressed
        original_replace = os.replace
        captured = {}

        def capture_temporary_file(temporary_file, **values):
            captured["file"] = temporary_file
            original_savez(temporary_file, **values)

        def inspect_then_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            assert captured["file"].closed
            assert source_path.parent == destination_path.parent == tmp_path
            assert source_path != destination_path
            with np.load(source_path, allow_pickle=True) as temporary_data:
                assert len(temporary_data["vectors"]) == 2
            with np.load(destination_path, allow_pickle=True) as current_data:
                assert len(current_data["vectors"]) == 1
            captured["temporary_path"] = source_path
            original_replace(source_path, destination_path)

        monkeypatch.setattr(np, "savez_compressed", capture_temporary_file)
        monkeypatch.setattr(os, "replace", inspect_then_replace)

        store.save()

        assert not captured["temporary_path"].exists()
        assert len(VectorStore.open(file_path)) == 2

    def test_save_write_failure_preserves_destination_and_cleans_temp(
        self, tmp_path, monkeypatch
    ):
        """Test partial temporary writes cannot damage the current archive."""
        file_path = tmp_path / "vectors.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "persisted"}])
        store.save(file_path)
        original_archive = file_path.read_bytes()
        store.add([[0.0, 1.0]], [{"id": "unsaved"}])

        def fail_during_write(temporary_file, **values):
            temporary_file.write(b"partial archive")
            raise OSError("simulated archive write failure")

        monkeypatch.setattr(np, "savez_compressed", fail_during_write)

        with pytest.raises(OSError, match="simulated archive write failure"):
            store.save()

        assert file_path.read_bytes() == original_archive
        assert list(tmp_path.glob(f".{file_path.name}.*.tmp")) == []

    def test_save_replace_failure_preserves_destination_and_cleans_temp(
        self, tmp_path, monkeypatch
    ):
        """Test a failed replace leaves the prior archive and no temporary file."""
        file_path = tmp_path / "vectors.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "persisted"}])
        store.save(file_path)
        original_archive = file_path.read_bytes()
        captured = {}

        def fail_during_replace(source, destination):
            captured["temporary_path"] = Path(source)
            assert captured["temporary_path"].exists()
            raise PermissionError("simulated replace failure")

        monkeypatch.setattr(os, "replace", fail_during_replace)

        with pytest.raises(PermissionError, match="simulated replace failure"):
            store.save()

        assert file_path.read_bytes() == original_archive
        assert not captured["temporary_path"].exists()

    def test_temporary_name_collision_does_not_delete_existing_file(
        self, tmp_path, monkeypatch
    ):
        """Test exclusive temporary creation never cleans up another file."""
        file_path = tmp_path / "vectors.npz"
        collision_path = tmp_path / f".{file_path.name}.collision.tmp"
        collision_path.write_text("belongs to another process")
        collision_id = type("CollisionId", (), {"hex": "collision"})()
        monkeypatch.setattr(vector_store_module, "uuid4", lambda: collision_id)
        store = VectorStore(dimensions=2)

        with pytest.raises(FileExistsError):
            store.save(file_path)

        assert collision_path.read_text() == "belongs to another process"
        assert not file_path.exists()

    def test_atomic_save_preserves_existing_file_permissions(self, tmp_path):
        """Test replacement does not broaden a destination's permission bits."""
        file_path = tmp_path / "vectors.npz"
        store = VectorStore(dimensions=2)
        store.save(file_path)
        file_path.chmod(0o640)
        store.add([[1.0, 0.0]], [{"id": "updated"}])

        store.save()

        assert stat.S_IMODE(file_path.stat().st_mode) == 0o640

    def test_save_writes_versioned_persistence_contract(self):
        """Test saves contain the complete version 1 archive schema."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store = VectorStore[dict[str, str]](dimensions=2)
            store.add([[1.0, 2.0]], [{"id": "test"}])
            store.save(file_path)

            with np.load(file_path, allow_pickle=True) as data:
                assert set(data.files) == {
                    "format_version",
                    "dimensions",
                    "normalize",
                    "vectors",
                    "metadata",
                }
                assert data["format_version"].shape == ()
                assert data["format_version"].item() == 1
                assert data["dimensions"].shape == ()
                assert data["dimensions"].item() == 2
                assert data["normalize"].shape == ()
                assert data["normalize"].item() is True
                assert data["vectors"].dtype == np.float32
                assert data["metadata"].dtype == np.dtype(object)
                assert data["metadata"][0] == {"id": "test"}
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_opaque_metadata_payloads_round_trip(self, tmp_path):
        """Test each opaque payload remains one row after saving and loading."""
        file_path = tmp_path / "vectors.npz"
        payloads = [
            ("tuple", 1),
            ["list", 2],
            MetadataRecord(identifier=3, label="dataclass"),
            "scalar",
            {"id": 5},
        ]
        store = VectorStore[object](dimensions=2)
        store.add(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
            payloads,
        )
        store.save(file_path)

        loaded = VectorStore[object].open(file_path)

        assert loaded.metadata.shape == (5,)
        assert loaded.metadata.tolist() == payloads

    @pytest.mark.parametrize("normalize", [True, False])
    def test_open_uses_versioned_archive_configuration(self, tmp_path, normalize):
        """Test open constructs and binds a store from persisted configuration."""
        file_path = tmp_path / "vectors.npz"
        persisted = VectorStore[dict[str, str]](
            dimensions=2,
            normalize=normalize,
        )
        persisted.add([[3.0, 4.0]], [{"id": "persisted"}])
        persisted.save(file_path)

        opened = VectorStore[dict[str, str]].open(path=file_path)

        assert opened.dimensions == 2
        assert opened.normalize is normalize
        assert opened.file_path == file_path
        assert len(opened) == 1
        assert opened.get(0)[1] == {"id": "persisted"}
        expected = np.array([0.6, 0.8]) if normalize else np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(opened.get(0)[0], expected)

    def test_open_resolves_extensionless_path(self, tmp_path):
        """Test open uses the same extension resolution as save."""
        extensionless_path = tmp_path / "vectors"
        expected_path = tmp_path / "vectors.npz"
        persisted = VectorStore(dimensions=2)
        persisted.save(extensionless_path)

        opened = VectorStore.open(extensionless_path)

        assert opened.file_path == expected_path

    def test_open_requires_an_existing_archive(self, tmp_path):
        """Test open fails clearly when its resolved archive does not exist."""
        with pytest.raises(FileNotFoundError):
            VectorStore.open(tmp_path / "missing")

    def test_open_rejects_empty_path(self):
        """Test open requires a meaningful archive path."""
        with pytest.raises(ValueError, match="path"):
            VectorStore.open("")

    def test_open_rejects_unversioned_archive(self, tmp_path):
        """Test open requires the self-describing version 1 schema."""
        file_path = tmp_path / "unversioned.npz"
        np.savez_compressed(
            file_path,
            vectors=np.empty((0, 2), dtype=np.float32),
            metadata=np.array([], dtype=object),
        )

        with pytest.raises(ValueError, match="missing fields: format_version"):
            VectorStore.open(file_path)

    @pytest.mark.parametrize(
        ("archive_dimensions", "archive_normalize", "message"),
        [(3, True, "dimensions"), (2, False, "normalize")],
    )
    def test_reload_rejects_versioned_configuration_mismatch(
        self, tmp_path, archive_dimensions, archive_normalize, message
    ):
        """Test archive configuration cannot silently change store semantics."""
        file_path = tmp_path / "vectors.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "in-memory"}])
        store.save(file_path)
        replacement = VectorStore(
            dimensions=archive_dimensions,
            normalize=archive_normalize,
        )
        replacement.save(file_path)

        with pytest.raises(ValueError, match=message):
            store.reload()

        assert len(store) == 1
        assert store.get(0)[1] == {"id": "in-memory"}

    def test_open_rejects_unknown_archive_version(self, tmp_path):
        """Test unknown archive versions fail instead of being guessed at."""
        file_path = tmp_path / "vectors.npz"
        np.savez_compressed(
            file_path,
            format_version=np.array(2, dtype=np.int64),
            dimensions=np.array(2, dtype=np.int64),
            normalize=np.array(True, dtype=np.bool_),
            vectors=np.empty((0, 2), dtype=np.float32),
            metadata=np.array([], dtype=object),
        )
        with pytest.raises(ValueError, match="format version: 2"):
            VectorStore.open(file_path)

    def test_unknown_archive_version_is_checked_before_its_fields(self, tmp_path):
        """Test future-version fields do not obscure an unsupported version."""
        file_path = tmp_path / "vectors.npz"
        np.savez_compressed(
            file_path,
            format_version=np.array(2, dtype=np.int64),
            dimensions=np.array(2, dtype=np.int64),
            normalize=np.array(True, dtype=np.bool_),
            vectors=np.empty((0, 2), dtype=np.float32),
            metadata=np.array([], dtype=object),
            future_configuration=np.array("value"),
        )
        with pytest.raises(ValueError, match="format version: 2"):
            VectorStore.open(file_path)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("format_version", np.array([1], dtype=np.int64), "scalar integer"),
            ("dimensions", np.array(2.0), "scalar integer"),
            ("normalize", np.array(1, dtype=np.int64), "scalar boolean"),
        ],
    )
    def test_open_rejects_malformed_archive_configuration(
        self, tmp_path, field, value, message
    ):
        """Test versioned configuration values use their documented types."""
        file_path = tmp_path / "vectors.npz"
        archive = {
            "format_version": np.array(1, dtype=np.int64),
            "dimensions": np.array(2, dtype=np.int64),
            "normalize": np.array(True, dtype=np.bool_),
            "vectors": np.empty((0, 2), dtype=np.float32),
            "metadata": np.array([], dtype=object),
        }
        archive[field] = value
        np.savez_compressed(file_path, **archive)
        with pytest.raises(ValueError, match=message):
            VectorStore.open(file_path)

    @pytest.mark.parametrize(
        ("vectors", "metadata", "message"),
        [
            (
                np.empty((0, 2), dtype=np.float64),
                np.array([], dtype=object),
                "float32",
            ),
            (
                np.empty((0, 2), dtype=np.float32),
                np.array([], dtype=np.int64),
                "object dtype",
            ),
        ],
    )
    def test_open_rejects_malformed_archive_array_dtypes(
        self, tmp_path, vectors, metadata, message
    ):
        """Test versioned arrays use their documented storage dtypes."""
        file_path = tmp_path / "vectors.npz"
        np.savez_compressed(
            file_path,
            format_version=np.array(1, dtype=np.int64),
            dimensions=np.array(2, dtype=np.int64),
            normalize=np.array(True, dtype=np.bool_),
            vectors=vectors,
            metadata=metadata,
        )
        with pytest.raises(ValueError, match=message):
            VectorStore.open(file_path)

    def test_reload_preserves_state_after_array_validation_failure(self, tmp_path):
        """Test late archive validation cannot partially replace live rows."""
        file_path = tmp_path / "vectors.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[3.0, 4.0]], [{"id": "in-memory"}])
        store.save(file_path)
        original_vectors = store.vectors.copy()
        original_metadata = store.metadata.copy()
        np.savez_compressed(
            file_path,
            format_version=np.array(1, dtype=np.int64),
            dimensions=np.array(2, dtype=np.int64),
            normalize=np.array(True, dtype=np.bool_),
            vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            metadata=np.array([{"id": "only-one"}], dtype=object),
        )

        with pytest.raises(ValueError, match="length mismatch"):
            store.reload()

        np.testing.assert_array_equal(store.vectors, original_vectors)
        np.testing.assert_array_equal(store.metadata, original_metadata)

    def test_context_manager_is_not_supported(self):
        """Test persistence has no implicit context-manager boundary."""
        store = VectorStore(dimensions=2)

        with pytest.raises(TypeError, match="context manager protocol"):
            with store:  # type: ignore[attr-defined]
                pass

    def test_reload_refreshes_an_open_store(self, tmp_path):
        """Test reload always replaces memory from the bound archive."""
        file_path = tmp_path / "vectors.npz"
        persisted = VectorStore[dict[str, str]](dimensions=2)
        persisted.add([[1.0, 0.0]], [{"id": "first"}])
        persisted.save(file_path)
        opened = VectorStore[dict[str, str]].open(file_path)

        persisted.add([[0.0, 1.0]], [{"id": "second"}])
        persisted.save()
        opened.add([[1.0, 1.0]], [{"id": "memory-only"}])
        opened.reload()

        assert len(opened) == 2
        assert opened.metadata.tolist() == [{"id": "first"}, {"id": "second"}]

    def test_reload_requires_a_bound_path(self):
        """Test reload rejects stores that have never been opened or saved."""
        store = VectorStore(dimensions=2)

        with pytest.raises(ValueError, match="bound file path"):
            store.reload()

    def test_reload_missing_file_preserves_in_memory_state(self, tmp_path):
        """Test strict reload raises without discarding rows when a file is gone."""
        file_path = tmp_path / "vectors.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "persisted"}])
        store.save(file_path)
        file_path.unlink()

        with pytest.raises(FileNotFoundError):
            store.reload()

        assert len(store) == 1
        assert store.get(0)[1] == {"id": "persisted"}

    def test_reload_invalid_archive_preserves_in_memory_state(self, tmp_path):
        """Test strict reload validates before changing the current rows."""
        file_path = tmp_path / "vectors.npz"
        store = VectorStore[dict[str, str]](dimensions=2)
        store.add([[1.0, 0.0]], [{"id": "persisted"}])
        store.save(file_path)
        np.savez_compressed(
            file_path,
            format_version=np.array(1, dtype=np.int64),
            dimensions=np.array(2, dtype=np.int64),
            normalize=np.array(True, dtype=np.bool_),
            vectors=np.array([[1.0, 0.0]], dtype=np.float32),
            metadata=np.array([], dtype=object),
        )

        with pytest.raises(ValueError, match="length mismatch"):
            store.reload()

        assert len(store) == 1
        assert store.get(0)[1] == {"id": "persisted"}

    def test_open_raises_on_vector_dimension_mismatch(self):
        """Test open rejects vectors that conflict with archive dimensions."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                format_version=np.array(1, dtype=np.int64),
                dimensions=np.array(3, dtype=np.int64),
                normalize=np.array(True, dtype=np.bool_),
                vectors=np.array([[1.0, 2.0]], dtype=np.float32),
                metadata=np.array([{"id": "test"}], dtype=object),
            )

            with pytest.raises(ValueError, match="Loaded vector dimension"):
                VectorStore.open(file_path)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_open_raises_on_metadata_length_mismatch(self):
        """Test open rejects inconsistent vector and metadata row counts."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                format_version=np.array(1, dtype=np.int64),
                dimensions=np.array(2, dtype=np.int64),
                normalize=np.array(True, dtype=np.bool_),
                vectors=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                metadata=np.array([{"id": "only-one"}], dtype=object),
            )

            with pytest.raises(ValueError, match="length mismatch"):
                VectorStore.open(file_path)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_open_raises_on_missing_required_arrays(self):
        """Test open fails fast when required persisted arrays are absent."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                format_version=np.array(1, dtype=np.int64),
                dimensions=np.array(2, dtype=np.int64),
                normalize=np.array(True, dtype=np.bool_),
                vectors=np.empty((0, 2), dtype=np.float32),
            )

            with pytest.raises(ValueError, match="metadata"):
                VectorStore.open(file_path)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_open_normalizes_non_zero_vectors(self):
        """Test open restores normalized storage semantics."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                format_version=np.array(1, dtype=np.int64),
                dimensions=np.array(2, dtype=np.int64),
                normalize=np.array(True, dtype=np.bool_),
                vectors=np.array([[2.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "test"}], dtype=object),
            )

            store = VectorStore.open(file_path)

            np.testing.assert_array_almost_equal(store.vectors[0], np.array([1.0, 0.0]))
            assert store.cosine_search([1.0, 0.0])[0].value == pytest.approx(1.0)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_open_preserves_raw_vectors_when_normalize_false(self):
        """Test open preserves valid vectors when normalize=False."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                format_version=np.array(1, dtype=np.int64),
                dimensions=np.array(2, dtype=np.int64),
                normalize=np.array(False, dtype=np.bool_),
                vectors=np.array([[2.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "test"}], dtype=object),
            )

            store = VectorStore.open(file_path)

            np.testing.assert_array_equal(store.vectors[0], np.array([2.0, 0.0]))
            assert store.cosine_search([1.0, 0.0])[0].value == pytest.approx(1.0)
            assert store.dot_search([1.0, 0.0])[0].value == pytest.approx(2.0)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_open_raises_on_zero_norm_vectors(self):
        """Test open rejects persisted zero-norm vectors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                format_version=np.array(1, dtype=np.int64),
                dimensions=np.array(2, dtype=np.int64),
                normalize=np.array(True, dtype=np.bool_),
                vectors=np.array([[0.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "zero"}], dtype=object),
            )

            with pytest.raises(ValueError, match="zero-norm"):
                VectorStore.open(file_path)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_open_allows_zero_norm_vectors_when_normalize_false(self):
        """Test raw stores load zero rows for dot and Euclidean search."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                format_version=np.array(1, dtype=np.int64),
                dimensions=np.array(2, dtype=np.int64),
                normalize=np.array(False, dtype=np.bool_),
                vectors=np.array([[0.0, 0.0]], dtype=np.float32),
                metadata=np.array([{"id": "zero"}], dtype=object),
            )

            store = VectorStore.open(file_path)

            np.testing.assert_array_equal(store.get(0)[0], np.zeros(2))
            assert store.dot_search([1.0, 0.0])[0].value == pytest.approx(0.0)
            assert store.euclidean_search([3.0, 4.0])[0].value == pytest.approx(5.0)
            with pytest.raises(ValueError, match="zero-norm stored vectors"):
                store.cosine_search([1.0, 0.0])
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
    def test_open_raises_on_non_finite_vectors(self, non_finite):
        """Test open rejects persisted non-finite vectors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            np.savez_compressed(
                file_path,
                format_version=np.array(1, dtype=np.int64),
                dimensions=np.array(2, dtype=np.int64),
                normalize=np.array(True, dtype=np.bool_),
                vectors=np.array([[non_finite, 1.0]], dtype=np.float32),
                metadata=np.array([{"id": "invalid"}], dtype=object),
            )

            with pytest.raises(ValueError, match="non-finite"):
                VectorStore.open(file_path)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_save_no_file_path(self):
        """Test save requires a path when the store is not yet bound."""
        store = VectorStore(dimensions=2)
        add_single_vector(store, np.array([1.0, 2.0]), {"id": "test"})

        with pytest.raises(ValueError, match="unbound store"):
            store.save()

    def test_save_rejects_empty_file_path(self):
        """Test an explicit empty path cannot bind a store."""
        store = VectorStore(dimensions=2)

        with pytest.raises(ValueError, match="file path"):
            store.save("")

    def test_save_empty_vectors(self):
        """Test save with empty vectors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store = VectorStore(dimensions=2)
            store.save(file_path)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_clear_then_save_overwrites_persisted_data(self):
        """Test clear() + save() persists an empty store to disk."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store1 = VectorStore(dimensions=2)
            add_single_vector(store1, np.array([1.0, 2.0]), {"id": "test"})
            store1.save(file_path)

            store1.clear()
            store1.save()

            store2 = VectorStore.open(file_path)
            assert len(store2.vectors) == 0
            assert len(store2.metadata) == 0
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_save_and_load_raw_vectors_round_trip(self):
        """Test normalize=False saves and loads raw vectors without mode metadata."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            file_path = tmp.name

        try:
            store1 = VectorStore(dimensions=2, normalize=False)
            store1.add([[3.0, 4.0]], [{"id": "raw"}])
            store1.save(file_path)

            with np.load(file_path, allow_pickle=True) as data:
                assert data["format_version"].item() == 1
                assert data["dimensions"].item() == 2
                assert data["normalize"].item() is False
                np.testing.assert_array_equal(data["vectors"][0], np.array([3.0, 4.0]))

            store2 = VectorStore.open(file_path)

            np.testing.assert_array_equal(store2.get(0)[0], np.array([3.0, 4.0]))
            assert store2.get(0)[1] == {"id": "raw"}
        finally:
            Path(file_path).unlink(missing_ok=True)
