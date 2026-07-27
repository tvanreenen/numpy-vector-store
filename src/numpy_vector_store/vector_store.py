from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

TMetadata = TypeVar("TMetadata")


@dataclass(frozen=True, slots=True)
class VectorHit(Generic[TMetadata]):
    """A typed vector-search result."""

    index: int
    value: float
    metadata: TMetadata


class VectorStore(Generic[TMetadata]):
    """
    A small in-memory vector store using NumPy exact vector search.

    Metadata is an opaque row payload returned with vector hits. The store does
    not implement a metadata query language; pass row indexes to within_rows for
    prefiltered vector search.
    """

    def __init__(
        self,
        dimensions: int,
        file_path: str | Path | None = None,
        *,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            dimensions: The number of dimensions for vectors to be stored.
            file_path: Optional path to save/load vectors from.
            normalize: Whether to store vectors normalized to unit length.
        """
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

        self.dimensions = dimensions
        self.file_path = Path(file_path) if file_path else None
        self.normalize = normalize
        self.vectors: npt.NDArray[np.float32] = np.empty(
            (0, dimensions), dtype=np.float32
        )
        self._loaded = False
        self.metadata: npt.NDArray[Any] = np.array([], dtype=object)

    def add(
        self, vectors: npt.ArrayLike, metadata: Sequence[TMetadata] | npt.NDArray[Any]
    ) -> None:
        """
        Add vectors and row metadata payloads.

        When normalize=True, vectors are normalized at insert time. Metadata
        items are stored as opaque payloads and returned with vector hits.
        """
        vectors_2d = np.asarray(vectors, dtype=np.float32)
        if vectors_2d.ndim != 2:
            raise ValueError("vectors must be a 2D array")

        if vectors_2d.shape[1] != self.dimensions:
            raise ValueError(
                f"Vector dimensions {vectors_2d.shape[1]} doesn't match store dimensions {self.dimensions}"
            )

        metadata_array = self._metadata_to_array(metadata)
        if metadata_array.ndim != 1:
            raise ValueError("metadata must be a 1D sequence")

        if len(vectors_2d) != len(metadata_array):
            raise ValueError("Number of vectors must match number of metadata items")

        vectors_to_store = self._prepare_vectors_for_storage(
            vectors_2d,
            zero_norm_error_message="Cannot add zero-norm vectors",
            non_finite_error_message="Cannot add vectors containing non-finite values",
        )

        if len(self.vectors) == 0:
            self.vectors = vectors_to_store
        else:
            self.vectors = np.vstack([self.vectors, vectors_to_store]).astype(
                np.float32, copy=False
            )

        self.metadata = np.append(self.metadata, metadata_array)

    def load(self) -> None:
        """Load vectors from file if file_path is specified and exists."""
        if self._loaded or not self.file_path:
            return

        if self.file_path.exists():
            with np.load(self.file_path, allow_pickle=True) as data:
                files = set(data.files)
                self._validate_required_fields(files)

                loaded_vectors = np.asarray(data["vectors"], dtype=np.float32)
                loaded_metadata = np.array(data["metadata"], copy=True)

            self._validate_loaded_arrays(loaded_vectors, loaded_metadata)
            loaded_vectors = self._prepare_vectors_for_storage(
                loaded_vectors,
                zero_norm_error_message="Loaded vectors contain zero-norm vectors",
                non_finite_error_message="Loaded vectors contain non-finite values",
            )

            self.vectors = loaded_vectors
            self.metadata = loaded_metadata

        self._loaded = True

    def save(self) -> None:
        """Save vectors and metadata if file_path is specified."""
        if not self.file_path:
            return

        np.savez_compressed(
            self.file_path,
            vectors=self.vectors.astype(np.float32, copy=False),
            metadata=np.array(self.metadata, copy=True),
        )

    def cosine_search(
        self,
        query: npt.ArrayLike,
        *,
        top_k: int = 10,
        min_value: float | None = None,
        within_rows: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    ) -> list[VectorHit[TMetadata]]:
        """
        Return the most similar rows using cosine similarity.

        Args:
            query: 1D query vector.
            top_k: Maximum number of hits to return.
            min_value: Optional minimum cosine similarity value.
            within_rows: Optional row indexes to restrict search to.
        """
        return self._metric_search(
            query,
            top_k=top_k,
            within_rows=within_rows,
            values_fn=self._cosine_values,
            descending=True,
            min_value=min_value,
            max_value=None,
        )

    def dot_search(
        self,
        query: npt.ArrayLike,
        *,
        top_k: int = 10,
        min_value: float | None = None,
        within_rows: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    ) -> list[VectorHit[TMetadata]]:
        """
        Return rows ranked by dot product.

        With normalize=True, this is the dot product of unit vectors. With
        normalize=False, this is the true dot product over original vectors.
        """
        return self._metric_search(
            query,
            top_k=top_k,
            within_rows=within_rows,
            values_fn=self._dot_values,
            descending=True,
            min_value=min_value,
            max_value=None,
        )

    def euclidean_search(
        self,
        query: npt.ArrayLike,
        *,
        top_k: int = 10,
        max_value: float | None = None,
        within_rows: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    ) -> list[VectorHit[TMetadata]]:
        """
        Return rows ranked by Euclidean distance.

        With normalize=True, this is distance between normalized directions.
        With normalize=False, this is true Euclidean distance over original
        vectors.
        """
        return self._metric_search(
            query,
            top_k=top_k,
            within_rows=within_rows,
            values_fn=self._euclidean_values,
            descending=False,
            min_value=None,
            max_value=max_value,
        )

    def _cosine_values(
        self, query: npt.NDArray[np.float32], vectors: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Compute cosine similarity values."""
        query_norm = self._normalize_query(query)

        if self.normalize:
            values = np.dot(vectors, query_norm)
        else:
            vector_norms = self._row_norms(vectors)
            if np.any(vector_norms == 0):
                raise ValueError("Cannot cosine search zero-norm stored vectors")
            values = (
                np.einsum(
                    "ij,j->i",
                    vectors,
                    query_norm,
                    dtype=np.float64,
                )
                / vector_norms
            )

        return np.asarray(values, dtype=np.float32)

    def _dot_values(
        self, query: npt.NDArray[np.float32], vectors: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float64]:
        """Compute dot product values."""
        if self.normalize:
            query = self._normalize_query(query)
            values = np.dot(vectors, query)
        else:
            values = np.einsum(
                "ij,j->i",
                vectors,
                query,
                dtype=np.float64,
            )
        return np.asarray(values, dtype=np.float64)

    def _euclidean_values(
        self, query: npt.NDArray[np.float32], vectors: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float64]:
        """Compute Euclidean distance values."""
        if self.normalize:
            query = self._normalize_query(query)
            differences = vectors - query
        else:
            differences = vectors.astype(np.float64) - query.astype(np.float64)
        return np.asarray(np.linalg.norm(differences, axis=1), dtype=np.float64)

    def get(self, index: int) -> tuple[npt.NDArray[np.float32], TMetadata] | None:
        """Get a stored vector and metadata payload by row index."""
        if 0 <= index < len(self.vectors):
            return (self.vectors[index], self.metadata[index])
        return None

    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        self.vectors = np.empty((0, self.dimensions), dtype=np.float32)
        self.metadata = np.array([], dtype=object)

    def __len__(self) -> int:
        """Return the number of stored vector rows."""
        return len(self.vectors)

    def __enter__(self) -> VectorStore[TMetadata]:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, auto-save if file_path is specified."""
        if self.file_path:
            self.save()

    def _metadata_to_array(
        self, metadata: Sequence[TMetadata] | npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        return np.asarray(metadata, dtype=object)

    def _prepare_vectors_for_storage(
        self,
        vectors: npt.NDArray[np.float32],
        *,
        zero_norm_error_message: str,
        non_finite_error_message: str,
    ) -> npt.NDArray[np.float32]:
        if not np.all(np.isfinite(vectors)):
            raise ValueError(non_finite_error_message)

        if self.normalize:
            norms = self._validate_non_zero_vectors(
                vectors,
                zero_norm_error_message=zero_norm_error_message,
            )
            return np.asarray(vectors / norms[:, np.newaxis], dtype=np.float32)
        return vectors.astype(np.float32, copy=True)

    def _validate_non_zero_vectors(
        self,
        vectors: npt.NDArray[np.float32],
        *,
        zero_norm_error_message: str,
    ) -> npt.NDArray[np.float64]:
        norms = self._row_norms(vectors)
        if np.any(norms == 0):
            raise ValueError(zero_norm_error_message)
        return norms

    def _row_norms(self, vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        squared_norms = np.einsum(
            "ij,ij->i",
            vectors,
            vectors,
            dtype=np.float64,
        )
        return np.asarray(np.sqrt(squared_norms), dtype=np.float64)

    def _validate_query(self, query: npt.ArrayLike) -> npt.NDArray[np.float32]:
        query_vector = np.asarray(query, dtype=np.float32)
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be a 1D array")
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} doesn't match store dimensions {self.dimensions}"
            )
        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Query vector must contain only finite values")
        return query_vector

    def _validate_search_threshold(
        self, value: float | None, *, name: str
    ) -> float | None:
        if value is None:
            return None

        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _normalize_query(
        self, query: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        query_magnitude = np.linalg.norm(query.astype(np.float64))
        if query_magnitude == 0:
            raise ValueError("Cannot search with zero-norm query vector")
        return np.asarray(
            query.astype(np.float64) / query_magnitude,
            dtype=np.float32,
        )

    def _normalize_within_rows(
        self, within_rows: Sequence[int] | npt.NDArray[np.integer[Any]] | None
    ) -> npt.NDArray[np.intp]:
        if within_rows is None:
            return np.arange(len(self.vectors), dtype=np.intp)

        rows = np.asarray(within_rows)
        if rows.ndim != 1:
            raise ValueError("within_rows must be a 1D sequence of row indexes")
        if len(rows) == 0:
            return np.array([], dtype=np.intp)
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("within_rows must contain integer row indexes")

        rows = rows.astype(np.intp, copy=False)
        if np.any(rows < 0) or np.any(rows >= len(self.vectors)):
            raise IndexError("within_rows contains row indexes outside the store")
        return rows

    def _metric_search(
        self,
        query: npt.ArrayLike,
        *,
        top_k: int,
        within_rows: Sequence[int] | npt.NDArray[np.integer[Any]] | None,
        values_fn: Callable[
            [npt.NDArray[np.float32], npt.NDArray[np.float32]],
            npt.NDArray[np.float32] | npt.NDArray[np.float64],
        ],
        descending: bool,
        min_value: float | None,
        max_value: float | None,
    ) -> list[VectorHit[TMetadata]]:
        query_vector = self._validate_query(query)
        min_value = self._validate_search_threshold(min_value, name="min_value")
        max_value = self._validate_search_threshold(max_value, name="max_value")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if len(self.vectors) == 0:
            return []

        row_indices = self._normalize_within_rows(within_rows)
        if len(row_indices) == 0:
            return []

        values = values_fn(query_vector, self.vectors[row_indices])

        valid_local_indices = np.arange(len(values))
        if min_value is not None:
            valid_local_indices = valid_local_indices[
                values[valid_local_indices] >= min_value
            ]
        if max_value is not None:
            valid_local_indices = valid_local_indices[
                values[valid_local_indices] <= max_value
            ]

        if len(valid_local_indices) == 0:
            return []

        valid_values = values[valid_local_indices]
        result_count = min(top_k, len(valid_local_indices))
        if result_count < len(valid_local_indices):
            if descending:
                top_valid_unsorted = np.argpartition(valid_values, -result_count)[
                    -result_count:
                ]
            else:
                top_valid_unsorted = np.argpartition(valid_values, result_count - 1)[
                    :result_count
                ]
        else:
            top_valid_unsorted = np.arange(len(valid_values))

        sort_order = np.argsort(valid_values[top_valid_unsorted])
        if descending:
            sort_order = sort_order[::-1]

        top_valid_sorted = top_valid_unsorted[sort_order]
        local_indices = valid_local_indices[top_valid_sorted]
        original_indices = row_indices[local_indices]

        return [
            VectorHit(
                index=int(original_idx),
                value=float(values[local_idx]),
                metadata=self.metadata[original_idx],
            )
            for local_idx, original_idx in zip(
                local_indices, original_indices, strict=True
            )
        ]

    def _validate_required_fields(self, files: set[str]) -> None:
        if "vectors" not in files:
            raise ValueError("Persisted vector store is missing vectors")
        if "metadata" not in files:
            raise ValueError("Persisted vector store is missing metadata")

    def _validate_loaded_arrays(
        self,
        loaded_vectors: npt.NDArray[np.float32],
        loaded_metadata: npt.NDArray[Any],
    ) -> None:
        if loaded_vectors.ndim != 2:
            raise ValueError("Loaded vectors must be a 2D array")

        if loaded_vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"Loaded vector dimension {loaded_vectors.shape[1]} doesn't match store dimensions {self.dimensions}"
            )

        if loaded_metadata.ndim != 1:
            raise ValueError("Loaded metadata must be a 1D array")

        if len(loaded_vectors) != len(loaded_metadata):
            raise ValueError("Loaded vectors and metadata length mismatch")
