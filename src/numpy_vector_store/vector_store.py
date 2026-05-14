from __future__ import annotations

import warnings
from collections.abc import Sequence
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
    A small in-memory vector store using NumPy cosine similarity.

    Metadata is an opaque row payload returned with vector hits. The store does
    not implement a metadata query language; pass row indexes to within_rows for
    prefiltered similarity search.
    """

    def __init__(
        self,
        dimensions: int,
        file_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            dimensions: The number of dimensions for vectors to be stored.
            file_path: Optional path to save/load vectors from.
        """
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

        self.dimensions = dimensions
        self.file_path = Path(file_path) if file_path else None
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

        Vectors are normalized at insert time. Metadata items are stored as
        opaque payloads and returned with vector hits.
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

        normalized_vectors = self._normalize_vectors(
            vectors_2d,
            error_message="Cannot add zero-norm vectors",
        )

        if len(self.vectors) == 0:
            self.vectors = normalized_vectors
        else:
            self.vectors = np.vstack([self.vectors, normalized_vectors]).astype(
                np.float32, copy=False
            )

        self.metadata = np.append(self.metadata, metadata_array)

    def add_vectors(self, vectors_2d: np.ndarray, metadata_array: np.ndarray) -> None:
        """
        Deprecated API for adding vectors and metadata arrays.

        Use add(...) instead.
        """
        warnings.warn(
            "VectorStore.add_vectors() is deprecated and will be removed in a future 0.x release. "
            "Use VectorStore.add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if vectors_2d.ndim != 2:
            raise ValueError("vectors_2d must be a 2D NumPy array")

        if metadata_array.ndim != 1:
            raise ValueError("metadata_array must be a 1D NumPy array")

        self.add(vectors_2d, metadata_array.tolist())

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
            loaded_vectors = self._normalize_vectors(
                loaded_vectors,
                error_message="Loaded vectors contain zero-norm vectors",
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
        query_vector = self._validate_query(query)

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if len(self.vectors) == 0:
            return []

        row_indices = self._normalize_within_rows(within_rows)
        if len(row_indices) == 0:
            return []

        similarities = self._cosine_similarity_numpy(
            query_vector, self.vectors[row_indices]
        )

        if min_value is not None:
            valid_local_indices = np.flatnonzero(similarities >= min_value)
        else:
            valid_local_indices = np.arange(len(similarities))

        if len(valid_local_indices) == 0:
            return []

        valid_similarities = similarities[valid_local_indices]
        result_count = min(top_k, len(valid_local_indices))
        if result_count < len(valid_local_indices):
            top_local_unsorted = np.argpartition(valid_similarities, -result_count)[
                -result_count:
            ]
        else:
            top_local_unsorted = np.arange(len(valid_similarities))

        top_local_sorted = top_local_unsorted[
            np.argsort(valid_similarities[top_local_unsorted])[::-1]
        ]
        local_indices = valid_local_indices[top_local_sorted]
        original_indices = row_indices[local_indices]

        return [
            VectorHit(
                index=int(original_idx),
                value=float(similarities[local_idx]),
                metadata=self.metadata[original_idx],
            )
            for local_idx, original_idx in zip(
                local_indices, original_indices, strict=True
            )
        ]

    def search(
        self, query_vector: np.ndarray, top_k: int = 10, score_cutoff: float = 0.0
    ) -> list[tuple[int, float, TMetadata]]:
        """
        Deprecated tuple-returning similarity search.

        Use cosine_search(...), which returns VectorHit objects and uses
        min_value instead.
        """
        warnings.warn(
            "VectorStore.search() is deprecated and will be removed in a future 0.x release. "
            "Use VectorStore.cosine_search() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        hits = self.cosine_search(query_vector, top_k=top_k, min_value=score_cutoff)
        return [(hit.index, hit.value, hit.metadata) for hit in hits]

    def _cosine_similarity_numpy(
        self, query: npt.NDArray[np.float32], vectors: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Compute cosine similarity against normalized vectors."""
        query_magnitude = np.linalg.norm(query)
        if query_magnitude == 0:
            raise ValueError("Cannot search with zero-norm query vector")
        query_norm = query / query_magnitude
        similarities = np.asarray(np.dot(vectors, query_norm), dtype=np.float32)
        return similarities

    def get(self, index: int) -> tuple[npt.NDArray[np.float32], TMetadata] | None:
        """Get a normalized vector and metadata payload by row index."""
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

    def _normalize_vectors(
        self, vectors: npt.NDArray[np.float32], *, error_message: str
    ) -> npt.NDArray[np.float32]:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError(error_message)
        return np.asarray(vectors / norms, dtype=np.float32)

    def _validate_query(self, query: npt.ArrayLike) -> npt.NDArray[np.float32]:
        query_vector = np.asarray(query, dtype=np.float32)
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be a 1D array")
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} doesn't match store dimensions {self.dimensions}"
            )
        return query_vector

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
