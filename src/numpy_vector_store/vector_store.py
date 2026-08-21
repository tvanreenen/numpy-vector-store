from __future__ import annotations

import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import uuid4

import numpy as np
import numpy.typing as npt

TMetadata = TypeVar("TMetadata")

_ARCHIVE_FORMAT_VERSION = 1
_ARCHIVE_FIELDS = frozenset(
    {"format_version", "dimensions", "normalize", "vectors", "metadata"}
)


@dataclass(frozen=True, slots=True)
class _LoadedArchive:
    dimensions: int
    normalize: bool
    vectors: npt.NDArray[Any]
    metadata: npt.NDArray[Any]


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
        *,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            dimensions: The number of dimensions for vectors to be stored.
            normalize: Whether to store vectors normalized to unit length.
        """
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

        self._dimensions = dimensions
        self._file_path: Path | None = None
        self._normalize = normalize
        self._vectors: npt.NDArray[np.float32] = np.empty(
            (0, dimensions), dtype=np.float32
        )
        self._metadata: npt.NDArray[Any] = np.array([], dtype=object)
        self._row_count = 0
        self._lock_row_storage()

    @property
    def dimensions(self) -> int:
        """Return the configured vector width."""
        return self._dimensions

    @property
    def normalize(self) -> bool:
        """Return whether vectors use normalized storage semantics."""
        return self._normalize

    @property
    def file_path(self) -> Path | None:
        """Return the currently bound archive path, if any."""
        return self._file_path

    @property
    def vectors(self) -> npt.NDArray[np.float32]:
        """Return a zero-copy, non-writeable view of the current vectors."""
        view = self._vectors[: self._row_count]
        view.flags.writeable = False
        return view

    @property
    def metadata(self) -> npt.NDArray[Any]:
        """Return a zero-copy, non-writeable view of the current metadata rows."""
        view = self._metadata[: self._row_count]
        view.flags.writeable = False
        return view

    @classmethod
    def open(cls, path: str | Path) -> VectorStore[TMetadata]:
        """Open a store from a versioned, self-describing archive."""
        resolved_path = cls._resolve_file_path(path)
        if resolved_path is None:
            raise ValueError("path must not be empty")

        archive = cls._read_archive(resolved_path)
        store = cls(dimensions=archive.dimensions, normalize=archive.normalize)
        store._file_path = resolved_path
        store._replace_with_archive(archive)
        return store

    def add(
        self, vectors: npt.ArrayLike, metadata: Sequence[TMetadata] | npt.NDArray[Any]
    ) -> None:
        """
        Add vectors and row metadata payloads.

        When normalize=True, vectors are normalized at insert time. Metadata
        items are stored as opaque payloads and returned with vector hits.
        """
        vectors_2d = self._to_float32_array(vectors)
        if vectors_2d.ndim != 2:
            raise ValueError("vectors must be a 2D array")

        if vectors_2d.shape[1] != self._dimensions:
            raise ValueError(
                f"Vector dimensions {vectors_2d.shape[1]} doesn't match store dimensions {self._dimensions}"
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

        self._append_rows(vectors_to_store, metadata_array)

    def reload(self) -> None:
        """Refresh this store from its bound archive path."""
        if self._file_path is None:
            raise ValueError("reload() requires a bound file path")

        archive = self._read_archive(self._file_path)
        self._replace_with_archive(archive)

    def save(self, path: str | Path | None = None) -> None:
        """Save the store and bind an explicitly supplied destination path."""
        destination = self._file_path if path is None else self._resolve_file_path(path)
        if destination is None:
            raise ValueError("save() requires a file path for an unbound store")

        self._write_archive(destination)
        self._file_path = destination

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

        if self._normalize:
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
        if self._normalize:
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
        if self._normalize:
            query = self._normalize_query(query)
            differences = vectors - query
        else:
            differences = np.empty(vectors.shape, dtype=np.float64)
            np.subtract(vectors, query, out=differences, dtype=np.float64)
        return np.asarray(np.linalg.norm(differences, axis=1), dtype=np.float64)

    def get(self, index: int) -> tuple[npt.NDArray[np.float32], TMetadata] | None:
        """Get a stored vector and metadata payload by row index."""
        if 0 <= index < self._row_count:
            return (self._vectors[index].copy(), self._metadata[index])
        return None

    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        self._replace_rows(
            np.empty((0, self._dimensions), dtype=np.float32),
            np.array([], dtype=object),
        )

    def __len__(self) -> int:
        """Return the number of stored vector rows."""
        return self._row_count

    def _append_rows(
        self,
        vectors: npt.NDArray[np.float32],
        metadata: npt.NDArray[Any],
    ) -> None:
        added_count = len(vectors)
        if added_count == 0:
            return

        if self._row_count == 0 and len(self._vectors) == 0:
            self._replace_rows(vectors, metadata)
            return

        required_capacity = self._row_count + added_count
        if required_capacity > len(self._vectors):
            self._grow_and_append(vectors, metadata, required_capacity)
            return

        start = self._row_count
        self._vectors.flags.writeable = True
        try:
            self._metadata.flags.writeable = True
            self._vectors[start:required_capacity] = vectors
            self._metadata[start:required_capacity] = metadata
        finally:
            self._lock_row_storage()
        self._row_count = required_capacity

    def _grow_and_append(
        self,
        vectors: npt.NDArray[np.float32],
        metadata: npt.NDArray[Any],
        required_capacity: int,
    ) -> None:
        new_capacity = max(required_capacity, len(self._vectors) * 2)
        new_vectors = np.empty((new_capacity, self._dimensions), dtype=np.float32)
        new_metadata = np.empty(new_capacity, dtype=object)

        new_vectors[: self._row_count] = self._vectors[: self._row_count]
        new_metadata[: self._row_count] = self._metadata[: self._row_count]
        new_vectors[self._row_count : required_capacity] = vectors
        new_metadata[self._row_count : required_capacity] = metadata

        self._vectors = new_vectors
        self._metadata = new_metadata
        self._row_count = required_capacity
        self._lock_row_storage()

    def _replace_rows(
        self,
        vectors: npt.NDArray[np.float32],
        metadata: npt.NDArray[Any],
    ) -> None:
        self._vectors = vectors
        self._metadata = metadata
        self._row_count = len(vectors)
        self._lock_row_storage()

    def _lock_row_storage(self) -> None:
        self._vectors.flags.writeable = False
        self._metadata.flags.writeable = False

    def _metadata_to_array(
        self, metadata: Sequence[TMetadata] | npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        if isinstance(metadata, np.ndarray) and metadata.ndim != 1:
            return np.asarray(metadata, dtype=object)

        metadata_array = np.empty(len(metadata), dtype=object)
        for index, payload in enumerate(metadata):
            metadata_array[index] = payload
        return metadata_array

    @staticmethod
    def _resolve_file_path(file_path: str | Path | None) -> Path | None:
        if not file_path:
            return None

        path = Path(file_path)
        if path.suffix != ".npz":
            return Path(f"{path}.npz")
        return path

    def _to_float32_array(self, values: npt.ArrayLike) -> npt.NDArray[np.float32]:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(values, dtype=np.float32)

    def _prepare_vectors_for_storage(
        self,
        vectors: npt.NDArray[np.float32],
        *,
        zero_norm_error_message: str,
        non_finite_error_message: str,
    ) -> npt.NDArray[np.float32]:
        if not np.all(np.isfinite(vectors)):
            raise ValueError(non_finite_error_message)

        if self._normalize:
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
        query_vector = self._to_float32_array(query)
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be a 1D array")
        if len(query_vector) != self._dimensions:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} doesn't match store dimensions {self._dimensions}"
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
        self, within_rows: Sequence[int] | npt.NDArray[np.integer[Any]]
    ) -> npt.NDArray[np.intp]:
        rows = np.asarray(within_rows)
        if rows.ndim != 1:
            raise ValueError("within_rows must be a 1D sequence of row indexes")
        if len(rows) == 0:
            return np.array([], dtype=np.intp)
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("within_rows must contain integer row indexes")

        rows = rows.astype(np.intp, copy=False)
        if np.any(rows < 0) or np.any(rows >= self._row_count):
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

        if self._row_count == 0:
            return []

        row_indices = None
        selected_vectors = self._vectors[: self._row_count]
        if within_rows is not None:
            row_indices = self._normalize_within_rows(within_rows)
            if len(row_indices) == 0:
                return []
            selected_vectors = self._vectors[row_indices]

        values = values_fn(query_vector, selected_vectors)

        valid_local_indices = np.arange(len(values), dtype=np.intp)
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
        valid_original_indices = (
            valid_local_indices
            if row_indices is None
            else row_indices[valid_local_indices]
        )
        result_count = min(top_k, len(valid_local_indices))
        top_positions = self._top_result_positions(
            valid_values,
            valid_original_indices,
            result_count=result_count,
            descending=descending,
        )
        local_indices = valid_local_indices[top_positions]
        original_indices = valid_original_indices[top_positions]

        return [
            VectorHit(
                index=int(original_idx),
                value=float(values[local_idx]),
                metadata=self._metadata[original_idx],
            )
            for local_idx, original_idx in zip(
                local_indices, original_indices, strict=True
            )
        ]

    @staticmethod
    def _top_result_positions(
        values: npt.NDArray[np.float32] | npt.NDArray[np.float64],
        original_indices: npt.NDArray[np.intp],
        *,
        result_count: int,
        descending: bool,
    ) -> npt.NDArray[np.intp]:
        if result_count < len(values):
            partition_index = (
                len(values) - result_count if descending else result_count - 1
            )
            cutoff = np.partition(values, partition_index)[partition_index]
            better = values > cutoff if descending else values < cutoff
            selected = np.flatnonzero(better)
            tied = np.flatnonzero(values == cutoff)
            remaining = result_count - len(selected)
            if remaining < len(tied):
                tied = tied[
                    np.argpartition(original_indices[tied], remaining - 1)[:remaining]
                ]
            selected = np.concatenate((selected, tied))
        else:
            selected = np.arange(len(values), dtype=np.intp)

        primary_values = -values[selected] if descending else values[selected]
        order = np.lexsort((original_indices[selected], primary_values))
        return np.asarray(selected[order], dtype=np.intp)

    @classmethod
    def _read_archive(
        cls,
        file_path: Path,
    ) -> _LoadedArchive:
        with np.load(file_path, allow_pickle=True) as data:
            files = set(data.files)
            if "format_version" not in files:
                raise ValueError(
                    "Persisted vector store is missing fields: format_version"
                )
            format_version = cls._read_integer_scalar(
                data["format_version"], name="format_version"
            )
            if format_version != _ARCHIVE_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported vector store archive format version: {format_version}"
                )
            cls._validate_archive_fields(files)
            dimensions = cls._read_integer_scalar(data["dimensions"], name="dimensions")
            normalize = cls._read_boolean_scalar(data["normalize"], name="normalize")
            if dimensions <= 0:
                raise ValueError("Persisted dimensions must be greater than 0")

            persisted_vectors = np.array(data["vectors"], copy=True)
            loaded_metadata = np.array(data["metadata"], copy=True)

        cls._validate_archive_array_dtypes(persisted_vectors, loaded_metadata)
        return _LoadedArchive(
            dimensions=dimensions,
            normalize=normalize,
            vectors=persisted_vectors,
            metadata=loaded_metadata,
        )

    def _replace_with_archive(self, archive: _LoadedArchive) -> None:
        self._validate_archive_configuration(
            dimensions=archive.dimensions,
            normalize=archive.normalize,
        )
        loaded_vectors = self._to_float32_array(archive.vectors)
        self._validate_loaded_arrays(loaded_vectors, archive.metadata)
        loaded_vectors = self._prepare_vectors_for_storage(
            loaded_vectors,
            zero_norm_error_message="Loaded vectors contain zero-norm vectors",
            non_finite_error_message="Loaded vectors contain non-finite values",
        )

        self._replace_rows(loaded_vectors, archive.metadata)

    def _write_archive(self, destination: Path) -> None:
        try:
            destination_mode = stat.S_IMODE(destination.stat().st_mode)
        except FileNotFoundError:
            destination_mode = None

        temporary_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        temporary_created = False
        try:
            with temporary_path.open("xb") as temporary_file:
                temporary_created = True
                np.savez_compressed(
                    temporary_file,
                    format_version=np.array(_ARCHIVE_FORMAT_VERSION, dtype=np.int64),
                    dimensions=np.array(self._dimensions, dtype=np.int64),
                    normalize=np.array(self._normalize, dtype=np.bool_),
                    vectors=self._vectors[: self._row_count].astype(
                        np.float32, copy=False
                    ),
                    metadata=np.array(self._metadata[: self._row_count], copy=True),
                )
            if destination_mode is not None:
                temporary_path.chmod(destination_mode)
            os.replace(temporary_path, destination)
        finally:
            if temporary_created:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_archive_fields(files: set[str]) -> None:
        missing = _ARCHIVE_FIELDS - files
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Persisted vector store is missing fields: {names}")

        unexpected = files - _ARCHIVE_FIELDS
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"Persisted vector store has unexpected fields: {names}")

    @staticmethod
    def _read_integer_scalar(value: npt.NDArray[Any], *, name: str) -> int:
        scalar = np.asarray(value)
        if scalar.ndim != 0 or not np.issubdtype(scalar.dtype, np.integer):
            raise ValueError(f"Persisted {name} must be a scalar integer")
        return int(scalar.item())

    @staticmethod
    def _read_boolean_scalar(value: npt.NDArray[Any], *, name: str) -> bool:
        scalar = np.asarray(value)
        if scalar.ndim != 0 or scalar.dtype != np.dtype(np.bool_):
            raise ValueError(f"Persisted {name} must be a scalar boolean")
        return bool(scalar.item())

    def _validate_archive_configuration(
        self, *, dimensions: int, normalize: bool
    ) -> None:
        if dimensions <= 0:
            raise ValueError("Persisted dimensions must be greater than 0")
        if dimensions != self._dimensions:
            raise ValueError(
                f"Persisted dimensions {dimensions} do not match store dimensions {self._dimensions}"
            )
        if normalize != self._normalize:
            raise ValueError(
                f"Persisted normalize setting {normalize} does not match store normalize setting {self._normalize}"
            )

    @staticmethod
    def _validate_archive_array_dtypes(
        loaded_vectors: npt.NDArray[Any],
        loaded_metadata: npt.NDArray[Any],
    ) -> None:
        if loaded_vectors.dtype != np.dtype(np.float32):
            raise ValueError("Persisted vectors must use the float32 dtype")
        if loaded_metadata.dtype != np.dtype(object):
            raise ValueError("Persisted metadata must use the object dtype")

    def _validate_loaded_arrays(
        self,
        loaded_vectors: npt.NDArray[np.float32],
        loaded_metadata: npt.NDArray[Any],
    ) -> None:
        if loaded_vectors.ndim != 2:
            raise ValueError("Loaded vectors must be a 2D array")

        if loaded_vectors.shape[1] != self._dimensions:
            raise ValueError(
                f"Loaded vector dimension {loaded_vectors.shape[1]} doesn't match store dimensions {self._dimensions}"
            )

        if loaded_metadata.ndim != 1:
            raise ValueError("Loaded metadata must be a 1D array")

        if len(loaded_vectors) != len(loaded_metadata):
            raise ValueError("Loaded vectors and metadata length mismatch")
