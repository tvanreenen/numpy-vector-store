from pathlib import Path
from typing import Any, Optional

import numpy as np


class VectorStore:
    """
    A simple vector store implementation using NumPy.

    This class provides basic functionality for storing and searching
    vector embeddings using cosine similarity.
    """

    def __init__(
        self,
        dimensions: int,
        file_path: Optional[str] = None,
        metadata_schema: Optional[dict] = None,
    ):
        """
        Initialize the vector store.

        Args:
            dimensions: The number of dimensions for vectors to be stored.
            file_path: Optional path to save/load vectors from.
            metadata_schema: Optional schema for structured metadata operations.
        """
        self.dimensions = dimensions
        self.file_path = Path(file_path) if file_path else None
        # Initialize as empty 2D array with correct dimensions
        self.vectors: np.ndarray = np.array([]).reshape(0, dimensions)
        self._loaded = False

        if metadata_schema:
            # Use structured NumPy array for performance
            self.metadata = np.array([], dtype=self._create_dtype(metadata_schema))
            self._use_structured = True
            self._metadata_schema: Optional[dict] = metadata_schema
        else:
            # Use object array for flexibility (still NumPy)
            self.metadata = np.array([], dtype=object)
            self._use_structured = False
            self._metadata_schema = None

    def _create_dtype(self, schema: dict) -> list:
        """Convert schema dict to NumPy dtype."""
        dtype = []
        for field, field_type in schema.items():
            if isinstance(field_type, str):
                dtype.append((field, field_type))
            else:
                dtype.append((field, "O"))  # Object type for complex data
        return dtype

    def add_vectors(self, vectors_2d: np.ndarray, metadata_array: np.ndarray) -> None:
        """
        Add vectors and metadata directly as NumPy arrays (most efficient).

        Args:
            vectors_2d: 2D NumPy array of shape (n_vectors, dimensions)
            metadata_array: 1D NumPy array of metadata objects
        """
        if vectors_2d.shape[1] != self.dimensions:
            raise ValueError(
                f"Vector dimensions {vectors_2d.shape[1]} doesn't match store dimensions {self.dimensions}"
            )

        if len(vectors_2d) != len(metadata_array):
            raise ValueError("Number of vectors must match number of metadata items")

        # Normalize vectors in batch
        norms = np.linalg.norm(vectors_2d, axis=1, keepdims=True)
        normalized_vectors = vectors_2d / norms

        # Add vectors directly
        if len(self.vectors) == 0:
            self.vectors = normalized_vectors
        else:
            self.vectors = np.vstack([self.vectors, normalized_vectors])

        # Add metadata directly
        self.metadata = np.append(self.metadata, metadata_array)

    def load(self) -> None:
        """
        Load vectors from file if file_path is specified.
        """
        if self._loaded or not self.file_path:
            return

        if self.file_path.exists():
            data = np.load(self.file_path, allow_pickle=True)
            self.vectors = data["vectors"]
            self.metadata = data["metadata"]
            data.close()

        self._loaded = True

    def save(self) -> None:
        """
        Save vectors to file if file_path is specified.
        """
        if not self.file_path:
            return

        if len(self.vectors) > 0:
            vectors_array: np.ndarray = self.vectors.astype(np.float32)
            metadata_array = np.array(self.metadata, dtype=object)
            np.savez_compressed(
                self.file_path, vectors=vectors_array, metadata=metadata_array
            )

    def search(
        self, query_vector: np.ndarray, top_k: int = 10, score_cutoff: float = 0.0
    ) -> list[tuple[int, float, dict]]:
        """
        Search for the most similar vectors.

        Args:
            query_vector: The query vector to search with.
            top_k: Number of top results to return.
            score_cutoff: Minimum similarity score to include in results.

        Returns:
            List of tuples containing (index, similarity_score, metadata).
        """
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} doesn't match store dimensions {self.dimensions}"
            )

        if len(self.vectors) == 0:
            return []

        # Use efficient cosine similarity computation
        similarities = self._cosine_similarity_numpy(query_vector, self.vectors)

        # Filter by similarity cutoff first
        valid_indices = np.where(similarities >= score_cutoff)[0]
        if len(valid_indices) == 0:
            return []

        # Get top_k results from valid indices
        valid_similarities = similarities[valid_indices]
        top_valid_indices = np.argsort(valid_similarities)[::-1][:top_k]
        top_indices = valid_indices[top_valid_indices]

        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx]), self.metadata[idx]))

        return results

    def _cosine_similarity_numpy(
        self, query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """
        Efficient cosine similarity computation using NumPy.

        Args:
            query: Query vector.
            vectors: Array of vectors to compare against.

        Returns:
            Array of similarity scores.
        """
        query_norm = query / np.linalg.norm(query)
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = vectors / vector_norms
        return np.dot(vectors_norm, query_norm)  # type: ignore[no-any-return]

    def get(self, index: int) -> Optional[tuple[np.ndarray, dict]]:
        """
        Get vector and metadata by index.

        Args:
            index: The index of the vector to retrieve.

        Returns:
            Tuple of (vector, metadata) or None if index is out of bounds.
        """
        if 0 <= index < len(self.vectors):
            if self._use_structured and self._metadata_schema is not None:
                # Convert structured array row to dict
                metadata_row = self.metadata[index]
                metadata_dict = {
                    field: metadata_row[field] for field in self._metadata_schema.keys()
                }
                return (self.vectors[index], metadata_dict)
            else:
                # Object array - return the dict directly
                return (self.vectors[index], self.metadata[index])
        return None

    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        self.vectors = np.array([]).reshape(0, self.dimensions)
        if self._use_structured and self._metadata_schema is not None:
            self.metadata = np.array(
                [], dtype=self._create_dtype(self._metadata_schema)
            )
        else:
            self.metadata = np.array([], dtype=object)

    def __enter__(self) -> "VectorStore":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, auto-save if file_path is specified."""
        if self.file_path:
            self.save()
