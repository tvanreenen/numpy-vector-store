# NumPy Vector Store

A fast, lightweight, and zero-setup in-memory vector store powered by NumPy.

- **High-performance** index-free O(n) cosine similarity searches
- **Vectorized** metadata queries with NumPy operations
- **Easy persistence** to and from compressed NumPy binary files (.npz)
- **Zero dependency** on external services or data stores

## Why?

This library is purpose-built for small to medium-scale vector search tasks and offers a simple, lightweight alternative to heavyweight solutions like Pinecone, Qdrant, Weaviate, Postgres + pgvector, or Azure AI Search—no complex setup or infrastructure required.

> Sometimes you don't need a sledgehammer to crack a nut.

## When/Where?

Below are benchmark results for the module's search method to help you assess its suitability for your use case.

| Embedding Type | Dimensions | ~5ms | ~25ms | ~100ms | ~500ms |
|----------------|------------|------|-------|--------|--------|
| **Sentence Transformers** | 384 | 1K vectors<br/>1.5MB | 10K vectors<br/>15MB | 100K vectors<br/>147MB | 500K vectors<br/>732MB |
| **OpenAI Small** | 1536 | 500 vectors<br/>3MB | 5K vectors<br/>29MB | 25K vectors<br/>147MB | 100K vectors<br/>586MB |
| **OpenAI Large** | 3072 | 200 vectors<br/>2MB | 2.5K vectors<br/>29MB | 5K vectors<br/>59MB | 25K vectors<br/>293MB |

*Benchmarks performed on Apple M2 hardware*

## Installation

**⚠️ Pending submission to PyPI**

```bash
uv add numpy-vector-store
```

## Quick Start

```python
import numpy as np
from numpy_vector_store import VectorStore

# Load your vector store
store = VectorStore(dimensions=1536, file_path='vectors.npz')
store.load()

# Embed your search query
query = np.array([0.2, 0.3, 0.4, ...])

# Search using cosine similarity
results = store.search(query, top_k=3)

# Compare the results
for index, similarity, meta in results:
    print(f"{meta['title']}: {similarity:.3f}")
```

## Usage Examples

### Adding Vectors

#### Adding Vectors in Batch

The `add_vectors` method takes a **2D NumPy array** where each row is a vector, and a **1D NumPy array** of metadata objects.

```python
# 2D np.array of vectors
embeddings = np.array([
    [0.1, 0.2, 0.3, ...],  # Text embedding 1
    [0.4, 0.5, 0.6, ...],  # Text embedding 2
    [0.7, 0.8, 0.9, ...]   # Text embedding 3
])

# 1D np.array of metadata objects
metadata = np.array([
    {"title": "AI Overview", "word_count": 12},
    {"title": "Python Guide", "word_count": 10},
    {"title": "Vector DBs", "word_count": 8}
])

# Vectors and metadata added using efficient vectorized NumPy operations
store.add_vectors(embeddings, metadata)
```

#### Adding Vectors Individually

Batch operations are generally preferable and more efficient. But individual vectors can be added with the same method.

```python
store.add_vectors(
    np.array([new_embedding]), # or np.atleast_2d(new_embedding)
    np.array([{"title": "Neural Networks Paper", "word_count": 15}])
)
```

### Save the Vector Store

Save to a gzip compressed NumPy binary file directly

```python
store.save()
```

Or use context manager for automatic persistence

```python
with VectorStore(dimensions=3, file_path="vectors.npz") as store:
    store.add_vectors(vectors_2d, metadata_array)

# Automatically saves when exiting the context
```

### Working with Metadata

Work with flexible, unstructured metadata using standard Python operations. No schema required - perfect for getting started quickly:

```python
# Access individual metadata
first_metadata = store.metadata[0]
print(f"First entry: {first_metadata}")

# Iterate through all metadata
for i, metadata in enumerate(store.metadata):
    print(f"Entry {i}: {metadata}")
```

### Advanced: Structured Metadata for Performance

If you define a homogeneous NumPy schema upfront, you get **significant performance improvements** for metadata operations. Instead of Python loops and dictionary lookups, you get **vectorized NumPy operations** that are orders of magnitude faster.

> **Learn more**: [NumPy Structured Arrays Documentation](https://numpy.org/doc/stable/user/basics.rec.html)

```python
# Define schema for performance
store = VectorStore(
    dimensions=512,
    metadata_schema={'title': 'U200', 'year': 'i4', 'citations': 'i4'}
)
store.add_vectors(
    np.array([vector1, vector2]),
    np.array([
        {"title": "Paper 1", "year": 2023, "citations": 100},
        {"title": "Paper 2", "year": 2022, "citations": 50}
    ])
)

# Perform vectorized NumPy operations on metadata
recent_mask = store.metadata['year'] == 2023
recent_vectors = np.array(store.vectors)[recent_mask]

# Sort by citations
sorted_indices = np.argsort(store.metadata['citations'])[::-1]
top_vectors = np.array(store.vectors)[sorted_indices]

# Complex filtering
high_impact = store.metadata['citations'] > 100
recent = store.metadata['year'] > 2020
combined_mask = high_impact & recent
filtered_vectors = np.array(store.vectors)[combined_mask]
```

## Contributing

### Setup Development Environment

```bash
git clone https://github.com/tvanreenen/numpy-vector-store.git
cd numpy-vector-store
uv sync --frozen --group dev
```

### Before Submitting a Pull Request

Please ensure:

1. **Code Quality**: Run `uv run ruff check` - should show no issues
2. **Formatting**: Run `uv run ruff format` - should show "files left unchanged"
3. **Type Checking**: Run `uv run mypy src/` - should show no errors
4. **Tests**: Run `uv run pytest` - all tests should pass

## License

MIT License - see [LICENSE](LICENSE) file for details.
