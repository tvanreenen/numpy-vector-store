# NumPy Vector Store

A fast, lightweight, zero-setup in-memory vector store powered by NumPy.

- **Tiny local vector search** for projects that do not need a vector database
- **Fast exact vector search** using vectorized NumPy operations
- **Simple typed API** returning `VectorHit(index, value, metadata)`
- **Composable filtering** by passing prefiltered row indexes with `within_rows`
- **Portable persistence** as trusted local `.npz` files with `vectors` + `metadata`
- **No framework opinions**: bring your own embeddings, chunking, async, and metadata model

## Why?

This library is purpose-built for small to medium-scale vector search tasks and
offers a simple alternative to heavyweight vector databases when you do not need
network services, indexing infrastructure, ingestion pipelines, or domain-specific
metadata filtering.

## When/Where?

Below are benchmark results for cosine similarity search to help you assess its
suitability for your use case.

| Embedding Type | Dimensions | ~5ms | ~25ms | ~100ms | ~500ms |
|----------------|------------|------|--------|---------|---------|
| **Sentence Transformers** | 384 | 1K vectors<br/>1.5MB | 10K vectors<br/>15MB | 100K vectors<br/>147MB | 500K vectors<br/>732MB |
| **OpenAI Small** | 1536 | 500 vectors<br/>3MB | 5K vectors<br/>29MB | 25K vectors<br/>147MB | 100K vectors<br/>586MB |
| **OpenAI Large** | 3072 | 200 vectors<br/>2MB | 2.5K vectors<br/>29MB | 5K vectors<br/>59MB | 25K vectors<br/>293MB |

*Benchmarks performed on Apple M2 hardware.*

## Installation

```bash
uv add numpy-vector-store
```

## Quick Start

```python
import numpy as np
from numpy_vector_store import VectorStore

store = VectorStore[dict[str, str]](dimensions=3)

store.add(
    vectors=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]),
    metadata=[
        {"title": "x-axis"},
        {"title": "y-axis"},
        {"title": "z-axis"},
    ],
)

hits = store.cosine_search(
    query=np.array([0.9, 0.1, 0.0]),
    top_k=2,
)

for hit in hits:
    print(f"{hit.metadata['title']}: {hit.value:.3f}")
```

`metadata` is an opaque row payload returned with hits. It can be a dict,
dataclass, string, integer row ID, or any other Python object that fits your
application.

## Normalization

`VectorStore` defaults to `normalize=True`, which scales each stored vector to
length `1`. Normalization preserves vector direction while discarding magnitude:

```python
[3.0, 4.0] -> [0.6, 0.8]
```

This is the default because it makes cosine similarity fast and direction-only,
which is the common case for semantic embeddings. Use `normalize=False` when
vector length matters, such as when magnitude encodes strength, confidence,
counts, scale, or raw geometry.

Zero vectors are rejected when `normalize=True` because they cannot be scaled to
unit length. Raw stores accept zero vectors for dot-product and Euclidean
search. Because cosine similarity is undefined for zero vectors,
`cosine_search` raises an error when its selected rows include one; use
`within_rows` to exclude zero rows when needed.

### Numerical inputs

Stored vectors use `float32` to keep the store compact. Vectors and queries must
remain finite when converted to `float32`, and search thresholds must also be
finite. Invalid values are rejected before they can affect stored state or
ranking.

Norms and raw metric values use `float64` accumulation where `float32`
intermediate calculations could overflow or underflow. This allows finite
`float32` vectors across the representable magnitude range to be normalized and
compared reliably.

| Method | `normalize=True` default | `normalize=False` |
|---|---|---|
| `cosine_search` | True cosine similarity over stored unit vectors; fastest/default path for embeddings | True cosine similarity over raw vectors; computes vector norms during search |
| `dot_search` | Dot product of unit vectors, effectively equivalent to cosine similarity | True dot product over original vectors; use when magnitude should affect ranking |
| `euclidean_search` | Distance between normalized directions; useful only when direction-normalized distance is intended | True Euclidean distance over original vectors; use for geometric/feature-space nearest neighbors |
| `get` | Returns normalized vectors | Returns original vectors |
| `save` | Saves normalized vectors | Saves raw vectors |
| `load` | Loads and normalizes vectors | Loads vectors exactly as stored |

## Search Methods

Use `cosine_search` for semantic embeddings and direction-only similarity:

```python
hits = store.cosine_search(query, top_k=10, min_value=0.75)
```

Use `dot_search` with `normalize=False` when larger-magnitude vectors should
rank higher:

```python
store = VectorStore[dict[str, str]](dimensions=3, normalize=False)
store.add(vectors, metadata)
hits = store.dot_search(query, top_k=10, min_value=0.0)
```

Use `euclidean_search` with `normalize=False` for raw coordinate or feature-space
nearest-neighbor search:

```python
store = VectorStore[dict[str, str]](dimensions=3, normalize=False)
store.add(vectors, metadata)
hits = store.euclidean_search(query, top_k=10, max_value=1.5)
```

## Prefiltering

The store does not implement a metadata query language. To filter by metadata,
produce row indexes first, then pass them with `within_rows`.

```python
rows = [
    i
    for i, metadata in enumerate(store.metadata)
    if metadata["title"].startswith("x")
]

hits = store.cosine_search(query, top_k=10, within_rows=rows)
```

For structured NumPy metadata, use NumPy to produce the row indexes:

```python
metadata_table = np.array(
    [
        ("intro", "A", 2024),
        ("setup", "A", 2023),
        ("guide", "B", 2024),
    ],
    dtype=[("title", "U20"), ("product", "U10"), ("year", "i4")],
)

store = VectorStore[int](dimensions=3)
store.add(vectors, metadata=np.arange(len(metadata_table)))

mask = (metadata_table["product"] == "A") & (metadata_table["year"] >= 2024)
rows = np.flatnonzero(mask)

hits = store.cosine_search(query, within_rows=rows)

for hit in hits:
    row = metadata_table[hit.metadata]
    print(row["title"], hit.value)
```

## Persistence

Pass a `file_path` and call `save()` / `load()` explicitly:

```python
store = VectorStore[dict[str, str]](dimensions=1536, file_path="vectors.npz")
store.add(embeddings, metadata)
store.save()

loaded = VectorStore[dict[str, str]](dimensions=1536, file_path="vectors.npz")
loaded.load()
```

If you save with `normalize=False`, load with `normalize=False` too:

```python
store = VectorStore[dict[str, str]](
    dimensions=1536,
    file_path="raw-vectors.npz",
    normalize=False,
)
store.add(raw_vectors, metadata)
store.save()

loaded = VectorStore[dict[str, str]](
    dimensions=1536,
    file_path="raw-vectors.npz",
    normalize=False,
)
loaded.load()
```

Context manager usage auto-saves on exit:

```python
with VectorStore[dict[str, str]](dimensions=1536, file_path="vectors.npz") as store:
    store.add(embeddings, metadata)
```

Persistence uses a minimal NumPy `.npz` contract with `vectors` and `metadata`
arrays. The `.npz` file does not encode the `normalize` setting; choose the same
setting when loading that you used when saving. Loading validates shape,
dimensions, row counts, and zero-norm vectors. It also uses `allow_pickle=True`
for flexible Python metadata payloads, so only load files generated by your own
application or another trusted local process. Loading untrusted `.npz` files is
not a supported security model.

## Compatibility

This project is still pre-1.0, so occasional breaking changes are expected while
the API stabilizes. Breaking changes are documented in GitHub release notes.
Deprecated APIs will keep warning for at least one point release before removal.

Supported Python versions are listed in the package classifiers and exercised
in CI. The project generally retains stable CPython versions until their
upstream end-of-life, adds new versions after its dependencies and CI support
them, and drops versions only in minor releases.

See the [project roadmap](ROADMAP.md) for the planned path to stable API and
persistence contracts.

## Contributing

```bash
git clone https://github.com/tvanreenen/numpy-vector-store.git
cd numpy-vector-store
uv sync --frozen --group dev
```

Before submitting a pull request:

1. Run `uv run ruff check`
2. Run `uv run ruff format --check`
3. Run `uv run mypy src/`
4. Run `uv run pytest`

## License

MIT License - see [LICENSE](LICENSE) file for details.
