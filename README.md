# NumPy Vector Store

A fast, lightweight, zero-setup in-memory vector store powered by NumPy.

- **Tiny local vector search** for projects that do not need a vector database
- **Fast exact vector search** using vectorized NumPy operations
- **Simple typed API** returning `VectorHit(index, value, metadata)`
- **Composable filtering** by passing prefiltered row indexes with `within_rows`
- **Portable persistence** as versioned, self-describing trusted local `.npz` files
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

`metadata` is an outer sequence with one opaque payload for each vector row.
Each payload can be a dict, dataclass, tuple, list, string, integer row ID, or
another Python object that fits your application. Tuple and list payloads remain
single row values rather than being interpreted as additional array dimensions.

## State ownership

The store owns its configuration and row structure. `dimensions`, `normalize`,
and `file_path` are readable properties, but callers cannot assign them
directly. Use the constructor for configuration, `open(path)` to open an
archive, and `save(path)` to establish or change a binding.

`store.vectors` and `store.metadata` are zero-copy, non-writeable NumPy views
for inspection and metadata prefiltering. Direct item assignment and ordinary
attempts to enable writes are rejected. Request a fresh view after `add()`,
`clear()`, or `reload()` when current rows are required.

These views prevent accidental mutation through the supported API; they are not
tamper-proof snapshots. NumPy exposes shared buffers, and Python private state
can be reached deliberately. Mutating backing storage through `.base`, private
attributes, `ctypes`, or similar escape hatches is unsupported and can corrupt
store invariants. Use `.copy()` when code needs an independently mutable
full-array snapshot:

```python
vectors = store.vectors.copy()
metadata_rows = store.metadata.copy()
```

Copying `metadata` isolates the outer row array but still shares the opaque
payload objects stored inside it.

`get(index)` has a narrower ownership boundary:

```python
vector, payload = store.get(0)

vector[0] = 10.0           # Independent copy; the store is unchanged.
payload["reviewed"] = True  # Shared application metadata object.
```

The returned vector is an independent `float32` copy. The metadata payload is
the same opaque object supplied to `add()` or restored from the archive; the
store protects the metadata row structure but does not deep-copy arbitrary
dicts, lists, dataclasses, or application objects. Applications that require
immutable payloads can use frozen objects or copy them at their own boundary.

## Repeated additions

`add()` accepts one row or a batch, and preserves insertion order in either
case. The store keeps private spare capacity so repeated small additions do not
copy every existing row on every call. When that capacity is full, the vector
and metadata arrays grow together and the active rows are copied once.

Spare capacity is internal. `len(store)`, inspection views, search, `get()`, and
saved archives contain only rows that were added. `clear()` releases both the
active rows and any reserved capacity held by the store. As with any NumPy
view, an older inspection view keeps its previous buffer alive until that view
is released. Adding a batch is still preferable when the application already
has one because it also reduces per-call validation and Python overhead.

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
| `open` and `reload` | Restore normalized storage semantics | Restore raw vectors exactly as stored |

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

### Result ordering

Cosine and dot-product results are ordered from larger values to smaller
values. Euclidean results are ordered from smaller distances to larger ones.
When two computed values are exactly equal, the row with the lower original
store index comes first.

The same rule determines which tied rows cross the `top_k` boundary. It also
applies to `within_rows`: the original store index breaks a tie, regardless of
the order in which filtered row indexes were supplied. Values that are close
but not exactly equal remain ordered by their computed metric value.

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

Searches without `within_rows` compute directly against the stored vector matrix
and do not make a full copy of it. A filtered search gathers the selected rows
into a temporary matrix, so its additional memory use scales with the number of
selected rows and the vector dimensions. Omit `within_rows` when every row
should be searched; passing every row explicitly would create an unnecessary
full-size temporary matrix.

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

## Thread safety

`VectorStore` does not use internal locks. Multiple threads may call search,
`get()`, or the inspection properties on the same instance while its rows,
configuration, binding, and metadata payloads remain unchanged.

If any thread may call `add()`, `clear()`, `reload()`, or `save()`, every access
to that store must use the same application-level lock or another external
synchronization mechanism. The same rule applies when application code mutates
a metadata payload shared with the store. In particular, do not overlap a save
with an in-memory mutation: an atomic file replacement cannot turn two separate
array reads into a consistent store snapshot.

Separate store instances writing the same path also need external writer
coordination. Atomic replacement prevents readers from seeing a partially
written archive, but concurrent writers can replace one another and the last
successful replacement wins.

## Persistence

Create a new store normally, then supply its destination on the first save:

```python
store = VectorStore[dict[str, str]](dimensions=1536)
store.add(embeddings, metadata)
store.save("vectors.npz")
```

`save(path)` writes the archive and binds that path to the store. Later
`save()` calls update the bound archive. Passing another path performs a Save
As operation; the new path becomes the binding only after the write succeeds.

Open an existing store directly from its self-describing archive:

```python
store = VectorStore[dict[str, str]].open("vectors.npz")
```

`open()` restores `dimensions` and `normalize` from the archive and binds its
path. Use `reload()` when the file may have changed externally and you
explicitly want to discard current in-memory changes:

```python
store.reload()
```

`reload()` always rereads the bound archive and raises if the store is unbound,
the file is missing, or the archive is invalid. A failed reload leaves the
current in-memory vectors and metadata unchanged.

The `.npz` suffix may be omitted. An extensionless path such as `"vectors"` is
resolved to `"vectors.npz"` for saving, opening, and reloading.

Raw-vector configuration is also restored from the archive:

```python
store = VectorStore[dict[str, str]](
    dimensions=1536,
    normalize=False,
)
store.add(raw_vectors, metadata)
store.save("raw-vectors.npz")

loaded = VectorStore[dict[str, str]].open("raw-vectors.npz")
assert loaded.normalize is False
```

Archives written by 0.4 use format version 1 and contain `format_version`,
`dimensions`, `normalize`, `vectors`, and `metadata`. The stored configuration
prevents an archive from being loaded with different dimensions or normalization
semantics.

Opening and reloading validate the complete schema, array dtypes and shapes, row
counts, finite vector values, and zero-norm behavior before changing in-memory
state. Opaque metadata values remain individual row payloads across persistence
round trips.

Each save writes a uniquely named temporary archive in the destination
directory, closes it, and then replaces the destination with `os.replace`.
Readers opening the destination path therefore see either the previous complete
archive or the new complete archive rather than a partially written file. If
writing or replacement fails, the previous destination remains in place and the
temporary file is removed.

Atomic replacement is not file locking or multi-writer coordination. Concurrent
writers can still replace one another, and the library does not promise that a
successful save has reached durable hardware storage across every operating
system or power failure.

Version 0.5 reads only self-describing format version 1 archives. Unversioned
archives containing only `vectors` and `metadata` cannot be opened because they
do not record dimensions or normalization semantics. Recreate those archives
from source data, or convert them with NumPy Vector Store 0.4 before upgrading.

Constructor `file_path=`, instance `load()`, and context-manager persistence
were removed in 0.5. See the [persistence migration guide](MIGRATION.md) for the
direct replacements and the pre-upgrade procedure for an unversioned archive.

Metadata persistence uses `allow_pickle=True` for flexible Python payloads, so
only load files generated by your own application or another trusted local
process. Loading untrusted `.npz` files is not a supported security model.

## Compatibility

This project is still pre-1.0, so occasional breaking changes are expected while
the API stabilizes. Changes are documented in the [changelog](CHANGELOG.md) and
GitHub release notes. Deprecated APIs will keep warning for at least one point
release before removal.

Version 0.4 supports Python 3.11 through 3.14 and NumPy 1.23.2 or newer. These
versions are listed in the package metadata and exercised in CI, including a
dedicated check against the minimum NumPy version. Python 3.10 remains supported
by the 0.3 release series but is not supported by 0.4.

The project generally retains stable CPython versions until their upstream
end-of-life, adds new versions after its dependencies and CI support them, and
drops versions only in minor releases.

See the [changelog](CHANGELOG.md) for release history and the
[project roadmap](ROADMAP.md) for the planned path to stable API and persistence
contracts. Persistence users upgrading from the 0.3 API should also read the
[migration guide](MIGRATION.md).

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
