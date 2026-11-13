# NumPy Vector Store

[Documentation](https://vectors.tvr.works/) · [API reference](https://vectors.tvr.works/reference/)

A fast, lightweight, zero-setup in-memory vector store powered by NumPy.

- **Tiny local vector search** for projects that do not need a vector database
- **Fast exact vector search** using vectorized NumPy operations
- **Simple typed API** returning `VectorHit(index, value, metadata)`
- **Composable filtering** by passing prefiltered row indexes with `within_rows`
- **Single-file persistence** as versioned, self-describing trusted local
  `.npz` archives
- **No framework opinions**: bring your own embeddings, chunking, async, and metadata model

## Is this a fit?

Use NumPy Vector Store when your collection fits comfortably in one Python
process, exact search meets your latency needs, and your application already
owns its embeddings and metadata. It is designed for small-to-medium collections
that do not need a separate database or search service.

Choose an indexed or service-backed system when you routinely work with millions
of vectors, need approximate search, or require shared access across processes
or machines. Transactions, coordinated concurrent writers, and a metadata query
engine are outside this library's scope. See [Performance](#performance) for
measured reference points and memory costs.

## Installation

```bash
uv add numpy-vector-store
```

With pip:

```bash
python -m pip install numpy-vector-store
```

## Quick Start

<!-- --8<-- [start:quickstart] -->

```python
from numpy_vector_store import VectorStore

store = VectorStore[str](dimensions=3)
store.add(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ["x-axis", "y-axis", "z-axis"],
)

for hit in store.cosine_search([0.9, 0.1, 0.0], top_k=2):
    print(f"{hit.metadata}: {hit.value:.3f}")
```

<!-- --8<-- [end:quickstart] -->

`metadata` is an outer sequence with one opaque payload for each vector row.
Each payload can be a dict, dataclass, tuple, list, string, integer row ID, or
another Python object that fits your application. Tuple and list payloads remain
single row values rather than being interpreted as additional array dimensions.

The supported top-level package names are `VectorStore`, `VectorHit`, and
`__version__`. See the [public API and compatibility policy](COMPATIBILITY.md)
for the complete interface that the project intends to stabilize at 1.0.

## Core concepts

### Search and normalization

Search by direction with `cosine_search`, by dot product with `dot_search`, or
by distance with `euclidean_search`. All searches score every selected row.
Cosine and dot results rank larger values first; Euclidean results rank smaller
distances first. Exact computed-value ties use the lower original row index,
including at the `top_k` boundary.

The default `normalize=True` stores unit vectors and normalizes queries.
Choose `normalize=False` for magnitude-sensitive dot products or raw Euclidean
geometry. Vectors use `float32` and must remain finite after conversion.
Complex values are rejected. Normalized stores and cosine queries reject zero
vectors. See the [search guide](https://vectors.tvr.works/search/) for metric
selection, thresholds, and input rules.

### Filter candidate rows

Filter metadata in your application, then pass unique integer row indexes as
`within_rows`. Unfiltered search uses the stored matrix directly; filtered
search gathers a temporary matrix. There is no metadata query language.

```python
rows = [i for i, payload in enumerate(store.metadata) if payload.startswith("x")]
hits = store.cosine_search([1.0, 0.0, 0.0], within_rows=rows)
```

See [filtering examples](https://vectors.tvr.works/search/#filter-candidates-before-searching)
for application predicates and NumPy Boolean masks.

### Persist trusted local archives

Save explicitly, then open a new store or refresh one already bound to a file:

```python
store.save("vectors.npz")       # Write and bind a destination.
store.save()                    # Update the bound archive.
loaded = VectorStore[str].open("vectors.npz")
loaded.reload()                 # Discard memory and reread the archive.
```

Only open trusted archives: metadata uses NumPy's pickle-backed object arrays.
Loading untrusted data can execute code. Format-version-1 archives are
self-describing and retain dimensions and normalization settings. Relative paths
are anchored when bound; later working-directory changes do not redirect them.
Writes are always explicit; there is no autosave context manager. See
[persistence](https://vectors.tvr.works/persistence/) for path handling, archive
compatibility, and failure behavior.

### State and concurrency

Configuration and row inspection properties are read-only. `get(index)` returns
an independent vector copy and the shared metadata payload, or `None` when the
integer row index is outside the store. Inspection views describe the rows
present when requested; refresh them after mutation. Mutable metadata remains
application-owned and shared by reference.

Concurrent reads are supported only while store state and shared metadata stay
unchanged. If any thread may mutate or save, synchronize **every access** to the
store with the same application-level lock. Separate writers to the same archive
also need coordination; atomic file replacement is not a lock or a transaction.
See [state, views, and threading](https://vectors.tvr.works/ownership/) for
ownership and synchronization rules.

## Performance

<!-- --8<-- [start:performance] -->

`VectorStore` performs exact search over every selected row. The following
measurements are reference points, not latency guarantees:

Each cell shows the median time per cosine query followed by the stored vector
matrix size:

| Rows | 384 dimensions | 1,536 dimensions | 3,072 dimensions |
|---:|---:|---:|---:|
| 1,000 | 0.032 ms · 1.5 MB | 0.049 ms · 6.1 MB | 0.088 ms · 12.3 MB |
| 10,000 | 0.286 ms · 15.4 MB | 1.065 ms · 61.4 MB | 2.122 ms · 122.9 MB |
| 100,000 | 2.983 ms · 153.6 MB | 9.877 ms · 614.4 MB | 20.509 ms · 1.23 GB |

These benchmarks intentionally stop at 100,000 rows. NumPy Vector Store is
designed for small-to-medium, in-process exact search; 100,000 rows is an upper
reference, not a promised limit or a target for continued scaling. The practical
boundary depends on vector dimensions, metadata, available memory, and latency
requirements. Workloads that routinely reach millions of vectors generally
need an indexed or service-backed system.

These are unfiltered `top_k=10` searches on a normalized store. Each row divides
the median duration of seven measured 20-query trials by 20, after two discarded
warmup trials. The vector matrix size excludes metadata, temporary search
arrays, and Python process overhead.

The measurements were taken from commit `4b23810` on a 24 GB Apple M4 Mac mini
with macOS 26.6.1, CPython 3.13.5, NumPy 2.3.3, and Accelerate BLAS. Hardware,
operating system activity, Python and NumPy versions, BLAS implementation, and
thread settings can all change the result.

The repository includes benchmark commands that emit the inputs, environment,
raw samples, and median as JSON:

```bash
uv run python benchmarks/benchmark.py search \
  --rows 10000 --dimensions 384 --queries 20 --top-k 10 \
  --warmup 2 --repetitions 7 > /tmp/nvs-search.json

uv run python benchmarks/benchmark.py ingest \
  --rows 10000 --dimensions 384 --batch-size 1 \
  --warmup 2 --repetitions 7 > /tmp/nvs-ingest.json
```

For the same 10,000-by-384 prepared input, repeated single-row ingestion had a
median of 78.6 ms, or about 127,000 rows per second. Supplying 1,000 rows per
`add()` call had a median of 6.63 ms, or about 1.51 million rows per second.
Both measurements include construction of a fresh normalized store and every
`add()` call, but exclude input generation.

See [the benchmark guide](benchmarks/README.md) for the timed regions, complete
options, output fields, and interpretation notes. The project keeps structural
complexity checks in CI, but does not fail shared runners on wall-clock timing.

<!-- --8<-- [end:performance] -->

## Compatibility

This project is still pre-1.0, so occasional breaking changes are expected while
the API stabilizes. Changes are documented in the [changelog](CHANGELOG.md) and
GitHub release notes. Deprecated APIs will keep warning for at least one point
release before removal.

Version 0.7 publishes the proposed 1.x contract for an adoption cycle before
1.0. The [public API and compatibility policy](COMPATIBILITY.md) identifies the
supported top-level surface, deprecation rules, archive guarantees, runtime
support policy, and details that remain implementation-specific.

Version 0.7 supports Python 3.11 through 3.14 and NumPy 1.23.2 or newer. These
versions are listed in the package metadata and exercised in CI, including a
dedicated check against the minimum NumPy version. Python 3.10 remains supported
by the 0.3 release series but is not supported by 0.4 or later.

The project generally retains stable CPython versions until their upstream
end-of-life, adds new versions after its dependencies and CI support them, and
drops versions only in minor releases.

See the [changelog](CHANGELOG.md) for release history and the
[project roadmap](ROADMAP.md) for the planned path to stable API and persistence
contracts. Persistence users upgrading from the 0.3 API should also read the
[migration guide](MIGRATION.md).

## Contributing

```bash
git clone https://github.com/tvr-works/numpy-vector-store.git
cd numpy-vector-store
uv sync --frozen --group dev
```

Before submitting a pull request:

1. Run `uv lock --check`
2. Run `uv run --locked ruff check .`
3. Run `uv run --locked ruff format --check .`
4. Run `uv run --locked mypy src/ benchmarks/ tools/`
5. Run `uv run --locked pytest`

The [documentation contributor guide](https://github.com/tvr-works/numpy-vector-store/blob/main/DOCUMENTATION.md)
explains the site sources, local preview, and documentation checks.

Maintainers should also read the [release and repository transfer runbook](RELEASING.md) before preparing a release or changing publication infrastructure.

## License

MIT License - see [LICENSE](LICENSE) file for details.
