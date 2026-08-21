# Roadmap

NumPy Vector Store is intentionally small: it provides exact in-memory vector
search without adding a service, indexing system, or metadata query language.
This roadmap explains how the project will make that focused core safer and
more predictable on the way to a stable 1.0 release.

The roadmap describes direction rather than delivery dates. Priorities may
change as the library receives real-world feedback, and each release will have
its final scope documented in its release notes.

## Versioning approach

The project follows semantic versioning while it is pre-1.0:

- Patch releases such as 0.3.2 fix defects and improve existing documented
  behavior without intentionally breaking valid usage.
- Minor releases such as 0.4.0 may change behavior or file formats when that is
  necessary to establish safer long-term contracts. Those changes will be
  called out clearly and will preserve compatibility where practical.
- The 1.0.0 release will mark a commitment to a stable public API and
  persistence contract.

Deprecated APIs will continue to warn for at least one point release before
removal. Persisted data will receive an explicit compatibility and migration
story before the project reaches 1.0.

## Runtime support policy

The Python versions listed in the package classifiers are part of the public
compatibility contract and are exercised in CI. The project generally:

- Supports stable CPython versions until their upstream end-of-life.
- Adds a newly stable Python version after NumPy and the project's test suite
  support it.
- Drops a Python version only in a minor release and calls out the change in
  release notes.

The 0.3 series supports Python 3.10 through 3.14. Version 0.4 supports Python
3.11 through 3.14, using the minor-release boundary to remove Python 3.10 ahead
of its upstream end-of-life in October 2026. NumPy 1.23.2 is the minimum for
0.4 because it is the earliest NumPy release that supports Python 3.11.

Supported NumPy versions are also part of the runtime contract. The declared
minimum should be installable on the oldest supported Python version and should
have a dedicated minimum-dependency CI check. The regular Python matrix will
continue to test the versions selected by the project's lockfile.

## 0.3.2: Reliability and performance

Status: complete

This patch release focuses on cases where the current API can silently produce
incorrect results, fail to reload data it saved, reject documented metadata
payloads, or allocate much more memory than an exact search requires.

Delivered changes:

- Validate vectors and queries for finite numeric values.
- Normalize large finite vectors without overflowing intermediate `float32`
  calculations.
- Accumulate raw metric values in `float64` when `float32` intermediates could
  overflow.
- Permit zero vectors in raw stores used for dot-product or Euclidean search,
  while keeping cosine similarity's undefined zero-vector behavior explicit.
- Make extensionless persistence paths save and load the same `.npz` file.
- Allow loading to be retried when the persistence file was initially absent.
- Preserve tuple, list, dataclass, scalar, and dictionary metadata as individual
  opaque row payloads.
- Avoid copying the complete vector matrix during an unfiltered search.
- Add continuous validation across every supported Python version and require
  equivalent checks before publishing.
- Align the minimum NumPy requirement with Python 3.10 support and add a
  minimum-dependency compatibility check.

These changes are intended to preserve results and behavior for valid existing
usage. Public state access, the persistence format, and context-manager
semantics will not change in this patch release.

## 0.4.0: Persistence and lifecycle

Status: complete

The earlier persistence format stored only vectors and metadata. That format
was compact, but callers had to separately remember dimensions and
normalization mode. Version 0.4 introduces a self-describing format that can
reject incompatible configuration instead of silently changing vector
semantics, with one deliberately short migration window for existing archives.

### Versioned archive contract

Every archive written by 0.4 uses format version 1 and contains five named
values:

- `format_version`: a scalar integer identifying version 1 of the archive
  contract. This version is independent of the package version.
- `dimensions`: a positive scalar integer matching the width of `vectors`.
- `normalize`: a scalar boolean recording whether stored vectors use normalized
  or raw semantics.
- `vectors`: a two-dimensional `float32` array.
- `metadata`: a one-dimensional object array with one payload per vector row.

Opening an archive validates the complete schema before changing live store
state. Missing fields, unsupported format versions, invalid configuration, and
inconsistent array shapes fail clearly instead of being guessed at.

Object metadata continues to rely on NumPy's pickle-backed object-array
loading. Persistence therefore remains a trusted-file feature: users must not
open archives from untrusted or unverifiable sources.

### Existing archive migration

Archives created before 0.4 contain only `vectors` and `metadata`, so they
cannot recover their original dimensions and normalization mode by themselves.
Version 0.4 keeps a temporary reader for these archives through the legacy
configuration-aware API. Opening one emits a `FutureWarning`, and its next save
rewrites it in the version 1 format.

This legacy reader will be removed in 0.5. Users who need an old archive after
upgrading should open and save it once with 0.4, or recreate it from its source
data. The project intentionally does not promise indefinite compatibility for
the incomplete two-array format.

### Preferred persistence API

Version 0.4 introduces the lifecycle intended to become the only persistence
API in 0.5:

```python
store = VectorStore(dimensions=1536, normalize=True)
store.add(vectors, metadata)
store.save("vectors.npz")

store = VectorStore.open("vectors.npz")
store.reload()
store.save()
```

The preferred constructor creates a new empty store and performs no disk I/O.
`VectorStore.open(path)` creates a store from the archive's own
configuration and binds that path. `save(path)` performs the first save or a
Save As operation and binds the supplied path; later `save()` calls update that
bound archive. `reload()` strictly refreshes a bound store from disk and leaves
the current in-memory state unchanged if reading or validation fails.

The generic parameter on `VectorStore` describes the application's metadata
payload type, not a built-in document abstraction. Examples that benefit from
an explicit type will use a descriptive application type such as
`ChunkMetadata`; otherwise they will omit the generic annotation.

### Short API compatibility window

The preferred API coexists in 0.4 with three old entry points: constructor
`file_path=`, instance `load()`, and direct context-manager use. Each emits a
`FutureWarning` in 0.4 and will be removed in 0.5. This single-release bridge is
intended to make migration obvious without carrying two persistence models
long term.

While the deprecated context manager remains, it saves only after normal
completion. If the managed block raises, the store does not save and the
exception propagates unchanged. No replacement autosave context manager is
planned; explicit `save()` calls make the persistence boundary easier to see
and reason about.

### Atomic save boundary

Saving writes and closes a temporary archive in the destination directory
before replacing the destination with `os.replace`. Readers therefore see
either the previous complete archive or the new complete archive, and a failed
write leaves the previous destination intact. Temporary files are cleaned up
after failures.

This is an atomic visibility guarantee, not a concurrency system. Version 0.4
does not add file locking, coordinate multiple writers, or promise survival of
every hardware or operating-system failure before data reaches durable storage.

## 0.5.0: State safety and ingestion

Status: in progress

The 0.4 release established the persistence lifecycle and archive contract.
Version 0.5 will finish that transition, then address a separate ownership
problem: public arrays currently let callers change normalized vectors, resize
row storage, or separate vectors from their metadata without validation.

### API at a glance

The intended public surface remains small. Most 0.4 call sites continue to work
unchanged:

```python
store = VectorStore(dimensions=1536, normalize=True)
store.add(vectors, metadata)

hits = store.cosine_search(query, top_k=10)
hits = store.dot_search(query, top_k=10)
hits = store.euclidean_search(query, top_k=10)

row = store.get(0)
store.clear()

store.save("vectors.npz")
store.save()

loaded = VectorStore.open("vectors.npz")
loaded.reload()
```

Configuration and rows remain available for inspection:

```python
store.dimensions
store.normalize
store.file_path
store.vectors
store.metadata
len(store)
```

The complete change from 0.4 is:

| Area | 0.4 | 0.5 |
|---|---|---|
| Create a store | `VectorStore(dimensions, normalize=...)` | Unchanged |
| Constructor `file_path=` | Works with `FutureWarning` | Removed |
| Open an archive | `VectorStore.open(path)` | Unchanged |
| Save and bind | `save(path)` | Unchanged |
| Save again | `save()` | Unchanged |
| Refresh from disk | `reload()` | Unchanged |
| Instance `load()` | Works with `FutureWarning` | Removed |
| Context manager | Works with `FutureWarning` | Removed |
| Unversioned archive | Temporary migration reader | Reader removed |
| Format version 1 archive | Supported | Supported unchanged |
| `dimensions`, `normalize`, `file_path` | Writable attributes | Read-only properties |
| `vectors`, `metadata` | Writable owning arrays | Read-only active-row views |
| Vector returned by `get()` | View into live storage | Independent copy |
| `add()` | Recopies existing rows | Amortized capacity growth |
| Equal search values | Unspecified order | Lower store row index first |
| Thread safety | Not formally defined | Read-only concurrency documented |

The package continues to expose `VectorStore` and `VectorHit`; 0.5 does not add
a document, state, snapshot, builder, or storage class.

### Persistence bridge removal

Version 0.5 will remove the four compatibility paths that warned throughout
0.4:

- Constructor `file_path=`. New stores use
  `VectorStore(dimensions, normalize=...)`, then `save(path)`.
- Instance `load()`. Existing version 1 archives use `VectorStore.open(path)`,
  and an already-open store uses `reload()`.
- Direct context-manager persistence. Explicit `save()` calls remain the only
  persistence boundary.
- The reader for unversioned archives containing only `vectors` and `metadata`.

The archive format itself does not change. Version 0.5 continues to read and
write format version 1. Anyone who still needs an unversioned archive must
migrate it once with 0.4 using the archive's original dimensions and
normalization mode, or recreate it from source data. Version 0.5 will not guess
configuration that the old file did not record.

The readable `file_path` attribute remains as the name of a store's current
binding; removing the constructor argument does not require a second `path`
convention. Only `open(path)` and a successful `save(path)` establish or change
that binding.

### Store-owned configuration and rows

Version 0.5 will move configuration, path binding, vectors, and metadata behind
private state. The existing public names remain available for inspection:

- `dimensions`, `normalize`, and `file_path` become read-only properties.
  Callers can inspect the store's configuration and binding but cannot bypass
  the constructor, archive validation, or Save As behavior by assigning to
  them.
- `vectors` returns a zero-copy, non-writeable NumPy view over the active vector
  rows. Spare ingestion capacity, if any, is never exposed.
- `metadata` returns a zero-copy, non-writeable one-dimensional object-array
  view over the active metadata rows.

These array views describe the store at the time they are requested. Code that
keeps a view across `add()`, `clear()`, or `reload()` must request a new view
before assuming it represents current rows. Direct item assignment and ordinary
attempts to enable writes are rejected.

The views protect against accidental mutation through the supported API; they
are not tamper-proof snapshots. Deliberately mutating backing storage through
`.base`, private attributes, `ctypes`, or similar escape hatches is unsupported
and can corrupt store invariants. Callers that need independently mutable
full-array snapshots can use `.copy()`. Copying `metadata` isolates its outer
row array but continues to share the opaque payload objects.

Read-only metadata protects the store's row structure, not the contents of an
opaque Python payload. A dict, list, dataclass, or application object remains
the same object supplied by the caller. The library will not deep-copy or
freeze arbitrary metadata objects.

In practice, matrix inspection and row retrieval have different ownership:

```python
vectors = store.vectors
metadata = store.metadata

vectors[0, 0] = 10.0       # Rejected: read-only view
metadata[0] = replacement  # Rejected: read-only row structure

vector, payload = store.get(0)
vector[0] = 10.0           # Allowed: independent copy
payload["reviewed"] = True  # Allowed: shared opaque metadata object
```

Applications that need immutable metadata can store frozen application objects
or copy payloads at their own boundary. The vector store will not impose one
copying policy on every metadata type.

### Safe row retrieval

`get(index)` keeps its existing return shape: a `(vector, metadata)` tuple for a
valid row and `None` for an index outside the store. The vector becomes an
independent `float32` copy. Changing it cannot change the stored row, and later
store operations cannot invalidate it.

The metadata payload remains a shared opaque object, matching the behavior of
search hits and the `metadata` view. This distinction avoids an expensive and
often incorrect promise that the library knows how to copy application-defined
objects.

### Amortized repeated ingestion

Through 0.4, the implementation concatenated vectors and metadata on every
noninitial `add()`. Repeated single-row additions therefore recopied all earlier
rows each time, even though search ultimately needs one contiguous matrix.

Version 0.5 keeps private contiguous vector and metadata storage with an active
row count and spare capacity. When an incoming batch fits, `add()` writes it
into available storage. When it does not fit, the store grows geometrically and
copies active rows once. This makes repeated additions amortized instead of
moving the full store for every call, while preserving efficient first and
large-batch insertion.

Capacity is an implementation detail. Public inspection, `len()`, search,
`get()`, and persistence operate only on active rows. `clear()` returns the
store to empty storage and releases retained row capacity rather than keeping
references to application data indefinitely. The `add()` signature,
normalization, validation, and metadata-row semantics do not change.

Chunked storage is deliberately out of scope. It would make writes cheap by
moving concatenation into search and persistence, or require a second cached
representation with invalidation rules. One contiguous representation better
fits a library whose primary operation is exact NumPy search.

### Deterministic equal-value ordering

Version 0.5 defines one ordering rule for exact metric ties:

- Cosine and dot searches order larger values first.
- Euclidean search orders smaller values first.
- Rows with equal computed values are ordered by ascending original store row
  index.

The row-index tie break also determines which rows are included when equal
values cross the `top_k` boundary. Filtered searches use the original store
index rather than the position or order supplied through `within_rows`. Search
results therefore remain reproducible without changing how non-tied values are
ranked.

### Thread-safety boundary

`VectorStore` does not add internal locking in 0.5. Multiple threads may use
search, `get()`, and the read-only inspection properties on the same instance
while its state and metadata payloads remain unchanged.

Applications must provide their own synchronization whenever another thread
may call `add()`, `clear()`, `reload()`, or `save()`, or mutate a metadata
payload shared with the store. This includes serializing a save with in-memory
mutation so the archive cannot observe vectors and metadata at different
logical moments.

Atomic archive replacement remains a file-visibility guarantee, not an object
or multi-writer lock. Separate stores writing the same destination can still
replace one another, and the last successful replacement wins.

### Explicit non-goals

Version 0.5 will not add update or delete operations, a separate builder,
streaming or async ingestion, internal locks, or multi-writer coordination. It
will not deep-copy opaque metadata. These features are not needed to establish
safe ownership and amortized addition, and adding them now would widen the API
before the existing core reaches stabilization.

State encapsulation and spare capacity do not change serialized data, so format
version 1 remains sufficient. Broader validation and exception consistency stay
in the 0.6 stabilization milestone.

## 0.6.0: API stabilization

This release is intended to consolidate the earlier changes rather than add a
new feature family.

Planned direction:

- Make validation and exception behavior consistent across methods.
- Add performance regression coverage for representative store sizes.
- Exercise persistence upgrades and backwards compatibility.
- Resolve known high- and medium-priority defects.
- Complete documentation of ordering, memory use, concurrency, and trusted-file
  requirements.

## 1.0.0: Stable contracts

The project will be ready for 1.0 when:

- The public API has completed at least one minor release cycle without
  structural redesign.
- Public access cannot silently invalidate vector-store invariants.
- Persistence is self-describing and has a documented migration policy.
- Supported numeric inputs behave reliably across supported Python and NumPy
  versions.
- Search and ingestion complexity are documented and covered by regression
  tests.

Reaching 1.0 does not require turning the project into a vector database. The
library will continue to favor a small exact-search API over indexing services,
framework integrations, or a built-in metadata query language.
