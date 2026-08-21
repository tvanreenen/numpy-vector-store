# Changelog

This changelog records user-visible changes to NumPy Vector Store. Earlier
release notes remain available on the
[GitHub releases page](https://github.com/tvanreenen/numpy-vector-store/releases).

## 0.5.0 - 2026-08-21

This release gives `VectorStore` clear ownership of its configuration and row
storage, makes repeated additions scale without recopying the complete store on
every call, and defines deterministic ordering for equal search values. It also
finishes the persistence transition announced in 0.4: the explicit
create/open/save/reload lifecycle is now the only persistence API.

### API at a glance

The core workflow remains small:

```python
store = VectorStore(dimensions=1536, normalize=True)
store.add(vectors, metadata)

hits = store.cosine_search(query, top_k=10)
row = store.get(0)

store.save("vectors.npz")
store.save()

loaded = VectorStore.open("vectors.npz")
loaded.reload()
```

`VectorStore` and `VectorHit` remain the only public classes. Version 0.5 does
not add a document wrapper, builder, snapshot object, metadata query language,
or another persistence abstraction.

### Store-owned configuration and rows

- Make `dimensions`, `normalize`, and `file_path` read-only properties. The
  constructor owns configuration, while `open(path)` and a successful
  `save(path)` own archive binding changes.
- Return zero-copy, non-writeable active-row views from `vectors` and
  `metadata`. Direct item assignment and ordinary attempts to enable writes are
  rejected.
- Keep inspection views moment-in-time. Code holding a view across `add()`,
  `clear()`, or `reload()` must request a new one to inspect current rows.
- Return an independent `float32` vector copy from `get(index)`, so changing a
  retrieved vector cannot change normalized storage or a later search.
- Preserve opaque metadata payloads by reference. The read-only metadata view
  protects row alignment, not the contents of a caller-owned dict, list,
  dataclass, or other application object.

The views prevent accidental mutation through the supported API; they are not
tamper-proof snapshots. Deliberately reaching backing storage through `.base`,
private attributes, `ctypes`, or similar escape hatches remains unsupported.
Call `.copy()` when code needs an independently mutable array.

### Amortized repeated additions

- Replace whole-store concatenation on every noninitial `add()` with private
  contiguous vector and metadata capacity plus an active row count.
- Reuse spare rows when a new batch fits. When it does not, grow vector and
  metadata storage together and copy active rows once.
- Keep spare capacity out of `len()`, inspection, search, `within_rows`,
  retrieval, and saved archives.
- Make `clear()` return the store to empty arrays and drop the store's retained
  capacity. A caller-held older NumPy view may still keep its previous buffer
  alive until that view is released.

The `add(vectors, metadata)` signature, insertion order, validation,
normalization, and opaque metadata behavior do not change. Passing a batch is
still useful when the application already has one, but repeated small
additions no longer move every earlier row on each call.

### Deterministic search ties

- Continue ordering cosine and dot-product results from larger values to
  smaller values and Euclidean results from smaller distances to larger ones.
- Break exact computed-value ties by ascending original store row index.
- Apply the row-index tie break when choosing which rows cross the `top_k`
  boundary, not only when ordering an already selected subset.
- Use original store indexes for filtered searches, so shuffling the same
  `within_rows` values does not change tied results.
- Preserve partial top-k selection rather than replacing it with a full-store
  sort.

Only exactly equal computed values use the row-index tie break. Close but
unequal values remain ordered by their metric value.

### Final persistence lifecycle

The 0.4 compatibility window is now closed:

- Remove constructor `file_path=`. Create an in-memory store, then call
  `save(path)` to write and bind it.
- Remove instance `load()`. Use `VectorStore.open(path)` to construct a store
  from an archive and `reload()` to refresh a bound store.
- Remove context-manager persistence. Call `save()` explicitly where the
  application intends to persist state.
- Remove the reader for unversioned archives containing only `vectors` and
  `metadata`.

Archive format version 1 is unchanged. Applications that already use the 0.4
`open()`, `save(path)`, `save()`, and `reload()` lifecycle need no persistence
changes. An older unversioned archive must be converted with 0.4 using its
original dimensions and normalization mode, or recreated from source data,
before upgrading. See the [persistence migration guide](MIGRATION.md) for the
side-by-side replacements and conversion procedure.

### Thread safety and persistence boundaries

- Support concurrent search, `get()`, and inspection on one instance only
  while its state and shared metadata payloads remain unchanged.
- Require application-level synchronization for every access when any thread
  may call `add()`, `clear()`, `reload()`, or `save()`, or mutate shared
  metadata.
- Keep atomic archive replacement as a destination-visibility guarantee, not a
  store snapshot, file lock, or multi-writer coordination system.

Separate store instances writing the same path can still replace one another;
applications with multiple writers must serialize them. Metadata persistence
continues to use NumPy's pickle-backed object arrays, so archives remain trusted
input and must not be opened from untrusted or unverifiable sources.

### Runtime compatibility and upgrade notes

- Continue supporting Python 3.11 through 3.14 and NumPy 1.23.2 or newer.
- Continue exercising every supported Python version in CI, with a dedicated
  minimum-NumPy job on Python 3.11.
- Keep archive format version 1 readable and writable without a file migration.
- Expect `AttributeError` from code that assigns public configuration or row
  arrays, and different ordering from code that relied on incidental NumPy
  partition order for exact ties.
- Expect a migration before upgrading code that still uses constructor
  `file_path=`, instance `load()`, context-manager persistence, or an
  unversioned two-array archive.

## 0.4.0 - 2026-08-09

This release makes persistence explicit, self-describing, and safer to update.
The earlier archive format stored vectors and metadata but omitted the settings
needed to interpret those vectors correctly. Version 0.4 records that
configuration in every new archive and introduces a lifecycle that clearly
separates creating, opening, saving, and reloading a store.

### Explicit persistence lifecycle

- Add `VectorStore.open(path)` to construct a store from a versioned archive.
  The archive supplies its own dimensions and normalization mode, so callers no
  longer need to repeat configuration that may be wrong.
- Let `save(path)` perform the first save or a Save As operation and bind that
  destination. Later `save()` calls update the bound archive.
- Add `reload()` as a deliberate refresh from disk. It always attempts to read
  the bound archive and leaves current in-memory state unchanged if reading or
  validation fails.
- Keep creating a new in-memory store separate from opening one on disk. This
  makes file access and persistence boundaries visible in application code.

### Versioned, self-describing archives

- Write archive format version 1 with `format_version`, `dimensions`,
  `normalize`, `vectors`, and `metadata` fields.
- Validate the complete archive before changing live store state, including
  field names, scalar configuration, array dtypes and shapes, row counts,
  finite vector values, and normalized-store zero-vector rules.
- Reject unsupported format versions and malformed archives clearly rather
  than inferring missing configuration or partially applying valid fields.
- Continue preserving each metadata item as one opaque row payload.

### Safer archive replacement

- Write each save to a uniquely named temporary archive in the destination
  directory, close it, and then replace the destination with `os.replace`.
- Preserve the previous complete archive when writing or replacement fails and
  clean up temporary files after failures.
- Bind a new Save As destination only after its archive has been written
  successfully.

This provides an atomic visibility boundary: a reader opening the destination
sees the previous complete archive or the new complete archive instead of a
partially written file. It does not provide file locking, multi-writer
coordination, or a universal power-loss durability guarantee.

### Short migration window

- Keep constructor `file_path=`, instance `load()`, and direct context-manager
  persistence for version 0.4 with `FutureWarning`. They will be removed in
  0.5.
- Make the deprecated context manager save only after a successful block. If
  the block raises, it does not save or suppress the exception.
- Keep a configuration-aware reader for older archives containing only
  `vectors` and `metadata`. Loading one warns, and its next save rewrites it as
  format version 1.
- Intentionally make `open()` reject an unversioned archive because that file
  cannot report its original dimensions or normalization semantics.
- Add a dedicated [persistence migration guide](MIGRATION.md) with side-by-side
  API replacements and one-time legacy archive conversion instructions.

The legacy API and unversioned archive reader are removed in 0.5. Applications
should migrate an old archive once with 0.4 or recreate it from source data;
indefinite compatibility with the incomplete two-array format is not planned.

### Runtime compatibility

- Support Python 3.11 through 3.14. Python 3.10 remains supported by the 0.3
  release series but is not supported by 0.4.
- Raise the minimum NumPy version from 1.21.3 to 1.23.2, the earliest release
  that supports Python 3.11.
- Exercise every supported Python version in CI and test NumPy 1.23.2 in a
  dedicated minimum-dependency job.

### Upgrade notes and boundaries

- Search, insertion, retrieval, clearing, normalization, and metadata behavior
  are unchanged from 0.3.2.
- Code using `VectorStore.open()`, `save(path)`, `save()`, and `reload()` is on
  the persistence API intended for 0.5.
- Code using a transitional entry point continues to work in 0.4 but emits a
  warning so the required 0.5 migration is visible during testing.
- Metadata still uses NumPy's pickle-backed object-array loading. Only open
  archives produced by your application or another trusted source.
- Mutable public state, repeated-add performance, deterministic tie ordering,
  and a formal thread-safety contract remain planned for later releases.

## 0.3.2 - 2026-07-27

This reliability and performance patch makes existing vector storage, search,
metadata, and persistence behavior safer and more predictable. It does not
intentionally break valid existing usage or change the `.npz` archive format.

### Numerical reliability

- Reject vectors, queries, and search thresholds that contain non-finite values
  or cannot remain finite when represented as `float32`. Invalid input now
  fails before it can corrupt stored state or ranking.
- Calculate norms and raw metric intermediates with `float64` where `float32`
  could overflow or underflow. Large and very small finite vectors can now be
  normalized and compared reliably.
- Allow zero vectors in stores created with `normalize=False`, where they are
  valid for dot-product and Euclidean search.
- Raise a clear error if a raw cosine search includes a zero vector, because
  cosine similarity is undefined for that row.
- Avoid duplicate full-size `float64` buffers when calculating raw Euclidean
  distance.

### Persistence

- Resolve a path without an `.npz` suffix to the same archive for both saving
  and loading. For example, `file_path="vectors"` consistently uses
  `vectors.npz`.
- Allow `load()` to be retried when the persistence file did not exist during
  an earlier attempt.
- Reset load state in `clear()` so a subsequent explicit `load()` can restore
  the saved rows.
- Keep repeated `load()` calls idempotent after a successful load.

### Metadata

- Preserve each item in the outer metadata sequence as one opaque row payload.
  Tuples and lists are no longer mistaken for extra NumPy array dimensions.
- Support dictionary, dataclass, tuple, list, string, integer, and other scalar
  payloads consistently through insertion, search results, saving, and loading.
- Continue rejecting explicitly multidimensional NumPy metadata arrays rather
  than silently flattening ambiguous input.

### Search memory use

- Search the stored vector matrix directly when `within_rows` is omitted,
  avoiding an unnecessary full-matrix copy on every unfiltered query.
- Preserve original store indexes and metadata when `within_rows` selects a
  filtered subset.
- Document that filtered searches allocate a temporary matrix proportional to
  the selected row count and vector dimensions.

### Compatibility and validation

- Test Python 3.10 through 3.14 in GitHub Actions.
- Test the minimum supported NumPy version in a dedicated Python 3.10 job.
- Raise the minimum NumPy requirement from 1.20 to 1.21.3 so it is compatible
  with the oldest supported Python version.
- Require linting, formatting, type checking, and the full Python test matrix
  before publishing to PyPI.

### Upgrade notes

- No public method signatures or persisted field names changed.
- Existing trusted `.npz` archives with `vectors` and `metadata` remain
  readable.
- Environments using NumPy 1.20 must upgrade to NumPy 1.21.3 or newer.
- Inputs that previously produced `nan`, `inf`, or unreliable rankings now
  raise `ValueError` instead.
