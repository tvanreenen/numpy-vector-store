# Changelog

This changelog records user-visible changes to NumPy Vector Store. Earlier
release notes remain available on the
[GitHub releases page](https://github.com/tvanreenen/numpy-vector-store/releases).

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
