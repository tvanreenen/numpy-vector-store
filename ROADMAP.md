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

The persistence format currently stores only vectors and metadata. That format
is compact, but callers must separately remember dimensions and normalization
mode. A safer format should be self-describing and able to reject incompatible
configuration instead of silently changing vector semantics. Version 0.4 will
introduce that safer format and provide one deliberately short migration window
for existing archives.

### Versioned archive contract

Every archive written by 0.4 will use format version 1 and contain five named
values:

- `format_version`: a scalar integer identifying version 1 of the archive
  contract. This version is independent of the package version.
- `dimensions`: a positive scalar integer matching the width of `vectors`.
- `normalize`: a scalar boolean recording whether stored vectors use normalized
  or raw semantics.
- `vectors`: a two-dimensional `float32` array.
- `metadata`: a one-dimensional object array with one payload per vector row.

Opening an archive will validate the complete schema before changing live store
state. Missing fields, unsupported format versions, invalid configuration, and
inconsistent array shapes will fail clearly instead of being guessed at.

Object metadata continues to rely on NumPy's pickle-backed object-array
loading. Persistence therefore remains a trusted-file feature: users must not
open archives from untrusted or unverifiable sources.

### Existing archive migration

Archives created before 0.4 contain only `vectors` and `metadata`, so they
cannot recover their original dimensions and normalization mode by themselves.
Version 0.4 will keep a temporary reader for these archives through the legacy
configuration-aware API. Opening one will emit a `FutureWarning`, and its next
save will rewrite it in the version 1 format.

This legacy reader will be removed in 0.5. Users who need an old archive after
upgrading should open and save it once with 0.4, or recreate it from its source
data. The project intentionally does not promise indefinite compatibility for
the incomplete two-array format.

### Preferred persistence API

Version 0.4 will introduce the lifecycle intended to become the only
persistence API in 0.5:

```python
store = VectorStore(dimensions=1536, normalize=True)
store.add(vectors, metadata)
store.save("vectors.npz")

store = VectorStore.open("vectors.npz")
store.reload()
store.save()
```

The constructor creates a new empty store and, in the final API, performs no
disk I/O. `VectorStore.open(path)` creates a store from the archive's own
configuration and binds that path. `save(path)` performs the first save or a
Save As operation and binds the supplied path; later `save()` calls update that
bound archive. `reload()` strictly refreshes a bound store from disk and leaves
the current in-memory state unchanged if reading or validation fails.

The generic parameter on `VectorStore` describes the application's metadata
payload type, not a built-in document abstraction. Examples that benefit from
an explicit type will use a descriptive application type such as
`ChunkMetadata`; otherwise they will omit the generic annotation.

### Short API compatibility window

The preferred API will coexist in 0.4 with three old entry points: constructor
`file_path=`, instance `load()`, and direct context-manager use. Each will emit
a `FutureWarning` in 0.4 and be removed in 0.5. This single-release bridge is
intended to make migration obvious without carrying two persistence models
long term.

While the deprecated context manager remains, it will save only after normal
completion. If the managed block raises, the store will not save and the
exception will propagate unchanged. No replacement autosave context manager is
planned; explicit `save()` calls make the persistence boundary easier to see
and reason about.

### Atomic save boundary

Saving will write and close a temporary archive in the destination directory
before replacing the destination with `os.replace`. Readers should therefore
see either the previous complete archive or the new complete archive, and a
failed write should leave the previous destination intact. Temporary files
will be cleaned up after failures.

This is an atomic visibility guarantee, not a concurrency system. Version 0.4
will not add file locking, coordinate multiple writers, or promise survival of
every hardware or operating-system failure before data reaches durable
storage.

## 0.5.0: State safety and ingestion

The current public NumPy arrays make inspection convenient, but they also let
callers mutate normalized vectors and break internal search assumptions.
Repeated small additions also copy all previously stored data.

Planned direction:

- Remove the unversioned two-array archive reader after its 0.4 migration
  window.
- Remove constructor `file_path=`, instance `load()`, and direct
  context-manager persistence after their 0.4 warning window.
- Keep vector storage behind private state.
- Expose read-only views or snapshots for inspection.
- Make row retrieval safe from accidental mutation.
- Improve ingestion for repeated additions without penalizing batch insertion.
- Define deterministic ordering for equal search values.
- Document thread-safety guarantees and limitations.

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
