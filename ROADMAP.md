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

## 0.3.2: Reliability and performance

Status: in progress

This patch release focuses on cases where the current API can silently produce
incorrect results, fail to reload data it saved, reject documented metadata
payloads, or allocate much more memory than an exact search requires.

Planned changes:

- Validate vectors and queries for finite numeric values.
- Normalize large finite vectors without overflowing intermediate `float32`
  calculations.
- Permit zero vectors in raw stores used for dot-product or Euclidean search,
  while keeping cosine similarity's undefined zero-vector behavior explicit.
- Make extensionless persistence paths save and load the same `.npz` file.
- Allow loading to be retried when the persistence file was initially absent.
- Preserve tuple, list, dataclass, scalar, and dictionary metadata as individual
  opaque row payloads.
- Avoid copying the complete vector matrix during an unfiltered search.
- Add continuous validation across every supported Python version and require
  equivalent checks before publishing.

These changes are intended to preserve results and behavior for valid existing
usage. Public state access, the persistence format, and context-manager
semantics will not change in this patch release.

## 0.4.0: Persistence and lifecycle

The persistence format currently stores only vectors and metadata. That format
is compact, but callers must separately remember dimensions and normalization
mode. A safer format should be self-describing and able to reject incompatible
configuration instead of silently changing vector semantics.

Planned direction:

- Introduce a versioned archive containing its dimensions and normalization
  mode.
- Continue reading the existing two-array archive format.
- Write archives through a temporary file and replace the destination
  atomically.
- Define the difference between initial loading and explicit reloading.
- Define whether context-manager exit saves after an exception.

## 0.5.0: State safety and ingestion

The current public NumPy arrays make inspection convenient, but they also let
callers mutate normalized vectors and break internal search assumptions.
Repeated small additions also copy all previously stored data.

Planned direction:

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
