# Changelog

This changelog records user-visible changes to NumPy Vector Store. Earlier
release notes remain available on the
[GitHub releases page](https://github.com/tvanreenen/numpy-vector-store/releases).

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
