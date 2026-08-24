# Public API and compatibility policy

NumPy Vector Store 0.7 publishes the contract intended for 1.0 so applications
can evaluate it before the project makes a long-term stability commitment. The
package remains pre-1.0 until the 0.7 API has completed a normal release cycle.

This policy separates the small supported interface from implementation details
that may continue to improve during 1.x.

## Supported public API

Import supported names from the top-level package:

```python
from numpy_vector_store import VectorHit, VectorStore, __version__
```

These are the package-level public names declared by `numpy_vector_store.__all__`:

- `VectorStore`, including its constructor, `len(store)`, and the documented
  methods and properties below.
- `VectorHit`, including its read-only `index`, `value`, and `metadata` fields.
- `__version__`, the installed package version as a string.

The supported `VectorStore` surface is:

| Area | Public interface |
|---|---|
| Create | `VectorStore(dimensions, *, normalize=True)` |
| Inspect configuration | `dimensions`, `normalize`, `file_path` |
| Inspect rows | `vectors`, `metadata`, `len(store)` |
| Change rows | `add(vectors, metadata)`, `clear()` |
| Retrieve one row | `get(index)` |
| Search | `cosine_search(...)`, `dot_search(...)`, `euclidean_search(...)` |
| Persist | `VectorStore.open(path)`, `save(path)`, `save()`, `reload()` |

The public contract includes documented signatures, accepted input categories,
return types, public type annotations, state and ownership behavior, search
ordering, and exception classes. The `VectorStore` generic parameter describes
its metadata payload type and flows through row retrieval and search results.
The contract also includes the documented `float32` vector representation and
format-version-1 archive schema. Exact exception messages are explanatory text,
not identifiers for application logic.

Importing from `numpy_vector_store.vector_store` may work, but the submodule
layout is an implementation detail. Applications should use top-level imports.
Names beginning with a single underscore, private object state, spare-capacity
layout, and internal calculation steps are not public API.

`VectorStore` is not designed as a subclass extension framework. Subclassing is
not prohibited, but overriding methods or depending on private helpers and state
is outside the compatibility promise. Prefer composition when application code
needs behavior around a store.

## `VectorHit` and opaque metadata

`VectorHit` is an immutable result record. Its fields cannot be reassigned, but
the `metadata` value remains the same opaque application object held by the
store. A mutable metadata payload can still be changed through that object.

Equality and hashability follow the values stored in the three fields. There is
no package-specific metadata comparison or hashing policy. In particular:

- A hit containing a dictionary or another unhashable payload is unhashable.
- A metadata type with non-Boolean equality, such as a NumPy array, can make
  whole-hit equality return a non-Boolean value or raise when converted to a
  truth value.
- A metadata type with application-defined equality keeps those semantics.

Use `hit.index` or an immutable application identifier inside `hit.metadata`
when code needs a stable key. Do not assume every possible metadata payload
makes the complete `VectorHit` suitable for a set, dictionary key, or Boolean
equality assertion.

## Semantic versioning after 1.0

Starting with 1.0, release numbers follow
[Semantic Versioning 2.0.0](https://semver.org/) and communicate changes to the
documented public API:

- Patch releases contain backward-compatible defect fixes, documentation
  corrections, and internal improvements. A patch may correct behavior that
  contradicts the documented contract, and the release notes will identify the
  affected case.
- Minor releases add backward-compatible functionality and may deprecate public
  API. They do not remove or incompatibly change documented public API.
- Major releases may remove deprecated API or make other incompatible public
  contract changes.

A public API deprecation will appear in the documentation and changelog and
will emit a warning where a runtime warning is practical. Removal waits for a
later major release and will have at least one minor release of prior
deprecation.

Exact error wording, object representations, private names, submodule layout,
internal algorithms, allocation strategy, benchmark timings, and undocumented
side effects may change in any release. Changes to these details must still
preserve the documented results and ownership rules.

## Archive compatibility

Archive format versions are independent of package versions. Format version 1
is the stable persistence contract for 1.x:

- Every 1.x release will continue reading and writing valid format-version-1
  archives.
- A mandatory replacement for format version 1 requires a major package
  release and a documented migration path.
- A new archive format may appear during 1.x only as an explicit opt-in while
  format-version-1 read and write support remains available.
- An older reader is not expected to understand an arbitrary newer archive
  format. Compatibility is a newer-reader promise, not a future-format promise
  for already published code.

Unversioned archives containing only `vectors` and `metadata` remain outside
this contract because they do not record dimensions or normalization semantics.

Format compatibility does not make every Python metadata object portable.
Opaque metadata uses pickle-backed NumPy object arrays and may depend on the
same application classes, import paths, and compatible dependencies being
available when it is opened. Archives must come from a trusted source. The
library does not treat untrusted pickle data as safe input.

## Python and NumPy support

The Python versions listed in package metadata are tested in CI. The declared
minimum NumPy version is tested with the oldest supported Python version.

Support for a Python or NumPy version may be removed in a minor release after
its relevant upstream support window or ecosystem compatibility makes continued
testing impractical. Such a change will be made in package metadata and called
out in the changelog and release notes. Dropping a runtime version does not
remove callable API from applications running on a supported interpreter.

## Performance expectations

Search results, metric definitions, threshold behavior, deterministic tie
ordering, and filtered-row semantics are public behavior. The implementation
may change how it produces those results.

Published benchmark measurements are reference evidence, not latency or memory
guarantees. Hardware, NumPy, BLAS, process load, vector shape, and metadata all
affect observed performance. Structural regressions such as repeated full-store
copies during ordinary ingestion or unnecessary unfiltered search copies remain
covered by tests rather than fixed wall-clock promises.
