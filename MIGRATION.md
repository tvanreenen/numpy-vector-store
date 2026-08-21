# Persistence migration guide

Version 0.5 completes the persistence lifecycle introduced in 0.4. Creating a
store, opening an archive, saving, and refreshing from disk now have separate,
explicit operations. Applications that adopted the recommended 0.4 API need no
further persistence changes.

The 0.4 compatibility paths are no longer present: constructor `file_path=`,
instance `load()`, context-manager persistence, and the unversioned archive
reader have been removed.

## API at a glance

```python
store = VectorStore(dimensions=1536, normalize=True)
store.add(vectors, metadata)
store.save("vectors.npz")

store.save()  # Update the bound archive.

loaded = VectorStore.open("vectors.npz")
loaded.reload()  # Deliberately discard memory and reread the archive.
```

The persistence changes from 0.4 to 0.5 are:

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
| `vectors`, `metadata` | Writable owning arrays | Read-only inspection views |
| Vector returned by `get()` | View into live storage | Independent `float32` copy |

## Store-owned state

Most code that reads configuration or uses NumPy operations for inspection and
prefiltering remains unchanged. The familiar property names are still present:

```python
store.dimensions
store.normalize
store.file_path
store.vectors
store.metadata
```

The difference is ownership. Configuration cannot be assigned directly, and
the vector and metadata views reject normal row mutation. Their views describe
the rows present when the property was requested, so code should request a
fresh view after `add()`, `clear()`, or `reload()`.

The views are a supported inspection boundary, not tamper-proof snapshots.
Deliberately mutating their backing storage through `.base`, private attributes,
`ctypes`, or similar escape hatches is unsupported and can corrupt the store.
Call `.copy()` when code needs an independently mutable full-array snapshot.
For metadata, this copies the outer row array while preserving the opaque
payload objects by reference.

Row retrieval deliberately treats vectors and metadata differently:

```python
vector, payload = store.get(0)

vector[0] = 10.0           # Independent copy; the store is unchanged.
payload["reviewed"] = True  # Shared application metadata object.
```

Vectors have one uniform NumPy representation, so copying a single row gives
the caller clear ownership at bounded cost. Metadata can be any Python object,
so the store preserves payload identity instead of imposing a potentially
expensive or invalid deep-copy policy. Applications that need immutable
metadata can use frozen application objects or copy payloads themselves.

## Creating and saving a new store

Previously, the destination was supplied while constructing the store:

```python
store = VectorStore(dimensions=1536, file_path="vectors.npz")
store.add(vectors, metadata)
store.save()
```

Create the in-memory store first, then bind its destination with the first
save:

```python
store = VectorStore(dimensions=1536)
store.add(vectors, metadata)
store.save("vectors.npz")
```

Later `save()` calls update the bound archive. Supplying another path performs
a Save As operation and binds the new destination after the write succeeds.
Calling `save()` before a store is bound raises `ValueError` rather than
silently leaving the data unsaved.

## Opening an existing archive

The old API required callers to repeat configuration that should belong to the
archive:

```python
store = VectorStore(
    dimensions=1536,
    file_path="vectors.npz",
    normalize=True,
)
store.load()
```

Open a version 1 archive directly:

```python
store = VectorStore.open("vectors.npz")
```

`open()` restores `dimensions` and `normalize` from the archive, validates its
contents, loads its rows, and binds its path. Applications no longer need to
keep archive configuration separately or risk loading the same vectors with
different semantics.

The generic parameter still describes application metadata. It can be kept
when useful:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkMetadata:
    source: str
    chunk_index: int


store = VectorStore[ChunkMetadata].open("vectors.npz")
```

`ChunkMetadata` is an example application type, not a class provided by this
library.

## Refreshing from disk

Use `reload()` when another process may have changed the bound archive and the
current in-memory changes should be discarded:

```python
store = VectorStore.open("vectors.npz")

# Later, after the file may have changed:
store.reload()
```

`reload()` always attempts to read. It raises if the store is unbound, the file
is missing, or the archive is invalid. A failed reload leaves the current
in-memory vectors and metadata unchanged.

## Replacing context-manager persistence

The earlier context manager saved automatically on exit:

```python
with VectorStore(dimensions=1536, file_path="vectors.npz") as store:
    store.add(vectors, metadata)
```

Use an explicit save after the work succeeds:

```python
store = VectorStore(dimensions=1536)
store.add(vectors, metadata)
store.save("vectors.npz")
```

Normal Python control flow already prevents the final line from running if
`add()` raises. The persistence boundary is visible, and readers do not need to
remember an implicit exit side effect.

There is no replacement autosave context manager. An explicit `save()` keeps
the persistence boundary visible and lets the application decide whether work
completed successfully enough to persist.

## Migrating an archive created before 0.4

Older archives contain only `vectors` and `metadata`. They do not record their
dimensions or whether vectors use normalized or raw semantics, so `open()`
cannot construct a correct store from them.

Before upgrading to 0.5, use the 0.4 compatibility API once with the archive's
original configuration:

```python
legacy = VectorStore(
    dimensions=1536,
    file_path="legacy-vectors.npz",
    normalize=True,
)
legacy.load()
legacy.save()
```

This code emits transition warnings in 0.4 by design. The final `save()`
rewrites the archive as format version 1 with `format_version`, `dimensions`,
`normalize`, `vectors`, and `metadata`. It can then be opened by 0.5:

```python
store = VectorStore.open("legacy-vectors.npz")
```

Applications that can recreate archives from source vectors and metadata may
do that instead. NumPy Vector Store 0.5 cannot perform this conversion because
the unversioned file does not contain enough information to reconstruct its
configuration safely.

## Removal schedule

| Transitional behavior | 0.4 | 0.5 |
|---|---|---|
| Constructor `file_path=` | Works with `FutureWarning` | Removed |
| Instance `load()` | Works with `FutureWarning` | Removed |
| Direct context-manager persistence | Saves only on successful exit and warns | Removed |
| Unversioned two-array archives | Load with known configuration and warn | Reader removed |
| `open()`, `save(path)`, `save()`, and `reload()` | Preferred | Supported |

## Persistence boundaries that do not change

Metadata is stored in a pickle-backed NumPy object array. Archives remain
trusted input and must not be opened from untrusted or unverifiable sources.

Saves use same-directory temporary files and atomic replacement, but the
library does not add file locking, coordinate concurrent writers, or promise
power-loss durability across every operating system and filesystem.
