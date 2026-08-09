# Persistence migration guide

Version 0.4 introduces a persistence lifecycle that separates creating a new
store, opening an existing archive, saving, and deliberately refreshing from
disk. It also provides a one-release bridge for applications and archives using
the earlier API.

The transition is intentionally short. Version 0.4 emits `FutureWarning` for
constructor `file_path=`, instance `load()`, direct context-manager use, and
unversioned archives. Version 0.5 removes those entry points and the
unversioned archive reader.

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

Unlike transitional `load()`, `reload()` always attempts to read. It raises if
the store is unbound, the file is missing, or the archive is invalid. A failed
reload leaves the current in-memory vectors and metadata unchanged.

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

During 0.4, the deprecated context manager saves only after normal completion.
It does not save while an exception is propagating and does not suppress the
exception. There is no planned replacement autosave context manager.

## Migrating an archive created before 0.4

Older archives contain only `vectors` and `metadata`. They do not record their
dimensions or whether vectors use normalized or raw semantics, so `open()`
cannot construct a correct store from them.

Use the 0.4 compatibility API once with the archive's original configuration:

```python
legacy = VectorStore(
    dimensions=1536,
    file_path="legacy-vectors.npz",
    normalize=True,
)
legacy.load()
legacy.save()
```

This code emits transition warnings by design. The final `save()` rewrites the
archive as format version 1 with `format_version`, `dimensions`, `normalize`,
`vectors`, and `metadata`. It can then use the preferred API:

```python
store = VectorStore.open("legacy-vectors.npz")
```

Applications that can recreate archives from source vectors and metadata may
choose to do that instead. The unversioned reader is removed in 0.5 rather than
maintained as a long-term compatibility format.

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
