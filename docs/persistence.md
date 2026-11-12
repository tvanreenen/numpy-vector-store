---
title: Persistence
description: Explicitly save, open, and reload trusted NumPy vector archives with a stable path binding.
---

# Persistence

Create a store in memory, then save it explicitly. Persistence uses a versioned,
self-describing `.npz` archive containing vectors, opaque metadata, dimensions,
and the normalization setting.

!!! warning "Open trusted archives only"
    Metadata uses NumPy's pickle-backed object arrays. Loading a malicious
    archive can execute code. Only open archives produced by your application
    or another trusted, verifiable source. Schema validation does not make
    untrusted pickle data safe.

## Save, open, and refresh

```python
--8<-- "docs/examples/persistence.py"
```

```text
2 2 True
['second', 'first']
```

| Operation | Effect |
| --- | --- |
| `VectorStore(dimensions, normalize=...)` | Create an unbound in-memory store |
| `save(path)` | Write the current state, then bind that destination |
| `save()` | Update the bound destination; raise `ValueError` if unbound |
| `VectorStore.open(path)` | Construct a store using the archive's configuration and rows |
| `reload()` | Reread the bound archive, discarding unsaved memory changes |

Passing a new destination to `save(path)` performs Save As. A failed write keeps
the old binding. `reload()` always attempts to read; an unbound store, missing
file, or invalid archive raises. Failed reloads leave current rows unchanged.

## Paths stay bound

Paths accept strings and string-valued `os.PathLike` objects such as
`pathlib.Path`. A path without the `.npz` suffix gets that suffix appended.
For example, `"vectors"` means `"vectors.npz"`.

A relative path is anchored to the current working directory when `save(path)`
or `open(path)` binds it. `file_path` returns that absolute path. Changing the
working directory later does not redirect `save()` or `reload()`.
Anchoring does not canonicalize symlinks or `..` components.

Empty paths raise `ValueError`; bytes and other non-path values raise
`TypeError`. `None` is meaningful only for `save()`, where it reuses the binding.

## Archive and failure boundaries

The current archive format is **version 1**, independently of package version
0.7.0. Version-1 archives from 0.4 remain readable. The complete field set is
`format_version`, `dimensions`, `normalize`, `vectors`, and `metadata`.
Unsupported versions and malformed schemas are rejected.

Metadata objects may need the same application classes, import paths, and
compatible dependencies to be importable on the machine that opens the archive.
An archive containing arbitrary Python metadata is not a language-neutral data
exchange format.

Saves write a same-directory temporary file, close it, and atomically replace
the destination. A reader sees the previous or new complete archive. This
visibility guarantee does not provide a consistent snapshot of concurrent
in-memory mutation, file locking, multi-writer coordination, or a universal
power-loss durability guarantee. Coordinate writers in your application.

Filesystem `OSError` subclasses and NumPy, pickle, or application deserialization
exceptions retain their original types. Schema errors use `ValueError`.

For concurrency rules, read [state and threading](ownership.md). For constructor
`file_path=`, `load()`, context managers, or archives from before 0.4, follow the
[migration guide](MIGRATION.md). Those older entry points are removed in 0.7.
