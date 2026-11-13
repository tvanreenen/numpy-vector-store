---
title: State, views, and threading
description: Understand read-only NumPy inspection views, vector copies, shared metadata, and application-owned synchronization.
---

# State, views, and threading {#state-and-threading}

The store owns its configuration and row structure. Your application owns the
meaning and mutability of its metadata payloads and coordinates concurrent use.

## Inspection and retrieval

`dimensions`, `normalize`, and `file_path` are read-only properties. Configure
dimensions and normalization in the constructor; use `open(path)` or `save(path)`
to establish a binding.

`vectors` and `metadata` return zero-copy, non-writeable NumPy views of active
rows. Ordinary assignment and enabling writes are rejected. Request fresh views
after `add()`, `clear()`, or `reload()`; a view describes rows present when it
was requested. It may keep an old buffer alive after the store releases it.

These are inspection views, not tamper-proof snapshots. Deliberately modifying
backing memory via `.base`, private attributes, or `ctypes` is unsupported.
Call `.copy()` when you need a mutable independent row array. A metadata array
copy still shares the payload objects inside it.

`get()` returns an independent `float32` vector copy and the original metadata
payload. Search hits share that metadata too:

```python
--8<-- "docs/examples/ownership.py"
```

```text
1.0
Reviewed
None
```

Use frozen application objects or copy metadata at your own boundary when you
need immutable payloads. The library does not deep-copy arbitrary Python objects.

## Concurrent reads

There are no internal locks. Search, `get()`, and inspection can run concurrently
on one instance **only while its state and shared metadata payloads are unchanged**.

If any thread may call `add()`, `clear()`, `reload()`, or `save()`, or mutate a
shared metadata payload, synchronize **every access** to that instance with the
same application lock or an equivalent mechanism. Locking only the writer is
insufficient. This also includes inspection and search while another thread saves.

A save must not overlap in-memory mutation. Atomic archive replacement protects
visibility at the destination path, not the consistency of arrays read from a
changing store. Separate instances or processes writing one destination also
need external writer coordination; the last successful replacement can win.

These ownership rules are part of the [public compatibility contract](COMPATIBILITY.md).

## Independent workers and replicas

Multiple processes can each open the same trusted archive and search their own
in-memory store. The library does not synchronize these stores: `add()`,
`clear()`, and metadata changes in one process do not change another process's
state. Read-mostly replicas do not need a shared writable archive. See
[shipping a store with your application](deployment.md) for an image-based
deployment pattern and update boundaries.
