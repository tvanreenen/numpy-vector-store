---
title: API reference
description: Source-backed reference for the supported VectorStore and VectorHit API in NumPy Vector Store 0.7.0.
---

# API reference

Import from the top-level package:

```python
from numpy_vector_store import VectorHit, VectorStore, __version__
```

These signatures and docstrings are rendered from the package source. Only the
supported public members are included. `TMetadata` is your application payload
type; `ArrayLike` refers to NumPy-compatible array input.

The source aliases `_PathInput` and `_RealScalar` appear in signatures.
`_PathInput` means `str | os.PathLike[str]`. `_RealScalar` means a Python or
NumPy integer or floating scalar; booleans are rejected. They describe accepted
inputs and are not additional supported package exports.

Read [search and filtering](search.md), [persistence](persistence.md), and
[state and threading](ownership.md) for the behavior behind these signatures.

::: numpy_vector_store.VectorStore
    options:
      heading_level: 2
      merge_init_into_class: true
      members:
        - __init__
        - dimensions
        - normalize
        - file_path
        - vectors
        - metadata
        - open
        - add
        - reload
        - save
        - cosine_search
        - dot_search
        - euclidean_search
        - get
        - clear
        - __len__

::: numpy_vector_store.VectorHit
    options:
      heading_level: 2
      members: [index, value, metadata]

A frozen result record. Fields cannot be reassigned, but a mutable metadata
payload is still shared with the application. Equality and hashability follow
the field values; a hit containing a dictionary is unhashable. Use `hit.index`
or an application identifier when you need a stable key.

## Package version

`__version__` is the installed version string. This documentation edition
describes **0.7.0**.
