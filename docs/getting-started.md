---
title: Getting started
description: Create a typed vector store, add embeddings and metadata, and read search results with NumPy Vector Store 0.7.0.
---

# Getting started

Install with `uv add numpy-vector-store` or
`python -m pip install numpy-vector-store`. The only runtime dependency is NumPy.
Python 3.11–3.14 are tested, and the minimum NumPy version is 1.23.2.

## Bring your own embeddings

The library takes vectors you already have. Generate document and query
embeddings with the same model and use its output width as `dimensions`.
The store does not call an embedding API, split documents, or download models.

`add()` accepts a two-dimensional real-valued array-like input with shape
`(rows, dimensions)`. Even one vector needs an outer row dimension:

```python
from numpy_vector_store import VectorStore

store = VectorStore[str](dimensions=3)
store.add([[1.0, 0.0, 0.0]], ["one row"])
```

Vectors become `float32`. Values must remain finite after conversion; NaN,
infinity, overflow to infinity, and complex inputs are rejected. With the
default `normalize=True`, zero vectors are also rejected.

## Keep metadata in your application

Supply one metadata payload per vector row. A payload can be a dict, dataclass,
tuple, list, string, integer ID, or another Python object. A tuple or list payload
remains one opaque row value. A NumPy metadata array must be one-dimensional.

`VectorStore[TMetadata]` carries your payload type through `get()` and search
results. The type parameter does not validate or transform metadata at runtime.
Use application identifiers as payloads if you keep the full records elsewhere.
Keep any external metadata table aligned with the store's row order.

## Search and read results

The [complete quick start](index.md#your-first-search) is runnable as written.
A search query is **one-dimensional**, with exactly `dimensions` real values.

All three metric methods return `list[VectorHit[TMetadata]]`:

| Field | Meaning |
| --- | --- |
| `index` | Original zero-based store row index |
| `value` | Cosine similarity, dot product, or Euclidean distance |
| `metadata` | Your payload for that row |

`top_k` is a positive integer and defaults to 10. You may receive fewer hits
when fewer rows qualify, or an empty list when no rows qualify. Search validates
its arguments even when the store or selected row set is empty.

Use [`get(index)`](reference.md#numpy_vector_store.VectorStore.get) to retrieve
one row without searching. It returns `(vector_copy, metadata)` or `None` for
an integer outside the current row range. `len(store)` gives the active row count.

## Grow or clear a store

Repeated `add()` calls append in insertion order and reuse private capacity.
Batch your inputs when they already arrive together to reduce per-call overhead.
`clear()` removes all active rows and releases the store's reserved capacity;
it retains the dimensions, normalization mode, and archive binding.

Mutation does not save automatically. Continue with [search and filtering](search.md),
[persistence](persistence.md), or the [state ownership rules](ownership.md).
