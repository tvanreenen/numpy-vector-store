---
title: Search and filtering
description: Choose cosine, dot product, or Euclidean distance; understand normalization, exact ties, thresholds, and row prefiltering.
---

# Search and filtering

Every search computes a metric for every selected row. Results are exact over
the stored `float32` representation, subject to floating-point arithmetic.
There is no approximate nearest-neighbor index.

## Choose a metric

| Method | Ranking | Inclusive threshold | Typical use |
| --- | --- | --- | --- |
| `cosine_search` | Larger similarity first | `min_value` | Direction-based embedding similarity |
| `dot_search` | Larger dot product first | `min_value` | Magnitude-sensitive scoring with raw vectors |
| `euclidean_search` | Smaller distance first | `max_value` | Nearest coordinates or features with raw vectors |

If your embedding model recommends cosine similarity, keep the default
`normalize=True` and use `cosine_search()`. Choose another metric or raw storage
when the model or your application's geometry requires it.

All methods accept `query`, keyword-only `top_k=10`, and `within_rows=None`.
A threshold removes rows whose computed values do not qualify, before choosing
the top results. It does not avoid computing the metric for selected rows.

## Normalization changes the meaning

`normalize=True` is the default. Insertion scales each vector to unit length;
`[3.0, 4.0]` becomes approximately `[0.6, 0.8]`. The magnitude is discarded.
Queries are normalized too. Dot product then has effectively the same meaning
as cosine similarity, and Euclidean distance measures distance between directions.

With `normalize=False`, stored vectors retain their converted `float32` values.
Dot product and Euclidean distance use their original magnitudes. Cosine still
compares directions, computing row norms during the search.

```python
--8<-- "docs/examples/metrics.py"
```

```text
cosine: ['near', 'strong']
dot: ['strong', 'near']
distance: ['near', 'other']
```

### Zero vectors and numerical inputs

Normalized stores reject zero vectors. Raw stores allow them for dot product
and Euclidean distance. Raw cosine search raises `ValueError` if a selected
stored row is zero; exclude those rows with `within_rows` when needed.

Cosine always requires a nonzero query. Dot product and Euclidean distance
require a nonzero query only in normalized stores. These query rules also apply
to empty stores and empty filters. Inputs and thresholds must be finite.
Norms and raw metric intermediates use `float64` where needed for numerical range.

## Deterministic ranking and ties

Exactly equal computed metric values are ordered by ascending **original store
row index**. This also chooses which tied rows cross the `top_k` boundary.
Close but unequal values keep their metric order.

The cosine example above gives the two collinear rows equal values. Row 0
(`near`) precedes row 1 (`strong`). Reversing those rows in `within_rows` would
not change the tie break. With the default normalization, you should not depend
on approximate unit-length arithmetic producing bit-for-bit equal values for
every mathematically equivalent expression.

## Filter candidates before searching

The store has no metadata query language. Evaluate your application's predicate
first and supply the resulting row indexes:

```python
--8<-- "docs/examples/filtering.py"
```

```text
[(0, 'Install'), (2, 'Deploy')]
```

`within_rows` must be a one-dimensional sequence or NumPy array of unique
integer row indexes. A Boolean mask is not a row selector; convert one with
`np.flatnonzero(mask)`. An empty sequence returns no hits after validation.
Returned `hit.index` values always refer to the original store.

Unfiltered search uses the stored vector matrix directly. Filtering gathers a
temporary matrix proportional to the selected rows and dimensions. Omit
`within_rows` when you mean all rows; explicitly passing every row adds a full
matrix copy. Keep row indexes current after `clear()` or `reload()`.

### NumPy predicates and an external metadata table

Keep structured records outside the store when NumPy predicates suit your
application. Store the record indexes as opaque payloads, then turn a Boolean
mask into candidate row indexes. This example keeps record order aligned with
store insertion order:

```python
--8<-- "docs/examples/numpy_filtering.py"
```

```text
0 intro
```

## Input errors

| Input | Accepted | Invalid input |
| --- | --- | --- |
| `dimensions`, `top_k`, `get(index)` | Python and NumPy integers, or integer-index objects | Booleans and non-integers raise `TypeError`; nonpositive dimensions and top-k raise `ValueError` |
| `normalize` | Python or NumPy Boolean | Other types raise `TypeError` |
| Threshold | Finite Python or NumPy real scalar | Booleans, strings, complex values, or arrays raise `TypeError`; non-finite values raise `ValueError` |
| `within_rows` | Unique integer indexes in range | Wrong shape or duplicates raise `ValueError`; non-integers or booleans raise `TypeError`; out-of-range indexes raise `IndexError` |

A Python selector such as `[0, True]` is rejected before NumPy can convert the
Boolean into an integer. Exact exception messages are not stable identifiers.
See the [API reference](reference.md) for signatures.
