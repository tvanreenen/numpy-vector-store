---
title: Vector search, without the infrastructure.
description: Exact, in-process vector search powered by NumPy. Get started with a small typed Python API and your own embeddings.
---

# Vector search, without the infrastructure.

<p class="intro">Exact, in-process vector search powered by NumPy.</p>

Keep embeddings and metadata in your Python process. Bring your own model;
search by cosine similarity, dot product, or Euclidean distance.

Made for small-to-medium collections in memory. Every search scores every
selected row, with no approximate index or background service.

## Install

=== "uv"

    ```sh
    uv add numpy-vector-store
    ```

=== "pip"

    ```sh
    python -m pip install numpy-vector-store
    ```

## Your first search

Use small vectors to see the complete workflow. Replace them with embeddings
from your model when you integrate the library.

--8<-- "README.md:quickstart"

```text
x-axis: 0.994
y-axis: 0.110
```

Each result is a `VectorHit` with the original row `index`, the metric `value`,
and your `metadata` payload. Vectors are stored as `float32` and normalized by
default. Choose `normalize=False` when magnitude matters.

## Keep going

| Task | Where to go |
| --- | --- |
| Work with embeddings and typed metadata | [Getting started](getting-started.md) |
| Choose a metric or restrict the candidate rows | [Search and filtering](search.md) |
| Save, open, or refresh a local archive | [Persistence](persistence.md) |
| Look up a signature | [API reference](reference.md) |
| Size a collection for your workload | [Performance and practical limits](performance.md) |

The package is pre-1.0. Version 0.7 publishes the contract intended for 1.0;
it does not yet make a 1.x stability commitment. Read the
[compatibility policy](COMPATIBILITY.md) before adopting it.
