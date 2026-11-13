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

## Is this a fit?

Use it when a collection fits comfortably in your Python process and exact
search meets your latency needs. Your application supplies the embeddings and
metadata; the store adds search and explicit local persistence without a service
to operate.

Choose an indexed or service-backed system for collections that routinely reach
millions of vectors, approximate search, or shared access across machines or
processes. This library does not provide transactions, coordinated writers, or
a metadata query engine. See [practical limits](performance.md) for memory costs
and reference measurements before sizing your collection.

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
| Understand vector shapes, metadata, and results | [Getting started](getting-started.md) |
| Choose a metric or restrict the candidate rows | [Search and filtering](search.md) |
| Save, open, or refresh a local archive | [Persistence](persistence.md) |
| Work with read-only views and concurrent access | [State, views, and threading](ownership.md) |
| Look up a signature | [API reference](reference.md) |
| Size a collection for your workload | [Performance and practical limits](performance.md) |
| Measure search or ingestion on your hardware | [Run benchmarks](benchmarks/README.md) |
| Upgrade from an older persistence API | [Migration guide](MIGRATION.md) |
| Review the supported API and stability commitment | [Compatibility policy](COMPATIBILITY.md) |

The package is pre-1.0. Version 0.7 publishes the contract intended for 1.0;
it does not yet make a 1.x stability commitment. Read the
[compatibility policy](COMPATIBILITY.md) before adopting it.
